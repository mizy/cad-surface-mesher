from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image

from .constants import DEFAULT_TARGET_LENGTH_M
from .mesh_io import read_mesh
from .normalization import normalize_vehicle_mesh
from .types import NormalizedVehicle, SideArtifacts
from .utils import resolve_path


def triangle_faces(mesh: pv.PolyData) -> np.ndarray:
    tri = mesh.triangulate()
    faces = np.asarray(tri.faces)
    if faces.size == 0 or faces.size % 4 != 0:
        raise ValueError("Triangulated mesh has no packed triangle faces")
    packed = faces.reshape(-1, 4)
    if not np.all(packed[:, 0] == 3):
        raise ValueError("Mesh triangulation did not produce triangles")
    return packed[:, 1:].astype(np.int64, copy=False)


def oriented_triangle_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = points[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid] = 0.0

    center = points.mean(axis=0)
    centroids = triangles.mean(axis=1)
    inward = np.einsum("ij,ij->i", normals, centroids - center) < 0.0
    normals[inward] *= -1.0
    return normals


def rasterize_side(mesh: pv.PolyData, *, width: int, height: int) -> SideArtifacts:
    points = np.asarray(mesh.points, dtype=np.float64)
    faces = triangle_faces(mesh)
    normals = oriented_triangle_normals(points, faces)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_span = max(x_max - x_min, 1e-12)
    z_span = max(z_max - z_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)

    margin = 6.0
    scale = min((width - 2.0 * margin) / x_span, (height - 2.0 * margin) / z_span)
    px = (x - x_min) * scale + margin
    pz = (height - 1.0) - ((z - z_min) * scale + margin)
    py = (y - y_min) / y_span

    depth = np.ones((height, width), dtype=np.float32) * np.inf
    normal = np.zeros((height, width, 3), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)

    for face, face_normal in zip(faces, normals):
        verts = np.column_stack((px[face], pz[face], py[face]))
        min_col = max(0, int(math.floor(float(verts[:, 0].min()))))
        max_col = min(width - 1, int(math.ceil(float(verts[:, 0].max()))))
        min_row = max(0, int(math.floor(float(verts[:, 1].min()))))
        max_row = min(height - 1, int(math.ceil(float(verts[:, 1].max()))))
        if min_col > max_col or min_row > max_row:
            continue

        x0, y0 = verts[0, 0], verts[0, 1]
        x1, y1 = verts[1, 0], verts[1, 1]
        x2, y2 = verts[2, 0], verts[2, 1]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denom)) < 1e-8:
            continue

        cols = np.arange(min_col, max_col + 1, dtype=np.float32) + 0.5
        rows = np.arange(min_row, max_row + 1, dtype=np.float32) + 0.5
        grid_x, grid_y = np.meshgrid(cols, rows)
        w0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denom
        w1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if not np.any(inside):
            continue

        tri_depth = w0 * verts[0, 2] + w1 * verts[1, 2] + w2 * verts[2, 2]
        current = depth[min_row : max_row + 1, min_col : max_col + 1]
        update = inside & (tri_depth < current)
        if not np.any(update):
            continue

        current[update] = tri_depth[update]
        mask_window = mask[min_row : max_row + 1, min_col : max_col + 1]
        normal_window = normal[min_row : max_row + 1, min_col : max_col + 1]
        mask_window[update] = True
        normal_window[update] = face_normal.astype(np.float32)

    depth_out = np.zeros((height, width), dtype=np.float32)
    depth_out[mask] = np.clip(depth[mask], 0.0, 1.0)
    return SideArtifacts(mask=mask, depth=depth_out, normal=normal)


def render_mesh_artifacts(
    mesh_path: Path,
    *,
    width: int,
    height: int,
    rotate_180_z: bool = False,
    target_length: float = DEFAULT_TARGET_LENGTH_M,
    force_target_length: bool = False,
    target_dimensions: tuple[float, float, float] | None = None,
) -> tuple[SideArtifacts, NormalizedVehicle]:
    mesh = read_mesh(mesh_path)
    normalized = normalize_vehicle_mesh(
        mesh,
        rotate_180_z=rotate_180_z,
        target_length=target_length,
        force_target_length=force_target_length,
        target_dimensions=target_dimensions,
    )
    return rasterize_side(normalized.mesh, width=width, height=height), normalized


def save_artifacts(artifacts: SideArtifacts, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_path = output_dir / f"{stem}_side_depth.png"
    normal_path = output_dir / f"{stem}_side_normal.png"

    depth_u8 = (np.clip(artifacts.depth, 0.0, 1.0) * 255).astype(np.uint8)
    depth_rgba = np.zeros((*depth_u8.shape, 4), dtype=np.uint8)
    depth_rgba[..., 0:3] = depth_u8[..., None]
    depth_rgba[..., 3] = artifacts.mask.astype(np.uint8) * 255
    Image.fromarray(depth_rgba, mode="RGBA").save(depth_path)

    normal_rgb = ((np.clip(artifacts.normal, -1.0, 1.0) * 0.5 + 0.5) * 255).astype(np.uint8)
    normal_rgb[~artifacts.mask] = 0
    Image.fromarray(normal_rgb, mode="RGB").save(normal_path)

    return {
        "depth": depth_path.name,
        "normal": normal_path.name,
    }


def load_artifacts(base_dir: Path, artifact_paths: dict[str, str]) -> SideArtifacts:
    depth_image = Image.open(resolve_path(artifact_paths["depth"], base=base_dir))
    if "A" in depth_image.getbands():
        depth_rgba = np.asarray(depth_image.convert("RGBA"))
        mask = depth_rgba[..., 3] >= 128
        depth = depth_rgba[..., 0].astype(np.float32) / 255.0
    else:
        mask_path = artifact_paths.get("mask")
        if mask_path is None:
            raise ValueError("Depth artifact has no alpha channel and no legacy mask artifact was provided")
        mask = np.asarray(Image.open(resolve_path(mask_path, base=base_dir)).convert("L")) >= 128
        depth = np.asarray(depth_image.convert("L"), dtype=np.float32) / 255.0
    depth[~mask] = 0.0

    normal_rgb = np.asarray(
        Image.open(resolve_path(artifact_paths["normal"], base=base_dir)).convert("RGB"),
        dtype=np.float32,
    )
    normal = normal_rgb / 255.0 * 2.0 - 1.0
    normal[~mask] = 0.0
    return SideArtifacts(mask=mask, depth=depth, normal=normal)


def resize_artifacts(artifacts: SideArtifacts, *, width: int, height: int) -> SideArtifacts:
    if artifacts.mask.shape == (height, width):
        return artifacts
    mask_img = Image.fromarray((artifacts.mask.astype(np.uint8) * 255), mode="L")
    depth_img = Image.fromarray((np.clip(artifacts.depth, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    normal_img = Image.fromarray(
        ((np.clip(artifacts.normal, -1.0, 1.0) * 0.5 + 0.5) * 255).astype(np.uint8),
        mode="RGB",
    )
    mask = np.asarray(mask_img.resize((width, height), Image.Resampling.NEAREST)) >= 128
    depth = np.asarray(depth_img.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    normal = np.asarray(normal_img.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0 * 2.0 - 1.0
    normal[~mask] = 0.0
    return SideArtifacts(mask=mask, depth=depth, normal=normal)
