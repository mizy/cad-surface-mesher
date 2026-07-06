#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw


VIEWS = {
    "front": {"u": 1, "v": 2, "depth": (1.0, 0.0, 0.0), "label": "front, vehicle nose at -X"},
    "rear": {"u": 1, "v": 2, "depth": (-1.0, 0.0, 0.0), "label": "rear"},
    "left": {"u": 0, "v": 2, "depth": (0.0, -1.0, 0.0), "label": "left side"},
    "right": {"u": 0, "v": 2, "depth": (0.0, 1.0, 0.0), "label": "right side"},
    "top": {"u": 0, "v": 1, "depth": (0.0, 0.0, -1.0), "label": "top"},
}


def read_surface(path: Path) -> pv.PolyData:
    mesh = pv.read(path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    surface = mesh.extract_surface().triangulate().clean()
    if surface.n_points == 0 or surface.n_cells == 0:
        raise ValueError(f"Empty mesh after surface extraction: {path}")
    return surface


def triangle_faces(mesh: pv.PolyData) -> np.ndarray:
    faces = np.asarray(mesh.faces)
    if faces.size == 0 or faces.size % 4 != 0:
        raise ValueError("Expected packed triangle faces")
    packed = faces.reshape(-1, 4)
    if not np.all(packed[:, 0] == 3):
        raise ValueError("Mesh is not fully triangulated")
    return packed[:, 1:].astype(np.int64, copy=False)


def bounds_dict(points: np.ndarray) -> dict[str, Any]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = maxs - mins
    return {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "extents": extents.tolist(),
        "length_x": float(extents[0]),
        "width_y": float(extents[1]),
        "height_z": float(extents[2]),
    }


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def edge_topology(faces: np.ndarray) -> tuple[dict[str, int], np.ndarray, dict[tuple[int, int], list[int]]]:
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, (a, b, c) in enumerate(faces):
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted((int(edge[0]), int(edge[1]))))
            edge_to_faces.setdefault(key, []).append(face_index)

    counts = np.array([len(face_ids) for face_ids in edge_to_faces.values()], dtype=np.int32)
    stats = {
        "edges": int(counts.size),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "manifold_edges": int(np.count_nonzero(counts == 2)),
        "non_manifold_edges": int(np.count_nonzero(counts > 2)),
    }
    boundary_edges = np.array([edge for edge, face_ids in edge_to_faces.items() if len(face_ids) == 1], dtype=np.int64)
    return stats, boundary_edges, edge_to_faces


def connected_components(face_count: int, edge_to_faces: dict[tuple[int, int], list[int]]) -> dict[str, Any]:
    uf = UnionFind(face_count)
    for face_ids in edge_to_faces.values():
        if len(face_ids) < 2:
            continue
        first = face_ids[0]
        for other in face_ids[1:]:
            uf.union(first, other)

    roots: dict[int, int] = {}
    for face_index in range(face_count):
        root = uf.find(face_index)
        roots[root] = roots.get(root, 0) + 1
    sizes = sorted(roots.values(), reverse=True)
    return {
        "count": len(sizes),
        "largest_faces": sizes[:10],
    }


def triangle_quality(points: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    triangles = points[faces]
    e0 = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    e1 = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    e2 = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    max_edge = np.maximum.reduce([e0, e1, e2])
    min_edge = np.minimum.reduce([e0, e1, e2])
    min_altitude = np.divide(2.0 * areas, max_edge, out=np.zeros_like(areas), where=max_edge > 1e-15)
    aspect = np.divide(max_edge, min_altitude, out=np.full_like(max_edge, np.inf), where=min_altitude > 1e-15)
    finite_aspect = aspect[np.isfinite(aspect)]
    eps_area = max(float(np.nanmedian(areas)) * 1e-12, 1e-18) if areas.size else 1e-18
    return {
        "faces": int(faces.shape[0]),
        "area": float(areas.sum()),
        "degenerate_faces": int(np.count_nonzero((areas <= eps_area) | (min_edge <= 1e-15))),
        "area_min": float(areas.min()) if areas.size else 0.0,
        "area_p50": float(np.percentile(areas, 50)) if areas.size else 0.0,
        "area_p95": float(np.percentile(areas, 95)) if areas.size else 0.0,
        "aspect_ratio_p50": float(np.percentile(finite_aspect, 50)) if finite_aspect.size else None,
        "aspect_ratio_p95": float(np.percentile(finite_aspect, 95)) if finite_aspect.size else None,
        "aspect_ratio_p99": float(np.percentile(finite_aspect, 99)) if finite_aspect.size else None,
        "aspect_ratio_max": float(finite_aspect.max()) if finite_aspect.size else None,
    }


def signed_volume(points: np.ndarray, faces: np.ndarray) -> float:
    triangles = points[faces]
    volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0
    return float(volume)


def oriented_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
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


def view_coordinates(points: np.ndarray, view: dict[str, Any], width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = points[:, int(view["u"])]
    v = points[:, int(view["v"])]
    depth_vec = np.array(view["depth"], dtype=np.float64)
    depth_raw = points @ depth_vec
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    d_min, d_max = float(depth_raw.min()), float(depth_raw.max())
    u_span = max(u_max - u_min, 1e-12)
    v_span = max(v_max - v_min, 1e-12)
    d_span = max(d_max - d_min, 1e-12)
    margin = 10.0
    scale = min((width - 2.0 * margin) / u_span, (height - 2.0 * margin) / v_span)
    px = (u - u_min) * scale + margin
    py = (height - 1.0) - ((v - v_min) * scale + margin)
    pd = (depth_raw - d_min) / d_span
    return px, py, pd


def rasterize_view(
    points: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    view: dict[str, Any],
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    px, py, pd = view_coordinates(points, view, width, height)
    depth = np.ones((height, width), dtype=np.float32) * np.inf
    normal = np.zeros((height, width, 3), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)

    for face, face_normal in zip(faces, normals):
        verts = np.column_stack((px[face], py[face], pd[face]))
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
        mask[min_row : max_row + 1, min_col : max_col + 1][update] = True
        normal[min_row : max_row + 1, min_col : max_col + 1][update] = face_normal.astype(np.float32)

    depth_out = np.zeros((height, width), dtype=np.float32)
    depth_out[mask] = np.clip(depth[mask], 0.0, 1.0)
    return mask, depth_out, normal, px, py


def save_view_images(
    output_dir: Path,
    stem: str,
    points: np.ndarray,
    faces: np.ndarray,
    boundary_edges: np.ndarray,
    normals: np.ndarray,
    *,
    width: int,
    height: int,
) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for view_name, view in VIEWS.items():
        mask, depth, normal, px, py = rasterize_view(points, faces, normals, view, width=width, height=height)
        depth_u8 = (np.clip(depth, 0.0, 1.0) * 255).astype(np.uint8)
        depth_rgba = np.zeros((*depth_u8.shape, 4), dtype=np.uint8)
        depth_rgba[..., :3] = depth_u8[..., None]
        depth_rgba[..., 3] = mask.astype(np.uint8) * 255
        depth_img = Image.fromarray(depth_rgba, mode="RGBA")
        depth_path = output_dir / f"{stem}_{view_name}_depth.png"
        depth_img.save(depth_path)

        normal_rgb = ((np.clip(normal, -1.0, 1.0) * 0.5 + 0.5) * 255).astype(np.uint8)
        normal_rgb[~mask] = 0
        normal_path = output_dir / f"{stem}_{view_name}_normal.png"
        Image.fromarray(normal_rgb, mode="RGB").save(normal_path)

        overlay = depth_img.convert("RGBA")
        draw = ImageDraw.Draw(overlay)
        for edge in boundary_edges:
            a, b = int(edge[0]), int(edge[1])
            draw.line([(float(px[a]), float(py[a])), (float(px[b]), float(py[b]))], fill=(255, 0, 0, 255), width=2)
        overlay_path = output_dir / f"{stem}_{view_name}_boundary.png"
        overlay.save(overlay_path)

        images.append(
            {
                "view": view_name,
                "label": str(view["label"]),
                "depth": str(depth_path),
                "normal": str(normal_path),
                "boundary_overlay": str(overlay_path),
            }
        )
    return images


def visual_checks(images: list[dict[str, str]], target: str) -> list[dict[str, Any]]:
    by_view = {image["view"]: image for image in images}
    return [
        {
            "check": "external_skin_completeness",
            "target": target,
            "images": [by_view[v]["depth"] for v in ("front", "rear", "left", "right", "top")],
            "question": "Does the rendered mesh look like a complete exterior aerodynamic vehicle skin for the target?",
        },
        {
            "check": "interior_part_residue",
            "target": target,
            "images": [by_view[v]["normal"] for v in ("front", "left", "right", "top")],
            "question": "Are visible cabin, engine-bay, underbody, or hidden internal parts still present on the exterior skin?",
        },
        {
            "check": "panel_gap_and_boundary_edges",
            "target": target,
            "images": [by_view[v]["boundary_overlay"] for v in ("front", "left", "right", "top")],
            "question": "Do red boundary overlays indicate unsealed hood, door, trunk, window cracks, or CAD assembly gaps that should be sealed?",
        },
        {
            "check": "functional_openings_policy",
            "target": target,
            "images": [by_view[v]["boundary_overlay"] for v in ("front", "top")],
            "question": "Are any apparent grille, intake, wheel-well, underbody, or exhaust openings present; should they be keep/cap/porous/ask under the target policy?",
        },
        {
            "check": "feature_preservation",
            "target": target,
            "images": [by_view[v]["depth"] for v in ("front", "rear", "left", "right")],
            "question": "Were aero-relevant exterior features such as wheels, mirrors, spoiler, and roofline preserved without obvious deletion?",
        },
    ]


def audit(path: Path, output_dir: Path, target: str, width: int, height: int) -> dict[str, Any]:
    surface = read_surface(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    surface_mesh_path = output_dir / "surface_mesh.vtp"
    surface.save(surface_mesh_path)

    points = np.asarray(surface.points, dtype=np.float64)
    faces = triangle_faces(surface)
    edge_stats, boundary_edges, edge_to_faces = edge_topology(faces)
    components = connected_components(faces.shape[0], edge_to_faces)
    quality = triangle_quality(points, faces)
    volume = signed_volume(points, faces)
    watertight_topology = edge_stats["boundary_edges"] == 0 and edge_stats["non_manifold_edges"] == 0
    normals = oriented_normals(points, faces)
    images = save_view_images(output_dir / "visual", "surface", points, faces, boundary_edges, normals, width=width, height=height)
    checks = visual_checks(images, target)
    checks_path = output_dir / "visual_checks.json"
    checks_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")

    report = {
        "input": str(path),
        "target": target,
        "surface_mesh": str(surface_mesh_path),
        "metrics": {
            "points": int(surface.n_points),
            "cells": int(surface.n_cells),
            "bounds": bounds_dict(points),
            "edge_topology": edge_stats,
            "components": components,
            "quality": quality,
            "signed_volume": volume,
            "volume_reliable": bool(watertight_topology),
            "self_intersections": {"checked": False, "reason": "not implemented in this first audit tool"},
        },
        "gates": {
            "watertight_topology": bool(watertight_topology),
            "no_degenerate_faces": quality["degenerate_faces"] == 0,
            "self_intersections_checked": False,
            "engineering_pass": False,
            "engineering_pass_reason": "self-intersection check and AI visual checks are required before final pass",
        },
        "visual_artifacts": images,
        "visual_checks": str(checks_path),
    }
    report_path = output_dir / "surface_mesh_quality.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a CAD-derived surface mesh and generate target-driven visual checks.")
    parser.add_argument("mesh", help="CAD-derived or vehicle surface mesh path readable by PyVista.")
    parser.add_argument("--output-dir", required=True, help="Output directory for mesh, metrics, and visual checks.")
    parser.add_argument("--target", default="external-aero-cfd-skin")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)
    args = parser.parse_args()

    report = audit(Path(args.mesh).resolve(), Path(args.output_dir).resolve(), args.target, args.width, args.height)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
