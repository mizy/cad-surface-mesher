from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from PIL import Image


def read_surface(path: Path) -> pv.PolyData:
    mesh = pv.read(path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    try:
        surface = mesh.extract_surface(algorithm="dataset_surface")
    except TypeError:
        surface = mesh.extract_surface()
    surface = surface.triangulate().clean()
    if surface.n_points == 0 or surface.n_cells == 0:
        raise ValueError(f"empty mesh after surface extraction: {path}")
    return surface


def triangle_faces(mesh: pv.PolyData) -> np.ndarray:
    faces = np.asarray(mesh.faces)
    if faces.size == 0 or faces.size % 4 != 0:
        raise ValueError("expected packed triangle faces")
    packed = faces.reshape(-1, 4)
    if not np.all(packed[:, 0] == 3):
        raise ValueError("surface is not fully triangulated")
    return packed[:, 1:].astype(np.int64, copy=False)


def write_vtp(
    path: Path,
    points: np.ndarray,
    faces: np.ndarray,
    cell_data: dict[str, np.ndarray] | None = None,
) -> None:
    packed = np.empty((faces.shape[0], 4), dtype=np.int64)
    packed[:, 0] = 3
    packed[:, 1:] = faces
    mesh = pv.PolyData(points, packed.ravel())
    for name, values in (cell_data or {}).items():
        mesh.cell_data[name] = values
    mesh.save(path)


def compact_mesh(
    points: np.ndarray,
    faces: np.ndarray,
    source_face_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected_faces = faces[source_face_indices]
    unique_points, inverse = np.unique(selected_faces.ravel(), return_inverse=True)
    compact_points = points[unique_points]
    compact_faces = inverse.reshape((-1, 3)).astype(np.int64, copy=False)
    return compact_points, compact_faces


def grid_shape(coords: np.ndarray, max_size: int) -> tuple[int, int, np.ndarray, np.ndarray]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-12)
    if spans[0] >= spans[1]:
        cols = max_size
        rows = max(64, int(round(max_size * spans[1] / spans[0])))
    else:
        rows = max_size
        cols = max(64, int(round(max_size * spans[0] / spans[1])))
    return rows, cols, mins, spans


def save_depth_preview(
    output_dir: Path,
    stage: str,
    points: np.ndarray,
    faces: np.ndarray,
    views: list[Any],
    *,
    size: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    centroids = points[faces].mean(axis=1)
    saved = []
    for view in [views[0], views[2], views[4]]:
        image = zbuffer_image(centroids, view, size)
        path = output_dir / f"{stage}_{view.name}_depth.png"
        image.save(path)
        saved.append(str(path))
    return saved


def zbuffer_image(centroids: np.ndarray, view: Any, max_size: int) -> Image.Image:
    coords = centroids[:, view.project_axes]
    rows, cols, mins, spans = grid_shape(coords, max_size)
    uv = np.clip((coords - mins) / spans, 0.0, 1.0)
    col = np.minimum((uv[:, 0] * cols).astype(np.int64), cols - 1)
    row = np.minimum((uv[:, 1] * rows).astype(np.int64), rows - 1)
    linear = row * cols + col

    depth = centroids @ np.asarray(view.depth_vector, dtype=np.float64)
    min_depth = np.full(rows * cols, np.inf, dtype=np.float64)
    np.minimum.at(min_depth, linear, depth)
    mask = np.isfinite(min_depth).reshape(rows, cols)
    depth_grid = min_depth.reshape(rows, cols)

    image = np.full((rows, cols, 3), 255, dtype=np.uint8)
    if np.any(mask):
        valid = depth_grid[mask]
        denom = max(float(valid.max() - valid.min()), 1e-12)
        shade = 235 - ((valid - valid.min()) / denom * 190).astype(np.uint8)
        image[mask] = np.column_stack((shade, shade, shade))
    return Image.fromarray(image)
