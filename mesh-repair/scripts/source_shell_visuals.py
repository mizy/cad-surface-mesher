from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PROJECTIONS = (
    ("top_xy", (0, 1)),
    ("side_xz", (0, 2)),
    ("front_yz", (1, 2)),
)


# @entry Render deterministic visual evidence for Codex component review.
def render_candidate_contact_sheets(
    points: np.ndarray,
    faces: np.ndarray,
    component_labels: np.ndarray,
    candidates: list[dict[str, Any]],
    output_dir: Path,
    *,
    page_size: int = 8,
    tile_width: int = 720,
    tile_height: int = 430,
    max_global_faces: int = 120_000,
    max_candidate_faces: int = 8_000,
) -> dict[str, Any]:
    coordinates, triangles, labels = _validate_inputs(points, faces, component_labels)
    if page_size < 1:
        raise ValueError("page_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    full_centroids = coordinates[triangles].mean(axis=1)
    global_ids = evenly_spaced_ids(triangles.shape[0], max_global_faces)
    global_samples = full_centroids[global_ids]
    global_bounds = {
        name: projection_bounds(coordinates[:, axes])
        for name, axes in PROJECTIONS
    }

    pages: list[dict[str, Any]] = []
    for page_index, start in enumerate(range(0, len(candidates), page_size), start=1):
        rows = candidates[start : start + page_size]
        columns = 2
        page_rows = int(np.ceil(len(rows) / columns))
        canvas = Image.new(
            "RGB",
            (tile_width * columns, tile_height * page_rows),
            (20, 25, 30),
        )
        for local_index, candidate in enumerate(rows):
            component_id = int(candidate["component_id"])
            face_ids = np.flatnonzero(labels == component_id)
            if face_ids.size == 0:
                raise ValueError(f"candidate component is absent: {component_id}")
            selected_ids = face_ids[evenly_spaced_ids(face_ids.size, max_candidate_faces)]
            candidate_points = coordinates[np.unique(triangles[face_ids].ravel())]
            candidate_centroids = full_centroids[selected_ids]
            tile = render_candidate_tile(
                candidate,
                global_samples,
                candidate_centroids,
                candidate_points,
                global_bounds,
                tile_width=tile_width,
                tile_height=tile_height,
            )
            left = (local_index % columns) * tile_width
            top = (local_index // columns) * tile_height
            canvas.paste(tile, (left, top))
        path = output_dir / f"candidate_contact_sheet_{page_index:03d}.png"
        canvas.save(path)
        pages.append(
            {
                "page": page_index,
                "path": str(path.resolve()),
                "candidate_ids": [candidate_key(row) for row in rows],
            }
        )

    manifest = {
        "schema": "source_shell_ai_evidence/v1",
        "method": "three_projection_global_location_and_isolated_shape_contact_sheets",
        "geometry_modified": False,
        "candidate_count": len(candidates),
        "page_size": page_size,
        "pages": pages,
        "candidates": [serializable_candidate(row) for row in candidates],
    }
    manifest_path = output_dir / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def render_candidate_tile(
    candidate: dict[str, Any],
    global_samples: np.ndarray,
    candidate_centroids: np.ndarray,
    candidate_points: np.ndarray,
    global_bounds: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    tile_width: int,
    tile_height: int,
) -> Image.Image:
    image = Image.new("RGB", (tile_width, tile_height), (245, 247, 249))
    draw = ImageDraw.Draw(image)
    identifier = candidate_key(candidate)
    summary = (
        f"{identifier}  faces={int(candidate.get('face_count', 0))}  "
        f"first_hit_faces={int(candidate.get('first_hit_face_count', 0))}  "
        f"first_hit_views={int(candidate.get('first_hit_view_max', 0))}"
    )
    draw.rectangle((0, 0, tile_width, 34), fill=(30, 41, 51))
    draw.text((9, 10), summary, fill=(240, 244, 247))

    panel_gap = 6
    panel_width = (tile_width - 4 * panel_gap) // 3
    panel_height = (tile_height - 58 - 3 * panel_gap) // 2
    isolated_bounds = {
        name: projection_bounds(candidate_points[:, axes], pad_ratio=0.08)
        for name, axes in PROJECTIONS
    }
    for column, (name, axes) in enumerate(PROJECTIONS):
        left = panel_gap + column * (panel_width + panel_gap)
        global_box = (left, 42, left + panel_width, 42 + panel_height)
        isolated_top = 42 + panel_height + panel_gap
        isolated_box = (
            left,
            isolated_top,
            left + panel_width,
            isolated_top + panel_height,
        )
        draw_projection_panel(
            image,
            global_box,
            global_samples[:, axes],
            candidate_centroids[:, axes],
            global_bounds[name],
            title=f"global {name}",
        )
        draw_projection_panel(
            image,
            isolated_box,
            np.empty((0, 2), dtype=np.float64),
            candidate_centroids[:, axes],
            isolated_bounds[name],
            title=f"isolated {name}",
        )
    return image


def draw_projection_panel(
    image: Image.Image,
    box: tuple[int, int, int, int],
    background: np.ndarray,
    candidate: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    title: str,
) -> None:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = box
    draw.rectangle(box, fill=(255, 255, 255), outline=(203, 211, 218))
    width = max(2, right - left - 4)
    height = max(2, bottom - top - 18)
    background_pixels = project_values(background, bounds, width, height)
    candidate_pixels = project_values(candidate, bounds, width, height)
    for x, y in background_pixels:
        draw.point((left + 2 + int(x), top + 16 + int(y)), fill=(190, 199, 207))
    for x, y in candidate_pixels:
        px = left + 2 + int(x)
        py = top + 16 + int(y)
        draw.rectangle((px - 1, py - 1, px + 1, py + 1), fill=(225, 43, 42))
    draw.text((left + 5, top + 3), title, fill=(56, 67, 77))


def projection_bounds(
    values: np.ndarray,
    *,
    pad_ratio: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.zeros(2, dtype=np.float64), np.ones(2, dtype=np.float64)
    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)
    span = np.maximum(maximum - minimum, 1e-12)
    padding = span * float(pad_ratio)
    return minimum - padding, maximum + padding


def project_values(
    values: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    if values.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    minimum, maximum = bounds
    normalized = np.clip(
        (values - minimum) / np.maximum(maximum - minimum, 1e-12),
        0.0,
        1.0,
    )
    pixels = np.rint(normalized * np.asarray((width - 1, height - 1))).astype(np.int64)
    pixels[:, 1] = height - 1 - pixels[:, 1]
    return pixels


def evenly_spaced_ids(count: int, limit: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if limit <= 0 or count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, limit, dtype=np.int64)


def candidate_key(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_id") or f"component_{int(candidate['component_id']):06d}")


def serializable_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in candidate.items()
        if key != "face_ids"
    }


def _validate_inputs(
    points: np.ndarray,
    faces: np.ndarray,
    component_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    labels = np.asarray(component_labels, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if labels.shape != (triangles.shape[0],):
        raise ValueError("component_labels must contain one id per face")
    return coordinates, triangles, labels
