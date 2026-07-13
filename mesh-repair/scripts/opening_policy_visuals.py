from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation


PROJECTION_AXES = {
    0: ("normal_x_view_yz", (1, 2)),
    1: ("normal_y_view_xz", (0, 2)),
    2: ("normal_z_view_xy", (0, 1)),
}
POLICY_CLASSIFICATIONS = {
    "large_opening_or_missing_surface",
    "pending_policy",
}


def attach_policy_evidence_views(
    points: np.ndarray,
    faces: np.ndarray,
    inventory: dict[str, Any],
    output_dir: Path,
    *,
    image_size: int = 720,
) -> dict[str, Any]:
    """Attach deterministic global/local opening images to policy-eligible loops.

    Images are evidence for semantic review only. They never alter geometry or
    decide whether an opening is capped.
    """

    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    policy_items = [
        item
        for item in inventory.get("boundary_regions", {}).get("items", [])
        if item.get("requires_policy")
        and item.get("classification") in POLICY_CLASSIFICATIONS
        and item.get("simple_closed_loop")
        and item.get("ordered_vertex_ids")
    ]
    evidence_dir = output_dir / "opening_policy_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    centroids = points[faces].mean(axis=1) if faces.size else np.empty((0, 3), dtype=np.float64)
    global_images = {
        axes: projected_occupancy(points, faces, axes, image_size)
        for _, axes in PROJECTION_AXES.values()
    }
    written = []
    for item in policy_items:
        normal = np.asarray(item.get("normal") or [0.0, 0.0, 1.0], dtype=np.float64)
        drop_axis = int(np.argmax(np.abs(normal))) if np.any(np.isfinite(normal)) else 2
        view_name, axes = PROJECTION_AXES[drop_axis]
        loop = np.asarray(item["ordered_vertex_ids"], dtype=np.int64)
        loop = loop[(loop >= 0) & (loop < points.shape[0])]
        if loop.size < 3:
            continue
        image = compose_policy_evidence(
            points,
            faces,
            centroids,
            loop,
            axes,
            global_images[axes],
            item,
            image_size,
        )
        path = evidence_dir / f"{item['id']}_{view_name}.png"
        image.save(path)
        resolved = str(path.resolve())
        item["evidence_views"] = [resolved]
        written.append({"source_region": item["id"], "view": view_name, "path": resolved})
    return {
        "method": "deterministic_global_context_plus_local_boundary_projection",
        "role": "semantic_policy_evidence_only",
        "requested_regions": len(policy_items),
        "written_regions": len(written),
        "image_size_per_panel": int(image_size),
        "artifacts": written,
    }


def compose_policy_evidence(
    points: np.ndarray,
    faces: np.ndarray,
    centroids: np.ndarray,
    loop: np.ndarray,
    axes: tuple[int, int],
    global_base: Image.Image,
    item: dict[str, Any],
    image_size: int,
) -> Image.Image:
    global_panel = global_base.copy()
    global_draw = ImageDraw.Draw(global_panel)
    global_bounds = projection_bounds(points[:, axes])
    draw_loop(global_draw, points, loop, axes, global_bounds, image_size, color=(239, 68, 68), width=3)

    loop_points = points[loop]
    loop_extent = np.maximum(loop_points.max(axis=0) - loop_points.min(axis=0), 0.0)
    diameter = max(float(np.linalg.norm(loop_extent)), float(item.get("mean_edge_length") or 0.0), 1e-12)
    local_min = loop_points.min(axis=0) - diameter * 0.65
    local_max = loop_points.max(axis=0) + diameter * 0.65
    local_faces = np.flatnonzero(np.all((centroids >= local_min) & (centroids <= local_max), axis=1))
    if local_faces.size > 8_000:
        stride = int(np.ceil(local_faces.size / 8_000))
        local_faces = local_faces[::stride]
    local_bounds = padded_bounds(loop_points[:, axes], pad_ratio=0.45)
    local_panel = Image.new("RGB", (image_size, image_size), (250, 251, 252))
    local_draw = ImageDraw.Draw(local_panel)
    for face_id in local_faces:
        triangle = faces[int(face_id)]
        pixels = [project_pixel(points[int(vertex_id)], axes, local_bounds, image_size) for vertex_id in triangle]
        local_draw.line([*pixels, pixels[0]], fill=(176, 188, 198), width=1)
    draw_loop(local_draw, points, loop, axes, local_bounds, image_size, color=(239, 68, 68), width=4)

    canvas = Image.new("RGB", (image_size * 2, image_size + 42), (18, 24, 29))
    canvas.paste(global_panel, (0, 42))
    canvas.paste(local_panel, (image_size, 42))
    title = (
        f"{item.get('id')}  global context | local loop    "
        f"d/bbox={feature_value(item, 'diameter_bbox_ratio')}  "
        f"compactness={feature_value(item, 'compactness')}  "
        f"planarity={feature_value(item, 'planarity')}"
    )
    ImageDraw.Draw(canvas).text((10, 13), title, fill=(232, 238, 242))
    return canvas


def projected_occupancy(
    points: np.ndarray,
    faces: np.ndarray,
    axes: tuple[int, int],
    image_size: int,
) -> Image.Image:
    bounds = projection_bounds(points[:, axes])
    centroids = points[faces].mean(axis=1)[:, axes]
    pixels = project_values(centroids, bounds, image_size)
    occupancy = np.zeros((image_size, image_size), dtype=bool)
    occupancy[pixels[:, 1], pixels[:, 0]] = True
    occupancy = binary_dilation(occupancy, iterations=1)
    array = np.full((image_size, image_size, 3), 250, dtype=np.uint8)
    array[occupancy] = (178, 191, 201)
    return Image.fromarray(array)


def projection_bounds(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return padded_bounds(values, pad_ratio=0.02)


def padded_bounds(values: np.ndarray, *, pad_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.asarray(values, dtype=np.float64).min(axis=0)
    maximum = np.asarray(values, dtype=np.float64).max(axis=0)
    span = np.maximum(maximum - minimum, 1e-12)
    pad = span * float(pad_ratio)
    return minimum - pad, maximum + pad


def project_values(
    values: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    image_size: int,
) -> np.ndarray:
    minimum, maximum = bounds
    normalized = np.clip((values - minimum) / np.maximum(maximum - minimum, 1e-12), 0.0, 1.0)
    pixels = np.rint(normalized * (image_size - 1)).astype(np.int64)
    pixels[:, 1] = image_size - 1 - pixels[:, 1]
    return pixels


def project_pixel(
    point: np.ndarray,
    axes: tuple[int, int],
    bounds: tuple[np.ndarray, np.ndarray],
    image_size: int,
) -> tuple[int, int]:
    pixel = project_values(np.asarray(point, dtype=np.float64)[None, axes], bounds, image_size)[0]
    return int(pixel[0]), int(pixel[1])


def draw_loop(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    loop: np.ndarray,
    axes: tuple[int, int],
    bounds: tuple[np.ndarray, np.ndarray],
    image_size: int,
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    pixels = [project_pixel(points[int(vertex_id)], axes, bounds, image_size) for vertex_id in loop]
    if pixels:
        draw.line([*pixels, pixels[0]], fill=color, width=width)


def feature_value(item: dict[str, Any], name: str) -> str:
    features = item.get("dimensionless_features", {})
    value = features.get(name, item.get(name))
    return "n/a" if value is None else f"{float(value):.4g}"
