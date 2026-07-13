from __future__ import annotations

from typing import Any

import numpy as np


def world_from_local(frame: Any, uv: np.ndarray, depth: np.ndarray | float) -> np.ndarray:
    """Lift local chart coordinates without changing any source boundary point."""

    uv_values = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    depth_values = np.broadcast_to(np.asarray(depth, dtype=np.float64), (uv_values.shape[0],))
    return (
        frame.center
        + uv_values[:, :1] * frame.u_axis
        + uv_values[:, 1:] * frame.v_axis
        + depth_values[:, None] * frame.normal
    )


def polygon_signed_area(points: np.ndarray) -> float:
    polygon = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return float(
        0.5
        * np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - polygon[:, 1] * np.roll(polygon[:, 0], -1)
        )
    )


def points_in_polygon(
    points: np.ndarray,
    polygon: np.ndarray,
    *,
    tolerance: float,
    include_boundary: bool = False,
) -> np.ndarray:
    samples = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ring = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    inside = np.zeros(samples.shape[0], dtype=bool)
    on_boundary = np.zeros(samples.shape[0], dtype=bool)
    for left, right in zip(ring, np.roll(ring, -1, axis=0), strict=True):
        segment = right - left
        relative = samples - left
        length_squared = float(np.dot(segment, segment))
        if length_squared > 0.0:
            fraction = np.clip((relative @ segment) / length_squared, 0.0, 1.0)
            distance = np.linalg.norm(relative - fraction[:, None] * segment, axis=1)
            on_boundary |= distance <= tolerance
        crosses = (left[1] > samples[:, 1]) != (right[1] > samples[:, 1])
        denominator = float(right[1] - left[1])
        if abs(denominator) > tolerance:
            x_cross = left[0] + (
                (samples[:, 1] - left[1]) * (right[0] - left[0]) / denominator
            )
            inside ^= crosses & (samples[:, 0] < x_cross)
    return (inside | on_boundary) if include_boundary else (inside & ~on_boundary)


def polygon_self_intersections(
    polygon: np.ndarray, tolerance: float
) -> list[list[int]]:
    ring = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    intersections: list[list[int]] = []
    for left in range(ring.shape[0]):
        left_next = (left + 1) % ring.shape[0]
        for right in range(left + 1, ring.shape[0]):
            right_next = (right + 1) % ring.shape[0]
            if left_next == right or right_next == left:
                continue
            if _segments_intersect(
                ring[left], ring[left_next], ring[right], ring[right_next], tolerance
            ):
                intersections.append([left, right])
    return intersections


def triangle_quality(points: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    triangles = np.asarray(points, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    edges = np.stack(
        (
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 1],
            triangles[:, 0] - triangles[:, 2],
        ),
        axis=1,
    )
    lengths = np.linalg.norm(edges, axis=2)
    areas = 0.5 * np.linalg.norm(np.cross(edges[:, 0], -edges[:, 2]), axis=1)
    altitude = 2.0 * areas / np.maximum(lengths.max(axis=1), 1e-30)
    aspect = lengths.max(axis=1) / np.maximum(altitude, 1e-30)
    return {
        "face_count": int(faces.shape[0]),
        "minimum_area": float(areas.min(initial=np.inf)),
        "maximum_aspect_ratio": float(aspect.max(initial=0.0)),
        "mean_aspect_ratio": float(aspect.mean()) if aspect.size else 0.0,
    }


def _segments_intersect(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, tolerance: float
) -> bool:
    def orient(left: np.ndarray, middle: np.ndarray, right: np.ndarray) -> float:
        first = middle - left
        second = right - left
        return float(first[0] * second[1] - first[1] * second[0])

    def on_segment(left: np.ndarray, right: np.ndarray, point: np.ndarray) -> bool:
        return bool(
            np.all(point >= np.minimum(left, right) - tolerance)
            and np.all(point <= np.maximum(left, right) + tolerance)
        )

    values = (orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b))
    segment_scale = max(
        float(np.linalg.norm(b - a)), float(np.linalg.norm(d - c)), tolerance
    )
    area_tolerance = tolerance * segment_scale
    if (
        values[0] * values[1] < -(area_tolerance * area_tolerance)
        and values[2] * values[3] < -(area_tolerance * area_tolerance)
    ):
        return True
    return bool(
        (abs(values[0]) <= area_tolerance and on_segment(a, b, c))
        or (abs(values[1]) <= area_tolerance and on_segment(a, b, d))
        or (abs(values[2]) <= area_tolerance and on_segment(c, d, a))
        or (abs(values[3]) <= area_tolerance and on_segment(c, d, b))
    )
