"""Face-level source-shell evidence from solid-triangle first-hit views.

The input directions point from the surface toward an orthographic observer.
This module only measures source triangles.  It does not expand evidence to a
connected component, change topology, or create replacement geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from solid_triangle_raster import ViewSpec, rasterize_solid_view


DEFAULT_DIRECTION_COUNT = 42
MAX_DIRECTION_COUNT = 4096
_CANONICAL_VIEW = ViewSpec(
    "directional_orthographic",
    (0, 1),
    (0.0, 0.0, 1.0),
)


@dataclass(frozen=True)
class DirectionalFirstHitEvidence:
    external_direction: np.ndarray
    face_first_hit_mask: np.ndarray
    face_first_hit_pixels: np.ndarray
    face_covered_pixels: np.ndarray
    face_min_depth_gap: np.ndarray
    report: dict[str, Any]


@dataclass(frozen=True)
class SourceShellVisibilityEvidence:
    view_directions: np.ndarray
    face_exterior_mask: np.ndarray
    face_first_hit_view_count: np.ndarray
    face_first_hit_pixel_support: np.ndarray
    face_external_direction: np.ndarray
    face_external_direction_resultant: np.ndarray
    report: dict[str, Any]


# @entry deterministic near-uniform directions for source-shell observation.
def fibonacci_sphere_directions(
    count: int = DEFAULT_DIRECTION_COUNT,
) -> np.ndarray:
    """Return deterministic unit vectors sampled over the whole sphere."""
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError("count must be an integer")
    if not 1 <= int(count) <= MAX_DIRECTION_COUNT:
        raise ValueError(f"count must be between 1 and {MAX_DIRECTION_COUNT}")
    indices = np.arange(int(count), dtype=np.float64)
    height = 1.0 - 2.0 * (indices + 0.5) / float(count)
    radius = np.sqrt(np.maximum(0.0, 1.0 - height * height))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    azimuth = indices * golden_angle
    directions = np.column_stack(
        (radius * np.cos(azimuth), radius * np.sin(azimuth), height)
    )
    return np.ascontiguousarray(directions)


# @entry one arbitrary-direction orthographic solid-triangle first-hit view.
def rasterize_directional_first_hits(
    points: np.ndarray,
    faces: np.ndarray,
    external_direction: Sequence[float],
    *,
    grid_size: int,
    depth_tolerance: float = 0.0,
    name: str = "directional",
) -> DirectionalFirstHitEvidence:
    """Measure faces first seen by an observer along ``external_direction``."""
    coordinates, triangles = _validate_mesh(points, faces)
    direction = _normalize_direction(external_direction)
    tolerance = _effective_depth_tolerance(coordinates, depth_tolerance)
    return _rasterize_direction(
        coordinates,
        triangles,
        direction,
        grid_size=grid_size,
        depth_tolerance=tolerance,
        name=name,
    )


# @entry multi-view face evidence for an open, source-accurate shell.
def build_source_shell_visibility(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    directions: np.ndarray | None = None,
    direction_count: int = DEFAULT_DIRECTION_COUNT,
    grid_size: int = 128,
    depth_tolerance: float = 0.0,
) -> SourceShellVisibilityEvidence:
    """Accumulate first-hit evidence without any component-level hard keep."""
    coordinates, triangles = _validate_mesh(points, faces)
    view_directions, direction_source = _resolve_directions(
        directions, direction_count
    )
    tolerance = _effective_depth_tolerance(coordinates, depth_tolerance)
    face_count = triangles.shape[0]
    hit_view_count = np.zeros(face_count, dtype=np.uint16)
    pixel_support = np.zeros(face_count, dtype=np.int64)
    direction_sum = np.zeros((face_count, 3), dtype=np.float64)
    view_reports: list[dict[str, Any]] = []

    for view_index, direction in enumerate(view_directions):
        view = _rasterize_direction(
            coordinates,
            triangles,
            direction,
            grid_size=grid_size,
            depth_tolerance=tolerance,
            name=f"source_shell_{view_index:04d}",
        )
        hit = view.face_first_hit_mask
        hit_view_count[hit] += 1
        pixel_support += view.face_first_hit_pixels
        direction_sum[hit] += direction
        view_reports.append(view.report)
        del view

    external_direction, resultant = _summarize_directions(
        direction_sum, hit_view_count
    )
    exterior = hit_view_count > 0
    return SourceShellVisibilityEvidence(
        view_directions=view_directions,
        face_exterior_mask=exterior,
        face_first_hit_view_count=hit_view_count,
        face_first_hit_pixel_support=pixel_support,
        face_external_direction=external_direction,
        face_external_direction_resultant=resultant,
        report=_visibility_report(
            face_count=face_count,
            exterior=exterior,
            pixel_support=pixel_support,
            direction_source=direction_source,
            view_directions=view_directions,
            grid_size=grid_size,
            requested_tolerance=depth_tolerance,
            effective_tolerance=tolerance,
            view_reports=view_reports,
        ),
    )


def _rasterize_direction(
    coordinates: np.ndarray,
    triangles: np.ndarray,
    direction: np.ndarray,
    *,
    grid_size: int,
    depth_tolerance: float,
    name: str,
) -> DirectionalFirstHitEvidence:
    frame = _orthographic_frame(direction)
    view_points = np.ascontiguousarray(coordinates @ frame.T)
    raster = rasterize_solid_view(
        view_points,
        triangles,
        _CANONICAL_VIEW,
        grid_size=grid_size,
        depth_tolerance=depth_tolerance,
        collect_face_evidence=True,
    )
    hit = raster.face_first_hit_pixels > 0
    report = dict(raster.report)
    report.update(
        {
            "view": name,
            "method": "arbitrary_direction_orthographic_solid_triangle_first_hit",
            "external_direction": direction.tolist(),
            "world_to_view_frame": frame.tolist(),
            "faces_with_first_hit": int(np.count_nonzero(hit)),
            "component_expansion_applied": False,
        }
    )
    return DirectionalFirstHitEvidence(
        external_direction=direction.copy(),
        face_first_hit_mask=hit,
        face_first_hit_pixels=raster.face_first_hit_pixels,
        face_covered_pixels=raster.face_covered_pixels,
        face_min_depth_gap=raster.face_min_depth_gap,
        report=report,
    )


def _orthographic_frame(external_direction: np.ndarray) -> np.ndarray:
    depth = -external_direction
    helper = np.zeros(3, dtype=np.float64)
    helper[int(np.argmin(np.abs(depth)))] = 1.0
    horizontal = np.cross(helper, depth)
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(depth, horizontal)
    return np.ascontiguousarray(np.vstack((horizontal, vertical, depth)))


def _resolve_directions(
    directions: np.ndarray | None,
    direction_count: int,
) -> tuple[np.ndarray, str]:
    if directions is None:
        return fibonacci_sphere_directions(direction_count), "fibonacci_sphere"
    values = np.asarray(directions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError("directions must have non-empty shape (view_count, 3)")
    if values.shape[0] > MAX_DIRECTION_COUNT:
        raise ValueError(f"at most {MAX_DIRECTION_COUNT} directions are supported")
    return np.ascontiguousarray(
        np.vstack([_normalize_direction(value) for value in values])
    ), "caller_supplied"


def _normalize_direction(direction: Sequence[float]) -> np.ndarray:
    value = np.asarray(direction, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("external_direction must contain three finite values")
    length = float(np.linalg.norm(value))
    if length <= 0.0:
        raise ValueError("external_direction must be non-zero")
    return np.ascontiguousarray(value / length)


def _summarize_directions(
    direction_sum: np.ndarray,
    hit_view_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.linalg.norm(direction_sum, axis=1)
    external_direction = np.divide(
        direction_sum,
        magnitude[:, None],
        out=np.zeros_like(direction_sum),
        where=magnitude[:, None] > 0.0,
    )
    resultant = np.divide(
        magnitude,
        hit_view_count,
        out=np.zeros_like(magnitude),
        where=hit_view_count > 0,
    )
    return external_direction, resultant


def _effective_depth_tolerance(
    points: np.ndarray,
    requested: float,
) -> float:
    if not np.isfinite(requested) or requested < 0.0:
        raise ValueError("depth_tolerance must be finite and non-negative")
    bbox_extent = float(np.max(np.ptp(points, axis=0)))
    return max(float(requested), bbox_extent * 1.0e-12)


def _validate_mesh(
    points: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not coordinates.size:
        raise ValueError("points must have non-empty shape (point_count, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (face_count, 3)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("points must be finite")
    if triangles.size and (
        int(np.min(triangles)) < 0 or int(np.max(triangles)) >= coordinates.shape[0]
    ):
        raise ValueError("faces contain out-of-range point indices")
    return np.ascontiguousarray(coordinates), np.ascontiguousarray(triangles)


def _visibility_report(
    *,
    face_count: int,
    exterior: np.ndarray,
    pixel_support: np.ndarray,
    direction_source: str,
    view_directions: np.ndarray,
    grid_size: int,
    requested_tolerance: float,
    effective_tolerance: float,
    view_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "source_shell_visibility/v1",
        "method": "multi_direction_orthographic_solid_triangle_first_hit",
        "role": "face_level_exterior_evidence",
        "geometry_modified": False,
        "selection_scope": "individual_source_face",
        "component_expansion_applied": False,
        "direction_source": direction_source,
        "direction_count": int(view_directions.shape[0]),
        "directions": view_directions.tolist(),
        "grid_size_longest_axis": int(grid_size),
        "requested_depth_tolerance": float(requested_tolerance),
        "effective_depth_tolerance": float(effective_tolerance),
        "face_count": int(face_count),
        "faces_with_first_hit": int(np.count_nonzero(exterior)),
        "first_hit_pixel_support": int(np.sum(pixel_support, dtype=np.int64)),
        "external_direction_model": "unit_resultant_of_first_hit_view_directions",
        "memory_contract": "views_are_rasterized_and_released_sequentially",
        "views": view_reports,
    }
