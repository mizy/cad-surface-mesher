from __future__ import annotations

from typing import Any

import numpy as np

from source_primary_patch_inputs import read_bounded_real
from source_primary_patch_parameterization import (
    points_in_polygon,
    polygon_self_intersections,
    polygon_signed_area,
)


def triangulate_constrained_polygon(
    boundary_uv: np.ndarray,
    boundary_vertex_ids: np.ndarray,
    *,
    source_point_count: int,
    steiner_rounds: int,
    max_appended_points: int,
    tolerance: float,
) -> dict[str, Any]:
    """Ear-clip a simple polygon, then add only strictly interior Steiner points."""

    try:
        raw_uv = np.asarray(boundary_uv)
        raw_ids = np.asarray(boundary_vertex_ids)
    except (OverflowError, TypeError, ValueError):
        return _failure("constrained_triangulation_boundary_invalid")
    if (
        raw_uv.ndim != 2
        or raw_uv.shape[1:] != (2,)
        or not np.issubdtype(raw_uv.dtype, np.number)
        or np.iscomplexobj(raw_uv)
        or raw_ids.ndim != 1
        or not np.issubdtype(raw_ids.dtype, np.integer)
    ):
        return _failure("constrained_triangulation_boundary_invalid")
    integer_values = (steiner_rounds, source_point_count, max_appended_points)
    integer_limit = np.iinfo(np.int64).max
    integer_values_valid = all(
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, np.integer))
        for value in integer_values
    )
    parsed_tolerance, tolerance_error = read_bounded_real(
        tolerance,
        name="tolerance",
        minimum=np.finfo(np.float64).tiny,
        maximum=np.finfo(np.float64).max,
    )
    if not integer_values_valid or tolerance_error is not None:
        return _failure("constrained_triangulation_limits_invalid")
    assert parsed_tolerance is not None
    if (
        not 0 <= steiner_rounds <= 3
        or not 0 <= source_point_count <= integer_limit
        or not 0 <= max_appended_points <= integer_limit - source_point_count
    ):
        return _failure("constrained_triangulation_limits_invalid")
    tolerance = parsed_tolerance
    uv = raw_uv.astype(np.float64, copy=False)
    if raw_ids.size and (
        np.min(raw_ids) < 0 or np.max(raw_ids) > integer_limit
    ):
        return _failure("constrained_triangulation_boundary_invalid")
    ids = raw_ids.astype(np.int64, copy=False)
    if (
        uv.shape[0] != ids.size
        or ids.size < 3
        or not np.all(np.isfinite(uv))
        or np.unique(ids).size != ids.size
        or np.any(ids >= source_point_count)
    ):
        return _failure("constrained_triangulation_boundary_invalid")
    if polygon_self_intersections(uv, tolerance):
        return _failure("constrained_triangulation_boundary_self_intersects")
    patch_uv = uv[::-1].copy()
    patch_ids = ids[::-1].copy()
    if polygon_signed_area(patch_uv) <= tolerance * tolerance:
        return _failure("constrained_triangulation_orientation_invalid")
    ears = _ear_clip(patch_uv, patch_ids, tolerance)
    if ears is None:
        return _failure("constrained_ear_clipping_failed")
    faces = ears
    known_uv = {
        int(vertex_id): patch_uv[index] for index, vertex_id in enumerate(patch_ids)
    }
    appended_uv: list[np.ndarray] = []
    for _ in range(steiner_rounds):
        if len(appended_uv) + faces.shape[0] > max_appended_points:
            return _failure(
                "planar_patch_appended_point_limit_exceeded",
                attempted_points=len(appended_uv) + int(faces.shape[0]),
                max_appended_points=int(max_appended_points),
            )
        refined: list[list[int]] = []
        for face in faces:
            triangle_uv = np.asarray([known_uv[int(value)] for value in face])
            centroid = triangle_uv.mean(axis=0)
            if not points_in_polygon(
                centroid[None, :], uv, tolerance=tolerance, include_boundary=False
            )[0]:
                return _failure("steiner_point_outside_true_footprint")
            new_id = source_point_count + len(appended_uv)
            appended_uv.append(centroid)
            known_uv[new_id] = centroid
            left, middle, right = (int(value) for value in face)
            refined.extend(
                ([left, middle, new_id], [middle, right, new_id], [right, left, new_id])
            )
        faces = np.asarray(refined, dtype=np.int64)
    if not _faces_have_positive_area(known_uv, faces, tolerance):
        return _failure("constrained_triangulation_non_positive_face")
    return {
        "success": True,
        "failure_reason_codes": [],
        "appended_uv": np.asarray(appended_uv, dtype=np.float64).reshape(-1, 2),
        "faces": faces,
        "diagnostics": {
            "method": "constrained_ear_clipping_then_interior_centroid_steiner",
            "boundary_vertices": int(ids.size),
            "base_faces": int(ears.shape[0]),
            "steiner_rounds": int(steiner_rounds),
            "appended_points": len(appended_uv),
            "generated_faces": int(faces.shape[0]),
            "boundary_constraints_preserved": True,
        },
    }


def _ear_clip(uv: np.ndarray, ids: np.ndarray, tolerance: float) -> np.ndarray | None:
    remaining = list(range(ids.size))
    output: list[list[int]] = []
    area_tolerance = tolerance * max(float(np.ptp(uv, axis=0).max()), tolerance)
    while len(remaining) > 3:
        candidates: list[tuple[float, int, list[int]]] = []
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following = remaining[(position + 1) % len(remaining)]
            a, b, c = uv[[previous, current, following]]
            cross = _cross_2d(b - a, c - b)
            if cross <= area_tolerance:
                continue
            if any(
                _point_in_triangle(uv[other], a, b, c, area_tolerance)
                for other in remaining
                if other not in {previous, current, following}
            ):
                continue
            lengths = np.asarray(
                [np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)]
            )
            score = cross / max(float(np.max(lengths) ** 2), 1e-30)
            candidates.append(
                (
                    -score,
                    int(ids[current]),
                    [int(ids[previous]), int(ids[current]), int(ids[following])],
                )
            )
        if not candidates:
            return None
        candidates.sort(key=lambda row: (row[0], row[1]))
        selected = candidates[0][2]
        position = next(
            index for index in remaining if int(ids[index]) == selected[1]
        )
        output.append(selected)
        remaining.remove(position)
    output.append([int(ids[index]) for index in remaining])
    return np.asarray(output, dtype=np.int64)


def _point_in_triangle(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    tolerance: float,
) -> bool:
    signs = np.asarray(
        (
            _cross_2d(b - a, point - a),
            _cross_2d(c - b, point - b),
            _cross_2d(a - c, point - c),
        )
    )
    return bool(np.all(signs >= -tolerance))


def _faces_have_positive_area(
    known_uv: dict[int, np.ndarray], faces: np.ndarray, tolerance: float
) -> bool:
    for face in faces:
        a, b, c = (known_uv[int(value)] for value in face)
        scale = max(
            float(np.linalg.norm(b - a)),
            float(np.linalg.norm(c - a)),
            tolerance,
        )
        if _cross_2d(b - a, c - a) <= tolerance * scale:
            return False
    return True


def _cross_2d(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def _failure(code: str, **diagnostics: Any) -> dict[str, Any]:
    return {
        "success": False,
        "failure_reason_codes": [code],
        "appended_uv": np.empty((0, 2), dtype=np.float64),
        "faces": np.empty((0, 3), dtype=np.int64),
        "diagnostics": diagnostics,
    }
