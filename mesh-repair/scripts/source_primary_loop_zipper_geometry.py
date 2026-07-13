from __future__ import annotations

from typing import Any

import numpy as np

from deterministic_hole_fill import validate_local_patch_intersections
from source_primary_patch_geometry import triangle_quality


def triangulate_ordered_loop_bridge(
    points: np.ndarray,
    source_faces: np.ndarray,
    first_analysis: dict[str, Any],
    second_analysis: dict[str, Any],
    *,
    characteristic_length: float,
    config: Any,
) -> dict[str, Any]:
    """Choose a cyclic phase by ordered position/normal correspondence."""

    del source_faces  # Full-source intersection auditing belongs to the transaction gate.
    first = np.asarray(first_analysis["loop"], dtype=np.int64)[::-1].copy()
    second_source = np.asarray(second_analysis["loop"], dtype=np.int64)
    first_normals = np.asarray(first_analysis["normal"]["boundary_vertex_normals"])[::-1]
    second_normals_source = np.asarray(second_analysis["normal"]["boundary_vertex_normals"])
    phases = _phase_candidates(points[first], points[second_source], config.phase_candidates)
    attempted: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for phase_value in phases:
        phase = int(phase_value)
        second = np.roll(second_source, -phase)
        second_normals = np.roll(second_normals_source, -phase, axis=0)
        faces = _arc_length_zipper(points, first, second)
        correspondence = _ordered_correspondence(
            points[first], points[second], first_normals, second_normals
        )
        quality = triangle_quality(points, faces)
        local_intersection = _local_intersection(points, faces, config)
        reasons = []
        if (
            correspondence["maximum"]
            > config.max_correspondence_distance_ratio * characteristic_length
        ):
            reasons.append("paired_loop_zipper_correspondence_distance_exceeded")
        if correspondence["minimum_normal_dot"] < config.min_loop_normal_abs_dot:
            reasons.append("paired_loop_zipper_normal_incompatible")
        if quality["maximum_aspect_ratio"] > config.max_aspect_ratio:
            reasons.append("paired_loop_zipper_aspect_ratio_exceeded")
        if not local_intersection["passed"]:
            reasons.append("paired_loop_zipper_self_intersection")
        cost = (
            correspondence["rms"] / max(characteristic_length, 1e-30)
            + max(0.0, 1.0 - correspondence["mean_normal_dot"])
            + quality["mean_aspect_ratio"] * 1e-3
        )
        attempted.append({"phase": phase, "cost": float(cost), "reason_codes": reasons})
        if not reasons and (best is None or (cost, phase) < (best["cost"], best["phase"])):
            best = {
                "faces": faces,
                "phase": phase,
                "cost": float(cost),
                "correspondence": correspondence,
                "local_intersection": local_intersection,
            }
    if best is None:
        reason = next(
            (
                code
                for row in attempted
                for code in row["reason_codes"]
                if code == "paired_loop_zipper_self_intersection"
            ),
            "paired_loop_zipper_triangulation_failed",
        )
        return _bridge_failure(reason, attempted_phases=attempted)
    return {
        "success": True,
        "failure_reason_codes": [],
        "faces": best["faces"],
        "diagnostics": {
            "method": "ordered_arc_length_annular_zipper",
            "first_vertices": int(first.size),
            "second_vertices": int(second_source.size),
            "generated_faces": int(best["faces"].shape[0]),
            "selected_phase": int(best["phase"]),
            "cost": float(best["cost"]),
            "correspondence": best["correspondence"],
            "local_intersection": best["local_intersection"],
            "attempted_phases": attempted,
            "source_boundaries_resampled": False,
        },
    }


def _arc_length_zipper(points: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_parameter = _loop_parameters(points[first])
    second_parameter = _loop_parameters(points[second])
    a, b = np.r_[first, first[0]], np.r_[second, second[0]]
    i = j = 0
    faces: list[list[int]] = []
    while i < first.size or j < second.size:
        take_first = j == second.size or (
            i < first.size and first_parameter[i + 1] <= second_parameter[j + 1]
        )
        faces.append(
            [int(a[i]), int(a[i + 1]), int(b[j])]
            if take_first
            else [int(a[i]), int(b[j + 1]), int(b[j])]
        )
        i += int(take_first)
        j += int(not take_first)
    return np.asarray(faces, dtype=np.int64)


def _ordered_correspondence(
    first_points: np.ndarray,
    second_points: np.ndarray,
    first_normals: np.ndarray,
    second_normals: np.ndarray,
) -> dict[str, float]:
    first_parameter = _loop_parameters(first_points)
    second_parameter = _loop_parameters(second_points)
    samples = np.unique(np.r_[first_parameter[:-1], second_parameter[:-1]])
    first_samples = _sample_closed(first_points, first_parameter, samples)
    second_samples = _sample_closed(second_points, second_parameter, samples)
    first_normal_samples = _normalize_rows(
        _sample_closed(first_normals, first_parameter, samples)
    )
    second_normal_samples = _normalize_rows(
        _sample_closed(second_normals, second_parameter, samples)
    )
    distances = np.linalg.norm(first_samples - second_samples, axis=1)
    normal_dots = np.einsum("ij,ij->i", first_normal_samples, second_normal_samples)
    return {
        "maximum": float(distances.max(initial=0.0)),
        "rms": float(np.sqrt(np.mean(distances * distances))),
        "minimum_normal_dot": float(normal_dots.min(initial=1.0)),
        "mean_normal_dot": float(normal_dots.mean()) if normal_dots.size else -1.0,
    }


def _sample_closed(values: np.ndarray, parameters: np.ndarray, samples: np.ndarray) -> np.ndarray:
    closed = np.vstack([values, values[0]])
    indices = np.searchsorted(parameters, samples, side="right") - 1
    indices = np.clip(indices, 0, values.shape[0] - 1)
    spans = parameters[indices + 1] - parameters[indices]
    fractions = (samples - parameters[indices]) / np.maximum(spans, 1e-30)
    return closed[indices] * (1.0 - fractions[:, None]) + closed[indices + 1] * fractions[:, None]


def _local_intersection(points: np.ndarray, faces: np.ndarray, config: Any) -> dict[str, Any]:
    try:
        local_points = points[np.unique(faces)]
        local_scale = max(float(np.linalg.norm(np.ptp(local_points, axis=0))), 1e-30)
        return validate_local_patch_intersections(
            points,
            np.empty((0, 3), dtype=np.int64),
            faces,
            geometric_tolerance=local_scale * 1.0e-12,
            max_candidate_pairs=config.max_intersection_candidate_pairs,
        )
    except Exception as exc:
        return {
            "status": "check_failed",
            "passed": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }


def _phase_candidates(first: np.ndarray, second: np.ndarray, limit: int) -> np.ndarray:
    distances = np.linalg.norm(second - first[0], axis=1)
    nearest_count = min((limit + 1) // 2, second.shape[0])
    even_count = min(limit - nearest_count, second.shape[0])
    nearest = np.argsort(distances, kind="stable")[:nearest_count]
    even = np.linspace(0, second.shape[0] - 1, even_count, dtype=np.int64)
    candidates = np.unique(np.r_[nearest, even])
    return candidates[np.lexsort((candidates, distances[candidates]))][:limit]


def _loop_parameters(points: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return np.r_[0.0, np.cumsum(lengths) / max(float(lengths.sum()), 1e-30)]


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=1)
    return np.divide(
        values,
        lengths[:, None],
        out=np.zeros_like(values),
        where=lengths[:, None] > 0.0,
    )


def _bridge_failure(code: str, **diagnostics: Any) -> dict[str, Any]:
    return {
        "success": False,
        "failure_reason_codes": [code],
        "faces": np.empty((0, 3), dtype=np.int64),
        "diagnostics": diagnostics,
    }
