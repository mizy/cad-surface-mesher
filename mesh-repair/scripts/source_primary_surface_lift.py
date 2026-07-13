from __future__ import annotations

from typing import Any

import numpy as np

from source_primary_patch_geometry import BoundaryFrame
from source_primary_patch_parameterization import points_in_polygon


def build_boundary_lift_controls(
    analysis: dict[str, Any],
    *,
    normal_control_offset_ratio: float,
    minimum_normal_graph_component: float,
    minimum_normal_control_coverage: float,
    maximum_normal_control_gap_ratio: float,
    maximum_controls: int,
) -> dict[str, Any]:
    """Build mandatory position/normal/curvature controls without dropping any."""

    frame: BoundaryFrame = analysis["frame"]
    uv = frame.boundary_uv
    depth = frame.boundary_depth
    normals = np.asarray(analysis["normal"]["boundary_vertex_normals"], dtype=np.float64)
    local_normals = np.column_stack(
        (normals @ frame.u_axis, normals @ frame.v_axis, normals @ frame.normal)
    )
    edge_lengths = np.linalg.norm(np.roll(uv, -1, axis=0) - uv, axis=1)
    base_offset = max(
        float(np.median(edge_lengths)) * normal_control_offset_ratio,
        frame.geometric_tolerance * 16.0,
    )
    hessian = np.asarray(analysis["curvature"].get("height_hessian", []), dtype=np.float64)
    curvature_reliable = bool(
        analysis["curvature"].get("reliable")
        and hessian.shape == (2, 2)
        and np.all(np.isfinite(hessian))
    )
    if not curvature_reliable:
        hessian = np.zeros((2, 2), dtype=np.float64)
    pseudo_uv: list[np.ndarray] = []
    pseudo_depth: list[float] = []
    usable = np.zeros(uv.shape[0], dtype=bool)
    for index, point in enumerate(uv):
        normal_component = float(local_normals[index, 2])
        if abs(normal_component) < minimum_normal_graph_component:
            continue
        tangent = uv[(index + 1) % uv.shape[0]] - uv[index - 1]
        tangent_length = float(np.linalg.norm(tangent))
        if tangent_length <= frame.geometric_tolerance:
            continue
        inward = np.asarray([tangent[1], -tangent[0]]) / tangent_length
        offset = base_offset
        candidate = point + inward * offset
        for _ in range(8):
            if points_in_polygon(
                candidate[None, :],
                uv,
                tolerance=frame.geometric_tolerance,
                include_boundary=False,
            )[0]:
                break
            offset *= 0.5
            candidate = point + inward * offset
        else:
            continue
        gradient = -local_normals[index, :2] / normal_component
        delta = candidate - point
        value = float(
            depth[index]
            + np.dot(gradient, delta)
            + 0.5 * np.dot(delta, hessian @ delta)
        )
        if not np.isfinite(value):
            continue
        usable[index] = True
        pseudo_uv.append(candidate)
        pseudo_depth.append(value)
    # Vertex counts are not a geometric coverage measure on non-uniform CAD rings.
    # Voronoi arc weights assign half of each adjacent edge to its endpoint.
    arc_weights = 0.5 * (edge_lengths + np.roll(edge_lengths, 1))
    perimeter = float(edge_lengths.sum())
    coverage = float(np.sum(arc_weights[usable]) / max(perimeter, 1.0e-30))
    gap_ratio = _maximum_cyclic_false_weight(usable, arc_weights) / max(
        perimeter, 1.0e-30
    )
    if (
        len(pseudo_uv) < 3
        or coverage < minimum_normal_control_coverage
        or gap_ratio > maximum_normal_control_gap_ratio
    ):
        return {
            "success": False,
            "failure_reason_codes": ["curved_patch_boundary_normal_constraints_insufficient"],
            "diagnostics": {
                "usable_normal_controls": len(pseudo_uv),
                "normal_control_arc_coverage": coverage,
                "maximum_missing_control_arc_ratio": gap_ratio,
            },
        }
    control_count = uv.shape[0] + len(pseudo_uv)
    if control_count > maximum_controls:
        return {
            "success": False,
            "failure_reason_codes": ["curved_patch_mandatory_control_limit_exceeded"],
            "diagnostics": {
                "mandatory_control_count": control_count,
                "maximum_controls": maximum_controls,
            },
        }
    return {
        "success": True,
        "failure_reason_codes": [],
        "uv": np.vstack([uv, np.asarray(pseudo_uv)]),
        "depth": np.concatenate([depth, np.asarray(pseudo_depth)]),
        "diagnostics": {
            "boundary_position_controls": int(uv.shape[0]),
            "boundary_normal_curvature_controls": len(pseudo_uv),
            "retained_controls": control_count,
            "normal_control_offset": base_offset,
            "normal_control_arc_coverage": coverage,
            "maximum_missing_control_arc_ratio": gap_ratio,
            "curvature_reliable": curvature_reliable,
            "curvature_model": "directional_quadratic_height_hessian",
        },
    }


def fit_thin_plate_surface(
    uv: np.ndarray, depth: np.ndarray, smoothing: float
) -> dict[str, Any]:
    """Fit a regularized thin-plate height field in a normalized local chart."""

    try:
        raw_controls = np.asarray(uv)
        raw_values = np.asarray(depth)
    except (TypeError, ValueError) as exc:
        return {"success": False, "diagnostics": {"message": str(exc)}}
    smoothing_valid = (
        not isinstance(smoothing, (bool, np.bool_))
        and isinstance(smoothing, (int, float, np.integer, np.floating))
        and np.isfinite(float(smoothing))
        and float(smoothing) >= 0.0
    )
    if (
        raw_controls.ndim != 2
        or raw_controls.shape[1:] != (2,)
        or raw_values.ndim != 1
        or not np.issubdtype(raw_controls.dtype, np.number)
        or np.iscomplexobj(raw_controls)
        or not np.issubdtype(raw_values.dtype, np.number)
        or np.iscomplexobj(raw_values)
        or not smoothing_valid
    ):
        return {
            "success": False,
            "diagnostics": {"message": "thin-plate controls and smoothing must be finite"},
        }
    controls = raw_controls.astype(np.float64, copy=False)
    values = raw_values.astype(np.float64, copy=False)
    if (
        controls.shape[0] < 3
        or controls.shape[0] != values.size
        or not np.all(np.isfinite(controls))
        or not np.all(np.isfinite(values))
    ):
        return {
            "success": False,
            "diagnostics": {"message": "thin-plate controls must have finite matching rows"},
        }
    center = controls.mean(axis=0)
    scale = max(float(np.linalg.norm(np.ptp(controls, axis=0))), 1e-30)
    normalized = (controls - center) / scale
    difference = normalized[:, None, :] - normalized[None, :, :]
    raw_kernel = _thin_plate_kernel(np.linalg.norm(difference, axis=2))
    kernel = raw_kernel.copy()
    kernel.flat[:: kernel.shape[0] + 1] += smoothing
    affine = np.column_stack((np.ones(controls.shape[0]), normalized))
    system = np.block(
        [[kernel, affine], [affine.T, np.zeros((3, 3), dtype=np.float64)]]
    )
    right = np.concatenate([values, np.zeros(3, dtype=np.float64)])
    try:
        solution, _, rank, singular = np.linalg.lstsq(system, right, rcond=1e-12)
    except np.linalg.LinAlgError as exc:
        return {"success": False, "diagnostics": {"message": str(exc)}}
    predicted = raw_kernel @ solution[: controls.shape[0]] + affine @ solution[controls.shape[0] :]
    residual = float(np.max(np.abs(predicted - values), initial=0.0))
    solved = bool(
        rank == system.shape[0]
        and np.all(np.isfinite(solution))
        and np.isfinite(residual)
    )
    return {
        "success": solved,
        "normalized_controls": normalized,
        "center": center,
        "scale": scale,
        "radial_weights": solution[: controls.shape[0]],
        "affine_weights": solution[controls.shape[0] :],
        "diagnostics": {
            "method": "regularized_thin_plate_spline",
            "control_count": int(controls.shape[0]),
            "system_rank": int(rank),
            "system_size": int(system.shape[0]),
            "singular_value_min": float(singular.min()) if singular.size else None,
            "control_residual_max": residual,
            "smoothing": float(smoothing),
        },
    }


def evaluate_thin_plate_surface(model: dict[str, Any], uv: np.ndarray) -> np.ndarray:
    queries = (np.asarray(uv, dtype=np.float64) - model["center"]) / model["scale"]
    difference = queries[:, None, :] - model["normalized_controls"][None, :, :]
    kernel = _thin_plate_kernel(np.linalg.norm(difference, axis=2))
    affine = np.column_stack((np.ones(queries.shape[0]), queries))
    return kernel @ model["radial_weights"] + affine @ model["affine_weights"]


def limit_surface_controls(
    uv: np.ndarray, depth: np.ndarray, limit: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if uv.shape[0] <= limit:
        selected = np.arange(uv.shape[0], dtype=np.int64)
        return np.asarray(uv).copy(), np.asarray(depth).copy(), selected
    selected = np.unique(np.linspace(0, uv.shape[0] - 1, limit, dtype=np.int64))
    return (
        np.asarray(uv)[selected].copy(),
        np.asarray(depth)[selected].copy(),
        selected,
    )


def validate_lift_depth(
    depth: np.ndarray,
    control_depth: np.ndarray,
    frame: BoundaryFrame,
    *,
    max_absolute_depth_ratio: float,
    max_overshoot_ratio: float,
) -> dict[str, Any]:
    absolute_limit = max_absolute_depth_ratio * frame.footprint_diagonal
    margin = max_overshoot_ratio * frame.footprint_diagonal
    lower = float(np.min(control_depth) - margin)
    upper = float(np.max(control_depth) + margin)
    finite = bool(np.all(np.isfinite(depth)))
    absolute_ok = bool(np.max(np.abs(depth), initial=0.0) <= absolute_limit)
    overshoot_ok = bool(np.all((depth >= lower) & (depth <= upper)))
    reason = None
    if not finite:
        reason = "curved_patch_depth_non_finite"
    elif not absolute_ok:
        reason = "curved_patch_depth_limit_exceeded"
    elif not overshoot_ok:
        reason = "curved_patch_thin_plate_overshoot"
    return {
        "passed": reason is None,
        "reason_code": reason,
        "depth_min": float(depth.min()) if depth.size else None,
        "depth_max": float(depth.max()) if depth.size else None,
        "absolute_limit": absolute_limit,
        "control_envelope_with_margin": [lower, upper],
    }


def _thin_plate_kernel(radius: np.ndarray) -> np.ndarray:
    squared = radius * radius
    return np.where(
        radius > 0.0,
        squared * np.log(np.maximum(radius, 1e-300)),
        0.0,
    )


def _maximum_cyclic_false_weight(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0 or np.all(values):
        return 0.0
    if not np.any(values):
        return float(np.sum(weights))
    doubled = np.concatenate([values, values])
    doubled_weights = np.concatenate([weights, weights])
    longest = current = 0.0
    for value, weight in zip(doubled, doubled_weights, strict=True):
        current = 0.0 if value else current + float(weight)
        longest = max(longest, current)
    return min(longest, float(np.sum(weights)))
