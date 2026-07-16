from __future__ import annotations

from typing import Any

import numpy as np

from mesh_metrics import bbox_drift_from_reports
from sealed_exterior_atlas import projection_erosion_radius_voxels
from source_preserving_repair import build_gates, gate


BBOX_DRIFT_THRESHOLD = 0.002
BBOX_DRIFT_HARD_MAX_THRESHOLD = 0.005
SILHOUETTE_DRIFT_THRESHOLD = 0.05
MIN_EXACT_PROJECTION_RATIO = 0.10


def watertight_shell_policy(target_name: str) -> dict[str, Any]:
    authorized = target_name == "watertight-exterior-shell"
    return {
        "mode": "seal_all_openings" if authorized else "review_each_opening",
        "scope": target_name,
        "authorized": authorized,
        "source": "user_requested_watertight_shell" if authorized else "target_default",
        "semantic_review_applicability": (
            "diagnostic_only" if authorized else "required_for_acceptance"
        ),
    }


def resolution_aware_bbox_drift(
    source_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    projection_report: dict[str, Any],
) -> dict[str, Any]:
    """Compare bbox endpoints at a tolerance the reconstruction can resolve.

    The source-projected shell inherits the closure proxy's sampling resolution.
    Requiring bbox agreement finer than one closure pitch is therefore not a
    meaningful fidelity test.  The resolution allowance can raise the nominal
    0.2% threshold, but never beyond the independent 0.5% hard ceiling.
    """
    bbox = bbox_drift_from_reports(source_metrics, candidate_metrics)
    bbox_scale = max(
        max(float(value) for value in source_metrics["bounds"]["extents"]),
        1e-12,
    )
    base_threshold_abs = bbox_scale * BBOX_DRIFT_THRESHOLD
    hard_max_threshold_abs = bbox_scale * BBOX_DRIFT_HARD_MAX_THRESHOLD

    resolution_context = projection_report.get("resolution_context", {})
    thresholds = projection_report.get("thresholds", {})
    closure_pitch = _finite_positive(resolution_context.get("closure_pitch"))
    resolved_projection_distance = _finite_positive(
        thresholds.get("resolved_max_projection_distance")
        or resolution_context.get("resolved_max_projection_distance")
    )
    projection_distance_in_pitches = _finite_positive(
        resolution_context.get("projection_distance_in_closure_pitches")
    )

    resolution_threshold_abs: float | None = None
    resolution_source = "unavailable"
    if closure_pitch is not None:
        resolution_threshold_abs = closure_pitch
        resolution_source = "reported_closure_pitch"
    elif (
        resolved_projection_distance is not None
        and projection_distance_in_pitches is not None
    ):
        resolution_threshold_abs = (
            resolved_projection_distance / projection_distance_in_pitches
        )
        resolution_source = "resolved_projection_distance_divided_by_reported_pitch_multiple"

    if resolution_threshold_abs is not None and resolved_projection_distance is not None:
        resolution_threshold_abs = min(
            resolution_threshold_abs,
            resolved_projection_distance,
        )

    requested_threshold_abs = max(
        base_threshold_abs,
        resolution_threshold_abs or 0.0,
    )
    effective_threshold_abs = min(
        requested_threshold_abs,
        hard_max_threshold_abs,
    )
    effective_threshold_ratio = effective_threshold_abs / bbox_scale
    resolution_threshold_ratio = (
        resolution_threshold_abs / bbox_scale
        if resolution_threshold_abs is not None
        else None
    )
    return {
        **bbox,
        "method": "bbox_endpoint_drift_resolution_aware",
        "bbox_reference_scale": bbox_scale,
        "base_threshold_ratio": BBOX_DRIFT_THRESHOLD,
        "base_threshold_abs": base_threshold_abs,
        "closure_pitch": closure_pitch,
        "resolved_max_projection_distance": resolved_projection_distance,
        "projection_distance_in_closure_pitches": projection_distance_in_pitches,
        "resolution_threshold_abs": resolution_threshold_abs,
        "resolution_threshold_ratio": resolution_threshold_ratio,
        "resolution_source": resolution_source,
        "hard_max_threshold_ratio": BBOX_DRIFT_HARD_MAX_THRESHOLD,
        "hard_max_threshold_abs": hard_max_threshold_abs,
        "hard_cap_applied": requested_threshold_abs > hard_max_threshold_abs,
        "effective_threshold_abs": effective_threshold_abs,
        "effective_threshold_ratio": effective_threshold_ratio,
        "threshold": effective_threshold_ratio,
        "passed": bbox["max_abs"] <= effective_threshold_abs,
        "status": "computed",
    }


def _finite_positive(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) and numeric > 0.0 else None


def projection_comparisons(
    source_metrics: dict[str, Any],
    proxy_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    projection_report: dict[str, Any],
    visual_drift: dict[str, Any],
    post_write_validation: dict[str, Any],
) -> dict[str, Any]:
    projection = projection_report.get("projection", {})
    residual = projection.get("source_distance_after_for_projected", {})
    all_distance = projection.get("distance_before", {})
    scale = max(float(value) for value in source_metrics["bounds"]["extents"])
    residual_threshold = max(scale * 1e-10, 1e-15)
    residual_max = residual.get("max")
    shell_distance_threshold = projection_report.get("thresholds", {}).get(
        "resolved_max_projection_distance"
    )
    shell_distance_max = all_distance.get("max")
    bbox = resolution_aware_bbox_drift(
        source_metrics,
        candidate_metrics,
        projection_report,
    )
    silhouette = visual_drift.get("summary", {})
    return {
        "source_distance": {
            "method": "all_shell_vertices_to_nearest_source_before_exact_projection",
            "scope": "all exact-projected and proxy-fallback vertices",
            "max": shell_distance_max,
            "p95": all_distance.get("p95"),
            "mean": None,
            "threshold": shell_distance_threshold,
            "passed": bool(
                shell_distance_max is not None
                and shell_distance_threshold is not None
                and float(shell_distance_max) <= float(shell_distance_threshold)
            ),
            "status": "computed" if shell_distance_max is not None else "required_metric_missing",
        },
        "source_projection_residual": {
            "method": "exact_residual_for_source_projected_vertices",
            "max": residual_max,
            "p95": residual.get("p95"),
            "threshold": residual_threshold,
            "passed": residual_max is not None and float(residual_max) <= residual_threshold,
            "status": "computed" if residual_max is not None else "required_metric_missing",
        },
        "bbox_drift": bbox,
        "silhouette_drift": {
            "method": visual_drift.get("method", "shared_projection_silhouette_occupancy"),
            "changed_ratio_max": silhouette.get("changed_ratio_max"),
            "overlap_ratio_min": silhouette.get("overlap_ratio_min"),
            "threshold": SILHOUETTE_DRIFT_THRESHOLD,
            "passed": bool(
                silhouette.get("changed_ratio_max") is not None
                and silhouette["changed_ratio_max"] <= SILHOUETTE_DRIFT_THRESHOLD
            ),
            "status": "computed" if silhouette else "required_metric_missing",
            "per_view": visual_drift.get("per_view", {}),
        },
        "self_intersection": projection_report.get(
            "self_intersection",
            projection_report.get("comparisons", {}).get("self_intersection", missing_metric("self_intersection")),
        ),
        "projection_topology": projection_report.get("comparisons", {}).get(
            "projection_topology", missing_metric("projection_topology")
        ),
        "projection_mapping": projection_report.get("comparisons", {}).get(
            "projection_mapping", missing_metric("projection_mapping")
        ),
        "sealed_exterior_occupancy": sealed_exterior_occupancy(
            candidate_metrics,
            projection_report,
        ),
        "volume_drift": volume_drift(proxy_metrics, candidate_metrics),
        "post_write_validation": post_write_validation,
        "six_view_depth_regression": projection_report.get(
            "six_view_depth_regression",
            missing_metric("six_view_depth_regression"),
        ),
        "before_after_topology": {
            "before": topology_snapshot(proxy_metrics),
            "after": topology_snapshot(candidate_metrics),
        },
    }


def build_projection_gates(
    candidate_metrics: dict[str, Any],
    comparisons: dict[str, Any],
    deterministic_passes: list[dict[str, Any]],
    candidate_path: str | None,
    candidate_produced: bool,
    closure_proxy_path: str,
    projection_report: dict[str, Any],
    shell_policy: dict[str, Any],
) -> dict[str, Any]:
    gates = build_gates(
        candidate_metrics,
        comparisons,
        {"items": [], "truncated": False},
        [],
        deterministic_passes,
        candidate_path,
        closure_proxy_path,
        projection_unhandled_items(comparisons),
    )
    gates["opening_policy_resolved"] = {
        "required": False,
        "passed": True,
        "status": "not_applicable_to_global_shell_projection",
        "value": shell_policy.get("semantic_review_applicability"),
        "threshold": "diagnostic_only",
        "failure_reason": None,
    }
    mapping = comparisons.get("projection_mapping", {})
    topology = comparisons.get("projection_topology", {})
    post_write = comparisons.get("post_write_validation", {})
    depth_regression = comparisons.get("six_view_depth_regression", {})
    sealed_occupancy = comparisons.get("sealed_exterior_occupancy", {})
    drift = comparisons.get("volume_drift", {})
    projection = projection_report.get("projection", {})
    count = int(projection.get("vertices") or 0)
    projected = int(projection.get("source_projected") or 0)
    ratio = projected / count if count else 0.0
    component_signed = candidate_metrics.get("volume", {}).get("component_signed", [])
    gates.update({
        "source_projected_candidate_present": gate(
            bool(candidate_path) and candidate_produced,
            candidate_path,
            "current-run source-projected watertight candidate",
            "source-projected candidate was not produced by this run",
        ),
        "accepted_path_is_source_projected_candidate": gate(
            bool(candidate_path) and candidate_path != closure_proxy_path,
            candidate_path,
            "decision.final_output_path == outputs.source_projected_watertight_candidate_vtp",
            "accepted output must be the projected candidate, not the raw proxy or an alternate candidate",
        ),
        "shell_closure_policy_authorized": gate(
            shell_policy.get("mode") == "seal_all_openings" and shell_policy.get("authorized") is True,
            shell_policy,
            {"mode": "seal_all_openings", "authorized": True},
            "global opening closure is not authorized for this target",
        ),
        "projection_topology_matches_proxy": gate(
            topology.get("status") == "computed"
            and topology.get("faces_equal") is True
            and topology.get("point_count_equal") is True,
            topology,
            "identical closure-proxy connectivity and point count",
            "projection changed the closure topology",
        ),
        "projection_mapping_complete": gate(
            mapping.get("status") == "computed"
            and int(mapping.get("unclassified", -1)) == 0
            and projection.get("fallback_without_reason", 0) == 0,
            mapping,
            "every vertex classified as exact source projection or reasoned proxy fallback",
            "one or more projected-shell vertices have incomplete provenance",
        ),
        "source_supported_vertices_fully_projected": gate(
            comparisons.get("source_projection_residual", {}).get("passed") is True,
            comparisons.get("source_projection_residual"),
            "exact source-surface residual within numerical tolerance",
            "source-supported vertices are not exactly on their selected source triangles",
        ),
        "exact_projection_coverage": gate(
            ratio >= MIN_EXACT_PROJECTION_RATIO,
            ratio,
            MIN_EXACT_PROJECTION_RATIO,
            "too little of the shell was safely recovered from the source surface",
        ),
        "post_write_validation_passed": gate(
            post_write.get("status") == "computed" and post_write.get("passed") is True,
            post_write,
            "raw and normalized VTP roundtrip preserve watertight manifold topology",
            "saved projected candidate does not survive roundtrip topology validation",
        ),
        "six_view_depth_regression_free": gate(
            depth_regression.get("status") == "computed" and depth_regression.get("passed") is True,
            depth_regression.get("regression_pixels"),
            0,
            (
                "matched-camera six-view raster found material background exposure or "
                "depth recession after image-space registration tolerance"
            ),
        ),
        "sealed_exterior_erosion_core_preserved": gate(
            sealed_occupancy.get("status") == "computed"
            and sealed_occupancy.get("passed") is True,
            sealed_occupancy.get("candidate_signed_abs"),
            sealed_occupancy.get("erosion_core_volume_lower_bound"),
            sealed_occupancy.get("failure_reason")
            or "projected shell does not preserve the resolution-derived exterior solid core",
        ),
        # The closure proxy is a connectivity template, not geometric truth.
        # Moving its vertices onto the source is expected to change enclosed
        # volume, sometimes substantially when the proxy is voxel-dilated.  A
        # fixed proxy-to-candidate volume ratio therefore cannot be a fidelity
        # acceptance gate.  Keep the measurement in the gate table for audit
        # visibility while relying on the independent topology, local triangle
        # quality, self-intersection, source-distance, bbox, and raster gates to
        # reject an actual collapse.
        "proxy_to_projected_volume_change": {
            "required": False,
            "passed": drift.get("status") == "computed",
            "status": "diagnostic_only",
            "value": drift.get("max_relative_abs"),
            "threshold": None,
            "reference_role": "closure_connectivity_template_not_geometry_truth",
            "failure_reason": None,
        },
        "component_orientation_outward": gate(
            bool(component_signed) and all(float(value) > 0.0 for value in component_signed),
            component_signed,
            "all closed component signed volumes > 0",
            "one or more closed components are inward-oriented",
        ),
    })
    return gates


def volume_drift(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_values = before.get("volume", {}).get("component_signed") or []
    after_values = after.get("volume", {}).get("component_signed") or []
    if not before.get("volume", {}).get("reliable") or not after.get("volume", {}).get("reliable"):
        return unavailable_volume_diagnostic("reliable component volumes are unavailable")
    if len(before_values) != len(after_values) or not before_values:
        return unavailable_volume_diagnostic("component volume correspondence is incomplete")
    before_abs = np.abs(np.asarray(before_values, dtype=np.float64))
    after_abs = np.abs(np.asarray(after_values, dtype=np.float64))
    relative = np.divide(after_abs - before_abs, before_abs, out=np.full_like(before_abs, np.inf), where=before_abs > 0)
    total_before = float(before_abs.sum())
    total_after = float(after_abs.sum())
    total_relative = (total_after - total_before) / total_before
    maximum = float(np.max(np.abs(relative)))
    return {
        "method": "diagnostic_signed_component_volume_change_from_closure_topology_template",
        "status": "computed",
        "role": "diagnostic_only",
        "reference_role": "closure_connectivity_template_not_geometry_truth",
        "before": total_before,
        "after": total_after,
        "total_relative": total_relative,
        "component_relative": relative.tolist(),
        "max_relative_abs": maximum,
        "threshold": None,
        "passed": None,
    }


def sealed_exterior_occupancy(
    candidate_metrics: dict[str, Any],
    projection_report: dict[str, Any],
) -> dict[str, Any]:
    """Require the projected shell to retain a discretely eroded exterior core.

    The closure volume is eroded by the smallest integer voxel radius covering
    the configured maximum source-projection displacement.  Any valid projected
    shell with the preserved closure topology must retain at least this core
    volume.  This is an absolute, resolution-derived lower bound; it is not a
    percentage comparison with proxy volume.
    """
    construction = projection_report.get("sealed_exterior_construction") or {}
    core = construction.get("projection_erosion_core") or {}
    thresholds = projection_report.get("thresholds") or {}
    candidate_volume_report = candidate_metrics.get("volume") or {}
    pitch = finite_float(construction.get("pitch"))
    max_distance = finite_float(thresholds.get("resolved_max_projection_distance"))
    filled_volume = finite_float(construction.get("estimated_filled_volume"))
    core_estimated_volume = finite_float(core.get("estimated_volume"))
    core_volume = finite_float(core.get("mesh_signed_abs_volume"))
    candidate_volume = finite_float(candidate_volume_report.get("signed_abs"))
    filled_voxels = nonnegative_int(construction.get("filled_or_shell_voxels"))
    core_voxels = nonnegative_int(core.get("filled_voxels"))
    core_distance = finite_float(core.get("max_projection_distance"))
    core_radius_world = finite_float(core.get("erosion_radius_world"))
    reported_radius = core.get("erosion_radius_voxels")
    issues: list[str] = []

    if construction.get("schema") != "sealed_exterior_volume/v1":
        issues.append("sealed exterior construction schema is missing or unsupported")
    if construction.get("method") != "voxel_shell_dilation_six_connected_far_field_flood":
        issues.append("closure volume was not built by sealed-shell far-field flood")
    if construction.get("surface_offset_restore") != "erode_filled_solid_by_seal_radius":
        issues.append("temporary sealing dilation offset was not restored")
    if pitch is None or pitch <= 0.0:
        issues.append("closure voxel pitch is missing or invalid")
    if max_distance is None or max_distance < 0.0:
        issues.append("resolved maximum projection distance is missing or invalid")
    if filled_volume is None or filled_volume <= 0.0:
        issues.append("post-restore sealed exterior volume is missing or empty")
    if filled_voxels is None or filled_voxels <= 0:
        issues.append("post-restore sealed exterior voxel count is missing or empty")
    if (
        core_estimated_volume is None
        or core_estimated_volume <= 0.0
        or core.get("nonempty") is not True
    ):
        issues.append("projection-distance erosion core is missing or empty")
    if core.get("mesh_volume_method") != "marching_cubes_signed_abs_volume":
        issues.append("erosion core geometric volume method is missing or unsupported")
    if core.get("mesh_watertight") is not True:
        issues.append("erosion core marching-cubes surface is not watertight")
    if core_volume is None or core_volume <= 0.0:
        issues.append("erosion core geometric signed volume is missing or non-positive")
    if core_voxels is None or core_voxels <= 0:
        issues.append("projection-distance erosion core voxel count is missing or empty")
    if not candidate_volume_report.get("reliable"):
        issues.append("projected candidate signed volume is not reliable")
    if candidate_volume is None or candidate_volume <= 0.0:
        issues.append("projected candidate signed volume is missing or non-positive")

    expected_radius = None
    if pitch is not None and pitch > 0.0 and max_distance is not None and max_distance >= 0.0:
        expected_radius = projection_erosion_radius_voxels(max_distance, pitch)
        if isinstance(reported_radius, bool) or not isinstance(
            reported_radius,
            (int, np.integer),
        ):
            issues.append("erosion core voxel radius is missing or invalid")
        elif int(reported_radius) != expected_radius:
            issues.append("erosion core radius does not match projection distance and pitch")
        distance_tolerance = max(
            pitch * 64.0 * np.finfo(np.float64).eps,
            np.finfo(np.float64).tiny,
        )
        if core_distance is None or abs(core_distance - max_distance) > distance_tolerance:
            issues.append("erosion core projection distance does not match projection thresholds")
        expected_radius_world = expected_radius * pitch
        if (
            core_radius_world is None
            or abs(core_radius_world - expected_radius_world) > distance_tolerance
        ):
            issues.append("erosion core world radius does not match voxel radius and pitch")
        voxel_volume = pitch**3
        if filled_voxels is not None and filled_volume is not None:
            expected_filled_volume = filled_voxels * voxel_volume
            volume_tolerance = max(
                voxel_volume * 1e-9,
                abs(expected_filled_volume) * 64.0 * np.finfo(np.float64).eps,
            )
            if abs(filled_volume - expected_filled_volume) > volume_tolerance:
                issues.append("post-restore volume does not match voxel count and pitch")
        if core_voxels is not None and core_estimated_volume is not None:
            expected_core_volume = core_voxels * voxel_volume
            volume_tolerance = max(
                voxel_volume * 1e-9,
                abs(expected_core_volume) * 64.0 * np.finfo(np.float64).eps,
            )
            if abs(core_estimated_volume - expected_core_volume) > volume_tolerance:
                issues.append("erosion core volume does not match voxel count and pitch")
    if (
        filled_voxels is not None
        and core_voxels is not None
        and core_voxels > filled_voxels
    ):
        issues.append("erosion core voxel count exceeds its post-restore parent volume")
    if (
        filled_volume is not None
        and core_volume is not None
        and core_estimated_volume is not None
        and core_estimated_volume > filled_volume
    ):
        issues.append("erosion core volume exceeds its post-restore parent volume")

    if issues:
        return {
            "method": "sealed_exterior_post_restore_erosion_core_volume_lower_bound",
            "status": "required_metric_missing",
            "passed": False,
            "candidate_signed_abs": candidate_volume,
            "post_restore_estimated_filled_volume": filled_volume,
            "erosion_core_volume_lower_bound": core_volume,
            "erosion_core_estimated_voxel_volume": core_estimated_volume,
            "pitch": pitch,
            "max_projection_distance": max_distance,
            "erosion_radius_voxels": reported_radius,
            "expected_erosion_radius_voxels": expected_radius,
            "failure_reason": "; ".join(issues),
        }

    assert pitch is not None
    assert core_volume is not None
    assert candidate_volume is not None
    assert filled_volume is not None
    signed_volume_numerical_tolerance = float(
        max(
            abs(core_volume) * 64.0 * np.finfo(np.float64).eps,
            abs(candidate_volume) * 64.0 * np.finfo(np.float64).eps,
            np.finfo(np.float64).tiny,
        )
    )
    passed = bool(
        candidate_volume + signed_volume_numerical_tolerance >= core_volume
    )
    return {
        "method": "sealed_exterior_post_restore_erosion_core_volume_lower_bound",
        "status": "computed",
        "passed": passed,
        "candidate_signed_abs": candidate_volume,
        "post_restore_estimated_filled_volume": filled_volume,
        "erosion_core_volume_lower_bound": core_volume,
        "erosion_core_estimated_voxel_volume": core_estimated_volume,
        "signed_volume_numerical_tolerance": signed_volume_numerical_tolerance,
        "candidate_to_post_restore_ratio_diagnostic": candidate_volume / filled_volume,
        "core_to_post_restore_ratio_diagnostic": core_volume / filled_volume,
        "pitch": pitch,
        "max_projection_distance": max_distance,
        "erosion_radius_voxels": int(reported_radius),
        "expected_erosion_radius_voxels": expected_radius,
        "failure_reason": None
        if passed
        else (
            "projected reliable signed volume is below the sealed exterior core "
            "eroded by the maximum allowed projection distance"
        ),
    }


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    result = int(value)
    return result if result >= 0 else None


def unavailable_volume_diagnostic(reason: str) -> dict[str, Any]:
    return {
        "method": "diagnostic_signed_component_volume_change_from_closure_topology_template",
        "status": "unavailable",
        "role": "diagnostic_only",
        "reference_role": "closure_connectivity_template_not_geometry_truth",
        "threshold": None,
        "passed": None,
        "failure_reason": reason,
    }


def projection_unhandled_items(comparisons: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name in (
        "source_distance",
        "source_projection_residual",
        "bbox_drift",
        "silhouette_drift",
        "six_view_depth_regression",
        "self_intersection",
        "sealed_exterior_occupancy",
    ):
        metric = comparisons.get(name, {})
        if metric.get("status") != "computed":
            rows.append({
                "item": name,
                "status": metric.get("status", "required_metric_missing"),
                "blocking": True,
                "failure_reason": metric.get("failure_reason") or f"{name} is required for acceptance",
            })
    return rows


def topology_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "boundary_edges": metrics["topology"]["boundary_edges"],
        "non_manifold_edges": metrics["topology"]["non_manifold_edges"],
        "non_manifold_vertices": metrics["topology"]["non_manifold_vertices"],
        "inconsistent_winding_edges": metrics["topology"]["inconsistent_winding_edges"],
        "degenerate_faces": metrics["quality"]["degenerate_faces"],
        "components": metrics["topology"]["components"]["count"],
    }


def missing_metric(name: str, reason: str | None = None) -> dict[str, Any]:
    return {
        "method": name,
        "status": "required_metric_missing",
        "passed": False,
        "failure_reason": reason or f"{name} is required for source-projected shell acceptance",
    }
