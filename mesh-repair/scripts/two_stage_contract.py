from __future__ import annotations

from pathlib import Path
from typing import Any

from mesh_metrics import bbox_drift_from_reports
from repair_inventory import inventory_truncation_items
from repair_report import two_stage_repair_report
from source_projected_contract import (
    build_projection_gates,
    local_policy_diagnostics,
    projection_comparisons,
    projection_unhandled_items,
    watertight_shell_policy,
)
from source_preserving_repair import (
    accepted_outputs,
    build_decision,
    build_ignored_outputs,
    input_truth,
)


SILHOUETTE_DRIFT_THRESHOLD = 0.05


def build_report(
    args: Any,
    stage_reports: dict[str, dict[str, Any]],
    extraction: dict[str, Any],
    deterministic_passes: list[dict[str, Any]],
    inventory_before: dict[str, Any],
    inventory_after: dict[str, Any],
    policy_packet: dict[str, Any],
    policy_decisions: list[dict[str, Any]],
    remesh: dict[str, Any],
    outputs: dict[str, Any],
    group_filter: dict[str, Any] | None,
    closure_visual_drift: dict[str, Any],
    hybrid_result: dict[str, Any],
    hybrid_visual_drift: dict[str, Any],
    projection_report: dict[str, Any],
    post_write_validation: dict[str, Any],
    projected_visual_drift: dict[str, Any],
) -> dict[str, Any]:
    candidate_path = outputs["source_preserving_candidate_vtp"]
    closure_proxy_path = outputs["closure_proxy_vtp"]
    hybrid_candidate_path = outputs.get("hybrid_fused_candidate_vtp")
    projected_candidate_path = outputs.get("source_projected_watertight_candidate_vtp")
    projected_candidate_produced = bool(outputs.get("source_projected_watertight_candidate_produced"))
    projected_candidate_metrics = stage_reports["source_projected_watertight_candidate"]
    patch_regions = hybrid_result.get("patch_regions", [])
    comparisons = projection_comparisons(
        stage_reports["deterministic_repair_candidate"],
        stage_reports["closure_proxy"],
        projected_candidate_metrics,
        projection_report,
        projected_visual_drift,
        post_write_validation,
    )
    hybrid_diagnostic_comparisons = hybrid_comparisons(
        stage_reports["stage1_exterior_candidate"],
        stage_reports["hybrid_fused_candidate"],
        hybrid_result.get("comparisons", {}),
        hybrid_visual_drift,
        closure_visual_drift,
    )
    hybrid_diagnostic_items = hybrid_unhandled_items(
        inventory_after,
        policy_packet,
        has_policy_file=args.policy_decisions is not None,
        patch_regions=patch_regions,
        comparisons=hybrid_diagnostic_comparisons,
    )
    shell_policy = watertight_shell_policy(args.target_name)
    unhandled_items = projection_unhandled_items(comparisons)
    gates = build_projection_gates(
        projected_candidate_metrics,
        comparisons,
        deterministic_passes,
        projected_candidate_path,
        projected_candidate_produced,
        closure_proxy_path,
        projection_report,
        shell_policy,
    )
    decision = build_decision(gates, projected_candidate_path)
    outputs = accepted_outputs(
        outputs,
        decision,
        projected_candidate_path or candidate_path,
        projected_candidate_path or candidate_path,
        source_fallback_path=candidate_path,
    )
    decision = {
        **decision,
        "outcome": outputs["mesh_result"]["status"],
        "source_fallback_path": outputs.get("source_fallback_vtp"),
    }
    ignored_outputs = build_ignored_outputs(
        closure_proxy_path=closure_proxy_path,
        stage1_path=outputs["stage1_exterior_candidate_vtp"],
        candidate_path=candidate_path,
        previews=outputs.get("previews", []),
        group_filter=group_filter,
        output_dir=args.output_dir,
        hybrid_candidate_path=hybrid_candidate_path,
        patch_regions=patch_regions,
        debug_artifacts=debug_artifact_paths(outputs),
    )
    if decision["status"] != "accepted" and projected_candidate_path:
        ignored_outputs.append({
            "path": projected_candidate_path,
            "kind": "rejected_source_projected_watertight_candidate",
            "reason": "source-projected shell did not pass every engineering acceptance gate",
            "safe_for_acceptance": False,
        })
    derived_artifacts = [outputs["stage1_exterior_candidate_vtp"], candidate_path, closure_proxy_path]
    if hybrid_candidate_path:
        derived_artifacts.append(hybrid_candidate_path)
    if projected_candidate_path:
        derived_artifacts.append(projected_candidate_path)
    truth = input_truth(Path(args.input), group_filter, derived_artifacts)
    truth["acceptance_source"] = "closure_proxy_connectivity_plus_deterministic_source_surface_positions"
    return {
        "decision": decision,
        "input": {
            "path": str(args.input),
            "kind": "mesh",
            "units": "unknown",
            "coordinate_convention": "unknown",
            "group_metadata_available": group_filter is not None,
        },
        "input_truth": truth,
        "output_contract": {
            "input_kind": "mesh",
            "output_kind": "watertight_mesh",
            "repair_domain": "mesh_domain_source_projected_watertight_shell",
            "cad_output": {"supported": False, "reason": "this prototype does not perform reverse CAD fitting"},
            "accepted_final_rule": (
                "decision.final_output_path == outputs.accepted_mesh_vtp == "
                "outputs.source_projected_watertight_candidate_vtp; closure_proxy_vtp supplies "
                "connectivity only and hybrid is an alternative diagnostic"
            ),
            "rejected_result_rule": (
                "outputs.mesh_result explicitly identifies a source-preserving fallback as unrepaired and not engineering-ready"
            ),
        },
        "target": {
            "name": args.target_name,
            "opening_policy": "seal all openings for the requested watertight exterior shell",
            "shell_policy": shell_policy,
            "component_count_role": "diagnostic_only",
        },
        "parameters": parameters_report(args),
        "limitations": limitations(group_filter),
        "group_filter": group_filter,
        "stages": stages_report(args, outputs, stage_reports, extraction, remesh),
        "inventory_before": inventory_before,
        "deterministic_passes": deterministic_passes,
        "unresolved_policy_packet": policy_packet,
        "policy_decisions": policy_decisions,
        "inventory_after": inventory_after,
        "patch_regions": patch_regions,
        "patch_region_summary": patch_region_summary(patch_regions),
        "hybrid_fusion": {
            "role": "alternative_diagnostic_not_eligible_for_acceptance",
            "face_provenance": hybrid_result.get("face_provenance"),
            "source_proxy_blend": hybrid_result.get("source_proxy_blend"),
            "debug_artifacts": hybrid_result.get("debug_artifacts", {}),
        },
        "source_projection": projection_report,
        "comparisons": comparisons,
        "hybrid_diagnostic_comparisons": hybrid_diagnostic_comparisons,
        "gates": gates,
        "outputs": outputs,
        "ignored_outputs": ignored_outputs,
        "unhandled_items": unhandled_items,
        "local_patch_policy_diagnostics": local_policy_diagnostics(hybrid_diagnostic_items),
        "repair_report": two_stage_repair_report(
            stage_reports,
            extraction,
            group_filter,
            remesh,
            deterministic_passes,
            patch_regions=patch_regions,
            decision=decision,
        ),
    }


def stages_report(
    args: Any,
    outputs: dict[str, Any],
    stage_reports: dict[str, dict[str, Any]],
    extraction: dict[str, Any],
    remesh: dict[str, Any],
) -> dict[str, Any]:
    hybrid_candidate_metrics = stage_reports.get("hybrid_fused_candidate")
    hybrid_candidate_path = outputs.get("hybrid_fused_candidate_vtp")
    hybrid_candidate_produced = outputs.get("hybrid_fused_candidate_produced")
    projected_path = outputs.get("source_projected_watertight_candidate_vtp")
    projected_metrics = stage_reports.get("source_projected_watertight_candidate")
    projected_produced = outputs.get("source_projected_watertight_candidate_produced")
    return {
        "original_input": {"path": str(args.input), "metrics": stage_reports["original_input"]},
        "stage1_exterior_candidate": {
            "path": outputs["stage1_exterior_candidate_vtp"],
            "source": "original_input",
            "provenance_fields": ["source_triangle_index"],
            "metrics": stage_reports["stage1_exterior_candidate"],
            "extraction": extraction,
        },
        "deterministic_repair_candidate": {
            "path": outputs["source_preserving_candidate_vtp"],
            "source": "stage1_exterior_candidate",
            "provenance_fields": ["source_triangle_index"],
            "metrics": stage_reports["deterministic_repair_candidate"],
        },
        "closure_proxy": {
            "path": outputs["closure_proxy_vtp"],
            "implicit_field_path": outputs.get("implicit_field_npz"),
            "source": "deterministic_repair_candidate",
            "metrics": stage_reports["closure_proxy"],
            "remesh": remesh,
            "accepted_final_geometry": False,
            "role": "immutable_watertight_connectivity_template",
        },
        "source_projected_watertight_candidate": {
            "path": projected_path,
            "source": "closure_proxy_topology_plus_deterministic_source_surface_positions",
            "metrics": projected_metrics,
            "status": (
                "accepted"
                if outputs.get("accepted_mesh_vtp") == projected_path
                else "candidate_produced_not_accepted"
                if projected_produced
                else "missing"
            ),
            "provenance_fields": [
                "source_projection_applied",
                "source_projection_fallback",
                "nearest_source_face_id",
                "source_projection_rejection_mask",
            ],
            "accepted_final_geometry": outputs.get("accepted_mesh_vtp") == projected_path,
        },
        "hybrid_fused_candidate": {
            "path": hybrid_candidate_path,
            "source": "deterministic_repair_candidate_plus_local_proxy_patches",
            "metrics": hybrid_candidate_metrics,
            "status": "alternative_diagnostic" if hybrid_candidate_produced else "missing",
            "provenance_fields": [
                "face_origin",
                "source_triangle_index",
                "fusion_region_id",
                "proxy_weight",
                "sdf_blend_weight",
            ],
            "accepted_final_geometry": False,
        },
    }


def parameters_report(args: Any) -> dict[str, Any]:
    return {
        "visibility_grid": args.visibility_grid,
        "visibility_min_views": getattr(args, "visibility_min_views", 1),
        "outside_flood_grid": getattr(args, "outside_flood_grid", None),
        "depth_tolerance": args.depth_tolerance,
        "requested_depth_tolerance": getattr(args, "requested_depth_tolerance", args.depth_tolerance),
        "depth_tolerance_bbox_ratio": getattr(args, "depth_tolerance_bbox_ratio", None),
        "depth_tolerance_edge_ratio": getattr(args, "depth_tolerance_edge_ratio", None),
        "dilate_rings": args.dilate_rings,
        "voxel_pitch": args.voxel_pitch,
        "requested_voxel_pitch": getattr(args, "requested_voxel_pitch", args.voxel_pitch),
        "voxel_pitch_source": getattr(args, "voxel_pitch_source", "explicit"),
        "voxel_pitch_bbox_divisor": getattr(args, "voxel_pitch_bbox_divisor", None),
        "voxel_pitch_bbox_max_extent": getattr(args, "voxel_pitch_bbox_max_extent", None),
        "sdf_band_voxels": getattr(args, "sdf_band_voxels", None),
        "sdf_smoothing_sigma": getattr(args, "sdf_smoothing_sigma", None),
        "max_sdf_memory_gb": getattr(args, "max_sdf_memory_gb", None),
        "remove_name_regex": args.remove_name_regex,
        "policy_decisions": str(args.policy_decisions) if args.policy_decisions else None,
        "policy_item_limit": args.policy_item_limit,
        "component_filter_thresholds": getattr(args, "component_filter_thresholds", None),
    }


def hybrid_comparisons(
    source_metrics: dict[str, Any],
    hybrid_metrics: dict[str, Any],
    fusion_comparisons: dict[str, Any],
    hybrid_visual_drift: dict[str, Any],
    closure_visual_drift: dict[str, Any],
) -> dict[str, Any]:
    bbox = bbox_drift_from_reports(source_metrics, hybrid_metrics)
    silhouette_summary = hybrid_visual_drift.get("summary", {})
    return {
        "source_distance": fusion_comparisons.get("source_distance", required_metric_missing("source_distance")),
        "patch_local_drift": fusion_comparisons.get("patch_local_drift", required_metric_missing("patch_local_drift")),
        "self_intersection": fusion_comparisons.get(
            "self_intersection",
            required_metric_missing("self_intersection"),
        ),
        "bbox_drift": {
            **bbox,
            "threshold": 0.002,
            "passed": bbox["max_ratio"] <= 0.002,
            "status": "computed",
        },
        "silhouette_drift": {
            "method": hybrid_visual_drift.get("method", "shared_projection_silhouette_occupancy"),
            "changed_ratio_max": silhouette_summary.get("changed_ratio_max"),
            "overlap_ratio_min": silhouette_summary.get("overlap_ratio_min"),
            "threshold": SILHOUETTE_DRIFT_THRESHOLD,
            "passed": bool(
                silhouette_summary.get("changed_ratio_max") is not None
                and silhouette_summary.get("changed_ratio_max") <= SILHOUETTE_DRIFT_THRESHOLD
            ),
            "status": "computed" if silhouette_summary else "required_metric_missing",
            "per_view": hybrid_visual_drift.get("per_view", {}),
        },
        "closure_proxy_silhouette_drift": closure_visual_drift,
        "before_after_topology": fusion_comparisons.get("before_after_topology", {}),
    }


def required_metric_missing(name: str) -> dict[str, Any]:
    return {
        "method": name,
        "max": None,
        "p95": None,
        "mean": None,
        "threshold": None,
        "passed": False,
        "status": "required_metric_missing",
        "failure_reason": f"{name} is required for hybrid candidate acceptance",
    }


def hybrid_unhandled_items(
    inventory: dict[str, Any],
    policy_packet: dict[str, Any],
    *,
    has_policy_file: bool,
    patch_regions: list[dict[str, Any]],
    comparisons: dict[str, Any],
) -> list[dict[str, Any]]:
    items = list(inventory.get("not_checked", []))
    items.extend(inventory_truncation_items(inventory, stage_label="inventory_after"))
    if policy_packet.get("truncated"):
        items.append(
            {
                "item": "policy_packet_full_inventory",
                "status": "truncated",
                "blocking": True,
                "reason_codes": ["opening_inventory_unresolved"],
                "failure_reason": "not all boundary regions were exported to the policy packet",
            }
        )
    if policy_packet.get("items") and not has_policy_file:
        items.append(
            {
                "item": "ai_policy_review",
                "status": "pending_review",
                "blocking": True,
                "failure_reason": "semantic opening policy decisions were not supplied for this run",
            }
        )
    for region in patch_regions:
        if region.get("selection_status") in {"hold_for_policy", "reject_patch"}:
            reason_codes = list(region.get("accept_reject_reason_codes") or [])
            if not reason_codes and region.get("rejection_reason"):
                reason_codes = [str(region["rejection_reason"])]
            items.append(
                {
                    "item": f"patch_region:{region.get('id')}",
                    "status": region.get("selection_status"),
                    "blocking": bool(region.get("blocking", True)),
                    "reason_codes": reason_codes,
                    "failure_reason": region.get("rejection_reason") or "patch region is not resolved",
                }
            )
    for name in ("source_distance", "bbox_drift", "silhouette_drift", "self_intersection"):
        metric = comparisons.get(name, {})
        if metric.get("status") != "computed":
            items.append(
                {
                    "item": name,
                    "status": metric.get("status", "required_metric_missing"),
                    "blocking": True,
                    "failure_reason": metric.get("failure_reason") or f"{name} metric is required for acceptance",
                }
            )
    return items


def limitations(group_filter: dict[str, Any] | None) -> list[str]:
    result = [
        "The external-flow shell policy seals every opening; local functional-opening decisions remain diagnostic for the alternative hybrid branch.",
        "A locally sealed voxel shell is partitioned by six-connected far-field flood before marching cubes supplies immutable watertight connectivity and proxy fallback positions; the proxy is never accepted without source projection and validation.",
        "Only correspondence-safe vertices are placed exactly on source triangles; every rejected match retains its proxy position with a reason code.",
        "The source-projected candidate must survive raw and normalized roundtrip topology checks plus a fail-closed global triangle-intersection audit.",
    ]
    if group_filter:
        result.append("GLTF group/name mapping is diagnostic metadata only and never removes primary mesh triangles.")
    else:
        result.extend([
            "No CAD/GLTF group tree was provided for this run.",
            "Stage 1 uses mesh-only multi-view outermost visibility instead of group hide/show.",
        ])
    return result


def patch_region_summary(patch_regions: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_classification: dict[str, int] = {}
    blocking = 0
    for row in patch_regions:
        status = str(row.get("selection_status"))
        classification = str(row.get("classification"))
        by_status[status] = by_status.get(status, 0) + 1
        by_classification[classification] = by_classification.get(classification, 0) + 1
        if status in {"hold_for_policy", "reject_patch"}:
            blocking += 1
    return {
        "total_regions": len(patch_regions),
        "blocking_regions": blocking,
        "by_selection_status": by_status,
        "by_classification": by_classification,
        "proxy_weight": patch_region_weight_summary(patch_regions),
    }


def patch_region_weight_summary(patch_regions: list[dict[str, Any]]) -> dict[str, float | None]:
    values = [
        float(row.get("proxy_weight", {}).get("mean", 0.0))
        for row in patch_regions
    ]
    if not values:
        return {"min": None, "mean": None, "max": None, "p95": None}
    values = sorted(values)
    p95_index = min(int(round((len(values) - 1) * 0.95)), len(values) - 1)
    return {
        "min": values[0],
        "mean": sum(values) / len(values),
        "max": values[-1],
        "p95": values[p95_index],
    }


def debug_artifact_paths(outputs: dict[str, Any]) -> list[str]:
    return [
        value
        for key, value in outputs.items()
        if key.startswith("debug_") and not key.endswith("_dir") and isinstance(value, str)
    ]
