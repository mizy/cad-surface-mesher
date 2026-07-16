from __future__ import annotations

from pathlib import Path
from typing import Any

from repair_report import two_stage_repair_report
from source_projected_contract import (
    build_projection_gates,
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
    projection_report: dict[str, Any],
    post_write_validation: dict[str, Any],
    projected_visual_drift: dict[str, Any],
) -> dict[str, Any]:
    candidate_path = outputs["source_preserving_candidate_vtp"]
    closure_proxy_path = outputs["closure_proxy_vtp"]
    projected_path = outputs.get("source_projected_watertight_candidate_vtp")
    projected_produced = bool(outputs.get("source_projected_watertight_candidate_produced"))
    projected_metrics = stage_reports["source_projected_watertight_candidate"]
    comparisons = projection_comparisons(
        stage_reports["deterministic_repair_candidate"],
        stage_reports["closure_proxy"],
        projected_metrics,
        projection_report,
        projected_visual_drift,
        post_write_validation,
    )
    shell_policy = watertight_shell_policy(args.target_name)
    gates = build_projection_gates(
        projected_metrics,
        comparisons,
        deterministic_passes,
        projected_path,
        projected_produced,
        closure_proxy_path,
        projection_report,
        shell_policy,
    )
    decision = build_decision(gates, projected_path)
    outputs = accepted_outputs(
        outputs,
        decision,
        projected_path or candidate_path,
        projected_path or candidate_path,
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
    )
    if decision["status"] != "accepted" and projected_path:
        ignored_outputs.append({
            "path": projected_path,
            "kind": "rejected_source_projected_watertight_candidate",
            "reason": "source-projected shell did not pass every engineering acceptance gate",
            "safe_for_acceptance": False,
        })
    derived_artifacts = [
        outputs["stage1_exterior_candidate_vtp"],
        candidate_path,
        closure_proxy_path,
    ]
    if projected_path:
        derived_artifacts.append(projected_path)
    truth = input_truth(Path(args.input), group_filter, derived_artifacts)
    truth["acceptance_source"] = (
        "closure_proxy_connectivity_plus_deterministic_source_surface_positions"
    )
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
            "cad_output": {
                "supported": False,
                "reason": "this worker does not perform reverse CAD fitting",
            },
            "accepted_final_rule": (
                "decision.final_output_path == outputs.accepted_mesh_vtp == "
                "outputs.source_projected_watertight_candidate_vtp; "
                "closure_proxy_vtp supplies connectivity only"
            ),
            "rejected_result_rule": (
                "outputs.mesh_result identifies the source-preserving fallback as unrepaired"
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
        "source_projection": projection_report,
        "comparisons": comparisons,
        "gates": gates,
        "outputs": outputs,
        "ignored_outputs": ignored_outputs,
        "unhandled_items": projection_unhandled_items(comparisons),
        "repair_report": two_stage_repair_report(
            stage_reports,
            extraction,
            group_filter,
            remesh,
            deterministic_passes,
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
    projected_path = outputs.get("source_projected_watertight_candidate_vtp")
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
            "metrics": stage_reports.get("source_projected_watertight_candidate"),
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


def limitations(group_filter: dict[str, Any] | None) -> list[str]:
    result = [
        "The watertight-shell target seals every opening; semantic evidence cannot weaken geometry gates.",
        "A sealed voxel shell is partitioned by six-connected far-field flood before marching cubes supplies immutable watertight connectivity and proxy fallback positions.",
        "Only correspondence-safe vertices are placed on source triangles; every rejected match retains its certified proxy position with a reason code.",
        "The source-projected candidate must survive serialized readback and exact all-face self-intersection certification.",
    ]
    if group_filter:
        result.append(
            "GLTF names and flattened face ranges are diagnostic metadata only and never remove primary mesh triangles."
        )
    else:
        result.extend([
            "No CAD/GLTF group tree was provided for this run.",
            "Stage 1 uses mesh-only multi-view outermost visibility instead of group hide/show.",
        ])
    return result
