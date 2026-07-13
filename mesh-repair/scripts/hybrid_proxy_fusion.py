from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from coincident_loop_weld import CoincidentLoopWeldConfig, weld_coincident_boundary_loops
from deterministic_hole_fill import run_deterministic_hole_fill
from hybrid_proxy_geometry import (
    FACE_ORIGIN,
    artifact_paths,
    bidirectional_distance,
    conformal_loop_stitch,
    conformal_same_mesh_loop_stitch,
    extract_ordered_boundary_loops,
    face_adjacency,
    face_component_ids,
    patch_local_drift,
    provenance_summary,
    topology_summary,
)
from hybrid_proxy_regions import build_patch_graph, build_patch_region, build_proxy_face_index
from local_proxy_patch import build_source_locked_proxy_patch
from mesh_io import compact_mesh, write_vtp
from mesh_metrics import edge_topology, inconsistent_winding_edges, mesh_report, self_intersection_report


@dataclass(frozen=True)
class FusionThresholds:
    voxel_pitch: float
    bbox_expand_scale: float = 2.5
    seam_contact_scale: float = 2.0
    source_distance_ratio: float = 0.002
    max_distance_samples: int = 2500
    max_target_samples: int = 4000
    seam_belt_rings: int = 2
    max_patch_area_ratio: float = 80.0


# @entry
def run_hybrid_proxy_fusion(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_indices: np.ndarray,
    proxy_points: np.ndarray,
    proxy_faces: np.ndarray,
    inventory: dict[str, Any],
    policy_packet: dict[str, Any],
    policy_decisions: list[dict[str, Any]],
    output_dir: Path,
    *,
    voxel_pitch: float,
) -> dict[str, Any]:
    thresholds = FusionThresholds(voxel_pitch=max(float(voxel_pitch), 1e-12))
    patches_dir = output_dir / "patches"
    debug_dir = output_dir / "debug"
    patches_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    _, edge_faces = edge_topology(source_faces)
    proxy_index = build_proxy_face_index(proxy_points, proxy_faces)
    graph = build_patch_graph(
        source_points,
        source_faces,
        source_indices,
        face_component_ids(source_faces, edge_faces),
        inventory,
        policy_packet,
        policy_decisions,
        thresholds,
    )

    adjacency = face_adjacency(source_faces)
    patch_regions = []
    committed_transactions = []
    rejected_transactions = []
    current_points = np.asarray(source_points, dtype=np.float64).copy()
    current_faces = np.asarray(source_faces, dtype=np.int64).copy()
    current_provenance = initial_provenance(source_indices)
    current_transaction_counts = transaction_topology_counts(current_points, current_faces)
    trace_steps = [{"step": "patch_graph", "nodes": len(graph["nodes"]), "edges": len(graph["edges"])}]
    for region in graph["components"]:
        operator = effective_region_operator(region)
        if operator == "policy_reject":
            patch = prepare_policy_rejected_patch(
                source_points,
                source_faces,
                source_indices,
                region,
                patches_dir,
            )
        elif operator == "loop_pair_zipper":
            patch = prepare_source_zipper_patch(
                source_points,
                source_faces,
                source_indices,
                region,
                patches_dir,
            )
        elif operator == "constrained_loop_triangulation":
            patch = prepare_small_hole_patch(
                source_points,
                source_faces,
                source_indices,
                region,
                patches_dir,
            )
        elif operator == "source_locked_proxy_patch":
            patch = prepare_source_locked_proxy_patch(
                source_points,
                source_faces,
                source_indices,
                proxy_points,
                proxy_faces,
                region,
                patches_dir,
            )
        else:
            patch = build_patch_region(
                source_points,
                source_faces,
                source_indices,
                adjacency,
                proxy_points,
                proxy_faces,
                proxy_index,
                region,
                thresholds,
                patches_dir,
            )
        patch_regions.append(patch["report"])
        trace_steps.append(patch["trace"])
        if not patch["accepted_for_final"]:
            continue
        transaction = attempt_patch_transaction(
            current_points,
            current_faces,
            current_provenance,
            patch,
            thresholds,
            patches_dir,
            before_counts=current_transaction_counts,
        )
        apply_transaction_to_report(patch["report"], transaction)
        trace_steps.append(transaction["trace"])
        if transaction["committed"]:
            current_points = transaction["points"]
            current_faces = transaction["faces"]
            current_provenance = transaction["provenance"]
            current_transaction_counts = transaction["topology_counts"]
            committed_transactions.append(transaction["summary"])
        else:
            rejected_transactions.append(transaction["summary"])

    final_points, final_faces, provenance = current_points, current_faces, current_provenance
    topology_filter = transaction_summary(
        mesh_report(source_points, source_faces),
        mesh_report(final_points, final_faces),
        committed_transactions,
        rejected_transactions,
    )
    trace_steps.append(topology_filter)
    source_proxy_blend = source_proxy_blend_report(final_points, final_faces, provenance, patch_regions)
    final_path = output_dir / "hybrid_fused_candidate.vtp"
    write_vtp(final_path, final_points, final_faces, provenance)

    comparisons = build_fusion_comparisons(
        source_points,
        source_faces,
        final_points,
        final_faces,
        provenance["face_origin"],
        thresholds,
    )
    graph_path = debug_dir / "region_patch_graph.json"
    trace_path = debug_dir / "hybrid_fusion_trace.json"
    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    trace = {
        "thresholds": asdict(thresholds),
        "steps": [
            *trace_steps,
            {
                "step": "write_hybrid_candidate",
                "path": str(final_path),
                "source_faces": int(source_faces.shape[0]),
                "output_faces": int(final_faces.shape[0]),
                "committed_patch_transactions": len(committed_transactions),
            },
        ],
        "comparisons": comparisons,
        "source_proxy_blend": source_proxy_blend,
    }
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return {
        "hybrid_fused_candidate_vtp": str(final_path),
        "hybrid_fused_candidate_produced": True,
        "final_metrics": mesh_report(final_points, final_faces),
        "patch_regions": patch_regions,
        "comparisons": comparisons,
        "face_provenance": provenance_summary(provenance),
        "source_proxy_blend": source_proxy_blend,
        "patch_artifacts": artifact_paths(patch_regions),
        "debug_artifacts": {
            "region_patch_graph_json": str(graph_path),
            "hybrid_fusion_trace_json": str(trace_path),
        },
        "fusion_trace": trace,
    }


def effective_region_operator(region: dict[str, Any]) -> str | None:
    operator = region.get("operator")
    decision = region.get("policy_decision") or {}
    if decision.get("status") == "decided" and decision.get("decision") == "reject":
        return "policy_reject"
    if (
        operator == "proxy_conformal_patch_after_cap_decision"
        and decision.get("status") == "decided"
        and decision.get("decision") == "cap"
    ):
        return (
            "constrained_loop_triangulation"
            if decision.get("shape_prior") == "planar"
            else "source_locked_proxy_patch"
        )
    return operator


def prepare_policy_rejected_patch(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_indices: np.ndarray,
    region: dict[str, Any],
    patches_dir: Path,
) -> dict[str, Any]:
    region_id = int(region["fusion_region_id"])
    face_ids = np.asarray(region.get("source_face_ids") or [], dtype=np.int64)
    face_ids = face_ids[(face_ids >= 0) & (face_ids < source_faces.shape[0])]
    artifacts: dict[str, str] = {}
    if face_ids.size:
        seam_points, seam_faces = compact_mesh(source_points, source_faces, face_ids)
        seam_path = patches_dir / f"region_{region_id:04d}_seam_belt.vtp"
        write_vtp(
            seam_path,
            seam_points,
            seam_faces,
            {
                "source_triangle_index": source_indices[face_ids],
                "fusion_region_id": np.full(seam_faces.shape[0], region_id, dtype=np.int32),
            },
        )
        artifacts["seam_belt_vtp"] = str(seam_path)
    reason = "policy_region_rejected"
    report = {
        "id": f"region_{region_id:04d}",
        "component_id": region.get("component_id"),
        "source_region_ids": region["source_region_ids"],
        "nearby_region_ids": region.get("nearby_region_ids", []),
        "local_scale": region["local_scale"],
        "classification": region["classification"],
        "semantic_classifications": region.get("semantic_classifications", []),
        "operator": "policy_reject",
        "operator_reason": "explicit_semantic_policy_rejected_automatic_cap",
        "requires_policy": region.get("requires_policy", False),
        "blocking": True,
        "selection_status": "reject_patch",
        "source_trust": {
            "status": "kept_unmodified",
            "reason_codes": [reason],
            "values": {"source_face_count": int(face_ids.size)},
        },
        "proxy_trust": {
            "status": "not_applicable",
            "reason_codes": [],
            "values": {"proxy_face_count": 0},
        },
        "proxy_weight": weight_summary(0.0),
        "sdf_blend_weight": weight_summary(0.0),
        "source_weight": weight_summary(1.0),
        "policy_decision": region.get("policy_decision"),
        "graph": {
            "fusion_region_id": region_id,
            "node_ids": region["source_region_ids"],
            "edge_reasons": region.get("edge_reasons", []),
        },
        "artifacts": artifacts,
        "final_provenance": {
            "consumed_by_final": False,
            "fusion_region_id": region_id,
            "face_origin_values": ["source"],
            "source_triangle_index_required_for_source_faces": True,
        },
        "accepted": False,
        "rejection_reason": reason,
        "accept_reject_reason_codes": [reason],
        "seam_results": {
            "status": "skipped_by_explicit_policy_reject",
            "method": "none",
            "boundary_edges_after": None,
            "non_manifold_edges_after": None,
            "reason": "policy rejected automatic geometry generation for this region",
        },
    }
    return {
        "report": report,
        "accepted_for_final": False,
        "operator": "policy_reject",
        "fusion_region_id": region_id,
        "local_scale": float(region["local_scale"]),
        "trace": {
            "step": "policy_rejected_patch",
            "region_id": report["id"],
            "selection_status": "reject_patch",
            "reason_code": reason,
            "artifacts": artifacts,
        },
    }


def initial_provenance(source_indices: np.ndarray) -> dict[str, np.ndarray]:
    count = int(source_indices.size)
    zeros = np.zeros(count, dtype=np.float32)
    return {
        "face_origin": np.full(count, FACE_ORIGIN["source"], dtype=np.int16),
        "source_triangle_index": np.asarray(source_indices, dtype=np.int64).copy(),
        "fusion_region_id": np.zeros(count, dtype=np.int32),
        "proxy_weight": zeros.copy(),
        "sdf_blend_weight": zeros.copy(),
    }


def prepare_source_zipper_patch(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_indices: np.ndarray,
    region: dict[str, Any],
    patches_dir: Path,
) -> dict[str, Any]:
    region_id = int(region["fusion_region_id"])
    face_ids = np.asarray(region.get("source_face_ids") or [], dtype=np.int64)
    face_ids = face_ids[(face_ids >= 0) & (face_ids < source_faces.shape[0])]
    loops = [np.asarray(loop, dtype=np.int64) for loop in region.get("ordered_boundary_loops", [])]
    artifacts: dict[str, str] = {}
    if face_ids.size:
        seam_points, seam_faces = compact_mesh(source_points, source_faces, face_ids)
        seam_path = patches_dir / f"region_{region_id:04d}_seam_belt.vtp"
        write_vtp(
            seam_path,
            seam_points,
            seam_faces,
            {
                "source_triangle_index": source_indices[face_ids],
                "fusion_region_id": np.full(seam_faces.shape[0], region_id, dtype=np.int32),
            },
        )
        artifacts["seam_belt_vtp"] = str(seam_path)
    valid = region.get("classification") == "patch_required" and len(loops) == 2
    rejection = None if valid else "zipper_requires_exactly_two_classified_boundary_loops"
    report = {
        "id": f"region_{region_id:04d}",
        "component_id": region.get("component_id"),
        "source_region_ids": region["source_region_ids"],
        "nearby_region_ids": region.get("nearby_region_ids", []),
        "local_scale": region["local_scale"],
        "classification": region["classification"],
        "semantic_classifications": region.get("semantic_classifications", []),
        "operator": "loop_pair_zipper",
        "requires_policy": region.get("requires_policy", False),
        "blocking": not valid,
        "selection_status": "transaction_pending" if valid else "reject_patch",
        "source_trust": {
            "status": "trusted" if valid else "untrusted",
            "reason_codes": [] if valid else [rejection],
            "values": {"source_face_count": int(face_ids.size), "ordered_loop_count": len(loops)},
        },
        "proxy_trust": {"status": "not_applicable", "reason_codes": [], "values": {}},
        "proxy_weight": weight_summary(0.0),
        "sdf_blend_weight": weight_summary(0.0),
        "source_weight": weight_summary(1.0),
        "policy_decision": region.get("policy_decision"),
        "graph": {
            "fusion_region_id": region_id,
            "node_ids": region["source_region_ids"],
            "edge_reasons": region.get("edge_reasons", []),
        },
        "artifacts": artifacts,
        "final_provenance": {
            "consumed_by_final": False,
            "fusion_region_id": region_id,
            "face_origin_values": ["source"],
            "source_triangle_index_required_for_source_faces": True,
        },
        "accepted": False,
        "rejection_reason": rejection,
        "seam_results": {
            "status": "not_run",
            "method": "paired_arc_length_edge_split_annular_bridge",
            "boundary_edges_after": None,
            "non_manifold_edges_after": None,
        },
    }
    return {
        "report": report,
        "accepted_for_final": valid,
        "operator": "loop_pair_zipper",
        "ordered_boundary_loops": loops,
        "fusion_region_id": region_id,
        "local_scale": float(region["local_scale"]),
        "trace": {
            "step": "source_loop_pair_candidate",
            "region_id": report["id"],
            "selection_status": report["selection_status"],
            "ordered_loop_count": len(loops),
            "artifacts": artifacts,
        },
    }


def prepare_source_locked_proxy_patch(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_indices: np.ndarray,
    proxy_points: np.ndarray,
    proxy_faces: np.ndarray,
    region: dict[str, Any],
    patches_dir: Path,
) -> dict[str, Any]:
    region_id = int(region["fusion_region_id"])
    face_ids = np.asarray(region.get("source_face_ids") or [], dtype=np.int64)
    face_ids = face_ids[(face_ids >= 0) & (face_ids < source_faces.shape[0])]
    loops = [np.asarray(loop, dtype=np.int64) for loop in region.get("ordered_boundary_loops", [])]
    decision = region.get("policy_decision") or {}
    policy_authorized = not region.get("requires_policy") or (
        decision.get("status") == "decided" and decision.get("decision") == "cap"
    )
    valid = region.get("classification") == "patch_required" and len(loops) == 1 and policy_authorized
    rejection = None
    if not policy_authorized:
        rejection = "proxy_patch_requires_cap_policy_decision"
    elif len(loops) != 1:
        rejection = "proxy_patch_requires_one_classified_simple_boundary_loop"
    elif region.get("classification") != "patch_required":
        rejection = "proxy_patch_region_not_authorized"

    artifacts: dict[str, str] = {}
    if face_ids.size:
        seam_points, seam_faces = compact_mesh(source_points, source_faces, face_ids)
        seam_path = patches_dir / f"region_{region_id:04d}_seam_belt.vtp"
        write_vtp(
            seam_path,
            seam_points,
            seam_faces,
            {
                "source_triangle_index": source_indices[face_ids],
                "fusion_region_id": np.full(seam_faces.shape[0], region_id, dtype=np.int32),
            },
        )
        artifacts["seam_belt_vtp"] = str(seam_path)

    report = {
        "id": f"region_{region_id:04d}",
        "component_id": region.get("component_id"),
        "source_region_ids": region["source_region_ids"],
        "nearby_region_ids": region.get("nearby_region_ids", []),
        "local_scale": region["local_scale"],
        "classification": region["classification"],
        "semantic_classifications": region.get("semantic_classifications", []),
        "operator": "source_locked_proxy_patch",
        "operator_reason": "policy_authorized_source_locked_closure_sdf_patch",
        "policy_shape_prior": decision.get("shape_prior") or "voxel_sdf",
        "requires_policy": region.get("requires_policy", False),
        "blocking": not valid,
        "selection_status": "transaction_pending" if valid else "reject_patch",
        "source_trust": {
            "status": "trusted" if valid else "untrusted",
            "reason_codes": [] if valid else [rejection],
            "values": {"source_face_count": int(face_ids.size), "ordered_loop_count": len(loops)},
        },
        "proxy_trust": {
            "status": "transaction_pending" if valid else "untrusted",
            "reason_codes": [] if valid else [rejection],
            "values": {"proxy_surface_face_count": int(proxy_faces.shape[0])},
        },
        "proxy_weight": weight_summary(1.0 if valid else 0.0),
        "sdf_blend_weight": weight_summary(1.0 if valid else 0.0),
        "source_weight": weight_summary(0.0 if valid else 1.0),
        "policy_decision": region.get("policy_decision"),
        "graph": {
            "fusion_region_id": region_id,
            "node_ids": region["source_region_ids"],
            "edge_reasons": region.get("edge_reasons", []),
        },
        "artifacts": artifacts,
        "final_provenance": {
            "consumed_by_final": False,
            "fusion_region_id": region_id,
            "face_origin_values": ["source"],
            "source_triangle_index_required_for_source_faces": True,
        },
        "accepted": False,
        "rejection_reason": rejection,
        "seam_results": {
            "status": "not_run",
            "method": "local_proxy_disk_crop_arc_length_resample_annular_bridge",
            "boundary_edges_after": None,
            "non_manifold_edges_after": None,
        },
    }
    return {
        "report": report,
        "accepted_for_final": valid,
        "operator": "source_locked_proxy_patch",
        "ordered_boundary_loops": loops,
        "fusion_region_id": region_id,
        "local_scale": float(region["local_scale"]),
        "proxy_surface_points": np.asarray(proxy_points, dtype=np.float64),
        "proxy_surface_faces": np.asarray(proxy_faces, dtype=np.int64),
        "trace": {
            "step": "source_locked_proxy_patch_candidate",
            "region_id": report["id"],
            "selection_status": report["selection_status"],
            "ordered_loop_count": len(loops),
            "policy_shape_prior": report["policy_shape_prior"],
            "artifacts": artifacts,
        },
    }


def prepare_small_hole_patch(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_indices: np.ndarray,
    region: dict[str, Any],
    patches_dir: Path,
) -> dict[str, Any]:
    region_id = int(region["fusion_region_id"])
    face_ids = np.asarray(region.get("source_face_ids") or [], dtype=np.int64)
    face_ids = face_ids[(face_ids >= 0) & (face_ids < source_faces.shape[0])]
    loops = [np.asarray(loop, dtype=np.int64) for loop in region.get("ordered_boundary_loops", [])]
    decision = region.get("policy_decision") or {}
    artifacts: dict[str, str] = {}
    if face_ids.size:
        seam_points, seam_faces = compact_mesh(source_points, source_faces, face_ids)
        seam_path = patches_dir / f"region_{region_id:04d}_seam_belt.vtp"
        write_vtp(
            seam_path,
            seam_points,
            seam_faces,
            {
                "source_triangle_index": source_indices[face_ids],
                "fusion_region_id": np.full(seam_faces.shape[0], region_id, dtype=np.int32),
            },
        )
        artifacts["seam_belt_vtp"] = str(seam_path)
    valid = region.get("classification") == "patch_required" and len(loops) == 1
    rejection = None if valid else "hole_fill_requires_one_classified_simple_boundary_loop"
    report = {
        "id": f"region_{region_id:04d}",
        "component_id": region.get("component_id"),
        "source_region_ids": region["source_region_ids"],
        "nearby_region_ids": region.get("nearby_region_ids", []),
        "local_scale": region["local_scale"],
        "classification": region["classification"],
        "semantic_classifications": region.get("semantic_classifications", []),
        "operator": "constrained_loop_triangulation",
        "operator_reason": (
            "policy_authorized_planar_cap"
            if decision.get("status") == "decided" and decision.get("shape_prior") == "planar"
            else "strict_small_hole_geometry"
        ),
        "policy_shape_prior": decision.get("shape_prior"),
        "requires_policy": region.get("requires_policy", False),
        "blocking": not valid,
        "selection_status": "transaction_pending" if valid else "reject_patch",
        "source_trust": {
            "status": "trusted" if valid else "untrusted",
            "reason_codes": [] if valid else [rejection],
            "values": {"source_face_count": int(face_ids.size), "ordered_loop_count": len(loops)},
        },
        "proxy_trust": {"status": "not_applicable", "reason_codes": [], "values": {}},
        "proxy_weight": weight_summary(0.0),
        "sdf_blend_weight": weight_summary(0.0),
        "source_weight": weight_summary(1.0),
        "policy_decision": region.get("policy_decision"),
        "graph": {
            "fusion_region_id": region_id,
            "node_ids": region["source_region_ids"],
            "edge_reasons": region.get("edge_reasons", []),
        },
        "artifacts": artifacts,
        "final_provenance": {
            "consumed_by_final": False,
            "fusion_region_id": region_id,
            "face_origin_values": ["source"],
            "source_triangle_index_required_for_source_faces": True,
        },
        "accepted": False,
        "rejection_reason": rejection,
        "seam_results": {
            "status": "not_run",
            "method": "fixed_source_boundary_multi_chart_constrained_ear_clipping",
            "boundary_edges_after": None,
            "non_manifold_edges_after": None,
        },
    }
    return {
        "report": report,
        "accepted_for_final": valid,
        "operator": "constrained_loop_triangulation",
        "ordered_boundary_loops": loops,
        "fusion_region_id": region_id,
        "local_scale": float(region["local_scale"]),
        "trace": {
            "step": "small_hole_fill_candidate",
            "region_id": report["id"],
            "selection_status": report["selection_status"],
            "ordered_loop_count": len(loops),
            "artifacts": artifacts,
        },
    }


def attempt_patch_transaction(
    points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, np.ndarray],
    patch: dict[str, Any],
    thresholds: FusionThresholds,
    patches_dir: Path,
    *,
    before_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    if patch.get("operator") == "loop_pair_zipper":
        trial = source_zipper_trial(points, faces, provenance, patch, thresholds)
    elif patch.get("operator") == "constrained_loop_triangulation":
        trial = small_hole_fill_trial(points, faces, provenance, patch)
    elif patch.get("operator") == "source_locked_proxy_patch":
        trial = source_locked_proxy_patch_trial(points, faces, provenance, patch)
    else:
        trial = proxy_patch_trial(points, faces, provenance, patch, thresholds)
    if not trial["success"]:
        return rejected_transaction(points, faces, provenance, patch, trial)

    local_self_intersection = trial.get("certified_local_intersection")
    if local_self_intersection is None:
        local_self_intersection = self_intersection_report(
            trial["points"],
            trial["faces"],
            focus_face_ids=trial["stitch_face_ids"],
            max_candidate_pairs=250_000,
        )
    if not local_self_intersection["passed"]:
        reason = (
            "patch_local_self_intersection_detected"
            if local_self_intersection.get("status") == "computed"
            else "patch_local_self_intersection_check_incomplete"
        )
        failed_trial = {
            **trial,
            "success": False,
            "failure_reason_codes": [reason],
            "diagnostics": {
                **trial.get("diagnostics", {}),
                "local_self_intersection": local_self_intersection,
            },
            "local_self_intersection": local_self_intersection,
        }
        return rejected_transaction(points, faces, provenance, patch, failed_trial)

    before_state = before_counts or transaction_topology_counts(points, faces)
    after_state = transaction_topology_counts(trial["points"], trial["faces"])
    topology_passed = counts_not_worse(before_state, after_state)
    boundary_improved = (
        after_state["boundary_edges"]
        < before_state["boundary_edges"]
    )
    committed = bool(topology_passed and boundary_improved)
    reason_codes = []
    if not topology_passed:
        reason_codes.append("patch_topology_worsened")
    if not boundary_improved:
        reason_codes.append("patch_did_not_reduce_boundary_edges")
    topology_gate = {
        "passed": committed,
        "topology_not_worse": topology_passed,
        "boundary_edges_reduced": boundary_improved,
        "before": before_state,
        "after": after_state,
        "reason_codes": reason_codes,
    }
    if not committed:
        trial = {**trial, "failure_reason_codes": reason_codes, "topology_gate": topology_gate}
        return rejected_transaction(points, faces, provenance, patch, trial)

    artifact_role = str(trial.get("artifact_role", "stitch_band"))
    artifact_path = write_generated_patch_artifact(
        trial["points"],
        trial["faces"],
        trial["provenance"],
        trial["stitch_face_ids"],
        int(patch["fusion_region_id"]),
        patches_dir,
        artifact_role=artifact_role,
    )
    additional_artifacts: dict[str, str] = {}
    proxy_patch_faces = np.asarray(trial.get("proxy_patch_faces", []), dtype=np.int64)
    proxy_patch_points = np.asarray(trial.get("proxy_patch_points", []), dtype=np.float64)
    if proxy_patch_faces.size and proxy_patch_points.size:
        proxy_path = patches_dir / f"region_{int(patch['fusion_region_id']):04d}_proxy_patch.vtp"
        proxy_face_ids = np.asarray(trial.get("proxy_face_ids", []), dtype=np.int64)
        write_vtp(
            proxy_path,
            proxy_patch_points,
            proxy_patch_faces,
            {
                "face_origin": np.full(
                    proxy_patch_faces.shape[0],
                    FACE_ORIGIN["proxy_patch"],
                    dtype=np.int16,
                ),
                "proxy_triangle_index": proxy_face_ids,
                "fusion_region_id": np.full(
                    proxy_patch_faces.shape[0],
                    int(patch["fusion_region_id"]),
                    dtype=np.int32,
                ),
                "proxy_weight": np.ones(proxy_patch_faces.shape[0], dtype=np.float32),
            },
        )
        additional_artifacts["proxy_patch_vtp"] = str(proxy_path)
    summary = {
        "region_id": int(patch["fusion_region_id"]),
        "operator": patch.get("operator") or "proxy_conformal_patch",
        "status": "committed",
        "before": before_state,
        "after": after_state,
        f"{artifact_role}_vtp": artifact_path,
        **additional_artifacts,
    }
    return {
        "committed": True,
        "points": trial["points"],
        "faces": trial["faces"],
        "provenance": trial["provenance"],
        "stitch_face_ids": trial["stitch_face_ids"],
        "generated_artifact_key": f"{artifact_role}_vtp",
        "generated_artifact_path": artifact_path,
        "additional_artifacts": additional_artifacts,
        "diagnostics": trial["diagnostics"],
        "topology_gate": topology_gate,
        "failure_reason_codes": [],
        "local_self_intersection": local_self_intersection,
        "topology_counts": after_state,
        "summary": summary,
        "trace": {
            "step": "patch_transaction",
            **summary,
            "topology_gate": topology_gate,
            "local_self_intersection": local_self_intersection,
        },
    }


def source_locked_proxy_patch_trial(
    points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, np.ndarray],
    patch: dict[str, Any],
) -> dict[str, Any]:
    loops = [np.asarray(loop, dtype=np.int64) for loop in patch.get("ordered_boundary_loops", [])]
    if len(loops) != 1:
        return trial_failure(
            "proxy_patch_requires_one_classified_simple_boundary_loop",
            "source-locked proxy patch requires exactly one ordered source loop",
        )
    result = build_source_locked_proxy_patch(
        points,
        faces,
        loops[0],
        np.asarray(patch.get("proxy_surface_points"), dtype=np.float64),
        np.asarray(patch.get("proxy_surface_faces"), dtype=np.int64),
        source_triangle_indices=np.asarray(provenance["source_triangle_index"], dtype=np.int64),
        region_id=int(patch["fusion_region_id"]),
    )
    if not result["success"]:
        diagnostics = result.get("diagnostics", {})
        return {
            "success": False,
            "failure_reason_codes": result["failure_reason_codes"],
            "diagnostics": diagnostics,
            "local_self_intersection": diagnostics.get("local_self_intersection"),
        }
    next_provenance = propagate_proxy_stitch_provenance(
        provenance,
        result,
        int(patch["fusion_region_id"]),
    )
    return {
        "success": True,
        "points": result["points"],
        "faces": result["faces"],
        "provenance": next_provenance,
        "stitch_face_ids": result["stitch_face_ids"],
        "diagnostics": result["diagnostics"],
        "certified_local_intersection": result["diagnostics"]["local_self_intersection"],
        "proxy_patch_points": result["proxy_patch_points"],
        "proxy_patch_faces": result["proxy_patch_faces"],
        "proxy_face_ids": result["proxy_face_ids"],
    }


def proxy_patch_trial(
    points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, np.ndarray],
    patch: dict[str, Any],
    thresholds: FusionThresholds,
) -> dict[str, Any]:
    proxy_points = np.asarray(patch.get("proxy_points"), dtype=np.float64)
    proxy_faces = np.asarray(patch.get("proxy_faces"), dtype=np.int64)
    if proxy_points.size == 0 or proxy_faces.size == 0:
        return trial_failure("proxy_patch_extraction_failed", "proxy patch is empty")
    loops = patch.get("ordered_boundary_loops") or []
    if len(loops) == 1:
        source_loop = np.asarray(loops[0], dtype=np.int64)
    else:
        extraction = extract_ordered_boundary_loops(points, faces)
        if not extraction["success"] or len(extraction["loops"]) != 1:
            return trial_failure(
                "source_boundary_loop_count_mismatch",
                "proxy fusion requires one explicitly classified source loop",
                source_loop_extraction=extraction.get("diagnostics"),
            )
        source_loop = extraction["loops"][0]
    proxy_extraction = extract_ordered_boundary_loops(proxy_points, proxy_faces)
    if not proxy_extraction["success"] or len(proxy_extraction["loops"]) != 1:
        return trial_failure(
            "proxy_boundary_loop_count_mismatch",
            "trimmed proxy patch must have exactly one simple boundary loop",
            proxy_loop_extraction=proxy_extraction.get("diagnostics"),
        )
    max_distance = max(
        thresholds.voxel_pitch * 4.0,
        float(patch.get("local_scale") or 0.0) * 0.5,
    )
    stitch = conformal_loop_stitch(
        points,
        faces,
        proxy_points,
        proxy_faces,
        source_loop=source_loop,
        proxy_loop=proxy_extraction["loops"][0],
        max_correspondence_distance=max_distance,
    )
    if not stitch["success"]:
        return {
            "success": False,
            "failure_reason_codes": stitch["failure_reason_codes"],
            "diagnostics": stitch["diagnostics"],
        }
    next_provenance = propagate_proxy_stitch_provenance(
        provenance,
        stitch,
        int(patch["fusion_region_id"]),
    )
    return {
        "success": True,
        "points": stitch["points"],
        "faces": stitch["faces"],
        "provenance": next_provenance,
        "stitch_face_ids": stitch["stitch_face_ids"],
        "diagnostics": stitch["diagnostics"],
    }


def small_hole_fill_trial(
    points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, np.ndarray],
    patch: dict[str, Any],
) -> dict[str, Any]:
    loops = [np.asarray(loop, dtype=np.int64) for loop in patch.get("ordered_boundary_loops", [])]
    if len(loops) != 1:
        return trial_failure(
            "hole_fill_requires_one_classified_simple_boundary_loop",
            "constrained hole fill requires exactly one ordered source loop",
        )
    fill = run_deterministic_hole_fill(
        points,
        faces,
        loops[0],
        provenance,
        fusion_region_id=int(patch["fusion_region_id"]),
        face_origin_value=FACE_ORIGIN["hole_fill"],
    )
    if not fill["committed"]:
        reason_codes = [
            "patch_local_self_intersection_detected"
            if code == "local_self_intersection_detected"
            else "patch_local_self_intersection_check_incomplete"
            if code == "local_self_intersection_check_incomplete"
            else code
            for code in fill["failure_reason_codes"]
        ]
        return {
            "success": False,
            "failure_reason_codes": reason_codes,
            "diagnostics": fill["diagnostics"],
            "local_self_intersection": fill.get("local_intersection"),
        }
    return {
        "success": True,
        "points": fill["points"],
        "faces": fill["faces"],
        "provenance": fill["provenance"],
        "stitch_face_ids": fill["generated_face_ids"],
        "artifact_role": "hole_fill",
        "diagnostics": fill["diagnostics"],
        "certified_local_intersection": fill["local_intersection"],
    }


def hole_fill_values(name: str, dtype: np.dtype[Any], count: int, region_id: int) -> np.ndarray:
    if name == "face_origin":
        return np.full(count, FACE_ORIGIN["hole_fill"], dtype=dtype)
    if name == "source_triangle_index":
        return np.full(count, -1, dtype=dtype)
    if name == "fusion_region_id":
        return np.full(count, region_id, dtype=dtype)
    return np.zeros(count, dtype=dtype)


def source_zipper_trial(
    points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, np.ndarray],
    patch: dict[str, Any],
    thresholds: FusionThresholds,
) -> dict[str, Any]:
    loops = [np.asarray(loop, dtype=np.int64) for loop in patch.get("ordered_boundary_loops", [])]
    if len(loops) != 2:
        return trial_failure(
            "zipper_requires_exactly_two_classified_boundary_loops",
            "source zipper transaction requires exactly two ordered loops",
        )
    _, edge_faces = edge_topology(faces)
    labels = face_component_ids(faces, edge_faces)
    component_ids = []
    for loop in loops:
        component_id = boundary_loop_component(loop, edge_faces, labels)
        if component_id is None:
            return trial_failure("zipper_loop_not_on_current_boundary", "ordered zipper loop is no longer a mesh boundary")
        component_ids.append(component_id)
    if component_ids[0] == component_ids[1]:
        max_distance = max(float(patch.get("local_scale") or 0.0) * 4.0, 1e-12)
        stitch = conformal_same_mesh_loop_stitch(
            points,
            faces,
            source_loop=loops[0],
            target_loop=loops[1],
            max_correspondence_distance=max_distance,
            normal_score_weight=0.0,
        )
        if not stitch["success"]:
            return {
                "success": False,
                "failure_reason_codes": stitch["failure_reason_codes"],
                "diagnostics": stitch["diagnostics"],
            }
        next_provenance = propagate_same_mesh_stitch_provenance(
            provenance,
            stitch,
            int(patch["fusion_region_id"]),
        )
        return {
            "success": True,
            "points": stitch["points"],
            "faces": stitch["faces"],
            "provenance": next_provenance,
            "stitch_face_ids": stitch["stitch_face_ids"],
            "diagnostics": stitch["diagnostics"],
        }
    component_face_ids = [np.flatnonzero(labels == component_id) for component_id in component_ids]
    first = compact_component(points, faces, component_face_ids[0], loops[0])
    second = compact_component(points, faces, component_face_ids[1], loops[1])
    if first is None or second is None:
        return trial_failure("zipper_loop_component_mapping_failed", "failed to map ordered loops into component meshes")
    max_distance = max(float(patch.get("local_scale") or 0.0) * 4.0, 1e-12)
    authority, weld_target = (
        (first, second)
        if first["global_face_ids"].size >= second["global_face_ids"].size
        else (second, first)
    )
    weld = weld_coincident_boundary_loops(
        authority["points"],
        authority["faces"],
        authority["loop"],
        weld_target["points"],
        weld_target["faces"],
        weld_target["loop"],
        config=CoincidentLoopWeldConfig(
            max_target_displacement=max_distance,
            allow_target_face_flip=True,
        ),
        source_triangle_indices=provenance["source_triangle_index"][authority["global_face_ids"]],
        target_triangle_indices=provenance["source_triangle_index"][weld_target["global_face_ids"]],
        region_id=int(patch["fusion_region_id"]),
    )
    if weld["success"]:
        merged = merge_coincident_weld_result(
            points,
            faces,
            provenance,
            authority,
            weld_target,
            weld,
            int(patch["fusion_region_id"]),
        )
        return {
            "success": True,
            "diagnostics": weld["diagnostics"],
            "certified_local_intersection": weld["diagnostics"]["local_self_intersection"],
            "artifact_role": "coincident_weld",
            **merged,
        }
    stitch = conformal_loop_stitch(
        first["points"],
        first["faces"],
        second["points"],
        second["faces"],
        source_loop=first["loop"],
        proxy_loop=second["loop"],
        max_correspondence_distance=max_distance,
        normal_score_weight=0.0,
    )
    if not stitch["success"]:
        return {
            "success": False,
            "failure_reason_codes": stitch["failure_reason_codes"],
            "diagnostics": {
                **stitch["diagnostics"],
                "coincident_weld_preflight": {
                    "failure_reason_codes": weld["failure_reason_codes"],
                    "diagnostics": weld["diagnostics"],
                },
            },
        }
    merged = merge_source_zipper_result(
        points,
        faces,
        provenance,
        first,
        second,
        stitch,
        int(patch["fusion_region_id"]),
    )
    return {"success": True, "diagnostics": stitch["diagnostics"], **merged}


def compact_component(
    points: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
    loop: np.ndarray,
) -> dict[str, Any] | None:
    selected_faces = faces[face_ids]
    unique_points, inverse = np.unique(selected_faces.ravel(), return_inverse=True)
    lookup = {int(global_id): local_id for local_id, global_id in enumerate(unique_points)}
    if any(int(vertex_id) not in lookup for vertex_id in loop):
        return None
    return {
        "points": points[unique_points],
        "faces": inverse.reshape((-1, 3)).astype(np.int64, copy=False),
        "global_point_ids": unique_points.astype(np.int64, copy=False),
        "global_face_ids": face_ids.astype(np.int64, copy=False),
        "loop": np.asarray([lookup[int(vertex_id)] for vertex_id in loop], dtype=np.int64),
    }


def boundary_loop_component(
    loop: np.ndarray,
    edge_faces: dict[tuple[int, int], list[int]],
    labels: np.ndarray,
) -> int | None:
    for index, left in enumerate(loop):
        right = int(loop[(index + 1) % loop.size])
        incident = edge_faces.get(tuple(sorted((int(left), right))), [])
        if len(incident) == 1:
            return int(labels[incident[0]])
    return None


def merge_source_zipper_result(
    points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, np.ndarray],
    first: dict[str, Any],
    second: dict[str, Any],
    stitch: dict[str, Any],
    region_id: int,
) -> dict[str, Any]:
    point_map = np.full(stitch["points"].shape[0], -1, dtype=np.int64)
    first_original = first["global_point_ids"].size
    second_original = second["global_point_ids"].size
    point_map[:first_original] = first["global_point_ids"]
    proxy_offset = int(stitch["proxy_point_offset"])
    point_map[proxy_offset:proxy_offset + second_original] = second["global_point_ids"]
    inserted_ids = np.flatnonzero(point_map < 0)
    point_map[inserted_ids] = np.arange(points.shape[0], points.shape[0] + inserted_ids.size, dtype=np.int64)
    next_points = np.vstack([points, stitch["points"][inserted_ids]])
    stitched_faces = point_map[stitch["faces"]]

    replaced = np.unique(np.concatenate([first["global_face_ids"], second["global_face_ids"]]))
    keep = np.ones(faces.shape[0], dtype=bool)
    keep[replaced] = False
    next_faces = np.vstack([faces[keep], stitched_faces])
    stitched_provenance = provenance_for_source_zipper(stitch, provenance, first, second, region_id)
    next_provenance = {
        name: np.concatenate([values[keep], stitched_provenance[name]])
        for name, values in provenance.items()
    }
    stitch_ids = np.arange(int(np.count_nonzero(keep)) + stitch["stitch_face_ids"][0], next_faces.shape[0], dtype=np.int64)
    return {
        "points": next_points,
        "faces": next_faces,
        "provenance": next_provenance,
        "stitch_face_ids": stitch_ids,
    }


def merge_coincident_weld_result(
    points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, np.ndarray],
    authority: dict[str, Any],
    target: dict[str, Any],
    weld: dict[str, Any],
    region_id: int,
) -> dict[str, Any]:
    """Replace two source components with a validated index-welded result."""

    point_map = np.full(weld["points"].shape[0], -1, dtype=np.int64)
    source_output_ids = np.asarray(weld["source_original_point_to_output"], dtype=np.int64)
    target_output_ids = np.asarray(weld["target_original_point_to_output"], dtype=np.int64)
    point_map[source_output_ids] = authority["global_point_ids"]
    for local_target_id, output_id in enumerate(target_output_ids):
        if point_map[int(output_id)] < 0:
            point_map[int(output_id)] = target["global_point_ids"][local_target_id]
    inserted_ids = np.flatnonzero(point_map < 0)
    point_map[inserted_ids] = np.arange(
        points.shape[0],
        points.shape[0] + inserted_ids.size,
        dtype=np.int64,
    )
    next_points = np.vstack([points, weld["points"][inserted_ids]])
    welded_faces = point_map[np.asarray(weld["faces"], dtype=np.int64)]

    replaced = np.unique(
        np.concatenate([authority["global_face_ids"], target["global_face_ids"]])
    )
    keep = np.ones(faces.shape[0], dtype=bool)
    keep[replaced] = False
    next_faces = np.vstack([faces[keep], welded_faces])

    local_seam_face_ids = welded_seam_face_ids(
        np.asarray(weld["faces"], dtype=np.int64),
        np.asarray(weld["welded_seam_edges"], dtype=np.int64),
    )
    welded_provenance = provenance_for_coincident_weld(
        weld,
        provenance,
        authority,
        target,
        local_seam_face_ids,
        region_id,
    )
    next_provenance = {
        name: np.concatenate([values[keep], welded_provenance[name]])
        for name, values in provenance.items()
    }
    global_seam_face_ids = local_seam_face_ids + int(np.count_nonzero(keep))
    return {
        "points": next_points,
        "faces": next_faces,
        "provenance": next_provenance,
        "stitch_face_ids": global_seam_face_ids,
    }


def welded_seam_face_ids(faces: np.ndarray, seam_edges: np.ndarray) -> np.ndarray:
    _, edge_faces = edge_topology(faces)
    face_ids = {
        int(face_id)
        for left, right in seam_edges
        for face_id in edge_faces.get(tuple(sorted((int(left), int(right)))), [])
    }
    return np.asarray(sorted(face_ids), dtype=np.int64)


def provenance_for_coincident_weld(
    weld: dict[str, Any],
    provenance: dict[str, np.ndarray],
    authority: dict[str, Any],
    target: dict[str, Any],
    seam_face_ids: np.ndarray,
    region_id: int,
) -> dict[str, np.ndarray]:
    count = int(weld["faces"].shape[0])
    result = {name: np.empty(count, dtype=values.dtype) for name, values in provenance.items()}
    source_parent = np.asarray(weld["source_face_parent"], dtype=np.int64)
    target_parent = np.asarray(weld["target_face_parent"], dtype=np.int64)
    source_mask = source_parent >= 0
    target_mask = target_parent >= 0
    if not np.all(source_mask ^ target_mask):
        raise ValueError("coincident weld faces must have exactly one source component parent")
    for name, values in provenance.items():
        result[name][source_mask] = values[
            authority["global_face_ids"][source_parent[source_mask]]
        ]
        result[name][target_mask] = values[target["global_face_ids"][target_parent[target_mask]]]
    result["fusion_region_id"][seam_face_ids] = region_id
    return result


def provenance_for_source_zipper(
    stitch: dict[str, Any],
    provenance: dict[str, np.ndarray],
    first: dict[str, Any],
    second: dict[str, Any],
    region_id: int,
) -> dict[str, np.ndarray]:
    count = stitch["faces"].shape[0]
    result = {
        name: np.empty(count, dtype=values.dtype)
        for name, values in provenance.items()
    }
    source_parent = stitch["source_face_parent"]
    proxy_parent = stitch["proxy_face_parent"]
    source_mask = source_parent >= 0
    proxy_mask = proxy_parent >= 0
    stitch_mask = ~(source_mask | proxy_mask)
    for name, values in provenance.items():
        result[name][source_mask] = values[first["global_face_ids"][source_parent[source_mask]]]
        result[name][proxy_mask] = values[second["global_face_ids"][proxy_parent[proxy_mask]]]
    result["face_origin"][stitch_mask] = FACE_ORIGIN["stitch_band"]
    result["source_triangle_index"][stitch_mask] = -1
    result["fusion_region_id"][stitch_mask] = region_id
    result["proxy_weight"][stitch_mask] = 0.0
    result["sdf_blend_weight"][stitch_mask] = 0.0
    return result


def propagate_same_mesh_stitch_provenance(
    provenance: dict[str, np.ndarray],
    stitch: dict[str, Any],
    region_id: int,
) -> dict[str, np.ndarray]:
    """Map split same-mesh faces to their one authoritative source parent."""

    count = stitch["faces"].shape[0]
    result = {
        name: np.empty(count, dtype=values.dtype)
        for name, values in provenance.items()
    }
    source_parent = stitch["source_face_parent"]
    source_mask = source_parent >= 0
    stitch_mask = ~source_mask
    for name, values in provenance.items():
        result[name][source_mask] = values[source_parent[source_mask]]
    result["face_origin"][stitch_mask] = FACE_ORIGIN["stitch_band"]
    result["source_triangle_index"][stitch_mask] = -1
    result["fusion_region_id"][stitch_mask] = region_id
    result["proxy_weight"][stitch_mask] = 0.0
    result["sdf_blend_weight"][stitch_mask] = 0.0
    return result


def propagate_proxy_stitch_provenance(
    provenance: dict[str, np.ndarray],
    stitch: dict[str, Any],
    region_id: int,
) -> dict[str, np.ndarray]:
    count = stitch["faces"].shape[0]
    result = {
        name: np.empty(count, dtype=values.dtype)
        for name, values in provenance.items()
    }
    source_parent = stitch["source_face_parent"]
    proxy_parent = stitch["proxy_face_parent"]
    source_mask = source_parent >= 0
    proxy_mask = proxy_parent >= 0
    stitch_mask = ~(source_mask | proxy_mask)
    for name, values in provenance.items():
        result[name][source_mask] = values[source_parent[source_mask]]
    result["face_origin"][proxy_mask] = FACE_ORIGIN["proxy_patch"]
    result["source_triangle_index"][proxy_mask] = -1
    result["fusion_region_id"][proxy_mask] = region_id
    result["proxy_weight"][proxy_mask] = 1.0
    result["sdf_blend_weight"][proxy_mask] = 1.0
    result["face_origin"][stitch_mask] = FACE_ORIGIN["stitch_band"]
    result["source_triangle_index"][stitch_mask] = -1
    result["fusion_region_id"][stitch_mask] = region_id
    result["proxy_weight"][stitch_mask] = 0.5
    result["sdf_blend_weight"][stitch_mask] = 0.5
    return result


def trial_failure(code: str, message: str, **diagnostics: Any) -> dict[str, Any]:
    return {
        "success": False,
        "failure_reason_codes": [code],
        "diagnostics": {"stage": "patch_transaction_precondition", "message": message, **diagnostics},
    }


def rejected_transaction(
    points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, np.ndarray],
    patch: dict[str, Any],
    trial: dict[str, Any],
) -> dict[str, Any]:
    reason_codes = list(trial.get("failure_reason_codes") or ["patch_transaction_failed"])
    summary = {
        "region_id": int(patch["fusion_region_id"]),
        "operator": patch.get("operator") or "proxy_conformal_patch",
        "status": "rolled_back",
        "reason_codes": reason_codes,
    }
    return {
        "committed": False,
        "points": points,
        "faces": faces,
        "provenance": provenance,
        "stitch_face_ids": np.zeros(0, dtype=np.int64),
        "stitch_band_vtp": None,
        "diagnostics": trial.get("diagnostics", {}),
        "topology_gate": trial.get("topology_gate"),
        "failure_reason_codes": reason_codes,
        "local_self_intersection": trial.get("local_self_intersection"),
        "summary": summary,
        "trace": {"step": "patch_transaction", **summary, "diagnostics": trial.get("diagnostics", {})},
    }


def write_generated_patch_artifact(
    points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, np.ndarray],
    stitch_face_ids: np.ndarray,
    region_id: int,
    patches_dir: Path,
    *,
    artifact_role: str,
) -> str:
    band_points, band_faces = compact_mesh(points, faces, stitch_face_ids)
    path = patches_dir / f"region_{region_id:04d}_{artifact_role}.vtp"
    write_vtp(
        path,
        band_points,
        band_faces,
        {
            "face_origin": provenance["face_origin"][stitch_face_ids],
            "source_triangle_index": provenance["source_triangle_index"][stitch_face_ids],
            "fusion_region_id": provenance["fusion_region_id"][stitch_face_ids],
            "proxy_weight": provenance["proxy_weight"][stitch_face_ids],
            "sdf_blend_weight": provenance["sdf_blend_weight"][stitch_face_ids],
        },
    )
    return str(path)


def apply_transaction_to_report(report: dict[str, Any], transaction: dict[str, Any]) -> None:
    committed = bool(transaction["committed"])
    is_zipper = report.get("operator") == "loop_pair_zipper"
    is_coincident_weld = (
        transaction.get("diagnostics", {}).get("stage")
        == "transactional_coincident_loop_weld"
    )
    is_hole_fill = report.get("operator") == "constrained_loop_triangulation"
    is_source_locked_proxy = report.get("operator") == "source_locked_proxy_patch"
    report["selection_status"] = (
        "use_source_zipper" if committed and is_zipper
        else "use_hole_fill" if committed and is_hole_fill
        else "use_proxy_patch" if committed
        else "reject_patch"
    )
    report["blocking"] = not committed
    report["accepted"] = committed
    report["rejection_reason"] = None if committed else transaction["failure_reason_codes"][0]
    artifact_key = transaction.get("generated_artifact_key")
    artifact_path = transaction.get("generated_artifact_path")
    if artifact_key and artifact_path:
        report.setdefault("artifacts", {})[artifact_key] = artifact_path
    report.setdefault("artifacts", {}).update(transaction.get("additional_artifacts", {}))
    if committed and report.get("operator") == "source_locked_proxy_patch":
        extraction = transaction.get("diagnostics", {}).get("extraction", {})
        selected_component = extraction.get("selected_component", {})
        report["proxy_trust"] = {
            "status": "trusted",
            "reason_codes": [],
            "values": {
                "proxy_face_count": selected_component.get("proxy_face_count"),
                "proxy_component_index": selected_component.get("component_index"),
                "nearest_point_scatter_used": False,
            },
        }
    report["seam_results"] = {
        "status": "passed_and_committed" if committed else "failed_and_rolled_back",
        "method": (
            "fixed_source_boundary_multi_chart_constrained_ear_clipping"
            if is_hole_fill
            else "local_proxy_disk_crop_arc_length_resample_annular_bridge"
            if is_source_locked_proxy
            else "arc_length_union_edge_split_fixed_source_ring_index_weld"
            if is_coincident_weld
            else "paired_arc_length_edge_split_annular_bridge"
        ),
        "boundary_edges_after": (
            transaction.get("topology_gate", {}).get("after", {}).get("boundary_edges")
            if transaction.get("topology_gate")
            else None
        ),
        "non_manifold_edges_after": (
            transaction.get("topology_gate", {}).get("after", {}).get("non_manifold_edges")
            if transaction.get("topology_gate")
            else None
        ),
        "diagnostics": transaction.get("diagnostics", {}),
        "topology_gate": transaction.get("topology_gate"),
        "failure_reason_codes": transaction.get("failure_reason_codes", []),
        "local_self_intersection": transaction.get("local_self_intersection"),
        "self_intersection_check": "deferred_to_global_closed_candidate_gate",
    }
    report["final_provenance"]["consumed_by_final"] = committed
    report["final_provenance"]["face_origin_values"] = (
        ["source", "hole_fill"] if is_hole_fill and committed
        else ["source"] if is_coincident_weld and committed
        else ["source", "stitch_band"] if is_zipper and committed
        else ["source", "proxy_patch", "stitch_band"] if committed
        else ["source"]
    )
    if not committed:
        report["proxy_weight"] = weight_summary(0.0)
        report["sdf_blend_weight"] = weight_summary(0.0)
        report["source_weight"] = weight_summary(1.0)


def transaction_summary(
    before: dict[str, Any],
    after: dict[str, Any],
    committed: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "step": "per_patch_transaction_summary",
        "passed": topology_not_worse(before, after),
        "accepted": len(committed),
        "rejected": len(rejected),
        "before": topology_counts(before),
        "after": topology_counts(after),
        "committed_regions": committed,
        "rolled_back_regions": rejected,
    }


def weight_summary(value: float) -> dict[str, float]:
    return {"min": value, "mean": value, "max": value, "p95": value}


def topology_not_worse(before: dict[str, Any], after: dict[str, Any]) -> bool:
    before_counts = topology_counts(before)
    after_counts = topology_counts(after)
    return counts_not_worse(before_counts, after_counts)


def counts_not_worse(before: dict[str, int], after: dict[str, int]) -> bool:
    return all(after[name] <= before[name] for name in before)


def topology_counts(metrics: dict[str, Any]) -> dict[str, int]:
    return {
        "boundary_edges": int(metrics["topology"]["boundary_edges"]),
        "non_manifold_edges": int(metrics["topology"]["non_manifold_edges"]),
        "inconsistent_winding_edges": int(metrics["topology"].get("inconsistent_winding_edges", 0)),
        "degenerate_faces": int(metrics["quality"]["degenerate_faces"]),
        "components": int(metrics["topology"]["components"]["count"]),
    }


def transaction_topology_counts(points: np.ndarray, faces: np.ndarray) -> dict[str, int]:
    """Compute only the topology fields needed by a patch transaction gate."""
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    topology, edge_faces = edge_topology(faces)
    labels = face_component_ids(faces, edge_faces)
    if faces.size:
        triangles = points[faces]
        edge_lengths = np.stack(
            [
                np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
                np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
                np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
            ],
            axis=1,
        )
        areas = triangle_areas(points, faces)
        area_epsilon = max(float(np.nanmedian(areas)) * 1e-12, 1e-18)
        degenerate = int(
            np.count_nonzero(
                (areas <= area_epsilon)
                | (edge_lengths.min(axis=1) <= 1e-15)
            )
        )
    else:
        degenerate = 0
    return {
        "boundary_edges": int(topology["boundary_edges"]),
        "non_manifold_edges": int(topology["non_manifold_edges"]),
        "inconsistent_winding_edges": int(inconsistent_winding_edges(faces)),
        "degenerate_faces": degenerate,
        "components": int(labels.max()) + 1 if labels.size else 0,
    }


def source_proxy_blend_report(
    points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, np.ndarray],
    patch_regions: list[dict[str, Any]],
) -> dict[str, Any]:
    weights = np.clip(np.asarray(provenance["proxy_weight"], dtype=np.float64), 0.0, 1.0)
    areas = triangle_areas(points, faces)
    total_area = float(areas.sum())
    proxy_area = float(np.dot(areas, weights)) if areas.size else 0.0
    source_area = max(total_area - proxy_area, 0.0)
    proxy_face_ratio = float(weights.mean()) if weights.size else 0.0
    source_face_ratio = 1.0 - proxy_face_ratio
    origins = provenance["face_origin"]
    return {
        "field": "proxy_weight",
        "sdf_blend_weight_field": "sdf_blend_weight",
        "semantics": "0=source_preserving_geometry, 1=closure_proxy_geometry",
        "range": {
            "min": float(weights.min()) if weights.size else None,
            "max": float(weights.max()) if weights.size else None,
        },
        "area_weighted": {
            "source_ratio": source_area / total_area if total_area > 0.0 else None,
            "proxy_ratio": proxy_area / total_area if total_area > 0.0 else None,
            "source_area": source_area,
            "proxy_area": proxy_area,
            "total_area": total_area,
        },
        "face_count_weighted": {
            "source_ratio": source_face_ratio,
            "proxy_ratio": proxy_face_ratio,
            "face_count": int(weights.size),
        },
        "origin_face_counts": {
            "source": int(np.count_nonzero(origins == FACE_ORIGIN["source"])),
            "proxy_patch": int(np.count_nonzero(origins == FACE_ORIGIN["proxy_patch"])),
            "stitch_band": int(np.count_nonzero(origins == FACE_ORIGIN["stitch_band"])),
            "hole_fill": int(np.count_nonzero(origins == FACE_ORIGIN["hole_fill"])),
        },
        "patch_region_weight_summary": patch_region_weight_summary(patch_regions),
    }


def triangle_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return np.zeros(0, dtype=np.float64)
    triangles = points[faces]
    return np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    ) * 0.5


def patch_region_weight_summary(patch_regions: list[dict[str, Any]]) -> dict[str, Any]:
    means = np.asarray(
        [float(region.get("proxy_weight", {}).get("mean", 0.0)) for region in patch_regions],
        dtype=np.float64,
    )
    if means.size == 0:
        return {"min": None, "mean": None, "max": None, "p95": None}
    return {
        "min": float(means.min()),
        "mean": float(means.mean()),
        "max": float(means.max()),
        "p95": float(np.percentile(means, 95)),
    }


def build_fusion_comparisons(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    final_points: np.ndarray,
    final_faces: np.ndarray,
    face_origin: np.ndarray,
    thresholds: FusionThresholds,
) -> dict[str, Any]:
    patch_faces = np.flatnonzero(face_origin != FACE_ORIGIN["source"])
    source_metrics = mesh_report(source_points, source_faces)
    final_metrics = mesh_report(final_points, final_faces)
    final_topology = final_metrics["topology"]
    if (
        final_topology["boundary_edges"] == 0
        and final_topology["non_manifold_edges"] == 0
        and final_metrics["quality"]["degenerate_faces"] == 0
    ):
        self_intersection = self_intersection_report(final_points, final_faces)
    else:
        self_intersection = {
            "method": "vtk_static_cell_locator_triangle_intersection",
            "status": "deferred_until_closed_manifold_candidate",
            "passed": False,
            "intersection_pairs": None,
            "failure_reason": "self-intersection acceptance check runs after boundary and non-manifold gates close",
        }
    return {
        "source_distance": bidirectional_distance(source_points, source_faces, final_points, final_faces, thresholds),
        "patch_local_drift": patch_local_drift(source_points, source_faces, final_points, final_faces, patch_faces, thresholds),
        "self_intersection": self_intersection,
        "before_after_topology": {
            "before": topology_summary(source_metrics),
            "after": topology_summary(final_metrics),
        },
    }
