from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mesh_io import compact_mesh
from mesh_metrics import bbox_drift_from_reports, edge_topology, mesh_report
from repair_policy import policy_item_id
from repair_inventory import face_component_ids, inventory_truncation_items, topology_snapshot
from source_primary_transaction_record import array_sha256


SOURCE_DISTANCE_THRESHOLD = 0.002
BBOX_DRIFT_THRESHOLD = 0.002
SILHOUETTE_DRIFT_THRESHOLD = 0.05
MESH_OUTPUT_EXTENSIONS = {".vtp", ".vtk", ".stl", ".obj", ".ply", ".glb", ".gltf"}
REGION_CLASSIFICATIONS = {"patch_required", "preserve_opening", "pending_policy", "reject_region"}
SELECTION_STATUSES = {
    "keep_source",
    "use_proxy_patch",
    "use_source_zipper",
    "use_hole_fill",
    "transaction_pending",
    "hold_for_policy",
    "reject_patch",
}
TOPOLOGY_ZERO_GATES = (
    ("boundary_edges_zero", "topology", "boundary_edges"),
    ("non_manifold_edges_zero", "topology", "non_manifold_edges"),
    ("non_manifold_vertices_zero", "topology", "non_manifold_vertices"),
    ("inconsistent_winding_edges_zero", "topology", "inconsistent_winding_edges"),
    ("degenerate_faces_zero", "quality", "degenerate_faces"),
)


def run_deterministic_repair(
    points: np.ndarray,
    faces: np.ndarray,
    source_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    candidate_points = points
    candidate_faces = faces
    candidate_sources = source_indices.astype(np.int64, copy=True)
    passes: list[dict[str, Any]] = []

    candidate_points, candidate_faces, candidate_sources, pass_report = apply_keep_operator(
        "repair_pass_001",
        "remove_degenerate_source_triangles",
        "remove_zero_area_or_near_zero_edge_faces",
        candidate_points,
        candidate_faces,
        candidate_sources,
        degenerate_keep_mask(candidate_points, candidate_faces),
        {"area_epsilon": "max(median_area * 1e-12, 1e-18)", "min_edge_epsilon": 1e-15},
    )
    passes.append(pass_report)

    candidate_points, candidate_faces, candidate_sources, pass_report = apply_keep_operator(
        "repair_pass_002",
        "remove_exact_duplicate_source_triangles",
        "remove_faces_with_identical_vertex_sets_after_import_cleaning",
        candidate_points,
        candidate_faces,
        candidate_sources,
        duplicate_keep_mask(candidate_faces),
        {"duplicate_key": "sorted_cleaned_vertex_indices"},
    )
    passes.append(pass_report)
    return candidate_points, candidate_faces, candidate_sources, passes


def prune_classified_internal_components(
    points: np.ndarray,
    faces: np.ndarray,
    source_indices: np.ndarray,
    exterior_face_score: np.ndarray,
    inventory: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove only components whose complete atlas evidence says internal or flying.

    Boundary-bearing components still require unanimous semantic agreement.  A
    closed component may be removed without a boundary row only when the
    component-level exterior atlas independently marks it as a conservative
    automatic-removal candidate.
    """
    exterior_face_score = np.asarray(exterior_face_score)
    if exterior_face_score.shape != (faces.shape[0],):
        raise ValueError("exterior_face_score must contain one value per face")
    _, edge_faces = edge_topology(faces)
    labels = face_component_ids(faces, edge_faces)
    classifications: dict[int, set[str]] = {}
    for item in inventory.get("boundary_regions", {}).get("items", []):
        component_id = item.get("component_id")
        if component_id is None:
            continue
        classifications.setdefault(int(component_id), set()).add(str(item.get("classification")))
    boundary_internal_candidates = {
        component_id
        for component_id, values in classifications.items()
        if values == {"internal_or_fragment_component_perimeter"}
    }
    component_evidence = {
        int(row["component_id"]): row
        for row in inventory.get("boundary_classification", {}).get("component_evidence", [])
        if row.get("component_id") is not None
    }
    evidence_removable: set[int] = set()
    removal_reasons: dict[int, str] = {}
    protected_reasons: dict[int, str] = {}
    removable_classes = {
        "internal_or_fragment_component_perimeter",
        "isolated_floating_fragment_perimeter",
    }
    for component_id, evidence in component_evidence.items():
        expected = evidence.get("removal_classification")
        if not evidence.get("automatic_remove_candidate") or expected not in removable_classes:
            continue
        boundary_values = classifications.get(component_id, set())
        if boundary_values and boundary_values != {expected}:
            protected_reasons[component_id] = "mixed_boundary_semantics"
            continue
        component_face_ids = np.flatnonzero(labels == component_id)
        if component_face_ids.size == 0:
            protected_reasons[component_id] = "component_evidence_not_present_in_candidate"
            continue
        if np.any(exterior_face_score[component_face_ids] > 0) or evidence.get("visible_hard_keep"):
            protected_reasons[component_id] = "multi_view_first_hit_hard_keep"
            continue
        if expected == "internal_or_fragment_component_perimeter":
            if evidence.get("direct_and_sealed_internal_consensus") is not True:
                protected_reasons[component_id] = "missing_direct_and_sealed_internal_consensus"
                continue
            if evidence.get("physical_scale_small") is not True:
                protected_reasons[component_id] = "physical_scale_not_safe_for_internal_removal"
                continue
        evidence_removable.add(component_id)
        removal_reasons[component_id] = str(evidence.get("removal_reason") or expected)
    for component_id in sorted(boundary_internal_candidates - set(component_evidence)):
        protected_reasons[component_id] = "missing_component_level_atlas_evidence"
    removable = sorted(evidence_removable)
    before = mesh_report(points, faces)
    thresholds = {
        "required_boundary_classifications": sorted(removable_classes),
        "component_rule": "all boundary regions on the component must agree",
        "closed_components_without_boundary_evidence": (
            "remove only with explicit component-level atlas automatic_remove_candidate"
        ),
        "visibility_guard": "every removed component must have zero six-view first-hit faces",
        "contained_internal_rule": "direct and sealed negative evidence plus small physical scale",
        "protected_component_ids": sorted(protected_reasons),
        "protected_component_reasons": {
            str(component_id): reason for component_id, reason in sorted(protected_reasons.items())
        },
        "component_filter_thresholds": inventory.get("boundary_classification", {}).get(
            "component_thresholds",
            {},
        ),
    }
    if not removable:
        return (
            points,
            faces,
            source_indices,
            exterior_face_score,
            pass_row(
                "repair_pass_003",
                "remove_classified_internal_fragment_components",
                "skipped",
                "component_level_exterior_atlas_consensus",
                thresholds,
                before,
                before,
                np.zeros(0, dtype=np.int64),
            ),
        )
    keep = ~np.isin(labels, np.asarray(removable, dtype=np.int64))
    if not np.any(keep):
        return (
            points,
            faces,
            source_indices,
            exterior_face_score,
            pass_row(
                "repair_pass_003",
                "remove_classified_internal_fragment_components",
                "failed",
                "component_level_exterior_atlas_consensus",
                thresholds,
                before,
                before,
                source_indices,
                "component pruning would remove every face",
            ),
        )
    kept_face_ids = np.flatnonzero(keep)
    next_points, next_faces = compact_mesh(points, faces, kept_face_ids)
    next_sources = source_indices[kept_face_ids]
    next_scores = exterior_face_score[kept_face_ids]
    after = mesh_report(next_points, next_faces)
    topology_regressions = []
    for field in ("boundary_edges", "non_manifold_edges", "inconsistent_winding_edges"):
        if after["topology"][field] > before["topology"][field]:
            topology_regressions.append(field)
    if after["quality"]["degenerate_faces"] > before["quality"]["degenerate_faces"]:
        topology_regressions.append("degenerate_faces")
    if topology_regressions:
        row = pass_row(
            "repair_pass_003",
            "remove_classified_internal_fragment_components",
            "failed",
            "component_level_exterior_atlas_consensus",
            {
                **thresholds,
                "attempted_component_ids": removable,
                "rollback_reason_codes": topology_regressions,
            },
            before,
            before,
            np.zeros(0, dtype=np.int64),
            "component pruning would worsen source topology",
        )
        row["scope"]["attempted_removed_source_triangle_count"] = int(np.count_nonzero(~keep))
        row["scope"]["removed_component_count"] = 0
        return points, faces, source_indices, exterior_face_score, row
    row = pass_row(
        "repair_pass_003",
        "remove_classified_internal_fragment_components",
        "applied",
        "component_level_exterior_atlas_consensus",
        {
            **thresholds,
            "removed_component_ids": removable,
            "removed_component_reasons": {
                str(component_id): removal_reasons.get(
                    component_id,
                    "unanimous_internal_boundary_evidence",
                )
                for component_id in removable
            },
        },
        before,
        after,
        source_indices[~keep],
    )
    row["scope"]["removed_component_count"] = len(removable)
    return next_points, next_faces, next_sources, next_scores, row


def build_comparisons(
    source_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    closure_proxy_visual_drift: dict[str, Any],
) -> dict[str, Any]:
    bbox = bbox_drift_from_reports(source_metrics, candidate_metrics)
    return {
        "source_distance": {
            "method": "symmetric_surface_distance_to_source",
            "max": None,
            "p95": None,
            "mean": None,
            "threshold": SOURCE_DISTANCE_THRESHOLD,
            "passed": False,
            "status": "not_implemented",
            "failure_reason": "surface-distance sampling is required before accepting a repaired mesh",
        },
        "bbox_drift": {
            **bbox,
            "threshold": BBOX_DRIFT_THRESHOLD,
            "passed": bbox["max_ratio"] <= BBOX_DRIFT_THRESHOLD,
            "status": "computed",
        },
        "silhouette_drift": {
            "method": "source_triangle_preserving_candidate_no_vertex_motion",
            "changed_ratio_max": 0.0,
            "overlap_ratio_min": 1.0,
            "threshold": SILHOUETTE_DRIFT_THRESHOLD,
            "passed": True,
            "status": "computed_by_construction",
        },
        "closure_proxy_silhouette_drift": closure_proxy_visual_drift,
        "before_after_topology": {
            "before": topology_snapshot(source_metrics),
            "after": topology_snapshot(candidate_metrics),
        },
    }


def build_gates(
    candidate_metrics: dict[str, Any],
    comparisons: dict[str, Any],
    policy_packet: dict[str, Any],
    policy_decisions: list[dict[str, Any]],
    deterministic_passes: list[dict[str, Any]],
    potential_final_path: str | None,
    closure_proxy_path: str | None,
    unhandled_items: list[dict[str, Any]],
    *,
    hybrid_candidate_path: str | None = None,
    hybrid_candidate_produced: bool = False,
    patch_regions: list[dict[str, Any]] | None = None,
    require_hybrid_candidate: bool = False,
) -> dict[str, Any]:
    boundary_gate = zero_count_gate(candidate_metrics, "topology", "boundary_edges")
    non_manifold_gate = zero_count_gate(candidate_metrics, "topology", "non_manifold_edges")
    non_manifold_vertex_gate = zero_count_gate(
        candidate_metrics, "topology", "non_manifold_vertices"
    )
    winding_gate = zero_count_gate(candidate_metrics, "topology", "inconsistent_winding_edges")
    degenerate_gate = zero_count_gate(candidate_metrics, "quality", "degenerate_faces")
    pending_policy = [
        row for row in policy_decisions if row["status"] != "decided" or row.get("decision") == "defer"
    ]
    rejected_policy = [
        row
        for row in policy_decisions
        if row["status"] == "decided" and row.get("decision") == "reject"
    ]
    policy_packet_truncated = bool(policy_packet.get("truncated", False))
    failed_passes = [row for row in deterministic_passes if row["status"] == "failed"]
    blocking_unhandled = [row for row in unhandled_items if row.get("blocking")]
    unhandled_gate = gate(
        not blocking_unhandled,
        len(blocking_unhandled),
        0,
        "blocking unhandled repair or validation items remain" if blocking_unhandled else None,
    )
    if blocking_unhandled:
        unhandled_gate["reason_codes"] = unhandled_reason_codes(blocking_unhandled)
    source_distance = comparisons.get("source_distance", {})
    bbox = comparisons.get("bbox_drift", {})
    silhouette = comparisons.get("silhouette_drift", {})
    self_intersection = comparisons.get("self_intersection", {})
    required_metrics = required_metric_status(candidate_metrics, source_distance, bbox, silhouette)
    gates = {
        "boundary_edges_zero": boundary_gate,
        "non_manifold_edges_zero": non_manifold_gate,
        "non_manifold_vertices_zero": non_manifold_vertex_gate,
        "inconsistent_winding_edges_zero": winding_gate,
        "degenerate_faces_zero": degenerate_gate,
        "closed_component_volumes_reliable": gate(
            bool(candidate_metrics.get("volume", {}).get("reliable")),
            candidate_metrics.get("volume", {}).get("reliable"),
            True,
            "closed component volumes are not reliable",
        ),
        "component_orientation_consistent": gate(
            bool(candidate_metrics.get("volume", {}).get("orientation_consistent")),
            candidate_metrics.get("volume", {}).get("orientation_consistent"),
            True,
            "closed components do not share a consistent outward orientation",
        ),
        "source_distance_within_tolerance": gate(
            metric_is_computed(source_distance, {"computed"}) and bool(source_distance.get("passed")),
            source_distance.get("max"),
            source_distance.get("threshold"),
            source_distance.get("failure_reason"),
        ),
        "bbox_drift_within_tolerance": gate(
            metric_is_computed(bbox, {"computed"}) and bool(bbox.get("passed")),
            bbox.get("max_ratio"),
            bbox.get("threshold"),
            bbox.get("failure_reason") or "bbox drift metric is missing or exceeds tolerance",
        ),
        "silhouette_drift_within_tolerance": gate(
            metric_is_computed(silhouette, {"computed", "computed_by_construction"}) and bool(silhouette.get("passed")),
            silhouette.get("changed_ratio_max"),
            silhouette.get("threshold"),
            silhouette.get("failure_reason") or "silhouette drift metric is missing or exceeds tolerance",
        ),
        "self_intersection_free": gate(
            self_intersection.get("status") == "computed" and bool(self_intersection.get("passed")),
            self_intersection.get("intersection_pairs"),
            0,
            self_intersection.get("failure_reason") or "self-intersection check is missing, incomplete, or found intersections",
        ),
        "opening_policy_resolved": policy_resolution_gate(
            pending_count=len(pending_policy),
            rejected_count=len(rejected_policy),
            packet_truncated=policy_packet_truncated,
        ),
        "deterministic_repair_passes_succeeded": gate(
            not failed_passes,
            len(failed_passes),
            0,
            "one or more deterministic repair operators failed" if failed_passes else None,
        ),
        "required_metrics_present": gate(
            required_metrics["passed"],
            required_metrics["status"],
            "computed source_distance, bbox_drift, silhouette_drift, topology",
            required_metrics["failure_reason"],
        ),
        "closure_proxy_not_final": gate(
            bool(potential_final_path) and potential_final_path != closure_proxy_path,
            potential_final_path,
            "final_output_path_must_not_equal_outputs.closure_proxy_vtp",
            "closure proxy is diagnostic only" if potential_final_path == closure_proxy_path else None,
        ),
        "unhandled_blocking_items_zero": unhandled_gate,
        "visual_only_html_contract": gate(
            True,
            "html_report_contains_summary_diagnostics_contract_with_bounded_viewer_payload",
            "HTML companion report",
        ),
    }
    if require_hybrid_candidate:
        candidate_path = hybrid_candidate_path
        candidate_produced = hybrid_candidate_produced
        regions = patch_regions or []
        blocking_regions = [
            row for row in regions if row.get("selection_status") in {"hold_for_policy", "reject_patch"}
        ]
        missing_patch_provenance = [
            row
            for row in regions
            if row.get("selection_status") in {"use_proxy_patch", "use_source_zipper", "use_hole_fill"}
            and not row.get("final_provenance", {}).get("consumed_by_final")
        ]
        missing_patch_artifacts = [
            row
            for row in regions
            if row.get("selection_status") == "use_proxy_patch"
            and not all(row.get("artifacts", {}).get(key) for key in ("proxy_patch_vtp", "seam_belt_vtp", "stitch_band_vtp"))
        ]
        missing_zipper_artifacts = [
            row
            for row in regions
            if row.get("selection_status") == "use_source_zipper"
            and not (
                row.get("artifacts", {}).get("seam_belt_vtp")
                and (
                    row.get("artifacts", {}).get("stitch_band_vtp")
                    or row.get("artifacts", {}).get("coincident_weld_vtp")
                )
            )
        ]
        missing_patch_artifacts.extend(missing_zipper_artifacts)
        missing_hole_fill_artifacts = [
            row
            for row in regions
            if row.get("selection_status") == "use_hole_fill"
            and not all(row.get("artifacts", {}).get(key) for key in ("seam_belt_vtp", "hole_fill_vtp"))
        ]
        missing_patch_artifacts.extend(missing_hole_fill_artifacts)
        gates.update(
            {
                "hybrid_candidate_present": gate(
                    bool(candidate_path) and bool(candidate_produced),
                    candidate_path,
                    "current-run hybrid fused candidate artifact",
                    "hybrid candidate mesh was not produced by this run",
                ),
                "accepted_path_is_hybrid_candidate": gate(
                    bool(potential_final_path)
                    and bool(candidate_path)
                    and potential_final_path == candidate_path
                    and potential_final_path != closure_proxy_path,
                    potential_final_path,
                    "decision.final_output_path == outputs.hybrid_fused_candidate_vtp",
                    "accepted output must be the gated hybrid candidate, not the source fallback or closure proxy",
                ),
                "patch_regions_resolved": gate(
                    not blocking_regions,
                    len(blocking_regions),
                    0,
                    "one or more patch regions are rejected or waiting for policy",
                ),
                "patch_artifacts_present": gate(
                    not missing_patch_artifacts,
                    len(missing_patch_artifacts),
                    0,
                    "committed patch regions must list their real seam and stitch artifacts",
                ),
                "face_provenance_present": gate(
                    not missing_patch_provenance,
                    len(missing_patch_provenance),
                    0,
                    "all committed generated faces must carry face_origin, source_triangle_index, and fusion_region_id",
                ),
            }
        )
    return gates


def zero_count_gate(metrics: dict[str, Any], section: str, field: str) -> dict[str, Any]:
    value = nested_metric(metrics, section, field)
    if not stable_count(value):
        return gate(False, value, 0, f"{section}.{field} metric missing or not stable")
    return gate(int(value) == 0, int(value), 0)


def required_metric_status(
    metrics: dict[str, Any],
    source_distance: dict[str, Any],
    bbox: dict[str, Any],
    silhouette: dict[str, Any],
) -> dict[str, Any]:
    missing = []
    for _, section, field in TOPOLOGY_ZERO_GATES:
        if not stable_count(nested_metric(metrics, section, field)):
            missing.append(f"{section}.{field}")
    if not stable_count(nested_metric(metrics, "topology", "components", "count")):
        missing.append("topology.components.count")
    if not metric_is_computed(source_distance, {"computed"}):
        missing.append("source_distance")
    if not metric_is_computed(bbox, {"computed"}):
        missing.append("bbox_drift")
    if not metric_is_computed(silhouette, {"computed", "computed_by_construction"}):
        missing.append("silhouette_drift")
    return {
        "passed": not missing,
        "status": "computed" if not missing else {"missing": missing},
        "failure_reason": None if not missing else f"required metrics missing or unstable: {', '.join(missing)}",
    }


def metric_is_computed(metric: dict[str, Any], allowed_statuses: set[str]) -> bool:
    return metric.get("status") in allowed_statuses


def nested_metric(metrics: dict[str, Any], *keys: str) -> Any:
    value: Any = metrics
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def stable_count(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, np.integer)):
        return False
    return int(value) >= 0


def build_decision(gates: dict[str, Any], candidate_path: str | None) -> dict[str, Any]:
    failed_required = [name for name, row in gates.items() if row.get("required") and not row.get("passed")]
    if not failed_required and candidate_path:
        return {"status": "accepted", "reason_codes": [], "final_output_path": candidate_path}
    return {
        "status": "rejected",
        "reason_codes": reason_codes(failed_required, gates),
        "final_output_path": None,
    }


def accepted_outputs(
    outputs: dict[str, Any],
    decision: dict[str, Any],
    final_path: str,
    rejected_candidate_path: str | None = None,
    *,
    source_fallback_path: str | None = None,
) -> dict[str, Any]:
    result = dict(outputs)
    if decision["status"] == "accepted":
        result["accepted_mesh_vtp"] = final_path
        result["accepted_mesh_available"] = True
        result["rejected_candidate_vtp"] = None
        result["source_fallback_vtp"] = None
        result["mesh_result"] = {
            "status": "accepted_repair",
            "path": final_path,
            "role": "accepted_repaired_mesh",
            "accepted": True,
            "engineering_ready": True,
        }
    else:
        result["accepted_mesh_vtp"] = None
        result["accepted_mesh_available"] = False
        result["rejected_candidate_vtp"] = rejected_candidate_path or final_path
        fallback = source_fallback_path or result.get("source_preserving_candidate_vtp")
        result["source_fallback_vtp"] = fallback
        result["mesh_result"] = {
            "status": "repair_rejected_with_source_fallback" if fallback else "repair_rejected_no_mesh",
            "path": fallback,
            "role": "source_preserving_unrepaired_fallback" if fallback else None,
            "accepted": False,
            "engineering_ready": False,
            "warning": (
                "fallback preserves cleaned source geometry but did not pass watertight repair gates"
                if fallback
                else "no mesh passed repair acceptance gates"
            ),
        }
    return result


def build_patch_regions(
    inventory: dict[str, Any],
    policy_decisions: list[dict[str, Any]],
    *,
    patch_dir: str | Path,
) -> list[dict[str, Any]]:
    decision_by_source = policy_decisions_by_source(policy_decisions)
    patch_dir = Path(patch_dir)
    rows: list[dict[str, Any]] = []
    region_index = 1
    for section in ("boundary_regions", "gap_regions", "non_manifold_regions", "overlap_regions"):
        for source_region in inventory.get(section, {}).get("items", []):
            rows.append(
                patch_region_row(
                    region_index,
                    section,
                    source_region,
                    decision_by_source.get(source_region.get("id")),
                    patch_dir,
                )
            )
            region_index += 1
    return rows


def policy_decisions_by_source(policy_decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows = {}
    for decision in policy_decisions:
        item_id = str(decision.get("item_id", ""))
        for prefix in ("opening_",):
            if item_id.startswith(prefix):
                suffix = item_id[len(prefix):]
                rows[f"boundary_loop_{suffix}"] = decision
                rows[f"gap_or_opening_candidate_{suffix}"] = decision
                rows[f"semantic_opening_{suffix}"] = decision
        source_region = decision.get("source_region")
        if source_region:
            rows[str(source_region)] = decision
    return rows


def patch_region_row(
    index: int,
    section: str,
    source_region: dict[str, Any],
    policy_decision: dict[str, Any] | None,
    patch_dir: Path,
) -> dict[str, Any]:
    region_id = f"region_{index:04d}"
    classification, selection_status, reason_codes = classify_patch_region(section, source_region, policy_decision)
    artifacts = patch_artifacts(region_id, patch_dir)
    row = {
        "id": region_id,
        "source_region_id": source_region.get("id"),
        "source_region_type": source_region.get("type"),
        "classification": classification,
        "selection_status": selection_status,
        "selection_region": {
            "bbox": source_region.get("bbox"),
            "edge_count": source_region.get("edge_count"),
            "length": source_region.get("length"),
            "source_triangle_ids": source_region.get("source_triangle_ids", []),
            "source_triangle_count": source_region.get("source_triangle_count") or source_region.get("face_count"),
            "closed_chain": source_region.get("closed_chain"),
        },
        "source_trust": source_trust(section, source_region, classification),
        "proxy_trust": proxy_trust(selection_status),
        "patch_metadata": {
            "patch_type": "proxy_patch" if classification == "patch_required" else None,
            "source": "closure_proxy",
            "extraction_method": "not_implemented",
            "region_classifier": source_region.get("classification"),
            "policy_item_id": policy_item_id(source_region["id"]) if source_region.get("id") else None,
            "policy_decision": policy_decision,
        },
        "seam_results": {
            "status": "not_run",
            "boundary_edges_after": None,
            "non_manifold_edges_after": None,
            "max_seam_gap": None,
            "reason": "local patch extraction and stitch-band fusion have not run",
        },
        "gating_metrics": {
            "patch_extracted": False,
            "seam_closed": False,
            "source_distance_checked": False,
            "provenance_written": False,
        },
        "accept_reject_reason_codes": reason_codes,
        "artifacts": artifacts,
        "final_provenance": {
            "consumed_by_final": False,
            "face_origin": None,
            "source_triangle_index": None,
            "fusion_region_id": region_id,
        },
    }
    assert row["classification"] in REGION_CLASSIFICATIONS
    assert row["selection_status"] in SELECTION_STATUSES
    return row


def classify_patch_region(
    section: str,
    source_region: dict[str, Any],
    policy_decision: dict[str, Any] | None,
) -> tuple[str, str, list[str]]:
    if source_region.get("requires_policy"):
        if not policy_decision or policy_decision.get("status") != "decided":
            return "pending_policy", "hold_for_policy", ["policy_review_pending"]
        if policy_decision.get("decision") == "preserve":
            return "preserve_opening", "keep_source", ["semantic_opening_preserved_by_policy"]
        if policy_decision.get("decision") in {"defer"}:
            return "pending_policy", "hold_for_policy", ["policy_review_pending"]
        if policy_decision.get("decision") in {"reject"}:
            return "reject_region", "reject_patch", ["policy_region_rejected"]
        return "patch_required", "reject_patch", ["proxy_patch_extraction_failed"]
    if section in {"boundary_regions", "gap_regions", "non_manifold_regions"}:
        return "patch_required", "reject_patch", ["proxy_patch_extraction_failed"]
    return "reject_region", "reject_patch", ["patch_region_rejected"]


def patch_artifacts(region_id: str, patch_dir: Path) -> dict[str, Any]:
    return {
        "proxy_patch_vtp": str(patch_dir / f"{region_id}_proxy_patch.vtp"),
        "seam_belt_vtp": str(patch_dir / f"{region_id}_seam_belt.vtp"),
        "stitch_band_vtp": str(patch_dir / f"{region_id}_stitch_band.vtp"),
        "selection_visuals": [],
    }


def source_trust(section: str, source_region: dict[str, Any], classification: str) -> dict[str, Any]:
    if classification == "preserve_opening":
        return {
            "status": "trusted_policy_opening",
            "score": 1.0,
            "reasons": ["policy selected keep_source for this opening"],
        }
    reasons = [source_region.get("policy_reason_source") or section]
    if source_region.get("blocking"):
        reasons.append("region is blocking for acceptance")
    return {
        "status": "untrusted_topology" if classification == "patch_required" else "rejected_or_unknown",
        "score": 0.0 if classification == "patch_required" else None,
        "reasons": reasons,
    }


def proxy_trust(selection_status: str) -> dict[str, Any]:
    if selection_status == "use_proxy_patch":
        return {"status": "trusted_local_patch", "score": 1.0, "reasons": []}
    return {
        "status": "diagnostic_only",
        "score": None,
        "reasons": ["closure proxy is global diagnostic output until a local patch extractor accepts this region"],
    }


def write_hybrid_debug_artifacts(output_dir: Path, report: dict[str, Any]) -> None:
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    patch_regions = report.get("patch_regions", [])
    patch_graph = {
        "kind": "region_patch_graph",
        "status": "report_only_no_patch_extraction",
        "regions": [
            {
                "id": row.get("id"),
                "source_region_id": row.get("source_region_id"),
                "classification": row.get("classification"),
                "selection_status": row.get("selection_status"),
                "artifacts": row.get("artifacts"),
            }
            for row in patch_regions
        ],
        "edges": [],
    }
    fusion_trace = {
        "kind": "hybrid_fusion_trace",
        "status": "rejected" if report.get("decision", {}).get("status") != "accepted" else "accepted",
        "final_output_path": report.get("decision", {}).get("final_output_path"),
        "hybrid_fused_candidate_vtp": report.get("outputs", {}).get("hybrid_fused_candidate_vtp"),
        "closure_proxy_role": "diagnostic_only",
        "blocking_reason_codes": report.get("decision", {}).get("reason_codes", []),
        "patch_region_count": len(patch_regions),
    }
    (debug_dir / "region_patch_graph.json").write_text(json.dumps(patch_graph, indent=2), encoding="utf-8")
    (debug_dir / "hybrid_fusion_trace.json").write_text(json.dumps(fusion_trace, indent=2), encoding="utf-8")


def build_unhandled_items(
    inventory: dict[str, Any],
    policy_packet: dict[str, Any],
    *,
    has_policy_file: bool,
) -> list[dict[str, Any]]:
    items = [
        {
            "item": "source_preserving_gap_stitch_or_cap_geometry",
            "status": "not_implemented",
            "blocking": True,
            "failure_reason": "current deterministic operators only remove degenerate and exact duplicate source triangles",
        },
        {
            "item": "source_distance_metric",
            "status": "not_implemented",
            "blocking": True,
            "failure_reason": "symmetric surface-distance sampling is required for acceptance",
        },
    ]
    items.extend(inventory.get("not_checked", []))
    items.extend(inventory_truncation_items(inventory))
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
    return items


def build_ignored_outputs(
    *,
    closure_proxy_path: str,
    stage1_path: str,
    candidate_path: str,
    previews: list[str],
    group_filter: dict[str, Any] | None,
    output_dir: str | Path | None = None,
    hybrid_candidate_path: str | None = None,
    patch_regions: list[dict[str, Any]] | None = None,
    debug_artifacts: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows = [
        ignored(closure_proxy_path, "closure_proxy", "voxel marching-cubes proxy is not accepted final geometry"),
        ignored(stage1_path, "debug_stage", "exterior extraction candidate is an intermediate source artifact"),
        ignored(candidate_path, "rejected_candidate", "candidate must pass all gates before downstream consumption"),
    ]
    current_paths = [closure_proxy_path, stage1_path, candidate_path, *previews]
    rejected_hybrid_path = hybrid_candidate_path
    if rejected_hybrid_path:
        rows.append(
            ignored(
                rejected_hybrid_path,
                "rejected_hybrid_fused_candidate",
                "hybrid candidate is unsafe unless produced by current-run fusion and accepted by all gates",
            )
        )
        current_paths.append(rejected_hybrid_path)
    if group_filter and group_filter.get("output_vtp"):
        rows.append(ignored(group_filter["output_vtp"], "debug_stage", "group-filtered mesh is an intermediate artifact"))
        current_paths.append(group_filter["output_vtp"])
    for artifact_path in patch_artifact_paths(patch_regions or []):
        rows.append(ignored(artifact_path, "patch_only_artifact", "patch artifact is not a standalone final mesh"))
        current_paths.append(artifact_path)
    for artifact_path in debug_artifacts or []:
        rows.append(ignored(artifact_path, "debug_artifact", "debug audit artifact is not geometry for acceptance"))
        current_paths.append(artifact_path)
    rows.extend(ignored(path, "visual_preview", "preview image is visual evidence, not geometry") for path in previews)
    if output_dir:
        rows.extend(stale_mesh_outputs(Path(output_dir), current_paths))
    return rows


def patch_artifact_paths(patch_regions: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for row in patch_regions:
        artifacts = row.get("artifacts", {})
        for key in ("proxy_patch_vtp", "seam_belt_vtp", "stitch_band_vtp", "hole_fill_vtp"):
            if artifacts.get(key):
                paths.append(artifacts[key])
        paths.extend(artifacts.get("selection_visuals") or [])
    return paths


def stale_mesh_outputs(output_dir: Path, current_paths: list[str]) -> list[dict[str, Any]]:
    if not output_dir.exists():
        return []
    current_keys = {path_key(path) for path in current_paths if path}
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in MESH_OUTPUT_EXTENSIONS:
            continue
        if path_key(path) in current_keys:
            continue
        rows.append(
            ignored(
                str(path),
                "stale_output_file",
                "mesh file in output directory is not referenced by the current report",
            )
        )
    return rows


def path_key(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def input_truth(input_path: Path, group_filter: dict[str, Any] | None, derived_artifacts: list[str]) -> dict[str, Any]:
    metadata_paths = [group_filter["source"]] if group_filter else []
    notes = [
        "read_surface imports, extracts, triangulates, and cleans the mesh before audit.",
        "source_triangle_index refers to the normalized import-session triangle order, not persistent CAD face IDs.",
    ]
    if group_filter:
        notes.append("group metadata is accepted only because flattened triangle count matched the primary input.")
    return {
        "primary_geometry_path": str(input_path),
        "primary_geometry_kind": "mesh",
        "metadata_paths": metadata_paths,
        "derived_artifacts": derived_artifacts,
        "acceptance_source": "stage1_exterior_candidate_with_source_triangle_index",
        "notes": notes,
    }


def apply_keep_operator(
    pass_id: str,
    name: str,
    method: str,
    points: np.ndarray,
    faces: np.ndarray,
    source_indices: np.ndarray,
    keep: np.ndarray,
    thresholds: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    before = mesh_report(points, faces)
    removed_sources = source_indices[~keep]
    if np.all(keep):
        return points, faces, source_indices, pass_row(pass_id, name, "skipped", method, thresholds, before, before, [])
    if not np.any(keep):
        return points, faces, source_indices, pass_row(
            pass_id, name, "failed", method, thresholds, before, before, removed_sources, "operator would remove every face"
        )
    local_indices = np.flatnonzero(keep)
    next_points, next_faces = compact_mesh(points, faces, local_indices)
    next_sources = source_indices[local_indices]
    after = mesh_report(next_points, next_faces)
    return next_points, next_faces, next_sources, pass_row(
        pass_id, name, "applied", method, thresholds, before, after, removed_sources
    )


def pass_row(
    pass_id: str,
    name: str,
    status: str,
    method: str,
    thresholds: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    removed_sources: np.ndarray,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    removed_sources = np.asarray(removed_sources, dtype=np.int64)
    return {
        "id": pass_id,
        "name": name,
        "status": status,
        "method": method,
        "scope": {
            "source_region": "stage1_exterior_candidate",
            "removed_source_triangle_count": int(removed_sources.size),
            "removed_source_triangle_ids": removed_sources.astype(int).tolist(),
            "removed_source_triangle_index_sha256": array_sha256(removed_sources),
            "removed_source_triangle_index_complete": True,
            "removal_reason_code": _removal_reason_code(name),
        },
        "thresholds": thresholds,
        "before": topology_snapshot(before),
        "after": topology_snapshot(after),
        "outputs": [],
        "failure_reason": failure_reason,
    }


def _removal_reason_code(name: str) -> str:
    return {
        "remove_degenerate_source_triangles": "degenerate_source_triangle",
        "remove_exact_duplicate_source_triangles": "exact_duplicate_source_triangle",
        "remove_classified_internal_fragment_components": (
            "classified_internal_or_fragment_component"
        ),
    }.get(name, "unknown_pre_baseline_removal")


def degenerate_keep_mask(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = points[faces]
    edges = [
        np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
        np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
        np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
    ]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    area_eps = max(float(np.nanmedian(areas)) * 1e-12, 1e-18)
    min_edge = np.minimum.reduce(edges)
    return (areas > area_eps) & (min_edge > 1e-15)


def duplicate_keep_mask(faces: np.ndarray) -> np.ndarray:
    if faces.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    keys = np.sort(faces, axis=1)
    _, first_indices = np.unique(keys, axis=0, return_index=True)
    keep = np.zeros(faces.shape[0], dtype=bool)
    keep[np.sort(first_indices)] = True
    return keep


def gate(passed: bool, value: Any, threshold: Any, failure_reason: str | None = None) -> dict[str, Any]:
    return {
        "required": True,
        "passed": bool(passed),
        "value": value,
        "threshold": threshold,
        "failure_reason": None if passed else failure_reason or "required gate failed",
    }


def policy_resolution_gate(
    *,
    pending_count: int,
    rejected_count: int,
    packet_truncated: bool,
) -> dict[str, Any]:
    blocking_count = int(pending_count) + int(rejected_count)
    passed = blocking_count == 0 and not packet_truncated
    reasons: list[str] = []
    if pending_count:
        reasons.append("policy review or deferred decisions remain")
    if rejected_count:
        reasons.append("explicitly rejected regions remain unresolved for watertight acceptance")
    if packet_truncated:
        reasons.append("policy packet does not cover the complete opening inventory")
    row = gate(passed, blocking_count, 0, "; ".join(reasons) or None)
    row.update(
        {
            "pending_count": int(pending_count),
            "rejected_count": int(rejected_count),
            "packet_truncated": bool(packet_truncated),
        }
    )
    reason_codes: list[str] = []
    if pending_count:
        reason_codes.append("policy_review_pending")
    if rejected_count:
        reason_codes.append("policy_region_rejected")
    if packet_truncated:
        reason_codes.append("opening_inventory_unresolved")
    if reason_codes:
        row["reason_codes"] = reason_codes
    return row


def unhandled_reason_codes(items: list[dict[str, Any]]) -> list[str]:
    codes: list[str] = []
    for item in items:
        raw = item.get("reason_codes") or item.get("reason_code") or []
        if isinstance(raw, str):
            raw = [raw]
        codes.extend(str(code) for code in raw)
    return list(dict.fromkeys(codes))


def reason_codes(failed_required: list[str], gates: dict[str, Any] | None = None) -> list[str]:
    mapping = {
        "boundary_edges_zero": "topology_gate_failed",
        "non_manifold_edges_zero": "topology_gate_failed",
        "non_manifold_vertices_zero": "topology_gate_failed",
        "inconsistent_winding_edges_zero": "orientation_gate_failed",
        "degenerate_faces_zero": "degenerate_gate_failed",
        "closed_component_volumes_reliable": "volume_gate_failed",
        "component_orientation_consistent": "orientation_gate_failed",
        "source_distance_within_tolerance": "source_distance_gate_failed",
        "bbox_drift_within_tolerance": "bbox_drift_gate_failed",
        "silhouette_drift_within_tolerance": "silhouette_drift_gate_failed",
        "self_intersection_free": "self_intersection_gate_failed",
        "opening_policy_resolved": "opening_policy_gate_failed",
        "deterministic_repair_passes_succeeded": "deterministic_repair_failed",
        "required_metrics_present": "required_metric_missing",
        "closure_proxy_not_final": "closure_proxy_only",
        "unhandled_blocking_items_zero": "unhandled_blocking_items",
        "hybrid_candidate_present": "hybrid_candidate_missing",
        "accepted_path_is_hybrid_candidate": "hybrid_candidate_missing",
        "patch_artifacts_present": "patch_artifact_missing",
        "face_provenance_present": "patch_provenance_missing",
        "patch_regions_resolved": "patch_region_rejected",
        "patch_provenance_present": "patch_provenance_missing",
        "source_projected_candidate_present": "projected_candidate_missing",
        "accepted_path_is_source_projected_candidate": "projected_candidate_missing",
        "shell_closure_policy_authorized": "shell_closure_policy_missing",
        "projection_topology_matches_proxy": "projection_topology_failed",
        "projection_mapping_complete": "projection_mapping_failed",
        "source_supported_vertices_fully_projected": "projection_residual_failed",
        "exact_projection_coverage": "projection_coverage_failed",
        "post_write_validation_passed": "post_write_validation_failed",
        "six_view_depth_regression_free": "six_view_depth_regression_failed",
        "sealed_exterior_erosion_core_preserved": "sealed_exterior_core_failed",
        "component_orientation_outward": "orientation_gate_failed",
        "visual_only_html_contract": "visual_only_html_contract_failed",
    }
    codes = []
    for name in failed_required:
        codes.append(mapping.get(name, name))
        if gates:
            codes.extend(gates.get(name, {}).get("reason_codes", []))
    return list(dict.fromkeys(codes)) or ["not_implemented"]


def ignored(path: str, kind: str, reason: str) -> dict[str, Any]:
    return {"path": path, "kind": kind, "reason": reason, "safe_for_acceptance": False}
