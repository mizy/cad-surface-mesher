from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from mesh_io import compact_mesh
from mesh_metrics import edge_topology, mesh_report
from repair_inventory import face_component_ids, topology_snapshot
MESH_OUTPUT_EXTENSIONS = {".vtp", ".vtk", ".stl", ".obj", ".ply", ".glb", ".gltf"}
TOPOLOGY_ZERO_GATES = (
    ("boundary_edges_zero", "topology", "boundary_edges"),
    ("non_manifold_edges_zero", "topology", "non_manifold_edges"),
    ("non_manifold_vertices_zero", "topology", "non_manifold_vertices"),
    ("inconsistent_winding_edges_zero", "topology", "inconsistent_winding_edges"),
    ("degenerate_faces_zero", "quality", "degenerate_faces"),
)


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


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


def build_gates(
    candidate_metrics: dict[str, Any],
    comparisons: dict[str, Any],
    policy_packet: dict[str, Any],
    policy_decisions: list[dict[str, Any]],
    deterministic_passes: list[dict[str, Any]],
    potential_final_path: str | None,
    closure_proxy_path: str | None,
    unhandled_items: list[dict[str, Any]],
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


def build_ignored_outputs(
    *,
    closure_proxy_path: str,
    stage1_path: str,
    candidate_path: str,
    previews: list[str],
    group_filter: dict[str, Any] | None,
    output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    rows = [
        ignored(closure_proxy_path, "closure_proxy", "voxel marching-cubes proxy is not accepted final geometry"),
        ignored(stage1_path, "debug_stage", "exterior extraction candidate is an intermediate source artifact"),
        ignored(candidate_path, "rejected_candidate", "candidate must pass all gates before downstream consumption"),
    ]
    current_paths = [closure_proxy_path, stage1_path, candidate_path, *previews]
    if group_filter and group_filter.get("output_vtp"):
        rows.append(ignored(group_filter["output_vtp"], "debug_stage", "group-filtered mesh is an intermediate artifact"))
        current_paths.append(group_filter["output_vtp"])
    rows.extend(ignored(path, "visual_preview", "preview image is visual evidence, not geometry") for path in previews)
    if output_dir:
        rows.extend(stale_mesh_outputs(Path(output_dir), current_paths))
    return rows


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
