from __future__ import annotations

from typing import Any


def two_stage_repair_report(
    stage_reports: dict[str, dict[str, Any]],
    extraction: dict[str, Any],
    group_filter: dict[str, Any] | None,
    remesh: dict[str, Any],
    deterministic_passes: list[dict[str, Any]],
    patch_regions: list[dict[str, Any]] | None = None,
    decision: dict[str, Any] | None = None,
    source_projected_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original = stage_reports["original_input"]
    stage1 = stage_reports["stage1_exterior_candidate"]
    candidate = stage_reports["deterministic_repair_candidate"]
    closure_proxy = stage_reports["closure_proxy"]
    hybrid_candidate = stage_reports.get("hybrid_fused_candidate")
    projected_candidate = source_projected_candidate or stage_reports.get("source_projected_watertight_candidate")
    repair_accepted = bool(decision and decision.get("status") == "accepted")
    group_removed = int(group_filter.get("removed_triangles", 0)) if group_filter else 0
    total_removed = int(extraction["removed_triangles"])
    trace = [
        trace_row("original_input", original, "loaded triangle surface from source geometry"),
        {
            "stage": "stage0_group_filter",
            "operation": "remove named non-target groups before geometry repair",
            "removed_triangles": group_removed,
            "output": group_filter.get("output_vtp") if group_filter else None,
            "status": "applied" if group_filter else "not_provided",
        },
        trace_row("stage1_exterior_candidate", stage1, "outer-wall extraction from source triangles"),
        trace_row(
            "deterministic_repair_candidate",
            candidate,
            "source-preserving deterministic cleanup without moving source vertices",
        ),
        trace_row(
            "closure_proxy",
            closure_proxy,
            "sealed-shell far-field fill and marching-cubes diagnostic closure proxy",
        ),
    ]
    if hybrid_candidate:
        trace.append(
            trace_row(
                "hybrid_fused_candidate",
                hybrid_candidate,
                "ungated source candidate after independent zipper, hole-fill, and proxy-patch transactions",
            )
        )
    if projected_candidate:
        trace.append(
            trace_row(
                "source_projected_watertight_candidate",
                projected_candidate,
                "project closure-shell vertices onto source geometry while preserving watertight connectivity",
            )
        )
    return {
        "geometry_to_mesh_trace": trace,
        "change_summary": {
            "removed": {
                "group_name_removed_triangles": group_removed,
                "visibility_removed_triangles": max(0, total_removed - group_removed),
                "total_removed_triangles": total_removed,
                "removed_triangle_ratio": extraction["removed_triangle_ratio"],
                "largest_removed_groups": group_filter.get("largest_removed_groups", []) if group_filter else [],
            },
            "deterministic_source_preserving_passes": deterministic_passes,
            "diagnostic_closure_proxy": {
                "scope": "candidate topology diagnostic only",
                "method": "sealed_voxel_shell_far_field_fill_plus_marching_cubes",
                "pitch": remesh.get("pitch"),
                "accepted_final_geometry": False,
                "classification": "implicit_global_closure_not_per_gap",
            },
            "hybrid_fusion": hybrid_fusion_summary(patch_regions or []),
            "source_projected_watertight_shell": {
                "produced": projected_candidate is not None,
                "method": "fixed_closure_topology_with_accepted_vertices_projected_to_source_surface",
                "preferred_final_metrics": projected_candidate is not None,
                "accepted_final_geometry": bool(projected_candidate and repair_accepted),
            },
            "boundary_semantics": {
                "classified_individually": True,
                "source": "inventory_after.boundary_regions.items",
                "report_limit_applied_to_geometry": False,
            },
        },
        "defect_matrix": defect_matrix(
            original,
            stage1,
            candidate,
            closure_proxy,
            hybrid_candidate,
            source_projected_final=projected_candidate,
        ),
        "requested_capabilities": requested_capabilities(
            stage1,
            candidate,
            closure_proxy,
            hybrid_candidate,
            source_projected_final=projected_candidate,
            patch_regions=patch_regions or [],
            deterministic_passes=deterministic_passes,
            accepted_final_geometry=repair_accepted,
        ),
    }


def adaptive_refinement_change_report(
    source_metrics: dict[str, Any],
    refined_metrics: dict[str, Any],
    size_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "geometry_to_mesh_trace": [
            trace_row("source_exterior_candidate", source_metrics, "source geometry used for adaptive sizing"),
            trace_row("adaptive_refined_source", refined_metrics, "conforming edge split on source triangles"),
        ],
        "change_summary": {
            "fine_faces": size_report["fine_faces"],
            "transition_faces": size_report["transition_faces"],
            "base_faces": size_report["base_faces"],
            "merge_method": "conforming_edge_split_shared_edges_no_t_junction_overlay",
            "coarsening": "not_implemented",
            "watertight_sealing": "not_performed",
        },
        "defect_matrix": {
            "source": topology_snapshot(source_metrics),
            "adaptive_refined_source": topology_snapshot(refined_metrics),
        },
    }


def trace_row(stage: str, metrics: dict[str, Any], operation: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "operation": operation,
        "triangles": metrics["triangles"],
        "points": metrics["points"],
        "boundary_edges": metrics["topology"]["boundary_edges"],
        "non_manifold_edges": metrics["topology"]["non_manifold_edges"],
        "non_manifold_vertices": metrics["topology"]["non_manifold_vertices"],
        "components": metrics["topology"]["components"]["count"],
        "degenerate_faces": metrics["quality"]["degenerate_faces"],
    }


def defect_matrix(
    original: dict[str, Any],
    stage1: dict[str, Any],
    candidate: dict[str, Any],
    closure_proxy: dict[str, Any],
    hybrid_candidate: dict[str, Any] | None = None,
    *,
    source_projected_final: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repaired = source_projected_final or hybrid_candidate or candidate
    matrix = {
        "original": topology_snapshot(original),
        "stage1_after_removal": topology_snapshot(stage1),
        "source_preserving_candidate": topology_snapshot(candidate),
        "closure_proxy_diagnostic": topology_snapshot(closure_proxy),
        "overlapping_faces": {
            "checked": True,
            "status": "exact_duplicate_faces_removed_in_deterministic_pass",
            "scope": "identical cleaned vertex triplets; near-coincident surface overlap is not inferred as a hole",
        },
        "self_intersection": {
            "checked": None,
            "status": "reported_in_primary_comparisons_after_closed_candidate_gate",
            "blocking": True,
        },
        "normal_consistency": {
            "checked": True,
            "status": "computed",
            "inconsistent_winding_edges": repaired["topology"].get("inconsistent_winding_edges"),
            "passed": repaired["topology"].get("inconsistent_winding_edges") == 0,
            "component_orientation_consistent": repaired["volume"].get("orientation_consistent"),
        },
        "source_loop_hole_inventory": {"checked": True, "status": "reported_in_inventory_before_boundary_regions"},
    }
    if hybrid_candidate:
        matrix["hybrid_fused_candidate"] = topology_snapshot(hybrid_candidate)
    if source_projected_final:
        matrix["source_projected_watertight_candidate"] = topology_snapshot(source_projected_final)
    return matrix


def topology_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    topology = metrics["topology"]
    quality = metrics["quality"]
    return {
        "boundary_edges": topology["boundary_edges"],
        "manifold_edges": topology["manifold_edges"],
        "non_manifold_edges": topology["non_manifold_edges"],
        "non_manifold_vertices": topology["non_manifold_vertices"],
        "inconsistent_winding_edges": topology.get("inconsistent_winding_edges"),
        "components": topology["components"]["count"],
        "degenerate_faces": quality["degenerate_faces"],
        "volume_reliable": metrics["volume"]["reliable"],
    }


def requested_capabilities(
    stage1: dict[str, Any],
    candidate: dict[str, Any],
    closure_proxy: dict[str, Any],
    hybrid_candidate: dict[str, Any] | None = None,
    *,
    source_projected_final: dict[str, Any] | None = None,
    patch_regions: list[dict[str, Any]] | None = None,
    deterministic_passes: list[dict[str, Any]] | None = None,
    accepted_final_geometry: bool = False,
) -> dict[str, Any]:
    final_for_repair = source_projected_final or hybrid_candidate or candidate
    regions = patch_regions or []
    passes = deterministic_passes or []
    duplicate_pass = next(
        (row for row in passes if row.get("name") == "remove_exact_duplicate_source_triangles"),
        None,
    )
    return {
        "part_self_gap_closure": hybrid_repair_status(
            stage1,
            final_for_repair,
            regions,
            {"loop_pair_zipper"},
        ),
        "between_part_gap_closure": hybrid_repair_status(
            stage1,
            final_for_repair,
            regions,
            {"loop_pair_zipper"},
        ),
        "free_edge_repair": before_after(stage1, final_for_repair, "boundary_edges"),
        "micro_hole_filling": hybrid_repair_status(
            stage1,
            final_for_repair,
            regions,
            {"constrained_loop_triangulation"},
        ),
        "large_missing_surface_repair": hybrid_repair_status(
            stage1,
            final_for_repair,
            regions,
            {"proxy_conformal_patch_after_cap_decision"},
        ),
        "overlap_face_repair": {
            "checked": duplicate_pass is not None,
            "scope": "exact duplicate triangles",
            "repaired": bool(duplicate_pass and duplicate_pass.get("status") == "applied"),
            "pass_id": duplicate_pass.get("id") if duplicate_pass else None,
        },
        "shared_edge_repair": before_after(stage1, candidate, "non_manifold_edges"),
        "normal_inconsistency_repair": {
            "checked": True,
            "before": stage1["topology"].get("inconsistent_winding_edges"),
            "after": final_for_repair["topology"].get("inconsistent_winding_edges"),
            "passed": final_for_repair["topology"].get("inconsistent_winding_edges") == 0,
        },
        "surface_leak_check": {
            "checked": True,
            "method": (
                "source-projected closure topology plus diagnostic closure proxy topology"
                if source_projected_final
                else "hybrid candidate topology plus diagnostic closure proxy topology"
            ),
            "source_preserving_candidate_passed": candidate["topology"]["boundary_edges"] == 0
            and candidate["topology"]["non_manifold_edges"] == 0
            and candidate["topology"]["non_manifold_vertices"] == 0,
            "hybrid_fused_candidate_passed": (
                hybrid_candidate["topology"]["boundary_edges"] == 0
                and hybrid_candidate["topology"]["non_manifold_edges"] == 0
                and hybrid_candidate["topology"]["non_manifold_vertices"] == 0
                if hybrid_candidate
                else None
            ),
            "source_projected_watertight_candidate_passed": (
                source_projected_final["topology"]["boundary_edges"] == 0
                and source_projected_final["topology"]["non_manifold_edges"] == 0
                and source_projected_final["topology"]["non_manifold_vertices"] == 0
                if source_projected_final
                else None
            ),
            "closure_proxy_passed": closure_proxy["topology"]["boundary_edges"] == 0
            and closure_proxy["topology"]["non_manifold_edges"] == 0
            and closure_proxy["topology"]["non_manifold_vertices"] == 0,
            "accepted_final_geometry": accepted_final_geometry,
        },
    }


def hybrid_fusion_summary(patch_regions: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_operator: dict[str, int] = {}
    for region in patch_regions:
        status = str(region.get("selection_status"))
        by_status[status] = by_status.get(status, 0) + 1
        operator = str(region.get("operator"))
        by_operator[operator] = by_operator.get(operator, 0) + 1
    return {
        "method": "independent_conformal_patch_transactions",
        "patch_region_count": len(patch_regions),
        "by_selection_status": by_status,
        "by_operator": by_operator,
        "accepted_proxy_patch_regions": by_status.get("use_proxy_patch", 0),
        "accepted_zipper_regions": by_status.get("use_source_zipper", 0),
        "accepted_hole_fill_regions": by_status.get("use_hole_fill", 0),
    }


def hybrid_repair_status(
    stage1: dict[str, Any],
    final: dict[str, Any],
    patch_regions: list[dict[str, Any]],
    operators: set[str],
) -> dict[str, Any]:
    regions = [row for row in patch_regions if row.get("operator") in operators]
    committed_statuses = {"use_proxy_patch", "use_source_zipper", "use_hole_fill"}
    committed = [row for row in regions if row.get("selection_status") in committed_statuses]
    return {
        "checked": True,
        "checked_scope": "per_classified_boundary_region_transaction",
        "classified_individually": bool(regions),
        "classification_status": (
            "reported_in_inventory_and_patch_regions"
            if regions
            else "not_individually_classified"
        ),
        "repaired": bool(committed),
        "method": sorted(operators),
        "region_ids": [row.get("id") for row in regions],
        "committed_region_ids": [row.get("id") for row in committed],
        "policy_or_patch_decisions": [
            {
                "id": row.get("id"),
                "selection_status": row.get("selection_status"),
                "rejection_reason": row.get("rejection_reason"),
            }
            for row in regions
        ],
        "provenance": "face_origin + source_triangle_index + fusion_region_id",
        "topology_boundary_edges_improved": final["topology"]["boundary_edges"] < stage1["topology"]["boundary_edges"],
        "before_boundary_edges": stage1["topology"]["boundary_edges"],
        "after_boundary_edges": final["topology"]["boundary_edges"],
    }


def not_repaired_status(stage1: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "checked": True,
        "classified_individually": False,
        "repaired": False,
        "method": "source-preserving deterministic cleanup only",
        "before_boundary_edges": stage1["topology"]["boundary_edges"],
        "after_boundary_edges": candidate["topology"]["boundary_edges"],
    }


def before_after(stage1: dict[str, Any], stage2: dict[str, Any], field: str) -> dict[str, Any]:
    return {
        "before": stage1["topology"][field],
        "after": stage2["topology"][field],
        "improved": stage2["topology"][field] <= stage1["topology"][field],
    }
