from __future__ import annotations

from typing import Any


def two_stage_repair_report(
    stage_reports: dict[str, dict[str, Any]],
    extraction: dict[str, Any],
    group_filter: dict[str, Any] | None,
    remesh: dict[str, Any],
    deterministic_passes: list[dict[str, Any]],
    decision: dict[str, Any] | None = None,
    source_projected_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original = stage_reports["original_input"]
    stage1 = stage_reports["stage1_exterior_candidate"]
    candidate = stage_reports["deterministic_repair_candidate"]
    proxy = stage_reports["closure_proxy"]
    projected = source_projected_candidate or stage_reports.get(
        "source_projected_watertight_candidate"
    )
    accepted = bool(decision and decision.get("status") == "accepted")
    group_removed = int(group_filter.get("removed_triangles", 0)) if group_filter else 0
    total_removed = int(extraction["removed_triangles"])
    trace = [
        trace_row("original_input", original, "loaded triangle surface from source geometry"),
        {
            "stage": "stage0_group_filter",
            "operation": "diagnose named non-target groups without changing mesh geometry",
            "removed_triangles": group_removed,
            "output": group_filter.get("output_vtp") if group_filter else None,
            "status": "diagnostic_only" if group_filter else "not_provided",
        },
        trace_row("stage1_exterior_candidate", stage1, "extract trusted exterior source triangles"),
        trace_row(
            "deterministic_repair_candidate",
            candidate,
            "remove degenerate and exact-duplicate source triangles without moving vertices",
        ),
        trace_row(
            "closure_proxy",
            proxy,
            "build sealed-exterior TSDF and extract immutable closure connectivity",
        ),
    ]
    if projected:
        trace.append(
            trace_row(
                "source_projected_watertight_candidate",
                projected,
                "project safe closure vertices to source geometry and retain proxy fallbacks",
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
                "largest_removed_groups": (
                    group_filter.get("largest_removed_groups", []) if group_filter else []
                ),
            },
            "deterministic_source_preserving_passes": deterministic_passes,
            "closure_proxy": {
                "method": "sealed_exterior_tsdf_plus_marching_cubes",
                "pitch": remesh.get("pitch"),
                "accepted_final_geometry": False,
                "role": "topology_and_fallback_positions",
            },
            "source_projected_watertight_shell": {
                "produced": projected is not None,
                "method": "certified_proxy_topology_plus_gated_source_projection",
                "accepted_final_geometry": bool(projected and accepted),
            },
        },
        "defect_matrix": defect_matrix(
            original,
            stage1,
            candidate,
            proxy,
            source_projected_final=projected,
        ),
        "requested_capabilities": requested_capabilities(
            stage1,
            candidate,
            proxy,
            source_projected_final=projected,
            deterministic_passes=deterministic_passes,
            accepted_final_geometry=accepted,
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
    *,
    source_projected_final: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final = source_projected_final or candidate
    matrix = {
        "original": topology_snapshot(original),
        "stage1_after_removal": topology_snapshot(stage1),
        "source_preserving_candidate": topology_snapshot(candidate),
        "closure_proxy_diagnostic": topology_snapshot(closure_proxy),
        "overlapping_faces": {
            "checked": True,
            "status": "exact_duplicate_faces_removed_in_deterministic_pass",
        },
        "self_intersection": {
            "checked": None,
            "status": "reported_by_libigl_cgal_in_primary_comparisons",
            "blocking": True,
        },
        "normal_consistency": {
            "checked": True,
            "inconsistent_winding_edges": final["topology"].get("inconsistent_winding_edges"),
            "passed": final["topology"].get("inconsistent_winding_edges") == 0,
            "component_orientation_consistent": final["volume"].get("orientation_consistent"),
        },
    }
    if source_projected_final:
        matrix["source_projected_watertight_candidate"] = topology_snapshot(
            source_projected_final
        )
    return matrix


def topology_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    topology = metrics["topology"]
    return {
        "boundary_edges": topology["boundary_edges"],
        "manifold_edges": topology["manifold_edges"],
        "non_manifold_edges": topology["non_manifold_edges"],
        "non_manifold_vertices": topology["non_manifold_vertices"],
        "inconsistent_winding_edges": topology.get("inconsistent_winding_edges"),
        "components": topology["components"]["count"],
        "degenerate_faces": metrics["quality"]["degenerate_faces"],
        "volume_reliable": metrics["volume"]["reliable"],
    }


def requested_capabilities(
    stage1: dict[str, Any],
    candidate: dict[str, Any],
    closure_proxy: dict[str, Any],
    *,
    source_projected_final: dict[str, Any] | None = None,
    deterministic_passes: list[dict[str, Any]] | None = None,
    accepted_final_geometry: bool = False,
) -> dict[str, Any]:
    final = source_projected_final or candidate
    passes = deterministic_passes or []
    duplicate_pass = next(
        (row for row in passes if row.get("name") == "remove_exact_duplicate_source_triangles"),
        None,
    )
    return {
        "free_edge_repair": before_after(stage1, final, "boundary_edges"),
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
            "after": final["topology"].get("inconsistent_winding_edges"),
            "passed": final["topology"].get("inconsistent_winding_edges") == 0,
        },
        "surface_leak_check": {
            "checked": True,
            "method": "sealed TSDF proxy plus source-projected closure topology",
            "source_projected_watertight_candidate_passed": (
                source_projected_final["topology"]["boundary_edges"] == 0
                and source_projected_final["topology"]["non_manifold_edges"] == 0
                and source_projected_final["topology"]["non_manifold_vertices"] == 0
                if source_projected_final
                else None
            ),
            "closure_proxy_passed": (
                closure_proxy["topology"]["boundary_edges"] == 0
                and closure_proxy["topology"]["non_manifold_edges"] == 0
                and closure_proxy["topology"]["non_manifold_vertices"] == 0
            ),
            "accepted_final_geometry": accepted_final_geometry,
        },
    }


def before_after(stage1: dict[str, Any], stage2: dict[str, Any], field: str) -> dict[str, Any]:
    return {
        "before": stage1["topology"][field],
        "after": stage2["topology"][field],
        "improved": stage2["topology"][field] <= stage1["topology"][field],
    }
