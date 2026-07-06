from __future__ import annotations

from typing import Any


def two_stage_repair_report(
    stage_reports: dict[str, dict[str, Any]],
    extraction: dict[str, Any],
    group_filter: dict[str, Any] | None,
    remesh: dict[str, Any],
) -> dict[str, Any]:
    original = stage_reports["original"]
    stage1 = stage_reports["stage1_exterior_candidate"]
    stage2 = stage_reports["stage2_watertight_remesh"]
    group_removed = int(group_filter.get("removed_triangles", 0)) if group_filter else 0
    total_removed = int(extraction["removed_triangles"])
    return {
        "geometry_to_mesh_trace": [
            trace_row("original_input", original, "loaded triangle surface from source geometry"),
            {
                "stage": "stage0_group_filter",
                "operation": "remove named non-target groups before geometry repair",
                "removed_triangles": group_removed,
                "output": group_filter.get("output_vtp") if group_filter else None,
                "status": "applied" if group_filter else "not_provided",
            },
            trace_row("stage1_exterior_candidate", stage1, "outer-wall extraction from source triangles"),
            trace_row("stage2_watertight_remesh", stage2, "voxel fill and marching-cubes remesh"),
        ],
        "change_summary": {
            "removed": {
                "group_name_removed_triangles": group_removed,
                "visibility_removed_triangles": max(0, total_removed - group_removed),
                "total_removed_triangles": total_removed,
                "removed_triangle_ratio": extraction["removed_triangle_ratio"],
                "largest_removed_groups": group_filter.get("largest_removed_groups", []) if group_filter else [],
            },
            "sealed_or_filled": [
                {
                    "scope": "free edges, small holes, unresolved gaps, and openings present in Stage 1",
                    "method": "voxel_fill_plus_marching_cubes",
                    "pitch": remesh.get("pitch"),
                    "classification": "implicit_global_closure_not_per_gap",
                }
            ],
            "not_individually_classified": [
                "per-part self gaps",
                "between-part gaps",
                "micro holes by source loop id",
                "functional openings",
            ],
        },
        "defect_matrix": defect_matrix(original, stage1, stage2),
        "requested_capabilities": requested_capabilities(stage1, stage2),
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
        "components": metrics["topology"]["components"]["count"],
        "degenerate_faces": metrics["quality"]["degenerate_faces"],
    }


def defect_matrix(
    original: dict[str, Any],
    stage1: dict[str, Any],
    stage2: dict[str, Any],
) -> dict[str, Any]:
    return {
        "original": topology_snapshot(original),
        "stage1_after_removal": topology_snapshot(stage1),
        "stage2_after_watertight_remesh": topology_snapshot(stage2),
        "overlapping_faces": {"checked": False, "status": "not_implemented"},
        "normal_consistency": {"checked": False, "status": "not_implemented"},
        "source_loop_hole_inventory": {"checked": False, "status": "implicit_voxel_closure_only"},
    }


def topology_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    topology = metrics["topology"]
    quality = metrics["quality"]
    return {
        "boundary_edges": topology["boundary_edges"],
        "manifold_edges": topology["manifold_edges"],
        "non_manifold_edges": topology["non_manifold_edges"],
        "components": topology["components"]["count"],
        "degenerate_faces": quality["degenerate_faces"],
        "volume_reliable": metrics["volume"]["reliable"],
    }


def requested_capabilities(stage1: dict[str, Any], stage2: dict[str, Any]) -> dict[str, Any]:
    return {
        "part_self_gap_closure": implicit_status(stage1, stage2),
        "between_part_gap_closure": implicit_status(stage1, stage2),
        "free_edge_repair": before_after(stage1, stage2, "boundary_edges"),
        "micro_hole_filling": implicit_status(stage1, stage2),
        "overlap_face_repair": {"checked": False, "repaired": False, "reason": "overlap detector not implemented"},
        "shared_edge_repair": before_after(stage1, stage2, "non_manifold_edges"),
        "normal_inconsistency_repair": {"checked": False, "repaired": False, "reason": "normal audit not implemented"},
        "front_bumper_cas_offset": {
            "checked": False,
            "applied": False,
            "reason": "requires CAS metadata, target region labels, and offset distance policy",
        },
        "surface_leak_check": {
            "checked": True,
            "method": "boundary and non-manifold edge topology",
            "passed": stage2["topology"]["boundary_edges"] == 0
            and stage2["topology"]["non_manifold_edges"] == 0,
        },
    }


def implicit_status(stage1: dict[str, Any], stage2: dict[str, Any]) -> dict[str, Any]:
    return {
        "checked": True,
        "classified_individually": False,
        "method": "implicit voxel closure",
        "before_boundary_edges": stage1["topology"]["boundary_edges"],
        "after_boundary_edges": stage2["topology"]["boundary_edges"],
    }


def before_after(stage1: dict[str, Any], stage2: dict[str, Any], field: str) -> dict[str, Any]:
    return {
        "before": stage1["topology"][field],
        "after": stage2["topology"][field],
        "improved": stage2["topology"][field] <= stage1["topology"][field],
    }
