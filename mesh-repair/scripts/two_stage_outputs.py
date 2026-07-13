from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mesh_io import save_depth_preview, write_vtp


def write_visibility_labels(
    args: Any,
    points: np.ndarray,
    faces: np.ndarray,
    source_indices: np.ndarray,
    *,
    visible_score: np.ndarray,
    component_visible_hard_keep: np.ndarray,
    stage1_selected: np.ndarray,
    outside_exposed: np.ndarray,
) -> Path:
    """Persist full source geometry with face-level visibility and filtering labels."""
    path = args.output_dir / "visibility_labeled_source.vtp"
    direct_visible = np.asarray(visible_score) > 0
    component_keep = np.asarray(component_visible_hard_keep, dtype=bool)
    selected = np.asarray(stage1_selected, dtype=bool)
    write_vtp(
        path,
        points,
        faces,
        {
            "source_triangle_index": np.asarray(source_indices, dtype=np.int64),
            "visible_view_count": np.asarray(visible_score, dtype=np.int16),
            "direct_first_hit_visible": direct_visible.astype(np.int8),
            "component_visible_hard_keep": component_keep.astype(np.int8),
            "fully_unseen_component_candidate": (~component_keep).astype(np.int8),
            "stage1_selected": selected.astype(np.int8),
            "outside_flood_exposed": np.asarray(outside_exposed, dtype=np.int8),
        },
    )
    return path


def write_stage1(
    args: Any,
    points: np.ndarray,
    faces: np.ndarray,
    source_indices: np.ndarray,
    visible_score: np.ndarray,
    local_indices: np.ndarray,
    *,
    outside_exposed: np.ndarray | None = None,
) -> Path:
    path = args.output_dir / "stage1_exterior_candidate.vtp"
    write_vtp(
        path,
        points,
        faces,
        {
            "source_triangle_index": source_indices.astype(np.int64),
            "visible_view_count": visible_score[local_indices].astype(np.int16),
            **(
                {"outside_flood_exposed": np.asarray(outside_exposed[local_indices], dtype=np.int8)}
                if outside_exposed is not None
                else {}
            ),
        },
    )
    return path


def write_source_preserving_candidate(
    args: Any,
    points: np.ndarray,
    faces: np.ndarray,
    source_indices: np.ndarray,
    *,
    exterior_face_score: np.ndarray | None = None,
    sealed_exterior_face_mask: np.ndarray | None = None,
    face_external_directions: np.ndarray | None = None,
) -> Path:
    path = args.output_dir / "source_preserving_candidate.vtp"
    cell_data = {"source_triangle_index": source_indices.astype(np.int64)}
    if exterior_face_score is not None:
        cell_data["exterior_view_count"] = np.asarray(exterior_face_score, dtype=np.int16)
    if sealed_exterior_face_mask is not None:
        cell_data["sealed_exterior_support"] = np.asarray(
            sealed_exterior_face_mask,
            dtype=np.int8,
        )
    if face_external_directions is not None:
        directions = np.asarray(face_external_directions, dtype=np.float64)
        if directions.shape != (faces.shape[0], 3):
            raise ValueError("face_external_directions must have shape (face_count, 3)")
        cell_data["external_direction"] = directions
    write_vtp(path, points, faces, cell_data)
    return path


def write_source_projected_watertight_candidate(
    args: Any,
    points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, dict[str, np.ndarray]],
) -> Path:
    path = args.output_dir / "source_projected_watertight_candidate.vtp"
    write_vtp(
        path,
        points,
        faces,
        provenance.get("cell_data"),
        point_data=provenance.get("point_data"),
    )
    return path


def write_previews(
    args: Any,
    source_points: np.ndarray,
    source_faces: np.ndarray,
    proxy_points: np.ndarray,
    proxy_faces: np.ndarray,
    hybrid_points: np.ndarray,
    hybrid_faces: np.ndarray,
    views: list[Any],
    *,
    projected_points: np.ndarray | None = None,
    projected_faces: np.ndarray | None = None,
) -> list[str]:
    if args.skip_previews:
        return []
    visual_dir = args.output_dir / "visual"
    previews = save_depth_preview(visual_dir, "source_candidate", source_points, source_faces, views, size=args.preview_size)
    if projected_points is not None and projected_faces is not None:
        previews.extend(
            save_depth_preview(
                visual_dir,
                "source_projected_watertight_candidate",
                projected_points,
                projected_faces,
                views,
                size=args.preview_size,
            )
        )
    previews.extend(save_depth_preview(visual_dir, "hybrid_fused_candidate", hybrid_points, hybrid_faces, views, size=args.preview_size))
    previews.extend(save_depth_preview(visual_dir, "closure_proxy", proxy_points, proxy_faces, views, size=args.preview_size))
    return previews


def outputs_report(
    args: Any,
    visibility_labels_path: Path,
    stage1_path: Path,
    candidate_path: Path,
    closure_proxy_path: Path,
    policy_packet_path: Path,
    previews: list[str],
    hybrid_result: dict[str, Any],
    projected_candidate_path: Path,
    projection_report_path: Path,
) -> dict[str, Any]:
    hybrid_candidate_path = require_hybrid_candidate(hybrid_result)
    patches_dir = args.output_dir / "patches"
    debug_dir = args.output_dir / "debug"
    return {
        "visibility_labeled_source_vtp": str(visibility_labels_path),
        "stage1_exterior_candidate_vtp": str(stage1_path),
        "source_preserving_candidate_vtp": str(candidate_path),
        "source_projected_watertight_candidate_vtp": str(projected_candidate_path),
        "source_projected_watertight_candidate_produced": projected_candidate_path.exists(),
        "source_projection_report_json": str(projection_report_path),
        "hybrid_fused_candidate_vtp": str(hybrid_candidate_path),
        "hybrid_fused_candidate_produced": hybrid_candidate_path.exists(),
        "accepted_mesh_vtp": None,
        "accepted_mesh_available": False,
        "rejected_candidate_vtp": None,
        "source_fallback_vtp": None,
        "mesh_result": {"status": "acceptance_pending", "path": None, "accepted": False},
        "closure_proxy_vtp": str(closure_proxy_path),
        "implicit_field_npz": str(args.output_dir / "implicit_field.npz"),
        "patches_dir": str(patches_dir),
        "debug_dir": str(debug_dir),
        "visual_dir": str(args.output_dir / "visual"),
        "debug_region_patch_graph_json": str(debug_dir / "region_patch_graph.json"),
        "debug_hybrid_fusion_trace_json": str(debug_dir / "hybrid_fusion_trace.json"),
        "ai_policy_packet_json": str(policy_packet_path),
        "report_json": str(args.output_dir / "two_stage_report.json"),
        "html_report": str(args.output_dir / "two_stage_report.html"),
        "previews": previews,
    }


def require_hybrid_candidate(hybrid_result: dict[str, Any]) -> Path:
    raw_path = hybrid_result.get("hybrid_fused_candidate_vtp")
    if not raw_path:
        raise KeyError("hybrid fusion result did not provide a candidate mesh path")
    candidate_path = Path(raw_path)
    if candidate_path.name != "hybrid_fused_candidate.vtp":
        raise ValueError("ungated hybrid geometry must use the candidate filename")
    if not candidate_path.exists():
        raise FileNotFoundError(f"hybrid fusion candidate is missing: {candidate_path}")
    hybrid_result["hybrid_fused_candidate_produced"] = candidate_path.exists()
    return candidate_path


def extraction_report(
    view_reports: list[dict[str, Any]],
    visible_score: np.ndarray,
    stage1_indices: np.ndarray,
    original_faces: np.ndarray,
) -> dict[str, Any]:
    return {
        "method": "six_view_triangle_raster_first_hit_hard_keep",
        "view_reports": view_reports,
        "visible_before_dilation_triangles": int(np.count_nonzero(visible_score > 0)),
        "selected_after_dilation_triangles": int(stage1_indices.size),
        "removed_triangles": int(original_faces.shape[0] - stage1_indices.size),
        "removed_triangle_ratio": float(1.0 - stage1_indices.size / original_faces.shape[0]),
    }


def print_summary(
    report_path: Path,
    stage1_report: dict[str, Any],
    candidate_report: dict[str, Any],
    closure_proxy_report: dict[str, Any],
    report: dict[str, Any],
    projected_report: dict[str, Any] | None = None,
) -> None:
    print(json.dumps({
        "report": str(report_path),
        "html_report": report["outputs"].get("html_report"),
        "decision": report["decision"],
        "stage1_triangles": stage1_report["triangles"],
        "source_preserving_candidate_triangles": candidate_report["triangles"],
        "candidate_boundary_edges": candidate_report["topology"]["boundary_edges"],
        "candidate_non_manifold_edges": candidate_report["topology"]["non_manifold_edges"],
        "candidate_non_manifold_vertices": candidate_report["topology"]["non_manifold_vertices"],
        "hybrid_fused_candidate_vtp": report["outputs"].get("hybrid_fused_candidate_vtp"),
        "hybrid_fused_candidate_produced": report["outputs"].get("hybrid_fused_candidate_produced"),
        "source_projected_watertight_candidate_vtp": report["outputs"].get(
            "source_projected_watertight_candidate_vtp"
        ),
        "source_projected_boundary_edges": (
            projected_report["topology"]["boundary_edges"] if projected_report else None
        ),
        "source_projected_non_manifold_edges": (
            projected_report["topology"]["non_manifold_edges"] if projected_report else None
        ),
        "mesh_result": report["outputs"].get("mesh_result"),
        "closure_proxy_triangles": closure_proxy_report["triangles"],
        "closure_proxy_boundary_edges": closure_proxy_report["topology"]["boundary_edges"],
        "closure_proxy_non_manifold_edges": closure_proxy_report["topology"]["non_manifold_edges"],
        "closure_proxy_non_manifold_vertices": closure_proxy_report["topology"]["non_manifold_vertices"],
    }, indent=2))
