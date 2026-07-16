from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from html_report import write_html_report
from mesh_io import grid_shape, read_surface, triangle_faces
from watertight_issue_markers import (
    ISSUE_BOUNDARY,
    ISSUE_INCONSISTENT_WINDING,
    ISSUE_NON_MANIFOLD,
    ISSUE_NON_MANIFOLD_VERTEX,
    analyze_geometry,
)


VIEW_SPECS = (
    ("projection_xz", (0, 2)),
    ("projection_yz", (1, 2)),
    ("projection_xy", (0, 1)),
)
ISSUE_COLORS = {
    ISSUE_BOUNDARY: (255, 151, 20),
    ISSUE_NON_MANIFOLD: (235, 35, 207),
    ISSUE_INCONSISTENT_WINDING: (0, 166, 255),
    ISSUE_NON_MANIFOLD_VERTEX: (220, 24, 75),
}


def load_json(path: Path | None) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path else {}


def issue_artifacts(stage: str, audit: dict[str, Any]) -> list[dict[str, Any]]:
    summary = audit["summary"]
    paths = audit["artifacts"]
    return [
        artifact(stage, "annotated_surface", summary["triangles"], None, "triangle_surface_cell_arrays", paths["annotated_surface_vtp"]),
        artifact(
            stage,
            "all_watertight_issue_edges",
            summary["boundary_edges"] + summary["non_manifold_edges"] + summary["inconsistent_winding_edges"],
            summary["boundary_regions"] + summary["non_manifold_regions"] + summary["inconsistent_winding_regions"],
            "line_cells",
            paths["all_issue_edges_vtp"],
        ),
        artifact(stage, "boundary_edges", summary["boundary_edges"], summary["boundary_regions"], "line_cells", paths["boundary_edges_vtp"]),
        artifact(
            stage,
            "non_manifold_edges",
            summary["non_manifold_edges"],
            summary["non_manifold_regions"],
            "line_cells",
            paths["non_manifold_edges_vtp"],
        ),
        artifact(
            stage,
            "non_manifold_vertices",
            summary["non_manifold_vertices"],
            None,
            "vertex_cells",
            paths["non_manifold_vertices_vtp"],
        ),
        artifact(
            stage,
            "inconsistent_winding_edges",
            summary["inconsistent_winding_edges"],
            summary["inconsistent_winding_regions"],
            "line_cells",
            paths["inconsistent_winding_edges_vtp"],
        ),
        artifact(stage, "issue_adjacent_faces", summary["affected_faces"], None, "triangle_surface", paths["issue_faces_vtp"]),
    ]


def artifact(
    stage: str,
    kind: str,
    count: int,
    regions: int | None,
    representation: str,
    path: str,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "kind": kind,
        "count": int(count),
        "regions": regions,
        "representation": representation,
        "path": path,
        "blocking": kind in {
            "boundary_edges",
            "non_manifold_edges",
            "non_manifold_vertices",
            "inconsistent_winding_edges",
        } and int(count) > 0,
    }


def write_issue_overlays(
    input_path: Path,
    role: str,
    output_dir: Path,
    *,
    size: int,
) -> list[str]:
    mesh = read_surface(input_path)
    points = np.asarray(mesh.points, dtype=np.float64)
    faces = triangle_faces(mesh)
    analysis = analyze_geometry(points, faces)
    visual_dir = output_dir / "visual"
    visual_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for view_name, axes in VIEW_SPECS:
        image = projected_issue_overlay(
            points,
            faces,
            analysis["edges"],
            axes,
            max_size=size,
            non_manifold_vertices=analysis["non_manifold_vertex_ids"],
        )
        path = visual_dir / f"{role}_{view_name}_watertight_issues.png"
        image.save(path)
        paths.append(str(path.resolve()))
    return paths


def projected_issue_overlay(
    points: np.ndarray,
    faces: np.ndarray,
    typed_edges: dict[int, list[tuple[int, int]]],
    axes: tuple[int, int],
    *,
    max_size: int,
    non_manifold_vertices: np.ndarray | None = None,
) -> Image.Image:
    projected_points = points[:, axes]
    rows, cols, mins, spans = grid_shape(projected_points, max_size)
    centroids = points[faces].mean(axis=1)[:, axes]
    uv = np.clip((centroids - mins) / spans, 0.0, 1.0)
    col = np.minimum((uv[:, 0] * (cols - 1)).astype(np.int64), cols - 1)
    row = rows - 1 - np.minimum((uv[:, 1] * (rows - 1)).astype(np.int64), rows - 1)
    occupancy = np.zeros((rows, cols), dtype=bool)
    occupancy[row, col] = True
    occupancy = binary_dilation(occupancy, iterations=1)
    image_array = np.full((rows, cols, 3), 250, dtype=np.uint8)
    image_array[occupancy] = (190, 201, 209)
    image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image)

    def pixel(vertex_id: int) -> tuple[int, int]:
        value = np.clip((projected_points[vertex_id] - mins) / spans, 0.0, 1.0)
        return (
            int(round(value[0] * (cols - 1))),
            rows - 1 - int(round(value[1] * (rows - 1))),
        )

    for issue_type in (ISSUE_BOUNDARY, ISSUE_NON_MANIFOLD, ISSUE_INCONSISTENT_WINDING):
        color = ISSUE_COLORS[issue_type]
        width = 1 if issue_type == ISSUE_BOUNDARY else 2
        for left, right in typed_edges[issue_type]:
            draw.line((pixel(left), pixel(right)), fill=color, width=width)
    for vertex_id in np.asarray(
        non_manifold_vertices if non_manifold_vertices is not None else [],
        dtype=np.int64,
    ):
        x, y = pixel(int(vertex_id))
        radius = 3
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=ISSUE_COLORS[ISSUE_NON_MANIFOLD_VERTEX],
        )
    return image


def gate(actual: Any, expected: Any, passed: bool) -> dict[str, Any]:
    return {
        "required": True,
        "actual": actual,
        "expected": expected,
        "passed": bool(passed),
    }


def build_comparison_report(
    marker_report: dict[str, Any],
    marker_report_path: Path,
    two_stage: dict[str, Any],
    two_stage_report_path: Path | None,
    previews: list[str],
) -> dict[str, Any]:
    original = marker_report["original"]
    processed = marker_report["processed"]
    original_summary = original["summary"]
    processed_summary = processed["summary"]
    decision = two_stage.get("decision") or {
        "status": "accepted" if processed_summary["engineering_watertight"] else "rejected",
        "reason_codes": [] if processed_summary["engineering_watertight"] else ["watertight_topology_failed"],
        "final_output_path": None,
    }
    self_intersection = two_stage.get("comparisons", {}).get(
        "self_intersection",
        {
            "status": "not_checked",
            "passed": False,
            "failure_reason": "no source repair report was supplied",
        },
    )
    artifacts = [
        *issue_artifacts("original", original),
        *issue_artifacts("processed", processed),
    ]
    outputs = {
        "original_mesh_vtp": original["artifacts"]["annotated_surface_vtp"],
        "processed_mesh_vtp": processed["artifacts"]["annotated_surface_vtp"],
        "original_issue_faces_vtp": original["artifacts"]["issue_faces_vtp"],
        "processed_issue_faces_vtp": processed["artifacts"]["issue_faces_vtp"],
        "watertight_issue_artifacts": artifacts,
        "marker_report_json": str(marker_report_path.resolve()),
        "source_repair_report_json": str(two_stage_report_path.resolve()) if two_stage_report_path else None,
        "source_repair_candidate_vtp": (
            two_stage.get("outputs", {}).get("source_projected_watertight_candidate_vtp")
        ),
        "accepted_mesh_vtp": two_stage.get("outputs", {}).get("accepted_mesh_vtp"),
        "previews": [
            *previews,
            *two_stage.get("outputs", {}).get("previews", []),
        ],
    }
    gates = {
        "boundary_edges_zero": gate(processed_summary["boundary_edges"], 0, processed_summary["boundary_edges"] == 0),
        "non_manifold_edges_zero": gate(
            processed_summary["non_manifold_edges"],
            0,
            processed_summary["non_manifold_edges"] == 0,
        ),
        "non_manifold_vertices_zero": gate(
            processed_summary["non_manifold_vertices"],
            0,
            processed_summary["non_manifold_vertices"] == 0,
        ),
        "inconsistent_winding_edges_zero": gate(
            processed_summary["inconsistent_winding_edges"],
            0,
            processed_summary["inconsistent_winding_edges"] == 0,
        ),
        "degenerate_faces_zero": gate(
            processed_summary["degenerate_faces"],
            0,
            processed_summary["degenerate_faces"] == 0,
        ),
        "closed_manifold": gate(processed_summary["closed_manifold"], True, processed_summary["closed_manifold"]),
        "source_repair_acceptance": gate(decision.get("status"), "accepted", decision.get("status") == "accepted"),
    }
    unresolved_policy = two_stage.get("unresolved_policy_packet", {})
    opening_policy = two_stage.get("target", {}).get(
        "opening_policy",
        "review_each_opening",
    )
    return {
        "decision": decision,
        "input": {
            "kind": "mesh",
            "path": original["input_path"],
            "role": "original full-resolution mesh",
        },
        "input_truth": {
            "primary_geometry_path": original["input_path"],
            "processed_geometry_path": processed["input_path"],
            "processed_role": "processed mesh under audit",
            "marker_geometry_normalization": "surface_extract_triangulate_clean",
            "source_units": "unknown; STL carries no unit metadata and coordinates are unchanged",
            "preview_axis_assumption": "stored X/Y/Z axes; no semantic axis roles inferred",
        },
        "target": {
            "name": "watertight-exterior-shell",
            "output_goal": "watertight_mesh",
            "opening_policy": opening_policy,
        },
        "output_contract": {
            "input_kind": "mesh",
            "output_kind": "diagnostic_mesh_report",
            "repair_domain": "mesh_domain",
            "annotated_surface_arrays_are_diagnostic": True,
            "line_vtp_representation": "exact topological issue edges",
            "processed_is_accepted": decision.get("status") == "accepted",
            "html_3d_meshes_full_resolution": True,
        },
        "method": "full-resolution original-versus-processed watertight issue VTP audit",
        "parameters": {
            "issue_type_codes": marker_report["issue_type_codes"],
            "array_contract": marker_report["array_contract"],
            "html_surface_preview_triangle_limit": None,
            "html_surface_preview_resolution": "full",
            "html_surface_preview_decimated": False,
            "vtp_geometry_decimated": False,
            "preview_projections": {
                "projection_xz": "XZ",
                "projection_yz": "YZ",
                "projection_xy": "XY",
            },
        },
        "limitations": [
            "A boundary edge is a topological free edge, not proof that the region is a true exterior-shell opening.",
            "Non-manifold detection uses exact shared-edge incidence and does not detect near-overlapping faces.",
            "Stage-local component IDs are not stable across original and processed meshes; use source_triangle_index where available.",
            "Closed internal components may be target-invalid even though they are not leaks.",
            "The issue-face VTP is a visualization subset; line-cell VTP files are the exact edge markers.",
            f"Self-intersection status: {self_intersection.get('status')}; it is not treated as passed unless computed.",
        ],
        "checks": {
            **marker_report["checks"],
            "self_intersections": self_intersection.get("status", "not_checked"),
        },
        "comparisons": {
            **marker_report["comparison"],
            "self_intersection": self_intersection,
        },
        "gates": gates,
        "outputs": outputs,
        "unhandled_items": [
            {
                "item": "semantic_opening_policy",
                "count": unresolved_policy.get("item_count_total", 0),
                "blocking": unresolved_policy.get("item_count_total", 0) > 0,
            },
            {
                "item": "self_intersection",
                "status": self_intersection.get("status"),
                "blocking": self_intersection.get("status") != "computed",
            },
        ],
        "repair_report": {
            "geometry_to_mesh_trace": [
                trace_row("original_mesh", original_summary, "audit original assembly topology", original["input_path"]),
                trace_row(
                    "processed_candidate",
                    processed_summary,
                    "audit current-run source-projected watertight exterior shell",
                    processed["input_path"],
                ),
            ],
            "change_summary": {
                "watertight_issue_delta": marker_report["comparison"],
                "accepted_final_geometry": decision.get("status") == "accepted",
            },
            "defect_matrix": {
                "original": original_summary,
                "processed": processed_summary,
                "self_intersection": self_intersection,
            },
            "requested_capabilities": {
                "original_and_processed_mesh_visible": True,
                "boundary_issue_vtp": True,
                "non_manifold_issue_vtp": True,
                "winding_issue_vtp": True,
                "degenerate_issue_faces_vtp": True,
                "full_resolution_vtp": True,
                "processed_engineering_watertight": processed_summary["engineering_watertight"],
            },
        },
    }


def trace_row(stage: str, summary: dict[str, Any], operation: str, output: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "operation": operation,
        "status": "computed",
        "triangles": summary["triangles"],
        "points": summary["points"],
        "boundary_edges": summary["boundary_edges"],
        "non_manifold_edges": summary["non_manifold_edges"],
        "components": summary["connected_components"],
        "output": output,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a single-file HTML report for original/processed watertight issue VTP artifacts."
    )
    parser.add_argument("marker_report_json", type=Path)
    parser.add_argument("--two-stage-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview-size", type=int, default=1000)
    return parser


def comparison_json_path(output_path: Path, marker_report_path: Path) -> Path:
    candidate = output_path.with_suffix(".json")
    if candidate.resolve() == marker_report_path.resolve():
        return output_path.with_name(f"{output_path.stem}_comparison.json")
    return candidate


def main() -> None:
    args = build_parser().parse_args()
    marker_report = load_json(args.marker_report_json)
    two_stage = load_json(args.two_stage_report)
    output_path = args.output or args.marker_report_json.with_suffix(".html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_path = Path(marker_report["original"]["input_path"])
    processed_path = Path(marker_report["processed"]["input_path"])
    previews = [
        *write_issue_overlays(original_path, "original", output_path.parent, size=args.preview_size),
        *write_issue_overlays(processed_path, "processed", output_path.parent, size=args.preview_size),
    ]
    report = build_comparison_report(
        marker_report,
        args.marker_report_json,
        two_stage,
        args.two_stage_report,
        previews,
    )
    report_path = comparison_json_path(output_path, args.marker_report_json)
    report["outputs"]["comparison_report_json"] = str(report_path.resolve())
    report["outputs"]["comparison_report_html"] = str(output_path.resolve())
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.environ["CAD_SURFACE_MESHER_VIEWER_TRIANGLES"] = "0"
    write_html_report(report, output_path, "Watertight Exterior Shell Issue Comparison")
    print(
        json.dumps(
            {
                "html": str(output_path.resolve()),
                "json": str(report_path.resolve()),
                "decision": report["decision"],
                "processed": marker_report["processed"]["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
