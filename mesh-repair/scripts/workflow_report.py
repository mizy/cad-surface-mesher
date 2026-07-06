from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from html_report import write_html_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge mesh repair stage reports into one workflow report.")
    parser.add_argument("--two-stage-report", required=True, type=Path)
    parser.add_argument("--adaptive-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    two_stage = read_json(args.two_stage_report)
    adaptive = read_json(args.adaptive_report)
    report = build_report(two_stage, adaptive)

    json_path = args.output_dir / "workflow_report.json"
    html_path = args.output_dir / "workflow_report.html"
    report["outputs"]["report_json"] = str(json_path)
    report["outputs"]["html_report"] = str(html_path)
    report["source_reports"] = {
        "two_stage": str(args.two_stage_report),
        "adaptive_refinement": str(args.adaptive_report),
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_html_report(report, html_path, "CAD Surface Mesher Workflow Report")
    print(json.dumps({"report": str(json_path), "html_report": str(html_path)}, indent=2))
    return 0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(two_stage: dict[str, Any], adaptive: dict[str, Any]) -> dict[str, Any]:
    stages = {
        "original_input": two_stage["stages"]["original"],
        "stage1_exterior_candidate": two_stage["stages"]["stage1_exterior_candidate"],
        "adaptive_refined_source_diagnostic": adaptive["refined_metrics"],
        "stage2_watertight_remesh": two_stage["stages"]["stage2_watertight_remesh"],
    }
    return {
        "input": two_stage["input"],
        "target": two_stage.get("target"),
        "output_contract": two_stage["output_contract"],
        "method": "group_visibility_filter_to_exterior_candidate_then_adaptive_diagnostics_and_watertight_remesh",
        "workflow_decision": workflow_decision(),
        "parameters": {
            "two_stage_watertight": two_stage.get("parameters", {}),
            "adaptive_depth_refinement": adaptive.get("parameters", {}),
        },
        "limitations": merged_limitations(two_stage, adaptive),
        "group_filter": two_stage.get("group_filter"),
        "view_reports": adaptive.get("view_reports", []),
        "size_field": adaptive.get("size_field"),
        "refinement_iterations": adaptive.get("refinement_iterations"),
        "comparisons": two_stage.get("comparisons"),
        "stages": stages,
        "repair_report": {
            "geometry_to_mesh_trace": workflow_trace(two_stage, adaptive),
            "change_summary": workflow_changes(two_stage, adaptive),
            "defect_matrix": workflow_defects(two_stage, adaptive),
            "requested_capabilities": workflow_requested(two_stage, adaptive),
        },
        "gates": two_stage.get("gates", {}),
        "outputs": workflow_outputs(two_stage, adaptive),
        "source_reports": {},
    }


def workflow_decision() -> dict[str, Any]:
    return {
        "primary_report": "workflow_report.html",
        "final_output_path": "stage2_watertight_surface_vtp",
        "adaptive_refinement_role": "diagnostic_source_refinement_branch",
        "adaptive_refinement_used_as_final_watertight_input": False,
        "reason": "current prototype validates local depth-driven sizing separately; final closure still uses voxel remesh on Stage 1 exterior candidate",
    }


def merged_limitations(two_stage: dict[str, Any], adaptive: dict[str, Any]) -> list[str]:
    limitations = []
    limitations.extend(two_stage.get("limitations", []))
    limitations.extend(adaptive.get("limitations", []))
    limitations.append("Adaptive refinement is reported in the same workflow but is not yet consumed by final watertight remeshing.")
    return list(dict.fromkeys(limitations))


def workflow_trace(two_stage: dict[str, Any], adaptive: dict[str, Any]) -> list[dict[str, Any]]:
    trace = list(two_stage["repair_report"]["geometry_to_mesh_trace"])
    adaptive_rows = adaptive["change_report"]["geometry_to_mesh_trace"]
    trace.insert(
        max(0, len(trace) - 1),
        {
            **adaptive_rows[-1],
            "stage": "diagnostic_adaptive_refined_source",
            "operation": "depth-map local refinement on source exterior candidate; diagnostic branch only",
            "status": "diagnostic_only_not_final_input",
            "output": adaptive.get("outputs", {}).get("adaptive_refined_source_vtp"),
        },
    )
    return trace


def workflow_changes(two_stage: dict[str, Any], adaptive: dict[str, Any]) -> dict[str, Any]:
    changes = dict(two_stage["repair_report"]["change_summary"])
    changes["refined"] = adaptive["change_report"]["change_summary"]
    changes["workflow_note"] = {
        "adaptive_refinement_consumed_by_final_mesh": False,
        "final_watertight_method": "voxel_fill_plus_marching_cubes_on_stage1_exterior_candidate",
    }
    return changes


def workflow_defects(two_stage: dict[str, Any], adaptive: dict[str, Any]) -> dict[str, Any]:
    defects = dict(two_stage["repair_report"]["defect_matrix"])
    defects["adaptive_refined_source_diagnostic"] = adaptive["change_report"]["defect_matrix"]["adaptive_refined_source"]
    defects["final_stage2_watertight_topology"] = defects.get("stage2_after_watertight_remesh")
    return defects


def workflow_requested(two_stage: dict[str, Any], adaptive: dict[str, Any]) -> dict[str, Any]:
    requested = dict(two_stage["repair_report"].get("requested_capabilities", {}))
    requested["depth_map_local_refinement"] = {
        "checked": True,
        "source": adaptive.get("input"),
        "fine_faces": adaptive.get("size_field", {}).get("fine_faces"),
        "transition_faces": adaptive.get("size_field", {}).get("transition_faces"),
        "used_as_final_watertight_input": False,
    }
    return requested


def workflow_outputs(two_stage: dict[str, Any], adaptive: dict[str, Any]) -> dict[str, Any]:
    two_outputs = two_stage.get("outputs", {})
    adaptive_outputs = adaptive.get("outputs", {})
    return {
        "stage1_exterior_candidate_vtp": two_outputs.get("stage1_exterior_candidate_vtp"),
        "adaptive_refined_source_vtp": adaptive_outputs.get("adaptive_refined_source_vtp"),
        "refinement_field_vtp": adaptive_outputs.get("refinement_field_vtp"),
        "stage2_watertight_surface_vtp": two_outputs.get("stage2_watertight_surface_vtp"),
        "previews": two_outputs.get("previews", []),
    }


if __name__ == "__main__":
    raise SystemExit(main())
