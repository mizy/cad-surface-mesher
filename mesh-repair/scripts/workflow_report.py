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
    stages = dict(two_stage["stages"])
    stages["adaptive_refined_source_diagnostic"] = {
        "path": adaptive.get("outputs", {}).get("adaptive_refined_source_vtp"),
        "source": adaptive.get("input"),
        "metrics": adaptive["refined_metrics"],
        "accepted_final_geometry": False,
    }
    return {
        "decision": two_stage["decision"],
        "input": two_stage["input"],
        "input_truth": two_stage.get("input_truth"),
        "target": two_stage.get("target"),
        "output_contract": two_stage["output_contract"],
        "method": "visibility_source_truth_to_repaired_closure_topology_then_exact_source_projection",
        "workflow_decision": workflow_decision(two_stage),
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
        "inventory_before": two_stage.get("inventory_before", {}),
        "deterministic_passes": two_stage.get("deterministic_passes", []),
        "unresolved_policy_packet": two_stage.get("unresolved_policy_packet", {"items": []}),
        "policy_decisions": two_stage.get("policy_decisions", []),
        "inventory_after": two_stage.get("inventory_after", {}),
        "repair_report": {
            "geometry_to_mesh_trace": workflow_trace(two_stage, adaptive),
            "change_summary": workflow_changes(two_stage, adaptive),
            "defect_matrix": workflow_defects(two_stage, adaptive),
            "requested_capabilities": workflow_requested(two_stage, adaptive),
        },
        "gates": two_stage.get("gates", {}),
        "outputs": workflow_outputs(two_stage, adaptive),
        "ignored_outputs": workflow_ignored_outputs(two_stage, adaptive),
        "unhandled_items": two_stage.get("unhandled_items", []),
        "source_reports": {},
    }


def workflow_decision(two_stage: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_report": "workflow_report.html",
        "decision_status": two_stage.get("decision", {}).get("status"),
        "final_output_path": two_stage.get("decision", {}).get("final_output_path"),
        "closure_proxy_role": "watertight_connectivity_template_only",
        "adaptive_refinement_role": "diagnostic_source_refinement_branch",
        "adaptive_refinement_used_as_final_watertight_input": False,
        "reason": "adaptive refinement is reported for source-detail diagnostics and is not consumed by the accepted final output",
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
        "closure_proxy_method": "voxel_fill_plus_marching_cubes_on_source_preserving_candidate",
        "accepted_final_output": two_stage.get("outputs", {}).get("accepted_mesh_vtp"),
    }
    return changes


def workflow_defects(two_stage: dict[str, Any], adaptive: dict[str, Any]) -> dict[str, Any]:
    defects = dict(two_stage["repair_report"]["defect_matrix"])
    defects["adaptive_refined_source_diagnostic"] = adaptive["change_report"]["defect_matrix"]["adaptive_refined_source"]
    defects["closure_proxy_topology"] = defects.get("closure_proxy_diagnostic")
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
        "source_preserving_candidate_vtp": two_outputs.get("source_preserving_candidate_vtp"),
        "source_projected_watertight_candidate_vtp": two_outputs.get(
            "source_projected_watertight_candidate_vtp"
        ),
        "accepted_mesh_vtp": two_outputs.get("accepted_mesh_vtp"),
        "rejected_candidate_vtp": two_outputs.get("rejected_candidate_vtp"),
        "closure_proxy_vtp": two_outputs.get("closure_proxy_vtp"),
        "adaptive_refined_source_vtp": adaptive_outputs.get("adaptive_refined_source_vtp"),
        "refinement_field_vtp": adaptive_outputs.get("refinement_field_vtp"),
        "ai_policy_packet_json": two_outputs.get("ai_policy_packet_json"),
        "previews": two_outputs.get("previews", []),
    }


def workflow_ignored_outputs(two_stage: dict[str, Any], adaptive: dict[str, Any]) -> list[dict[str, Any]]:
    ignored = list(two_stage.get("ignored_outputs", []))
    adaptive_outputs = adaptive.get("outputs", {})
    for key in ("adaptive_refined_source_vtp", "refinement_field_vtp"):
        path = adaptive_outputs.get(key)
        if path:
            ignored.append(
                {
                    "path": path,
                    "kind": "adaptive_diagnostic",
                    "reason": "adaptive refinement is not consumed by the final watertight repair output",
                    "safe_for_acceptance": False,
                }
            )
    return ignored


if __name__ == "__main__":
    raise SystemExit(main())
