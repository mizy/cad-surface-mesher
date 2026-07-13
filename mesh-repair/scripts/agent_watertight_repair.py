#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import trimesh

from agent_repair_contract import (
    GLOBAL_REGION_ID,
    advance_repair_state,
    build_observation_packet,
    initial_repair_state,
    validate_agent_response,
)
from agent_repair_review import review_repaired_candidate
from agent_observation import render_registered_observation_bundle
from html_report import write_html_report
from mesh_io import read_surface, triangle_faces
from source_shell_extract import run_source_shell


DEFAULT_TARGET_POLICY = {
    "schema": "mesh_repair_target_policy/v1",
    "target": "watertight-exterior-shell",
    "remove": {
        "non_target_internal_parts": True,
        "hidden_internal_faces": True,
    },
    "seal": {"small_assembly_gaps": True},
    "openings": {
        "functional_openings": "ask",
        "through_holes": "ask",
    },
    "components": {"natural_watertight_components": "keep_separate"},
    "limits": {
        "max_bbox_drift_ratio": 0.002,
        "hard_max_bbox_drift_ratio": 0.005,
        "max_silhouette_changed_ratio": 0.05,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-faithful watertight mesh with a flood-signed TSDF closure "
            "and a fail-closed semantic Agent review loop."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-policy", type=Path)
    parser.add_argument("--voxel-pitch", type=float, default=0.0)
    parser.add_argument("--sdf-grid-size", type=int, default=256)
    parser.add_argument("--sdf-band-voxels", type=float, default=6.0)
    parser.add_argument("--sdf-smoothing-sigma", type=float, default=0.5)
    parser.add_argument("--max-sdf-memory-gb", type=float, default=4.0)
    parser.add_argument("--agent-mode", choices=("auto", "off"), default="auto")
    parser.add_argument("--max-agent-rounds", type=int, default=5)
    parser.add_argument("--agent-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--visibility-grid", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-previews", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepare_output_dir(args.output_dir, overwrite=args.overwrite)
    status = {
        "schema": "agent_mesh_repair_run_status/v1",
        "status": "running",
        "stage": "starting",
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "updated_at": timestamp(),
        "accepted_mesh_vtp": None,
    }
    write_json(args.output_dir / "run_status.json", status)
    try:
        report = run_agent_repair(args, status)
    except BaseException as error:
        status.update(
            {
                "status": "failed",
                "stage": "failed",
                "updated_at": timestamp(),
                "accepted_mesh_vtp": None,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        write_json(args.output_dir / "run_status.json", status)
        raise
    print(
        json.dumps(
            {
                "report": report["outputs"]["report_json"],
                "html_report": report["outputs"]["html_report"],
                "decision": report["decision"],
                "accepted_mesh_vtp": report["outputs"].get("accepted_mesh_vtp"),
            },
            indent=2,
        )
    )
    return 0


def run_agent_repair(args: argparse.Namespace, status: dict[str, Any]) -> dict[str, Any]:
    validate_args(args)
    policy = load_target_policy(args.target_policy)
    update_status(args, status, "source_shell_agent_review")
    pre_dir = args.output_dir / "pre_agent"
    shell_args = source_shell_args(args, pre_dir)
    source_shell_report = run_source_shell(
        shell_args,
        **({"ai_reviewer": conservative_off_reviewer} if args.agent_mode == "off" else {}),
    )
    source_shell_path = pre_dir / "source_shell.vtp"
    canonical_shell = args.output_dir / "source_shell.vtp"
    shutil.copy2(source_shell_path, canonical_shell)

    update_status(args, status, "tsdf_geometry_pipeline")
    geometry_dir = args.output_dir / "geometry"
    geometry_log = args.output_dir / "geometry_pipeline.log"
    command = geometry_command(args, canonical_shell, geometry_dir)
    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    geometry_log.write_text(
        "COMMAND\n" + json.dumps(command) + "\n\nSTDOUT\n" + completed.stdout + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"deterministic TSDF geometry pipeline failed with status {completed.returncode}; "
            f"see {geometry_log}"
        )
    geometry_report_path = geometry_dir / "two_stage_report.json"
    geometry_report = json.loads(geometry_report_path.read_text(encoding="utf-8"))
    geometry_outputs = geometry_report["outputs"]
    projected_path = Path(geometry_outputs["source_projected_watertight_candidate_vtp"])
    if not projected_path.is_file():
        raise FileNotFoundError(f"source-projected candidate is missing: {projected_path}")

    canonical_outputs = copy_canonical_geometry_outputs(args.output_dir, geometry_outputs, projected_path)
    update_status(args, status, "post_repair_observation")
    view_paths: list[str | dict[str, Any]] = observation_views(
        source_shell_report, geometry_outputs
    )
    registered_observations = None
    if not args.skip_previews:
        registered_observations = render_registered_observation_bundle(
            {
                "source_shell": canonical_shell,
                "closure_proxy": canonical_outputs["closure_proxy_vtp"],
                "source_projected_candidate": canonical_outputs[
                    "source_projected_candidate_vtp"
                ],
            },
            args.output_dir / "agent_visual",
            grid_size=max(64, min(int(args.visibility_grid), 720)),
        )
        view_paths = select_agent_views(registered_observations["views"])
    observation = build_observation_packet(
        source_path=args.input,
        candidate_path=canonical_outputs["source_projected_candidate_vtp"],
        source_shell_report=source_shell_report,
        geometry_report=geometry_report,
        view_paths=view_paths,
    )
    observation_path = args.output_dir / "observation_packet.json"
    write_json(observation_path, observation)
    write_json(args.output_dir / "region_inventory.json", geometry_report.get("inventory_after", {}))

    update_status(args, status, "post_repair_agent_review")
    if args.agent_mode == "auto":
        agent_review = review_repaired_candidate(
            observation,
            output_dir=args.output_dir / "post_agent",
            timeout_seconds=args.agent_timeout_seconds,
            codex_binary=args.codex_binary,
        )
    else:
        agent_review = deterministic_off_review(observation, geometry_report, policy)
    decisions = list(agent_review.get("decisions", []))
    decision_log = [
        {"phase": "pre_sdf_source_shell", **row}
        for row in source_shell_report.get("ai_review", {}).get("decisions", [])
    ]
    decision_log.extend({"phase": "post_sdf_audit", **row} for row in decisions)
    write_jsonl(args.output_dir / "agent_decisions.jsonl", decision_log)

    transactions = geometry_transactions(geometry_report, observation)
    write_jsonl(args.output_dir / "transactions.jsonl", transactions)
    state = initial_repair_state(observation, max_rounds=args.max_agent_rounds)
    state = advance_repair_state(
        state,
        candidate_geometry_hash=observation["candidate_geometry_hash"],
        defect_signature_value=observation["defect_signature"],
        committed_transactions=sum(row.get("status") == "committed" for row in transactions),
    )

    combined_gates = combined_acceptance_gates(
        geometry_report,
        agent_review,
        decisions,
        source_independent_watertight_components=independent_watertight_component_count(
            canonical_shell
        ),
    )
    accepted = all(row["passed"] for row in combined_gates.values())
    if accepted:
        accepted_path = args.output_dir / "watertight_mesh.vtp"
        shutil.copy2(canonical_outputs["source_projected_candidate_vtp"], accepted_path)
        decision = {
            "status": "accepted",
            "reason_codes": [],
            "final_output_path": str(accepted_path),
        }
        state.update({"status": "accepted", "stop_reason": "all_required_gates_passed"})
    else:
        accepted_path = None
        decision = {
            "status": "rejected",
            "reason_codes": [name for name, row in combined_gates.items() if not row["passed"]],
            "final_output_path": None,
        }
        state.update(
            {
                "status": "blocked",
                "stop_reason": state.get("stop_reason") or "blocking_gate_or_agent_action",
            }
        )
    write_json(args.output_dir / "repair_state.json", state)

    outputs = {
        **{name: str(path) for name, path in canonical_outputs.items()},
        "accepted_mesh_vtp": str(accepted_path) if accepted_path else None,
        "observation_packet_json": str(observation_path),
        "region_inventory_json": str(args.output_dir / "region_inventory.json"),
        "agent_decisions_jsonl": str(args.output_dir / "agent_decisions.jsonl"),
        "transactions_jsonl": str(args.output_dir / "transactions.jsonl"),
        "repair_state_json": str(args.output_dir / "repair_state.json"),
        "geometry_report_json": str(geometry_report_path),
        "geometry_pipeline_log": str(geometry_log),
        "report_json": str(args.output_dir / "agent_repair_report.json"),
        "html_report": str(args.output_dir / "agent_repair_report.html"),
    }
    report = {
        "schema": "agent_watertight_repair_report/v1",
        "decision": decision,
        "input": {
            "path": str(args.input),
            "kind": "mesh",
            "units": "unknown",
            "coordinate_convention": "unknown",
        },
        "output_contract": {
            "input_kind": "mesh",
            "output_kind": "watertight_mesh",
            "repair_domain": "mesh_domain_agent_supervised_tsdf_source_projection",
            "closure_proxy_is_never_accepted": True,
        },
        "target_policy": policy,
        "parameters": parameter_report(args),
        "geometry_to_mesh_trace": [
            {"stage": "source_shell", "path": str(canonical_shell), "owner": "agent_plus_deterministic_visibility"},
            {"stage": "implicit_field", "path": outputs.get("implicit_field_npz"), "owner": "deterministic_tsdf"},
            {"stage": "closure_proxy", "path": outputs.get("closure_proxy_vtp"), "owner": "deterministic_tsdf"},
            {"stage": "source_projected_candidate", "path": outputs.get("source_projected_candidate_vtp"), "owner": "deterministic_projection"},
            {"stage": "post_agent_audit", "path": outputs["agent_decisions_jsonl"], "owner": "semantic_agent"},
            {"stage": "accepted_mesh", "path": outputs["accepted_mesh_vtp"], "owner": "validator"},
        ],
        "source_shell_agent": source_shell_report,
        "observation_packet": observation,
        "post_repair_agent": agent_review,
        "registered_observations": registered_observations,
        "transactions": transactions,
        "repair_state": state,
        "gates": combined_gates,
        "geometry_decision": geometry_report.get("decision"),
        "geometry_gates": geometry_report.get("gates"),
        "unhandled_items": geometry_report.get("unhandled_items", []),
        "not_implemented": unsupported_requested_actions(decisions),
        "outputs": outputs,
        "limitations": [
            "The first implementation executes one full deterministic repair transaction round; additional Agent actions fail closed for a future round.",
            "Functional openings remain blocking unless the Agent or target policy explicitly chooses cap.",
            "Unknown input units use bbox- and voxel-relative thresholds and are topology-first approximations.",
        ],
    }
    write_json(Path(outputs["report_json"]), report)
    write_html_report(report, Path(outputs["html_report"]), "Agent-Supervised TSDF Watertight Repair")
    status.update(
        {
            "status": "completed_repair_accepted" if accepted else "completed_repair_rejected",
            "stage": "completed",
            "updated_at": timestamp(),
            "decision": decision,
            "accepted_mesh_vtp": outputs["accepted_mesh_vtp"],
            "outputs": outputs,
        }
    )
    write_json(args.output_dir / "run_status.json", status)
    return report


def source_shell_args(args: argparse.Namespace, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        input=args.input,
        output_dir=output_dir,
        direction_count=42,
        visibility_grid=max(32, int(args.visibility_grid)),
        depth_tolerance=0.0,
        continuity_rings=1,
        min_first_hit_views=1,
        min_first_hit_pixels=1,
        ai_max_candidates=48,
        ai_min_component_faces=2,
        ai_remove_confidence=0.85,
        ai_timeout_seconds=float(args.agent_timeout_seconds),
        codex_binary=args.codex_binary,
        deadline_seconds=max(600.0, float(args.agent_timeout_seconds) * 2.0),
    )


def conservative_off_reviewer(**kwargs: Any) -> dict[str, Any]:
    return {
        "status": "accepted",
        "decisions": [
            {
                "candidate_id": candidate_id,
                "decision": "keep_exterior",
                "confidence": 1.0,
                "semantic_role": "conservative_agent_off_keep",
                "evidence_view_ids": [],
                "reason_codes": ["agent_mode_off_no_destructive_semantic_removal"],
                "rationale": "Agent mode is off, so uncertain components are conservatively retained.",
            }
            for candidate_id in kwargs["candidate_ids"]
        ],
        "error": None,
        "artifacts": {},
    }


def geometry_command(
    args: argparse.Namespace,
    source_shell_path: Path,
    geometry_dir: Path,
) -> list[str]:
    script = Path(__file__).with_name("two_stage_watertight_remesh.py")
    command = [
        sys.executable,
        str(script),
        str(source_shell_path),
        "--output-dir",
        str(geometry_dir),
        "--visibility-grid",
        str(max(64, int(args.visibility_grid))),
        "--outside-flood-grid",
        str(max(32, min(int(args.sdf_grid_size), 192))),
        "--sealed-exterior-grid",
        str(max(32, min(int(args.sdf_grid_size), 192))),
        "--voxel-pitch",
        str(float(args.voxel_pitch)),
        "--voxel-pitch-bbox-divisor",
        str(float(args.sdf_grid_size)),
        "--sdf-band-voxels",
        str(float(args.sdf_band_voxels)),
        "--sdf-smoothing-sigma",
        str(float(args.sdf_smoothing_sigma)),
        "--max-sdf-memory-gb",
        str(float(args.max_sdf_memory_gb)),
    ]
    if args.skip_previews:
        command.append("--skip-previews")
    return command


def deterministic_off_review(
    packet: dict[str, Any],
    geometry_report: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    geometry_accepted = geometry_report.get("decision", {}).get("status") == "accepted"
    opening_default = str(policy.get("openings", {}).get("functional_openings", "ask"))
    opening_action = opening_default if opening_default in {"cap", "reconstruct_wall", "porous_surface", "ask", "reject"} else "ask"
    response = {
        "candidate_geometry_hash": packet["candidate_geometry_hash"],
        "decisions": [],
    }
    for item in packet["review_items"]:
        action = "accept" if item["region_id"] == GLOBAL_REGION_ID and geometry_accepted else "reject"
        if item["kind"] == "semantic_opening":
            action = opening_action
        response["decisions"].append(
            {
                "region_id": item["region_id"],
                "action": action,
                "confidence": 1.0,
                "evidence_view_ids": [],
                "rationale": "Deterministic Agent-off policy decision.",
            }
        )
    decisions = validate_agent_response(packet, response)
    return {"status": "accepted", "decisions": decisions, "error": None, "artifacts": {}}


def combined_acceptance_gates(
    geometry_report: dict[str, Any],
    agent_review: dict[str, Any],
    decisions: list[dict[str, Any]],
    *,
    source_independent_watertight_components: int = 0,
) -> dict[str, dict[str, Any]]:
    geometry_accepted = geometry_report.get("decision", {}).get("status") == "accepted"
    global_decision = next(
        (row for row in decisions if row.get("region_id") == GLOBAL_REGION_ID),
        None,
    )
    opening_decisions = [
        row for row in decisions if row.get("region_id") != GLOBAL_REGION_ID
    ]
    opening_passed = all(row.get("action") == "cap" for row in opening_decisions)
    output_components = (
        geometry_report.get("stages", {})
        .get("source_projected_watertight_candidate", {})
        .get("metrics", {})
        .get("topology", {})
        .get("components", {})
        .get("count")
    )
    separation_passed = bool(
        source_independent_watertight_components == 0
        or (
            isinstance(output_components, int)
            and output_components >= source_independent_watertight_components
        )
    )
    return {
        "deterministic_geometry_accepted": gate(
            geometry_accepted,
            geometry_report.get("decision"),
            "accepted",
        ),
        "agent_review_schema_valid": gate(
            agent_review.get("status") == "accepted",
            agent_review.get("status"),
            "accepted",
        ),
        "global_post_audit_accepted": gate(
            global_decision is not None and global_decision.get("action") == "accept",
            global_decision,
            "action == accept",
        ),
        "semantic_openings_resolved_for_watertight_shell": gate(
            opening_passed,
            opening_decisions,
            "every semantic opening action == cap",
        ),
        "independent_watertight_components_not_merged": gate(
            separation_passed,
            {
                "source_independently_watertight_components": source_independent_watertight_components,
                "output_components": output_components,
            },
            "output component count >= independently watertight source component count",
        ),
    }


def geometry_transactions(
    geometry_report: dict[str, Any],
    observation: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        {
            "transaction_id": "tx_tsdf_closure",
            "operator": "global_tsdf_closure",
            "input_geometry_hash": observation["input_geometry_hash"],
            "output_geometry_hash": observation["candidate_geometry_hash"],
            "status": "committed" if geometry_report.get("stages", {}).get("closure_proxy") else "rejected",
            "authority": "deterministic_geometry",
        },
        {
            "transaction_id": "tx_source_projection",
            "operator": "bounded_source_projection",
            "output_geometry_hash": observation["candidate_geometry_hash"],
            "status": "committed" if geometry_report.get("source_projection") else "rejected",
            "authority": "deterministic_geometry",
        },
    ]
    for index, patch in enumerate(geometry_report.get("patch_regions", []), start=1):
        selection = patch.get("selection_status")
        rows.append(
            {
                "transaction_id": f"tx_patch_{index:05d}",
                "region_id": patch.get("region_id") or patch.get("id"),
                "operator": patch.get("selected_operator") or selection,
                "status": "committed" if patch.get("final_provenance", {}).get("consumed_by_final") else "rolled_back_or_diagnostic",
                "local_gates": patch.get("seam_results") or patch.get("quality"),
                "authority": "deterministic_geometry",
            }
        )
    return rows


def copy_canonical_geometry_outputs(
    output_dir: Path,
    outputs: dict[str, Any],
    projected_path: Path,
) -> dict[str, Path]:
    mapping = {
        "implicit_field_npz": Path(outputs["implicit_field_npz"]),
        "closure_proxy_vtp": Path(outputs["closure_proxy_vtp"]),
        "source_projected_candidate_vtp": projected_path,
    }
    destinations = {
        "implicit_field_npz": output_dir / "implicit_field.npz",
        "closure_proxy_vtp": output_dir / "closure_proxy.vtp",
        "source_projected_candidate_vtp": output_dir / "source_projected_candidate.vtp",
        "candidate_iter_0_vtp": output_dir / "candidate_iter_0.vtp",
    }
    for name, source in mapping.items():
        if not source.is_file():
            raise FileNotFoundError(f"required geometry artifact is missing: {source}")
        shutil.copy2(source, destinations[name])
    shutil.copy2(projected_path, destinations["candidate_iter_0_vtp"])
    return destinations


def observation_views(
    source_shell_report: dict[str, Any],
    geometry_outputs: dict[str, Any],
) -> list[str]:
    paths = []
    paths.extend(source_shell_report.get("outputs", {}).get("previews", []))
    paths.extend(geometry_outputs.get("previews", []))
    return list(dict.fromkeys(str(path) for path in paths if Path(path).is_file()))


def select_agent_views(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("render_mode") == "depth"
        and row.get("mesh_role") in {"source_shell", "source_projected_candidate"}
    ]
    selected.extend(
        row
        for row in rows
        if row.get("render_mode") == "face_id_discontinuity"
        and row.get("mesh_role") == "source_projected_candidate"
    )
    selected.extend(
        row
        for row in rows
        if row.get("render_mode") == "face_id"
        and row.get("mesh_role") == "source_shell"
    )
    return selected[:24]


def independent_watertight_component_count(path: Path) -> int:
    surface = read_surface(path)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(surface.points, dtype=np.float64),
        faces=triangle_faces(surface),
        process=False,
    )
    return int(
        sum(
            bool(component.is_watertight)
            for component in mesh.split(only_watertight=False)
        )
    )


def unsupported_requested_actions(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    supported = {"accept", "cap"}
    return [
        {
            "region_id": row["region_id"],
            "action": row["action"],
            "blocking": True,
            "reason": "requested post-audit action requires a subsequent geometry transaction round",
        }
        for row in decisions
        if row.get("action") not in supported
    ]


def load_target_policy(path: Path | None) -> dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_TARGET_POLICY))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("target policy must be a JSON object")
    merged = json.loads(json.dumps(DEFAULT_TARGET_POLICY))
    deep_update(merged, data)
    return merged


def deep_update(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value


def parameter_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
        if key not in {"input", "output_dir"}
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.sdf_grid_size < 16:
        raise ValueError("sdf-grid-size must be at least 16")
    if args.voxel_pitch < 0.0:
        raise ValueError("voxel-pitch must be positive, or zero for bbox-derived resolution")
    if args.sdf_band_voxels < 2.0:
        raise ValueError("sdf-band-voxels must be at least two")
    if args.max_sdf_memory_gb <= 0.0:
        raise ValueError("max-sdf-memory-gb must be positive")
    if args.max_agent_rounds < 1:
        raise ValueError("max-agent-rounds must be positive")
    if args.agent_mode == "auto" and args.skip_previews:
        raise ValueError("agent-mode auto requires visual previews")


def prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(f"output directory is not empty: {path}; use --overwrite")
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def update_status(args: argparse.Namespace, status: dict[str, Any], stage: str) -> None:
    status.update({"stage": stage, "updated_at": timestamp()})
    write_json(args.output_dir / "run_status.json", status)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def gate(passed: bool, value: Any, threshold: Any) -> dict[str, Any]:
    return {
        "required": True,
        "passed": bool(passed),
        "value": value,
        "threshold": threshold,
    }


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
