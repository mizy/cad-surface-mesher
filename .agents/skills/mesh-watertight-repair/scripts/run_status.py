from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from run_status_lease import (
    acquire_run_lease,
    clean_output_artifacts,
    ensure_generation_id,
    release_run_lease,
)


def prepare_output_dir(args: Any) -> None:
    if args.output_dir.exists():
        if not args.output_dir.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {args.output_dir}")
        if not args.overwrite:
            raise FileExistsError(
                f"output directory already exists: {args.output_dir}; use --overwrite explicitly"
            )
    else:
        args.output_dir.mkdir(parents=True)
    fingerprint = command_fingerprint(args)
    acquire_run_lease(args, fingerprint)
    try:
        write_run_status(args.output_dir, in_progress_run_status(args, fingerprint))
        if args.overwrite:
            clean_output_artifacts(args.output_dir, preserve_run_status=True)
    except BaseException:
        release_run_lease(args)
        raise


def release_output_lease(args: Any) -> None:
    release_run_lease(args)


def in_progress_run_status(args: Any, fingerprint: str) -> dict[str, Any]:
    generation_id = ensure_generation_id(args)
    started_at = timestamp()
    return {
        "generation_id": generation_id,
        "run_id": f"mesh-repair-{generation_id}",
        "status": "running",
        "execution_status": "running",
        "repair_status": "not_accepted",
        "outcome_status": "in_progress_no_accepted_mesh",
        "stage": "preparing_output",
        "started_at": started_at,
        "updated_at": started_at,
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "command": sys.argv,
        "command_fingerprint": fingerprint,
        "decision": {"status": "rejected", "reason_codes": ["run_in_progress"]},
        "accepted_mesh_vtp": None,
        "accepted_mesh_available": False,
        "outputs": {"accepted_mesh_vtp": None, "accepted_mesh_available": False},
    }


def initial_run_status(args: Any) -> dict[str, Any]:
    fingerprint = command_fingerprint(args)
    started_at = timestamp()
    generation_id = ensure_generation_id(args)
    return {
        "_started_monotonic": time.monotonic(),
        "generation_id": generation_id,
        "run_id": f"mesh-repair-{generation_id}",
        "status": "running",
        "execution_status": "running",
        "repair_status": "not_decided",
        "outcome_status": "running",
        "stage": "starting",
        "started_at": started_at,
        "updated_at": started_at,
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "command": sys.argv,
        "command_fingerprint": fingerprint,
        "parameters": run_parameters(args),
        "voxel_pitch": pitch_status(args),
        "accepted_mesh_vtp": None,
        "accepted_mesh_available": False,
        "outputs": {"accepted_mesh_vtp": None, "accepted_mesh_available": False},
        "metrics": {},
    }

def update_run_status(
    args: Any,
    status: dict[str, Any],
    stage: str,
    *,
    outputs: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    voxel_pitch: dict[str, Any] | None = None,
    remesh: dict[str, Any] | None = None,
    accepted_mesh_vtp: str | None = None,
) -> None:
    status["stage"] = stage
    status["updated_at"] = timestamp()
    status["elapsed_seconds"] = elapsed_seconds(status)
    if outputs:
        status.setdefault("outputs", {}).update(outputs)
    if metrics:
        status.setdefault("metrics", {}).update(metrics)
    if voxel_pitch:
        status["voxel_pitch"] = voxel_pitch
    if remesh:
        status.setdefault("metrics", {})["closure_proxy_remesh"] = remesh
    status["accepted_mesh_vtp"] = accepted_mesh_vtp
    status["accepted_mesh_available"] = bool(accepted_mesh_vtp)
    status.setdefault("outputs", {})["accepted_mesh_vtp"] = accepted_mesh_vtp
    status["outputs"]["accepted_mesh_available"] = bool(accepted_mesh_vtp)
    if stage == "completed":
        status.update(completed_run_outcome(status))
        status["ended_at"] = status["updated_at"]
    write_run_status(args.output_dir, status)


def record_failure(args: Any, status: dict[str, Any], exc: BaseException) -> None:
    failure = failure_status(exc)
    status["status"] = "failed"
    status["execution_status"] = "failed"
    status["repair_status"] = "not_accepted"
    status["outcome_status"] = "run_failed"
    status["updated_at"] = timestamp()
    status["ended_at"] = status["updated_at"]
    status["elapsed_seconds"] = elapsed_seconds(status)
    status["failure"] = failure
    status["resource_failure"] = resource_failure(failure)
    status["feasible_upper_bound"] = feasible_upper_bound(args, status)
    status["accepted_mesh_vtp"] = None
    status["accepted_mesh_available"] = False
    status.setdefault("outputs", {}).update(partial_outputs(args.output_dir))
    status["outputs"]["accepted_mesh_vtp"] = None
    status["outputs"]["accepted_mesh_available"] = False
    write_run_status(args.output_dir, status)
    try:
        write_failure_audit_report(args, status)
    except Exception as report_exc:
        status["failure_report_error"] = failure_status(report_exc)
        write_run_status(args.output_dir, status)


def pitch_status(args: Any) -> dict[str, Any]:
    return {
        "actual": float(args.voxel_pitch),
        "requested": float(getattr(args, "requested_voxel_pitch", args.voxel_pitch)),
        "source": getattr(args, "voxel_pitch_source", "explicit"),
        "bbox_divisor": getattr(args, "voxel_pitch_bbox_divisor", None),
        "bbox_max_extent": getattr(args, "voxel_pitch_bbox_max_extent", None),
    }


def topology_brief(report: dict[str, Any]) -> dict[str, Any]:
    topology = report["topology"]
    return {
        "triangles": report["triangles"],
        "boundary_edges": topology["boundary_edges"],
        "non_manifold_edges": topology["non_manifold_edges"],
        "non_manifold_vertices": topology["non_manifold_vertices"],
        "inconsistent_winding_edges": topology.get("inconsistent_winding_edges"),
        "components": topology["components"]["count"],
        "volume_reliable": report["volume"]["reliable"],
    }


def completed_run_outcome(status: dict[str, Any]) -> dict[str, Any]:
    decision = status.get("metrics", {}).get("decision", {})
    decision_status = decision.get("status")
    accepted_path = status.get("accepted_mesh_vtp")
    mesh_result = status.get("outputs", {}).get("mesh_result", {})
    if decision_status == "accepted" and accepted_path:
        return {
            "status": "completed_repair_accepted",
            "execution_status": "completed",
            "repair_status": "accepted",
            "outcome_status": "accepted_repair",
            "run_outcome": {
                "execution": "completed",
                "repair": "accepted",
                "delivery": "accepted_mesh_available",
                "mesh_vtp": accepted_path,
            },
        }
    if decision_status == "rejected":
        fallback = mesh_result.get("path") if mesh_result.get("status") == "repair_rejected_with_source_fallback" else None
        return {
            "status": "completed_repair_rejected",
            "execution_status": "completed",
            "repair_status": "rejected",
            "outcome_status": mesh_result.get("status", "repair_rejected_no_mesh"),
            "run_outcome": {
                "execution": "completed",
                "repair": "rejected",
                "delivery": "source_fallback_available" if fallback else "no_accepted_mesh",
                "mesh_vtp": fallback,
                "engineering_ready": False,
            },
        }
    return {
        "status": "completed_outcome_unknown",
        "execution_status": "completed",
        "repair_status": "unknown",
        "outcome_status": "missing_repair_decision",
        "run_outcome": {
            "execution": "completed",
            "repair": "unknown",
            "delivery": "no_accepted_mesh",
            "mesh_vtp": None,
        },
    }


def write_run_status(output_dir: Path, status: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_status.json"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=output_dir
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                serializable_status(status), handle, indent=2, allow_nan=False
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def serializable_status(status: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in status.items() if not key.startswith("_")}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed_seconds(status: dict[str, Any]) -> float:
    started = float(status.get("_started_monotonic", time.monotonic()))
    return round(time.monotonic() - started, 3)


def failure_status(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def resource_failure(failure: dict[str, Any]) -> dict[str, Any] | None:
    text = f"{failure.get('type', '')} {failure.get('message', '')}".lower()
    if "memory" in text or "bad allocation" in text or "cannot allocate" in text:
        return {"kind": "memory", "failure_reason": failure.get("message") or failure.get("type")}
    if "timeout" in text or "timed out" in text:
        return {"kind": "timeout", "failure_reason": failure.get("message") or failure.get("type")}
    return None


def feasible_upper_bound(args: Any, status: dict[str, Any]) -> dict[str, Any] | None:
    outputs = status.get("outputs", {})
    if not outputs.get("closure_proxy_vtp"):
        return None
    return {
        "voxel_pitch": float(args.voxel_pitch),
        "voxel_pitch_bbox_divisor": getattr(args, "voxel_pitch_bbox_divisor", None),
        "basis": "closure_proxy was written before the failure",
    }


def partial_outputs(output_dir: Path) -> dict[str, Any]:
    paths = {
        "stage1_exterior_candidate_vtp": output_dir / "stage1_exterior_candidate.vtp",
        "source_preserving_candidate_vtp": output_dir / "source_preserving_candidate.vtp",
        "closure_proxy_vtp": output_dir / "closure_proxy.vtp",
        "source_projected_watertight_candidate_vtp": output_dir / "source_projected_watertight_candidate.vtp",
    }
    outputs: dict[str, Any] = {
        "accepted_mesh_vtp": None,
        "accepted_mesh_available": False,
        "report_json": str(output_dir / "two_stage_report.json"),
        "html_report": str(output_dir / "two_stage_report.html"),
    }
    for key, path in paths.items():
        if path.exists():
            outputs[key] = str(path)
    rejected = next(
        (
            path
            for path in (paths["source_projected_watertight_candidate_vtp"],)
            if path.exists()
        ),
        paths["source_preserving_candidate_vtp"],
    )
    outputs["rejected_candidate_vtp"] = str(rejected) if rejected.exists() else None
    fallback = paths["source_preserving_candidate_vtp"] if paths["source_preserving_candidate_vtp"].exists() else None
    outputs["source_fallback_vtp"] = str(fallback) if fallback else None
    outputs["mesh_result"] = {
        "status": "run_failed_with_source_fallback" if fallback else "run_failed_no_mesh",
        "path": str(fallback) if fallback else None,
        "role": "source_preserving_unrepaired_fallback" if fallback else None,
        "accepted": False,
        "engineering_ready": False,
    }
    return outputs


def write_failure_audit_report(args: Any, status: dict[str, Any]) -> None:
    report_path = args.output_dir / "two_stage_report.json"
    html_path = args.output_dir / "two_stage_report.html"
    outputs = partial_outputs(args.output_dir)
    report = {
        "decision": {
            "status": "rejected",
            "outcome": outputs["mesh_result"]["status"],
            "reason_codes": failure_reason_codes(status),
            "final_output_path": None,
            "source_fallback_path": outputs.get("source_fallback_vtp"),
        },
        "input": {"path": str(args.input), "kind": "mesh"},
        "parameters": run_parameters(args),
        "run_status": serializable_status(status),
        "gates": {
            "run_completed": {
                "required": True,
                "passed": False,
                "value": status.get("stage"),
                "threshold": "completed",
                "failure_reason": status.get("failure", {}).get("message"),
            },
            "no_accepted_mesh_claimed": {
                "required": True,
                "passed": outputs.get("accepted_mesh_vtp") is None,
                "value": outputs.get("accepted_mesh_vtp"),
                "threshold": None,
            },
        },
        "outputs": outputs,
        "ignored_outputs": ignored_partial_outputs(outputs),
        "unhandled_items": [{
            "item": "mesh_repair_run",
            "status": "failed",
            "blocking": True,
            "failure_reason": status.get("failure", {}).get("message"),
        }],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path.write_text(render_failure_html(report), encoding="utf-8")


def run_parameters(args: Any) -> dict[str, Any]:
    return {
        "input": str(args.input),
        "target_name": args.target_name,
        "group_source_gltf": str(args.group_source_gltf) if args.group_source_gltf else None,
        "remove_name_regex": args.remove_name_regex,
        "visibility_grid": args.visibility_grid,
        "visibility_min_views": getattr(args, "visibility_min_views", 1),
        "outside_flood_grid": getattr(args, "outside_flood_grid", None),
        "sealed_exterior_grid": getattr(args, "sealed_exterior_grid", None),
        "sealed_exterior_radius_voxels": getattr(args, "sealed_exterior_radius_voxels", None),
        "sealed_exterior_band_voxels": getattr(args, "sealed_exterior_band_voxels", None),
        "depth_tolerance": args.depth_tolerance,
        "requested_depth_tolerance": getattr(args, "requested_depth_tolerance", args.depth_tolerance),
        "depth_tolerance_bbox_ratio": getattr(args, "depth_tolerance_bbox_ratio", None),
        "depth_tolerance_edge_ratio": getattr(args, "depth_tolerance_edge_ratio", None),
        "dilate_rings": args.dilate_rings,
        "voxel_pitch": args.voxel_pitch,
        "requested_voxel_pitch": getattr(args, "requested_voxel_pitch", args.voxel_pitch),
        "voxel_pitch_source": getattr(args, "voxel_pitch_source", "explicit"),
        "voxel_pitch_bbox_divisor": getattr(args, "voxel_pitch_bbox_divisor", None),
        "voxel_pitch_bbox_max_extent": getattr(args, "voxel_pitch_bbox_max_extent", None),
        "policy_item_limit": args.policy_item_limit,
        "component_filter_thresholds": getattr(args, "component_filter_thresholds", None),
    }


def command_fingerprint(args: Any) -> str:
    payload = json.dumps(run_parameters(args), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def failure_reason_codes(status: dict[str, Any]) -> list[str]:
    resource = status.get("resource_failure")
    if resource:
        return [f"resource_{resource['kind']}_failed"]
    return ["run_failed_before_final_report"]


def ignored_partial_outputs(outputs: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    if outputs.get("closure_proxy_vtp"):
        rows.append({
            "path": outputs["closure_proxy_vtp"],
            "kind": "closure_proxy",
            "reason": "diagnostic only; not accepted final geometry",
        })
    if outputs.get("source_projected_watertight_candidate_vtp"):
        rows.append({
            "path": outputs["source_projected_watertight_candidate_vtp"],
            "kind": "failed_source_projected_watertight_candidate",
            "reason": "run failed before accepted report gates completed",
        })
    return rows


def render_failure_html(report: dict[str, Any]) -> str:
    content = escape(json.dumps(report, indent=2))
    status = escape(str(report["decision"]["status"]))
    return "\n".join([
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\"><title>Watertight Mesh Repair Run Failed</title></head>",
        "<body>",
        "<h1>Watertight Mesh Repair Run Failed</h1>",
        f"<p>decision.status: {status}</p>",
        "<h2>Machine Report</h2>",
        f"<pre>{content}</pre>",
        "</body></html>",
    ])
