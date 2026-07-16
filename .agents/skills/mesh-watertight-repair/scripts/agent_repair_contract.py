from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mesh_io import read_surface, triangle_faces


GLOBAL_REGION_ID = "global_post_audit"
GLOBAL_ACTIONS = (
    "accept",
    "rerun_tsdf_resolution",
    "rerun_projection_thresholds",
    "rerun_smoothing",
    "request_view",
    "ask",
    "reject",
)
OPENING_ACTIONS = (
    "cap",
    "preserve",
    "porous_surface",
    "defer",
    "ask",
    "reject",
)
FORBIDDEN_DECISION_KEYS = {
    "point",
    "points",
    "vertex",
    "vertices",
    "coordinates",
    "faces",
    "triangles",
}


def geometry_hash_from_file(path: Path) -> str:
    surface = read_surface(path)
    return geometry_hash(np.asarray(surface.points), triangle_faces(surface))


def geometry_hash(points: np.ndarray, faces: np.ndarray) -> str:
    point_array = np.ascontiguousarray(np.asarray(points, dtype="<f8"))
    face_array = np.ascontiguousarray(np.asarray(faces, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(np.asarray(point_array.shape, dtype="<i8").tobytes())
    digest.update(point_array.tobytes())
    digest.update(np.asarray(face_array.shape, dtype="<i8").tobytes())
    digest.update(face_array.tobytes())
    return digest.hexdigest()


def defect_signature(geometry_report: dict[str, Any]) -> str:
    decision = geometry_report.get("decision", {})
    gates = geometry_report.get("gates", {})
    unhandled = geometry_report.get("unhandled_items", [])
    payload = {
        "decision_status": decision.get("status"),
        "reason_codes": sorted(str(value) for value in decision.get("reason_codes", [])),
        "failed_gates": sorted(
            name
            for name, row in gates.items()
            if row.get("required", True) and row.get("passed") is not True
        ),
        "blocking_unhandled": sorted(
            str(row.get("item") or row.get("id") or row.get("reason"))
            for row in unhandled
            if row.get("blocking", True)
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def review_items(geometry_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "region_id": GLOBAL_REGION_ID,
            "kind": "global_post_audit",
            "allowed_actions": list(GLOBAL_ACTIONS),
        }
    ]
    packet = geometry_report.get("unresolved_policy_packet", {})
    for item in packet.get("items", []):
        rows.append(
            {
                "region_id": str(item["id"]),
                "kind": "semantic_opening",
                "source_region": item.get("source_region"),
                "allowed_actions": list(OPENING_ACTIONS),
                "geometry": {
                    "bbox": item.get("bbox"),
                    "edge_count": item.get("edge_count"),
                    "centroid": item.get("centroid"),
                    "normal": item.get("normal"),
                },
            }
        )
    return rows


def build_observation_packet(
    *,
    source_path: Path,
    candidate_path: Path,
    source_exterior_report: dict[str, Any],
    geometry_report: dict[str, Any],
    view_paths: Iterable[str | dict[str, Any]],
) -> dict[str, Any]:
    candidate_hash = geometry_hash_from_file(candidate_path)
    views = []
    for index, raw in enumerate(view_paths, start=1):
        if isinstance(raw, dict):
            row = dict(raw)
            row.setdefault("view_id", f"post_view_{index:03d}")
            row["path"] = str(row["path"])
            views.append(row)
        else:
            views.append({"view_id": f"post_view_{index:03d}", "path": str(raw)})
    return {
        "schema": "mesh_repair_observation/v1",
        "input_geometry_hash": geometry_hash_from_file(source_path),
        "candidate_geometry_hash": candidate_hash,
        "candidate_path": str(candidate_path),
        "target": geometry_report.get("target", {}),
        "source_exterior": {
            "decision": source_exterior_report.get("decision"),
            "components": source_exterior_report.get("components"),
            "selection": source_exterior_report.get("selection"),
        },
        "deterministic_decision": geometry_report.get("decision"),
        "gates": geometry_report.get("gates", {}),
        "defect_signature": defect_signature(geometry_report),
        "review_items": review_items(geometry_report),
        "views": views,
        "authority": {
            "agent": "semantic classification, observation requests, and declared rerun-policy selection",
            "geometry": "all point/face mutations",
            "validator": "final acceptance; agent decisions cannot override failed gates",
        },
    }


def decision_schema(packet: dict[str, Any]) -> dict[str, Any]:
    item_ids = [str(row["region_id"]) for row in packet.get("review_items", [])]
    actions = sorted(
        {
            action
            for row in packet.get("review_items", [])
            for action in row.get("allowed_actions", [])
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["candidate_geometry_hash", "decisions"],
        "properties": {
            "candidate_geometry_hash": {
                "type": "string",
                "const": packet["candidate_geometry_hash"],
            },
            "decisions": {
                "type": "array",
                "minItems": len(item_ids),
                "maxItems": len(item_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "region_id",
                        "action",
                        "confidence",
                        "evidence_view_ids",
                        "rationale",
                    ],
                    "properties": {
                        "region_id": {"type": "string", "enum": item_ids},
                        "action": {"type": "string", "enum": actions},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "evidence_view_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "rationale": {"type": "string", "maxLength": 500},
                    },
                },
            },
        },
    }


def validate_agent_response(
    packet: dict[str, Any],
    response: dict[str, Any],
) -> list[dict[str, Any]]:
    if response.get("candidate_geometry_hash") != packet.get("candidate_geometry_hash"):
        raise ValueError("agent decision geometry hash is stale or unknown")
    decisions = response.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("agent response must contain a decisions list")
    items = {str(row["region_id"]): row for row in packet.get("review_items", [])}
    view_ids = {str(row["view_id"]) for row in packet.get("views", [])}
    seen: set[str] = set()
    normalized = []
    for raw in decisions:
        if not isinstance(raw, dict):
            raise ValueError("every agent decision must be an object")
        forbidden = FORBIDDEN_DECISION_KEYS.intersection(raw)
        if forbidden:
            raise ValueError(f"agent decisions cannot contain raw geometry keys: {sorted(forbidden)}")
        region_id = str(raw.get("region_id", ""))
        if region_id not in items:
            raise ValueError(f"agent decision references an unknown region: {region_id}")
        if region_id in seen:
            raise ValueError(f"agent decision duplicates region: {region_id}")
        action = raw.get("action")
        if action not in items[region_id]["allowed_actions"]:
            raise ValueError(f"action {action!r} is not allowed for region {region_id}")
        confidence = raw.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("agent decision confidence must be numeric")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("agent decision confidence must be in [0, 1]")
        evidence = raw.get("evidence_view_ids")
        if not isinstance(evidence, list) or any(str(value) not in view_ids for value in evidence):
            raise ValueError("agent decision cites an unknown evidence view")
        rationale = raw.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("agent decision rationale must be non-empty")
        seen.add(region_id)
        normalized.append(
            {
                "region_id": region_id,
                "action": str(action),
                "confidence": float(confidence),
                "evidence_view_ids": [str(value) for value in evidence],
                "rationale": rationale.strip(),
                "candidate_geometry_hash": packet["candidate_geometry_hash"],
            }
        )
    missing = sorted(set(items) - seen)
    if missing:
        raise ValueError(f"agent response omitted regions: {missing}")
    return normalized


def initial_repair_state(
    packet: dict[str, Any],
    *,
    max_rounds: int,
) -> dict[str, Any]:
    return {
        "schema": "mesh_repair_state/v1",
        "round": 0,
        "max_rounds": int(max_rounds),
        "candidate_geometry_hash": packet["candidate_geometry_hash"],
        "defect_signature": packet["defect_signature"],
        "previous_candidate_hashes": [],
        "previous_defect_signatures": [],
        "no_progress_rounds": 0,
        "status": "awaiting_agent_review",
        "stop_reason": None,
    }


def advance_repair_state(
    state: dict[str, Any],
    *,
    candidate_geometry_hash: str,
    defect_signature_value: str,
    committed_transactions: int,
) -> dict[str, Any]:
    result = dict(state)
    previous_hashes = list(result.get("previous_candidate_hashes", []))
    previous_signatures = list(result.get("previous_defect_signatures", []))
    current_hash = result.get("candidate_geometry_hash")
    current_signature = result.get("defect_signature")
    previous_hashes.append(current_hash)
    previous_signatures.append(current_signature)
    no_progress = int(result.get("no_progress_rounds", 0))
    if candidate_geometry_hash == current_hash and defect_signature_value == current_signature and committed_transactions == 0:
        no_progress += 1
    else:
        no_progress = 0
    result.update(
        {
            "round": int(result.get("round", 0)) + 1,
            "candidate_geometry_hash": candidate_geometry_hash,
            "defect_signature": defect_signature_value,
            "previous_candidate_hashes": previous_hashes,
            "previous_defect_signatures": previous_signatures,
            "no_progress_rounds": no_progress,
        }
    )
    stop_reason = None
    if candidate_geometry_hash in previous_hashes[:-1] or defect_signature_value in previous_signatures[:-1]:
        stop_reason = "candidate_or_defect_signature_oscillation"
    elif no_progress >= 2:
        stop_reason = "two_rounds_without_progress"
    elif result["round"] >= int(result["max_rounds"]):
        stop_reason = "agent_round_budget_exhausted"
    result["stop_reason"] = stop_reason
    result["status"] = "stopped" if stop_reason else "ready_for_next_round"
    return result


def combined_acceptance_gates(
    geometry_report: dict[str, Any],
    skill_review: dict[str, Any],
    decisions: list[dict[str, Any]],
    *,
    source_independent_watertight_components: int = 0,
) -> dict[str, dict[str, Any]]:
    """Combine immutable geometry gates with the active Skill agent's review."""

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
        "deterministic_geometry_accepted": acceptance_gate(
            geometry_accepted,
            geometry_report.get("decision"),
            "accepted",
        ),
        "skill_review_schema_valid": acceptance_gate(
            skill_review.get("status") == "accepted",
            skill_review.get("status"),
            "accepted",
        ),
        "skill_post_audit_accepted": acceptance_gate(
            global_decision is not None and global_decision.get("action") == "accept",
            global_decision,
            "action == accept",
        ),
        "semantic_openings_resolved_for_watertight_shell": acceptance_gate(
            opening_passed,
            opening_decisions,
            "every semantic opening action == cap",
        ),
        "independent_watertight_components_not_merged": acceptance_gate(
            separation_passed,
            {
                "source_independently_watertight_components": source_independent_watertight_components,
                "output_components": output_components,
            },
            "output component count >= independently watertight source component count",
        ),
    }


def deterministic_transaction_log(
    geometry_report: dict[str, Any],
    observation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Describe committed deterministic geometry work for Skill-level audit logs."""

    rows = [
        {
            "transaction_id": "tx_tsdf_closure",
            "operator": "global_tsdf_closure",
            "input_geometry_hash": observation["input_geometry_hash"],
            "output_geometry_hash": observation["candidate_geometry_hash"],
            "status": (
                "committed"
                if geometry_report.get("stages", {}).get("closure_proxy")
                else "rejected"
            ),
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
    return rows


def acceptance_gate(passed: bool, value: Any, threshold: Any) -> dict[str, Any]:
    return {
        "required": True,
        "passed": bool(passed),
        "value": value,
        "threshold": threshold,
    }
