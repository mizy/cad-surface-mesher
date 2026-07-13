from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_OPTIONS = ["cap", "preserve", "porous_surface", "defer", "reject"]
SEMANTIC_OPTIONS = [
    "missing_cover",
    "grille_or_intake",
    "functional_opening",
    "part_perimeter",
    "unknown",
]
SHAPE_PRIOR_OPTIONS = [
    "mirrored_opposite_side",
    "local_curvature",
    "voxel_sdf",
    "planar",
    "none",
]
POLICY_ELIGIBLE_CLASSIFICATIONS = {
    "pending_policy",
    "large_opening_or_missing_surface",
}
def build_policy_packet(
    inventory: dict[str, Any],
    target_name: str,
    *,
    max_items: int,
    run_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = run_context or {}
    run_fingerprint = stable_fingerprint({"target": target_name, "run_context": context})
    regions = policy_regions(inventory)
    packet_items = []
    for region in regions:
        item = {
            "id": policy_item_id(region["id"]),
            "question": "cap_preserve_porous_or_reject",
            "target": target_name,
            "region_type": region.get("detector_reason") or region.get("classification", "pending_policy"),
            "geometry_classification": region.get("classification"),
            "recommended_operator": region.get("operator"),
            "source_region": region["id"],
            "source_triangle_ids": region.get("source_triangle_ids", []),
            "source_triangle_count": region.get("source_triangle_count"),
            "bbox": region.get("bbox"),
            "edge_count": region.get("edge_count"),
            "length": region.get("length"),
            "centroid": region.get("centroid"),
            "normal": region.get("normal"),
            "dimensionless_features": region.get("dimensionless_features", {}),
            "policy_reason_source": region.get("policy_reason_source"),
            "detector_reason": region.get("detector_reason"),
            "evidence_views": region.get("evidence_views", []),
            "options": POLICY_OPTIONS,
            "semantic_options": SEMANTIC_OPTIONS,
            "shape_prior_options": SHAPE_PRIOR_OPTIONS,
            "default_if_unreviewed": "reject",
            "blocking": True,
        }
        item["item_fingerprint"] = stable_fingerprint(
            {
                "run_fingerprint": run_fingerprint,
                "target": target_name,
                "source_region": item["source_region"],
                "source_triangle_ids": item["source_triangle_ids"],
                "source_triangle_count": item["source_triangle_count"],
                "bbox": item["bbox"],
                "edge_count": item["edge_count"],
                "length": item["length"],
                "centroid": item["centroid"],
                "normal": item["normal"],
                "dimensionless_features": item["dimensionless_features"],
                "region_type": item["region_type"],
                "geometry_classification": item["geometry_classification"],
                "recommended_operator": item["recommended_operator"],
                "policy_reason_source": item["policy_reason_source"],
                "detector_reason": item["detector_reason"],
            }
        )
        item["fingerprint"] = item["item_fingerprint"]
        packet_items.append(item)
    packet = {
        "kind": "ai_policy_packet",
        "review_scope": "semantic_opening_policy_only",
        "target": target_name,
        "run_context": context,
        "run_fingerprint": run_fingerprint,
        "items": packet_items,
        "item_count_total": len(regions),
        "reported_item_count": len(packet_items),
        "truncated": False,
        "geometry_truth_complete": True,
        "requested_report_item_limit": max(0, int(max_items)),
        "report_limit_applied_to_geometry": False,
        "item_source": "explicit_semantic_opening_candidates",
        "decision_schema": {
            "required": ["item_id", "status", "decision"],
            "semantic_label": SEMANTIC_OPTIONS,
            "shape_prior": SHAPE_PRIOR_OPTIONS,
            "decision": POLICY_OPTIONS,
            "confidence_range": [0.0, 1.0],
        },
        "policy_rules": [
            "AI policy may choose only among declared options.",
            "AI policy cannot override deterministic topology, source-distance, bbox, silhouette, volume, or orientation gates.",
            "A cap or porous decision still requires a deterministic geometry pass before acceptance.",
            "AI may select semantic_label and shape_prior, but neither may bypass geometric transaction gates.",
        ],
    }
    packet["packet_fingerprint"] = stable_fingerprint(
        {
            "kind": packet["kind"],
            "target": target_name,
            "run_fingerprint": run_fingerprint,
            "items": [
                {"id": item["id"], "item_fingerprint": item["item_fingerprint"]}
                for item in packet_items
            ],
            "item_count_total": packet["item_count_total"],
            "reported_item_count": packet["reported_item_count"],
            "truncated": packet["truncated"],
        }
    )
    for item in packet_items:
        item["decision_template"] = {
            "item_id": item["id"],
            "status": "decided",
            "decision": None,
            "semantic_label": None,
            "shape_prior": None,
            "confidence": None,
            "reason": None,
            "run_fingerprint": packet["run_fingerprint"],
            "packet_fingerprint": packet["packet_fingerprint"],
            "item_fingerprint": item["item_fingerprint"],
        }
    return packet


def load_policy_decisions(path: Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        unresolved = data.get("unresolved_policy_packet")
        packet = unresolved if isinstance(unresolved, dict) else data
        run_fingerprint = packet.get("run_fingerprint")
        packet_fingerprint = packet.get("packet_fingerprint")
        for key in ("policy_decisions", "decisions"):
            rows = data.get(key)
            if isinstance(rows, list):
                result = []
                for row in rows:
                    next_row = dict(row)
                    if run_fingerprint and "run_fingerprint" not in next_row:
                        next_row["run_fingerprint"] = run_fingerprint
                    if packet_fingerprint and "packet_fingerprint" not in next_row:
                        next_row["packet_fingerprint"] = packet_fingerprint
                    result.append(next_row)
                return result
    raise ValueError(f"policy decisions must be a list or object with decisions: {path}")


def resolve_policy_decisions(
    packet: dict[str, Any],
    supplied_decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    supplied_by_id = {
        str(row.get("item_id") or row.get("id")): row
        for row in supplied_decisions
        if row.get("item_id") or row.get("id")
    }
    decisions = []
    for item in packet.get("items", []):
        supplied = supplied_by_id.get(item["id"])
        decision = supplied.get("decision") if supplied else None
        options = item.get("options", POLICY_OPTIONS)
        valid = decision in options
        stale_reasons = stale_policy_reasons(packet, item, supplied) if supplied else []
        supplied_status = supplied.get("status", "decided") if supplied else None
        status = "decided" if supplied and valid and supplied_status == "decided" and not stale_reasons else "pending_review"
        reviewer = supplied.get("reviewer", "codex_ai_policy") if supplied and status == "decided" else None
        confidence = supplied.get("confidence") if supplied and status == "decided" else None
        semantic_label = supplied.get("semantic_label") if supplied and status == "decided" else None
        shape_prior = supplied.get("shape_prior") if supplied and status == "decided" else None
        if semantic_label not in SEMANTIC_OPTIONS:
            semantic_label = None
        if shape_prior not in SHAPE_PRIOR_OPTIONS:
            shape_prior = None
        decisions.append(
            {
                "item_id": item["id"],
                "status": status,
                "decision": decision if valid else None,
                "reviewer": reviewer,
                "reason": policy_decision_reason(supplied, valid, stale_reasons),
                "confidence": confidence,
                "semantic_label": semantic_label,
                "shape_prior": shape_prior,
                "applied_by_pass_id": None,
                "run_fingerprint": packet.get("run_fingerprint"),
                "packet_fingerprint": packet.get("packet_fingerprint"),
                "item_fingerprint": item.get("item_fingerprint"),
                "supplied_run_fingerprint": supplied.get("run_fingerprint") if supplied else None,
                "supplied_packet_fingerprint": supplied.get("packet_fingerprint") if supplied else None,
                "supplied_item_fingerprint": supplied_item_fingerprint(supplied) if supplied else None,
                "stale_policy_decision": bool(stale_reasons),
                "stale_reasons": stale_reasons,
            }
        )
    return decisions


def policy_regions(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    regions = []
    for section in ("semantic_opening_regions", "gap_regions", "boundary_regions"):
        for region in inventory.get(section, {}).get("items", []):
            if is_policy_eligible(region):
                regions.append(region)
    return regions


def is_policy_eligible(region: dict[str, Any]) -> bool:
    return bool(region.get("requires_policy")) and region.get("classification") in POLICY_ELIGIBLE_CLASSIFICATIONS


def policy_item_id(source_region_id: str) -> str:
    for prefix in ("boundary_loop", "gap_or_opening_candidate", "semantic_opening"):
        if source_region_id.startswith(f"{prefix}_"):
            return source_region_id.replace(prefix, "opening", 1)
    return f"opening_{source_region_id}"


def stale_policy_reasons(packet: dict[str, Any], item: dict[str, Any], supplied: dict[str, Any]) -> list[str]:
    checks = [
        ("run_fingerprint", packet.get("run_fingerprint"), supplied.get("run_fingerprint")),
        ("packet_fingerprint", packet.get("packet_fingerprint"), supplied.get("packet_fingerprint")),
        ("item_fingerprint", item.get("item_fingerprint"), supplied_item_fingerprint(supplied)),
    ]
    reasons = []
    for name, expected, actual in checks:
        if not expected:
            continue
        if actual is None:
            reasons.append(f"missing_{name}")
        elif actual != expected:
            reasons.append(f"mismatched_{name}")
    return reasons


def supplied_item_fingerprint(supplied: dict[str, Any]) -> str | None:
    return supplied.get("item_fingerprint") or supplied.get("fingerprint")


def policy_decision_reason(supplied: dict[str, Any] | None, valid: bool, stale_reasons: list[str]) -> str:
    if not supplied:
        return "no policy decision supplied"
    if stale_reasons:
        return "stale_policy_decision: " + ", ".join(stale_reasons)
    if not valid:
        return "invalid policy decision"
    if supplied.get("status", "decided") != "decided":
        return supplied.get("reason", "policy decision pending")
    return supplied.get("reason", "")


def stable_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
