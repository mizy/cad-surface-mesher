from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from source_primary_patch_contract import PatchCandidate, normalize_region_id
from source_primary_serialization import to_audit_value, to_json_value
from source_primary_transaction_data import transaction_state_sha256


def build_transaction_row(
    candidate: PatchCandidate,
    order: int,
    status: str,
    retained: tuple[str, ...],
    state_revision: int,
    quality: Mapping[str, Any],
    reason_codes: Sequence[str],
    *,
    region_ids: tuple[str, ...],
    component_id: str | int | None,
    dispatch_evidence_paths: tuple[str, ...],
    state_sha256_before: str,
    state_sha256_after: str,
    mapped_faces: np.ndarray | None = None,
    committed_points: np.ndarray | None = None,
    fusion_region_id: int,
) -> dict[str, Any]:
    return {
        "region_id": _audit_region_id(region_ids[0]) if region_ids else None,
        "region_ids": [_audit_region_id(value) for value in region_ids],
        "fusion_region_id": int(fusion_region_id),
        "component_id": component_id
        if component_id is not None
        else candidate.diagnostics.get("component_id"),
        "patch_method": candidate.method,
        "transaction_order": int(order),
        "transaction_status": status,
        "state_revision_before": state_revision,
        "state_sha256_before": state_sha256_before,
        "state_sha256_after": state_sha256_after,
        "retained_committed_region_ids_before": audit_region_ids(retained),
        "boundary_loops": [
            _boundary_row(mapping) for mapping in candidate.boundary_mapping
        ],
        "descriptor": {
            "normal": to_json_value(candidate.normal),
            "curvature": to_json_value(candidate.curvature),
            "repairer_diagnostics": to_json_value(candidate.diagnostics),
            "computed_local_geometry": to_json_value(quality.get("local_geometry", {})),
            "computed_curvature": to_json_value(
                quality.get("gates", {})
                .get("boundary_curvature_continuity", {})
                .get("actual")
            ),
            "planarity": to_json_value(_planarity_descriptor(candidate, quality)),
        },
        "provenance": {
            "source": to_audit_value(candidate.source_provenance),
            "proxy": to_json_value(candidate.proxy_provenance),
            "appended_points": to_audit_value(candidate.delta.point_provenance),
            "appended_faces": to_audit_value(candidate.delta.face_provenance),
        },
        "quality": to_json_value(quality),
        "delta": {
            "new_vertex_count": int(candidate.delta.appended_points.shape[0]),
            "new_face_count": int(candidate.delta.appended_faces.shape[0]),
            "repair_mask_area": float(quality.get("repair_mask_area", 0.0))
            if status == "committed"
            else 0.0,
            "appended_points_sha256": array_sha256(candidate.delta.appended_points),
            "committed_points_sha256": (
                array_sha256(committed_points) if committed_points is not None else None
            ),
            "candidate_faces_sha256": array_sha256(candidate.delta.appended_faces),
            "mapped_faces_sha256": array_sha256(mapped_faces)
            if mapped_faces is not None
            else None,
        },
        "rollback": {
            "applied": status != "committed",
            "scope": "current_regions_only",
            "reason_codes": list(dict.fromkeys(str(code) for code in reason_codes)),
        },
        "evidence_paths": sorted(
            {
                *dispatch_evidence_paths,
                *(
                    str(path)
                    for path in candidate.diagnostics.get("evidence_paths", [])
                ),
            }
        ),
    }


def build_job_rollback_row(
    job: Mapping[str, Any],
    order: int,
    retained: tuple[str, ...],
    state_revision: int,
    reason_code: str,
    failure_reason: str | None,
    current: Mapping[str, Any],
    *,
    fusion_region_id: int | None,
) -> dict[str, Any]:
    state_hash = transaction_state_sha256(current)
    region_ids = [_audit_region_id(value) for value in job.get("region_ids", ())]
    return {
        "region_id": region_ids[0] if region_ids else None,
        "region_ids": region_ids,
        "fusion_region_id": int(fusion_region_id)
        if fusion_region_id is not None
        else None,
        "component_id": job.get("component_id"),
        "patch_method": str(job.get("patch_method") or "unknown"),
        "transaction_order": int(order),
        "transaction_status": "rolled_back",
        "state_revision_before": state_revision,
        "state_revision_after": state_revision,
        "state_sha256_before": state_hash,
        "state_sha256_after": state_hash,
        "retained_committed_region_ids_before": audit_region_ids(retained),
        "retained_committed_region_ids_after": audit_region_ids(retained),
        "boundary_loops": [],
        "descriptor": {"status": "candidate_not_available"},
        "provenance": {
            "source": {},
            "proxy": {},
            "appended_points": {},
            "appended_faces": {},
        },
        "quality": {
            "status": "not_computed_candidate_generation_failed",
            "passed": False,
            "failure_reason": failure_reason,
            "reason_codes": [reason_code],
            "gates": {},
        },
        "delta": {
            "new_vertex_count": 0,
            "new_face_count": 0,
            "repair_mask_area": 0.0,
            "appended_points_sha256": None,
            "committed_points_sha256": None,
            "candidate_faces_sha256": None,
            "mapped_faces_sha256": None,
        },
        "rollback": {
            "applied": True,
            "scope": "current_regions_only",
            "reason_codes": [reason_code],
            "state_bitwise_unchanged": True,
        },
        "evidence_paths": sorted(str(path) for path in job.get("evidence_paths", ())),
    }


def build_failed_transaction_run(
    state: Mapping[str, Any],
    code: str,
    message: str | None,
    coverage: Mapping[str, Any],
    *,
    region_id_map: list[dict[str, Any]],
    fusion_map_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "failed_precondition",
        "points": state["points"],
        "faces": state["faces"],
        "point_data": state["point_data"],
        "cell_data": state["cell_data"],
        "transactions": [],
        "inventory_coverage": dict(coverage),
        "region_id_map": region_id_map,
        "source_baseline": {
            "point_count": int(state["points"].shape[0]),
            "face_count": int(state["faces"].shape[0]),
        },
        "summary": {
            "candidate_transactions": 0,
            "committed_transactions": 0,
            "rolled_back_transactions": 0,
            "inventory_coverage_passed": False,
        },
        "reason_codes": [code],
        "failure_reason": message,
        "fusion_region_id_map_validation": dict(fusion_map_diagnostics or {}),
    }


def array_sha256(values: np.ndarray | None) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    if array.dtype.hasobject:
        digest.update(
            json.dumps(
                array.tolist(), sort_keys=True, ensure_ascii=False, default=str
            ).encode("utf-8")
        )
    else:
        digest.update(array.tobytes())
    return digest.hexdigest()


def _boundary_row(mapping: Any) -> dict[str, Any]:
    payload = {
        "region_id": _audit_region_id(normalize_region_id(mapping.region_id)),
        "source_vertex_ids": list(mapping.source_vertex_ids),
        "incident_source_face_ids": list(mapping.source_edge_face_ids),
        "source_triangle_indices": list(mapping.source_triangle_indices),
        "edge_count": len(mapping.source_vertex_ids),
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _planarity_descriptor(
    candidate: PatchCandidate,
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    local_loops = quality.get("local_geometry", {}).get("boundary_loops", [])
    diagnostics = candidate.diagnostics
    boundary = (
        diagnostics.get("boundary", {}) if isinstance(diagnostics, Mapping) else {}
    )
    return {
        "repairer": to_json_value(
            diagnostics.get("planarity", boundary.get("planarity", boundary))
            if isinstance(diagnostics, Mapping)
            else {}
        ),
        "quality_boundary_loops": [
            {
                key: row.get(key)
                for key in (
                    "planarity_ratio",
                    "maximum_plane_deviation",
                    "pca_normal",
                    "newell_normal",
                )
            }
            for row in local_loops
            if isinstance(row, Mapping)
        ],
    }


def _audit_region_id(value: str) -> str | int:
    """Keep legacy integer IDs typed while preserving opaque inventory IDs."""

    return int(value) if value.lstrip("-").isdigit() else value


def audit_region_ids(values: Sequence[str]) -> list[str | int]:
    return [_audit_region_id(value) for value in values]
