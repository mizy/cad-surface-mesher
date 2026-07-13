from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from source_primary_patch_contract import (
    PatchCandidate,
    RegionId,
    validate_patch_candidate,
)
from source_primary_quality import (
    PatchQualityLimits,
    audit_source_prefix,
)
from source_primary_transaction_data import (
    append_candidate_delta,
    audit_data_prefix,
    audit_transaction_prefix,
    copy_transaction_state,
    initialize_transaction_state,
    map_candidate_faces,
    transaction_state_sha256,
)
from source_primary_transaction_record import (
    audit_region_ids,
    build_failed_transaction_run,
    build_job_rollback_row,
    build_transaction_row,
)
from source_primary_transaction_preflight import validate_transaction_candidate
from source_primary_transaction_quality import audit_transaction_patch_quality
from source_primary_transaction_jobs import (
    PatchTransactionJob,
    build_inventory_coverage,
    build_job_candidate,
    candidate_region_ids,
    duplicated_region_ids,
    materialize_transaction_jobs,
    region_sort_key,
    region_tuple_key,
    resolve_fusion_region_id_map,
)


# @entry
def run_patch_transactions(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_triangle_index: np.ndarray,
    candidates: Iterable[PatchCandidate | PatchTransactionJob],
    *,
    source_cell_data: Mapping[str, np.ndarray] | None = None,
    source_point_data: Mapping[str, np.ndarray] | None = None,
    limits: PatchQualityLimits | None = None,
    required_region_ids: Iterable[RegionId] | None = None,
    fusion_region_id_by_region: Mapping[RegionId, int] | None = None,
) -> dict[str, Any]:
    """Generate and audit patches while keeping JSON IDs separate from VTP tokens.

    Opaque canonical region IDs require a complete one-to-one
    ``fusion_region_id_by_region`` mapping. Legacy non-negative numeric IDs use
    identity tokens when no mapping is supplied.
    """

    jobs, stream_failure = materialize_transaction_jobs(candidates)
    jobs.sort(
        key=lambda job: (region_tuple_key(job["region_ids"]), job["patch_method"])
    )
    assigned_region_ids = [region_id for job in jobs for region_id in job["region_ids"]]
    duplicated = duplicated_region_ids(assigned_region_ids)
    coverage = build_inventory_coverage(
        required_region_ids,
        assigned_region_ids,
        duplicated,
        stream_failure,
    )

    initialized = initialize_transaction_state(
        source_points,
        source_faces,
        source_triangle_index,
        source_cell_data,
        source_point_data,
    )
    baseline = initialized["state"]
    if not initialized["success"]:
        return build_failed_transaction_run(
            baseline,
            "source_baseline_invalid",
            initialized["failure_reason"],
            coverage,
            region_id_map=[],
        )
    fusion_map = resolve_fusion_region_id_map(
        [*coverage.get("required_region_ids", []), *assigned_region_ids],
        fusion_region_id_by_region,
    )
    if not fusion_map["success"]:
        return build_failed_transaction_run(
            baseline,
            str(fusion_map["reason_code"]),
            str(fusion_map["failure_reason"]),
            coverage,
            region_id_map=[],
            fusion_map_diagnostics=fusion_map,
        )
    fusion_tokens: Mapping[str, int] = fusion_map["mapping"]
    current = copy_transaction_state(baseline)

    transactions: list[dict[str, Any]] = []
    committed_region_ids: list[str] = []
    state_revision = 0
    for order, job in enumerate(jobs, start=1):
        job_fusion_ids = {
            fusion_tokens[region_id]
            for region_id in job["region_ids"]
            if region_id in fusion_tokens
        }
        fusion_region_id = next(iter(job_fusion_ids)) if len(job_fusion_ids) == 1 else None
        if len(job["region_ids"]) != 1 or fusion_region_id is None:
            transactions.append(
                build_job_rollback_row(
                    job,
                    order,
                    tuple(committed_region_ids),
                    state_revision,
                    "patch_transaction_requires_single_composite_region_id",
                    (
                        "one append-only patch transaction must own one canonical composite "
                        "region and one numeric fusion token"
                    ),
                    current,
                    fusion_region_id=None,
                )
            )
            continue
        if set(job["region_ids"]).intersection(duplicated):
            transactions.append(
                build_job_rollback_row(
                    job,
                    order,
                    tuple(committed_region_ids),
                    state_revision,
                    "duplicate_region_transaction",
                    f"region is assigned more than once: {sorted(set(job['region_ids']).intersection(duplicated))}",
                    current,
                    fusion_region_id=fusion_region_id,
                )
            )
            continue
        candidate, generation_failure = build_job_candidate(job)
        if candidate is None:
            transactions.append(
                build_job_rollback_row(
                    job,
                    order,
                    tuple(committed_region_ids),
                    state_revision,
                    "patch_candidate_generation_failed",
                    generation_failure,
                    current,
                    fusion_region_id=fusion_region_id,
                )
            )
            continue
        result = apply_patch_transaction(
            baseline,
            current,
            candidate,
            declared_region_ids=job["region_ids"],
            component_id=job["component_id"],
            dispatch_evidence_paths=job["evidence_paths"],
            transaction_order=order,
            retained_region_ids=tuple(committed_region_ids),
            state_revision=state_revision,
            limits=limits,
            fusion_region_id=fusion_region_id,
        )
        transactions.append(result["transaction"])
        if result["committed"]:
            current = result["state"]
            committed_region_ids.extend(job["region_ids"])
            state_revision += 1

    if stream_failure is not None:
        transactions.append(
            build_job_rollback_row(
                {
                    "region_ids": (),
                    "patch_method": "candidate_stream",
                    "component_id": None,
                    "evidence_paths": (),
                },
                len(transactions) + 1,
                tuple(committed_region_ids),
                state_revision,
                "patch_candidate_stream_failed",
                stream_failure,
                current,
                fusion_region_id=None,
            )
        )

    rejected = [row for row in transactions if row["transaction_status"] != "committed"]
    committed = [
        row for row in transactions if row["transaction_status"] == "committed"
    ]
    final_source_data = audit_data_prefix(baseline, current)
    return {
        "status": "completed" if not rejected else "completed_with_local_rollbacks",
        "points": current["points"],
        "faces": current["faces"],
        "point_data": current["point_data"],
        "cell_data": current["cell_data"],
        "transactions": transactions,
        "source_baseline": {
            "point_count": int(baseline["points"].shape[0]),
            "face_count": int(baseline["faces"].shape[0]),
            "source_triangle_index_count": int(
                baseline["cell_data"]["source_triangle_index"].shape[0]
            ),
            "state_sha256": transaction_state_sha256(baseline),
        },
        "source_data_prefix": final_source_data,
        "inventory_coverage": coverage,
        "region_id_map": fusion_map["rows"],
        "summary": {
            "candidate_transactions": len(transactions),
            "committed_transactions": len(committed),
            "rolled_back_transactions": len(rejected),
            "committed_region_ids": sorted(
                set(committed_region_ids), key=region_sort_key
            ),
            "rolled_back_region_ids": sorted(
                {
                    region_id
                    for row in rejected
                    for region_id in row.get("region_ids", [])
                },
                key=region_sort_key,
            ),
            "required_region_ids": coverage["required_region_ids"],
            "inventory_coverage_passed": coverage["passed"],
            "added_vertices": int(
                current["points"].shape[0] - baseline["points"].shape[0]
            ),
            "added_faces": int(current["faces"].shape[0] - baseline["faces"].shape[0]),
            "repair_mask_area": float(
                sum(row["delta"]["repair_mask_area"] for row in committed)
            ),
            "rollback_isolation": "failed transaction leaves all earlier committed regions intact",
            "final_state_sha256": transaction_state_sha256(current),
        },
        "reason_codes": list(
            dict.fromkeys(
                code
                for row in rejected
                for code in row.get("rollback", {}).get("reason_codes", [])
            )
        ),
    }


def apply_patch_transaction(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    candidate: PatchCandidate,
    *,
    declared_region_ids: tuple[str, ...] | None = None,
    component_id: str | int | None = None,
    dispatch_evidence_paths: tuple[str, ...] = (),
    transaction_order: int,
    retained_region_ids: tuple[str, ...],
    state_revision: int = 0,
    limits: PatchQualityLimits | None = None,
    fusion_region_id: int,
) -> dict[str, Any]:
    """Commit one canonical PatchCandidate or return current unchanged."""

    region_ids = declared_region_ids or candidate_region_ids(candidate)
    before_hash = transaction_state_sha256(current)
    try:
        contract_errors = validate_patch_candidate(
            baseline["points"],
            baseline["faces"],
            baseline["cell_data"]["source_triangle_index"],
            candidate,
        )
        preflight = validate_transaction_candidate(
            candidate, contract_errors, region_ids
        )
        if preflight:
            return _rollback(
                current,
                candidate,
                transaction_order,
                retained_region_ids,
                state_revision,
                preflight,
                {
                    "status": "not_computed_candidate_rejected",
                    "passed": False,
                    "candidate_contract_errors": contract_errors,
                },
                region_ids=region_ids,
                component_id=component_id,
                dispatch_evidence_paths=dispatch_evidence_paths,
                before_hash=before_hash,
                fusion_region_id=fusion_region_id,
            )
        mapped_faces = map_candidate_faces(
            candidate,
            source_point_count=baseline["points"].shape[0],
            current_point_count=current["points"].shape[0],
        )
        quality, orientation_evidence = audit_transaction_patch_quality(
            baseline,
            current,
            candidate,
            mapped_faces,
            limits,
        )
        if quality is None:
            reason = str(orientation_evidence["reason_code"])
            return _rollback(
                current,
                candidate,
                transaction_order,
                retained_region_ids,
                state_revision,
                [reason],
                {
                    "status": "not_computed_outside_evidence_invalid",
                    "passed": False,
                    "reason_codes": [reason],
                    "orientation_evidence": orientation_evidence,
                },
                region_ids=region_ids,
                component_id=component_id,
                dispatch_evidence_paths=dispatch_evidence_paths,
                before_hash=before_hash,
                fusion_region_id=fusion_region_id,
            )
        if not quality["passed"]:
            return _rollback(
                current,
                candidate,
                transaction_order,
                retained_region_ids,
                state_revision,
                quality["reason_codes"],
                quality,
                region_ids=region_ids,
                component_id=component_id,
                dispatch_evidence_paths=dispatch_evidence_paths,
                before_hash=before_hash,
                fusion_region_id=fusion_region_id,
            )
        trial = append_candidate_delta(
            current,
            candidate,
            mapped_faces,
            region_ids=region_ids,
            fusion_region_id=fusion_region_id,
        )
        source_lock = audit_source_prefix(
            baseline["points"],
            baseline["faces"],
            baseline["cell_data"]["source_triangle_index"],
            trial["points"],
            trial["faces"],
            trial["cell_data"]["source_triangle_index"],
        )
        data_lock = audit_data_prefix(baseline, trial)
        retained_lock = audit_transaction_prefix(current, trial)
        if (
            not source_lock["passed"]
            or not data_lock["passed"]
            or not retained_lock["passed"]
        ):
            codes = [
                *source_lock["reason_codes"],
                *data_lock["reason_codes"],
                *retained_lock["reason_codes"],
            ]
            quality = {
                **quality,
                "passed": False,
                "reason_codes": list(dict.fromkeys(codes)),
                "source_prefix": source_lock,
                "source_data_prefix": data_lock,
                "retained_state_prefix": retained_lock,
            }
            return _rollback(
                current,
                candidate,
                transaction_order,
                retained_region_ids,
                state_revision,
                quality["reason_codes"],
                quality,
                region_ids=region_ids,
                component_id=component_id,
                dispatch_evidence_paths=dispatch_evidence_paths,
                before_hash=before_hash,
                fusion_region_id=fusion_region_id,
            )
        quality["source_prefix"] = source_lock
        quality["source_data_prefix"] = data_lock
        quality["retained_state_prefix"] = retained_lock
        after_hash = transaction_state_sha256(trial)
        row = build_transaction_row(
            candidate,
            transaction_order,
            "committed",
            retained_region_ids,
            state_revision,
            quality,
            [],
            region_ids=region_ids,
            component_id=component_id,
            dispatch_evidence_paths=dispatch_evidence_paths,
            state_sha256_before=before_hash,
            state_sha256_after=after_hash,
            mapped_faces=mapped_faces,
            committed_points=trial["points"][current["points"].shape[0] :],
            fusion_region_id=fusion_region_id,
        )
        row["state_revision_after"] = state_revision + 1
        row["retained_committed_region_ids_after"] = [
            *audit_region_ids(retained_region_ids),
            *audit_region_ids(region_ids),
        ]
        return {"committed": True, "state": trial, "transaction": row}
    except Exception as exc:  # One region must never terminate or undo the run.
        return _rollback(
            current,
            candidate,
            transaction_order,
            retained_region_ids,
            state_revision,
            ["patch_transaction_exception"],
            {
                "status": "check_failed",
                "passed": False,
                "failure_reason": f"{type(exc).__name__}: {exc}",
            },
            region_ids=region_ids,
            component_id=component_id,
            dispatch_evidence_paths=dispatch_evidence_paths,
            before_hash=before_hash,
            fusion_region_id=fusion_region_id,
        )


def _rollback(
    current: Mapping[str, Any],
    candidate: PatchCandidate,
    order: int,
    retained: tuple[str, ...],
    state_revision: int,
    reason_codes: list[str],
    quality: Mapping[str, Any],
    *,
    region_ids: tuple[str, ...],
    component_id: str | int | None,
    dispatch_evidence_paths: tuple[str, ...],
    before_hash: str,
    fusion_region_id: int,
) -> dict[str, Any]:
    after_hash = transaction_state_sha256(current)
    row = build_transaction_row(
        candidate,
        order,
        "rolled_back",
        retained,
        state_revision,
        quality,
        reason_codes,
        region_ids=region_ids,
        component_id=component_id,
        dispatch_evidence_paths=dispatch_evidence_paths,
        state_sha256_before=before_hash,
        state_sha256_after=after_hash,
        fusion_region_id=fusion_region_id,
    )
    row["state_revision_after"] = state_revision
    row["retained_committed_region_ids_after"] = audit_region_ids(retained)
    row["rollback"]["state_bitwise_unchanged"] = before_hash == after_hash
    if before_hash != after_hash:
        row["rollback"]["reason_codes"].append("transaction_rollback_state_changed")
    return {"committed": False, "state": current, "transaction": row}
