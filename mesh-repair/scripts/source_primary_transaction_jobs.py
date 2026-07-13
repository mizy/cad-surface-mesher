from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from source_primary_patch_contract import PatchCandidate, RegionId, normalize_region_id
from source_primary_transaction_data import validate_fusion_region_id


@dataclass(frozen=True)
class PatchTransactionJob:
    """One independently generated region patch with stable dispatch identity."""

    region_ids: tuple[RegionId, ...]
    patch_method: str
    build_candidate: Callable[[], PatchCandidate]
    component_id: str | int | None = None
    evidence_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(
            sorted(
                (normalize_region_id(value) for value in self.region_ids),
                key=region_sort_key,
            )
        )
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError(
                "a transaction job must own one or more distinct region IDs"
            )
        object.__setattr__(self, "region_ids", normalized)
        object.__setattr__(self, "patch_method", str(self.patch_method))
        object.__setattr__(
            self, "evidence_paths", tuple(str(path) for path in self.evidence_paths)
        )


# @entry
def resolve_fusion_region_id_map(
    region_ids: Iterable[RegionId],
    supplied: Mapping[RegionId, int] | None,
) -> dict[str, Any]:
    """Bind canonical region IDs to unique non-negative VTP integer tokens."""

    try:
        expected = {normalize_region_id(value) for value in region_ids}
    except (TypeError, ValueError) as exc:
        return _fusion_map_failure("fusion_region_id_map_region_id_invalid", str(exc))
    mapping: dict[str, int]
    if supplied is None:
        if any(not value.isdigit() for value in expected):
            return _fusion_map_failure(
                "fusion_region_id_map_required_for_opaque_region_ids",
                "opaque canonical region IDs require an explicit numeric fusion mapping",
            )
        try:
            mapping = {
                value: validate_fusion_region_id(int(value)) for value in expected
            }
        except ValueError as exc:
            return _fusion_map_failure("fusion_region_id_map_token_invalid", str(exc))
    else:
        mapping = {}
        try:
            for raw_region_id, raw_token in supplied.items():
                region_id = normalize_region_id(raw_region_id)
                if region_id in mapping:
                    return _fusion_map_failure(
                        "fusion_region_id_map_canonical_key_collision",
                        f"multiple mapping keys normalize to canonical region {region_id!r}",
                    )
                mapping[region_id] = validate_fusion_region_id(raw_token)
        except (AttributeError, TypeError, ValueError) as exc:
            return _fusion_map_failure("fusion_region_id_map_token_invalid", str(exc))
        missing = sorted(expected.difference(mapping), key=region_sort_key)
        unexpected = sorted(set(mapping).difference(expected), key=region_sort_key)
        if missing or unexpected:
            return _fusion_map_failure(
                "fusion_region_id_map_coverage_invalid",
                "fusion region mapping must exactly cover the canonical inventory",
                missing_region_ids=missing,
                unexpected_region_ids=unexpected,
            )
    if len(set(mapping.values())) != len(mapping):
        return _fusion_map_failure(
            "fusion_region_id_map_token_collision",
            "every canonical region must map to a distinct fusion_region_id",
        )
    ordered_ids = sorted(mapping, key=region_sort_key)
    return {
        "success": True,
        "mapping": {region_id: mapping[region_id] for region_id in ordered_ids},
        "rows": [
            {"region_id": region_id, "fusion_region_id": mapping[region_id]}
            for region_id in ordered_ids
        ],
        "reason_code": None,
        "failure_reason": None,
    }


def materialize_transaction_jobs(
    values: Iterable[PatchCandidate | PatchTransactionJob],
) -> tuple[list[dict[str, Any]], str | None]:
    jobs: list[dict[str, Any]] = []
    iterator = iter(values)
    while True:
        try:
            value = next(iterator)
        except StopIteration:
            return jobs, None
        except Exception as exc:
            return jobs, f"{type(exc).__name__}: {exc}"
        if isinstance(value, PatchTransactionJob):
            jobs.append(
                {
                    "region_ids": tuple(value.region_ids),
                    "patch_method": value.patch_method,
                    "build_candidate": value.build_candidate,
                    "candidate": None,
                    "component_id": value.component_id,
                    "evidence_paths": value.evidence_paths,
                }
            )
        elif isinstance(value, PatchCandidate):
            jobs.append(
                {
                    "region_ids": candidate_region_ids(value),
                    "patch_method": value.method,
                    "build_candidate": None,
                    "candidate": value,
                    "component_id": value.diagnostics.get("component_id"),
                    "evidence_paths": tuple(
                        value.diagnostics.get("evidence_paths", ())
                    ),
                }
            )
        else:
            return jobs, f"unsupported transaction item: {type(value).__name__}"


def build_job_candidate(
    job: Mapping[str, Any],
) -> tuple[PatchCandidate | None, str | None]:
    try:
        candidate = job.get("candidate")
        if candidate is None:
            candidate = job["build_candidate"]()
        if not isinstance(candidate, PatchCandidate):
            raise TypeError("candidate builder must return PatchCandidate")
        if candidate.method != job["patch_method"]:
            raise ValueError("candidate method does not match dispatch method")
        return candidate, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def build_inventory_coverage(
    required_values: Iterable[RegionId] | None,
    assigned_values: list[str],
    duplicated: list[str],
    stream_failure: str | None,
) -> dict[str, Any]:
    required_supplied = required_values is not None
    try:
        raw_required = [] if required_values is None else list(required_values)
        required = [normalize_region_id(value) for value in raw_required]
    except Exception as exc:
        return {
            "status": "invalid_inventory_region_ids",
            "passed": False,
            "coverage_source": "explicit_complete_inventory_required_region_ids",
            "required_region_ids": [],
            "assigned_region_ids": sorted(set(assigned_values), key=region_sort_key),
            "missing_region_ids": [],
            "duplicate_region_ids": duplicated,
            "unexpected_region_ids": [],
            "required_region_id_duplicates": [],
            "candidate_stream_failure": stream_failure,
            "reason_codes": ["repair_required_region_inventory_invalid"],
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
    required_duplicates = duplicated_region_ids(required)
    required_set = set(required)
    assigned_set = set(assigned_values)
    missing = sorted(required_set - assigned_set, key=region_sort_key)
    unexpected = (
        sorted(assigned_set - required_set, key=region_sort_key)
        if required_supplied
        else []
    )
    reasons = []
    if not required_supplied:
        reasons.append("repair_required_region_inventory_not_supplied")
    if required_duplicates:
        reasons.append("duplicate_required_region_id")
    if duplicated:
        reasons.append("duplicate_region_transaction")
    if missing:
        reasons.append("repair_required_region_unassigned")
    if unexpected:
        reasons.append("transaction_region_not_in_inventory")
    if stream_failure:
        reasons.append("patch_candidate_stream_failed")
    return {
        "status": "computed" if required_supplied else "not_computed_inventory_missing",
        "passed": not reasons,
        "coverage_source": "explicit_complete_inventory_required_region_ids",
        "required_region_ids": sorted(required_set, key=region_sort_key),
        "assigned_region_ids": sorted(assigned_set, key=region_sort_key),
        "missing_region_ids": missing,
        "duplicate_region_ids": duplicated,
        "unexpected_region_ids": unexpected,
        "required_region_id_duplicates": required_duplicates,
        "candidate_stream_failure": stream_failure,
        "reason_codes": reasons,
    }


def candidate_region_ids(candidate: PatchCandidate) -> tuple[str, ...]:
    mapped = sorted(
        {
            normalize_region_id(mapping.region_id)
            for mapping in candidate.boundary_mapping
        },
        key=region_sort_key,
    )
    if mapped:
        return tuple(mapped)
    diagnostic_ids = candidate.diagnostics.get("region_ids")
    if diagnostic_ids:
        return tuple(
            sorted(
                {normalize_region_id(value) for value in diagnostic_ids},
                key=region_sort_key,
            )
        )
    diagnostic_id = candidate.diagnostics.get("region_id")
    return (normalize_region_id(diagnostic_id),) if diagnostic_id is not None else ()


def duplicated_region_ids(region_ids: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for region_id in region_ids:
        counts[region_id] = counts.get(region_id, 0) + 1
    return sorted(
        (key for key, count in counts.items() if count > 1), key=region_sort_key
    )


def region_sort_key(value: str | int) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.lstrip("-").isdigit() else (1, text)


def region_tuple_key(values: tuple[str, ...]) -> tuple[tuple[int, int | str], ...]:
    return tuple(region_sort_key(value) for value in values)


def _fusion_map_failure(
    reason_code: str,
    failure_reason: str,
    **diagnostics: Any,
) -> dict[str, Any]:
    return {
        "success": False,
        "mapping": {},
        "rows": [],
        "reason_code": reason_code,
        "failure_reason": failure_reason,
        **diagnostics,
    }
