from __future__ import annotations

import numpy as np

from source_primary_patch_contract import PatchCandidate
from source_primary_transaction_jobs import candidate_region_ids


def validate_transaction_candidate(
    candidate: PatchCandidate,
    contract_errors: list[str],
    declared_region_ids: tuple[str, ...],
) -> list[str]:
    reasons = (
        list(candidate.failure_reason_codes) if candidate.status != "candidate" else []
    )
    if contract_errors:
        reasons.append("patch_candidate_contract_invalid")
    if candidate.method == "slit_weld":
        reasons.append("slit_weld_requires_source_connectivity_edit")
    mapped_region_ids = candidate_region_ids(candidate)
    if not declared_region_ids:
        reasons.append("patch_region_id_missing")
    if candidate.status == "candidate" and mapped_region_ids != declared_region_ids:
        reasons.append("patch_candidate_region_assignment_mismatch")
    if any(mapping.closed is not True for mapping in candidate.boundary_mapping):
        reasons.append("open_boundary_mapping_not_append_only_patchable")
    reasons.extend(_validate_proxy_reference(candidate))
    return list(dict.fromkeys(str(code) for code in reasons))


def _validate_proxy_reference(candidate: PatchCandidate) -> list[str]:
    proxy = candidate.proxy_provenance
    if not proxy.get("used"):
        return []
    reasons = []
    if candidate.method != "curved_conformal_patch":
        reasons.append("patch_proxy_reference_method_forbidden")
    if (
        proxy.get("role") != "local_normal_depth_reference_only"
        or proxy.get("geometry_consumed") is not False
        or proxy.get("footprint_policy") != "strict_polygon_interior_no_dilation"
        or proxy.get("sampling_method")
        != "oriented_normal_line_triangle_barycentric_depth"
    ):
        reasons.append("patch_proxy_reference_not_footprint_safe")
    expected_normal = np.asarray(
        candidate.normal.get("oriented_normal", []), dtype=np.float64
    )
    sampled_normal = np.asarray(proxy.get("oriented_normal", []), dtype=np.float64)
    if (
        expected_normal.shape != (3,)
        or sampled_normal.shape != (3,)
        or not np.allclose(expected_normal, sampled_normal, rtol=1e-8, atol=1e-10)
    ):
        reasons.append("patch_proxy_query_direction_mismatch")
    requested = proxy.get("requested_sample_count")
    selected = proxy.get("selected_sample_count")
    coverage = proxy.get("coverage")
    arrays_complete = all(
        key in proxy
        for key in (
            "sample_indices",
            "sample_uv",
            "signed_depth",
            "proxy_triangle_index",
            "proxy_component_id",
            "barycentric",
            "normal_dot",
            "missing_sample_indices",
        )
    )
    if (
        not isinstance(requested, int)
        or not isinstance(selected, int)
        or requested < 1
        or selected < 1
        or selected > requested
        or not isinstance(coverage, (float, int))
        or not np.isfinite(float(coverage))
        or float(coverage) < 0.5
        or not arrays_complete
    ):
        reasons.append("patch_proxy_sampling_provenance_incomplete")
    return reasons
