from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def summarize_incident_outside_directions(
    face_ids: Sequence[int],
    face_areas: np.ndarray,
    directions: np.ndarray | None,
    minimum_dot: float,
) -> dict[str, Any]:
    ids = np.asarray(sorted(set(int(value) for value in face_ids)), dtype=np.int64)
    incident_count = int(ids.size)
    if directions is None:
        return _result(
            "external_direction_unavailable",
            incident_count,
            0,
            None,
            None,
            None,
            None,
            "none",
        )
    values = np.asarray(directions, dtype=np.float64)
    if values.shape == (3,):
        selected = np.repeat(values[None, :], incident_count, axis=0)
        method = "global_external_direction"
    elif values.ndim == 2 and values.shape[1] == 3:
        selected = values[ids]
        method = "incident_face_area_weighted_external_directions"
    else:
        return _result(
            "external_direction_invalid",
            incident_count,
            0,
            None,
            None,
            None,
            None,
            "invalid",
        )
    lengths = np.linalg.norm(selected, axis=1)
    evidenced = np.isfinite(lengths) & (lengths > 0.0)
    evidenced_count = int(np.count_nonzero(evidenced))
    coverage = evidenced_count / max(incident_count, 1)
    if evidenced_count == 0:
        return _result(
            "external_direction_incomplete",
            incident_count,
            evidenced_count,
            coverage,
            None,
            0.0,
            None,
            method,
        )
    unit = selected[evidenced] / lengths[evidenced, None]
    weights = np.asarray(face_areas, dtype=np.float64)[ids[evidenced]]
    vector = np.sum(weights[:, None] * unit, axis=0)
    magnitude = float(np.linalg.norm(vector))
    total_weight = float(weights.sum())
    resultant_ratio = magnitude / max(total_weight, 1.0e-30)
    if magnitude <= 0.0:
        return _result(
            "external_direction_conflict",
            incident_count,
            evidenced_count,
            coverage,
            None,
            resultant_ratio,
            None,
            method,
        )
    direction = vector / magnitude
    dots = unit @ direction
    minimum = float(dots.min(initial=1.0))
    negative_count = int(np.count_nonzero(dots < 0.0))
    tangent_count = int(np.count_nonzero(dots < minimum_dot))
    if evidenced_count != incident_count:
        status = "external_direction_incomplete"
    elif negative_count:
        status = "external_direction_multivalued"
    elif tangent_count or resultant_ratio < minimum_dot:
        status = "external_direction_tangent_or_conflicting"
    else:
        status = "computed"
    return {
        **_result(
            status,
            incident_count,
            evidenced_count,
            coverage,
            minimum,
            resultant_ratio,
            direction.tolist(),
            method,
        ),
        "negative_evidence_count": negative_count,
        "tangent_or_conflicting_evidence_count": tangent_count,
        "minimum_required_dot": float(minimum_dot),
    }


def _result(
    status: str,
    incident_count: int,
    evidenced_count: int,
    coverage: float | None,
    minimum_dot: float | None,
    resultant_ratio: float | None,
    direction: list[float] | None,
    method: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "direction": direction,
        "method": method,
        "incident_face_count": incident_count,
        "nonzero_evidence_count": evidenced_count,
        "coverage": coverage,
        "minimum_dot_to_resultant": minimum_dot,
        "resultant_ratio": resultant_ratio,
        "negative_evidence_count": 0,
        "tangent_or_conflicting_evidence_count": 0,
    }
