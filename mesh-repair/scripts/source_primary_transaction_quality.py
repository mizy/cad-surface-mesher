from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from source_primary_expected_normals import derive_expected_patch_normals
from source_primary_patch_contract import PatchCandidate
from source_primary_quality import PatchQualityLimits, audit_source_primary_patch


def audit_transaction_patch_quality(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    candidate: PatchCandidate,
    mapped_faces: np.ndarray,
    limits: PatchQualityLimits | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    loops = tuple(
        np.asarray(item.source_vertex_ids, dtype=np.int64)
        for item in candidate.boundary_mapping
    )
    orientation_evidence = derive_expected_patch_normals(
        baseline["points"],
        baseline["faces"],
        mapped_faces,
        loops,
        baseline["cell_data"].get("external_direction"),
    )
    if not orientation_evidence["success"]:
        return None, orientation_evidence
    expected_normals = np.asarray(
        orientation_evidence["expected_face_normals"], dtype=np.float64
    )
    quality = audit_source_primary_patch(
        current["points"],
        current["faces"],
        candidate.delta.appended_points,
        mapped_faces,
        loops,
        expected_face_normals=expected_normals,
        limits=limits,
    )
    quality["independent_orientation_evidence"] = orientation_evidence
    return quality, orientation_evidence
