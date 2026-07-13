from __future__ import annotations

from typing import Any, Sequence

import numpy as np


AI_DECISION_CODES = {
    "not_reviewed": 0,
    "remove_internal": 1,
    "keep_exterior": 2,
    "split_required": 3,
    "ambiguous": 4,
}
SELECTION_REASONS = {
    "not_selected": 0,
    "direct_first_hit": 1,
    "continuity_ring": 2,
    "ai_keep_exterior": 3,
}


def attach_component_visibility(
    descriptions: list[dict[str, Any]],
    component_labels: np.ndarray,
    face_first_hit_view_count: np.ndarray,
    face_first_hit_pixel_support: np.ndarray,
) -> list[dict[str, Any]]:
    labels, view_count, pixel_support = _evidence_arrays(
        component_labels,
        face_first_hit_view_count,
        face_first_hit_pixel_support,
    )
    rows: list[dict[str, Any]] = []
    for description in descriptions:
        component_id = int(description["component_id"])
        face_ids = np.flatnonzero(labels == component_id)
        visible = view_count[face_ids] > 0
        rows.append(
            {
                **description,
                "candidate_id": f"component_{component_id:06d}",
                "first_hit_face_count": int(np.count_nonzero(visible)),
                "first_hit_face_ratio": float(np.count_nonzero(visible) / max(face_ids.size, 1)),
                "first_hit_view_max": int(np.max(view_count[face_ids], initial=0)),
                "first_hit_pixel_support": int(np.sum(pixel_support[face_ids], dtype=np.int64)),
            }
        )
    return rows


def select_ai_candidates(
    component_rows: list[dict[str, Any]],
    *,
    max_candidates: int,
    min_face_count: int = 1,
) -> list[dict[str, Any]]:
    if max_candidates < 0 or min_face_count < 1:
        raise ValueError("candidate limits must be non-negative and min_face_count positive")
    candidates = [
        row
        for row in component_rows
        if int(row.get("first_hit_face_count", 0)) == 0
        and int(row.get("face_count", 0)) >= min_face_count
    ]
    ordered = sorted(
        candidates,
        key=lambda row: (
            -float(row.get("surface_area", 0.0)),
            -int(row.get("face_count", 0)),
            int(row["component_id"]),
        ),
    )
    return ordered[:max_candidates] if max_candidates else []


# @entry Select one open source shell from face evidence and guarded AI decisions.
def select_source_shell_faces(
    faces: np.ndarray,
    component_labels: np.ndarray,
    face_first_hit_view_count: np.ndarray,
    face_first_hit_pixel_support: np.ndarray,
    ai_decisions: Sequence[dict[str, Any]],
    *,
    min_first_hit_views: int = 1,
    min_first_hit_pixels: int = 1,
    ai_remove_confidence: float = 0.85,
    continuity_rings: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    triangles = np.asarray(faces, dtype=np.int64)
    labels, view_count, pixel_support = _evidence_arrays(
        component_labels,
        face_first_hit_view_count,
        face_first_hit_pixel_support,
    )
    if triangles.shape != (labels.size, 3):
        raise ValueError("faces and component labels have incompatible shapes")
    if min_first_hit_views < 1 or min_first_hit_pixels < 1:
        raise ValueError("first-hit thresholds must be positive")
    if not 0.0 <= ai_remove_confidence <= 1.0:
        raise ValueError("ai_remove_confidence must be in [0, 1]")
    if continuity_rings < 0:
        raise ValueError("continuity_rings must be non-negative")

    direct = (view_count >= min_first_hit_views) & (pixel_support >= min_first_hit_pixels)
    selection = direct.copy()
    reason = np.where(direct, SELECTION_REASONS["direct_first_hit"], 0).astype(np.uint8)
    ai_code = np.zeros(labels.size, dtype=np.uint8)
    removed_components: set[int] = set()
    protected_components: set[int] = set()
    ignored_removals: list[dict[str, Any]] = []
    available_components = set(int(value) for value in np.unique(labels))

    for decision in ai_decisions:
        component_id = component_id_from_candidate(str(decision["candidate_id"]))
        if component_id not in available_components:
            raise ValueError(f"AI decision references an unknown component: {component_id}")
        kind = str(decision["decision"])
        if kind not in AI_DECISION_CODES:
            raise ValueError(f"unsupported AI decision: {kind}")
        component_faces = labels == component_id
        ai_code[component_faces] = AI_DECISION_CODES[kind]
        confidence = float(decision["confidence"])
        if kind == "keep_exterior":
            protected_components.add(component_id)
            selection[component_faces] = True
            reason[component_faces] = SELECTION_REASONS["ai_keep_exterior"]
        elif kind == "remove_internal" and confidence >= ai_remove_confidence:
            if np.any(direct[component_faces]):
                ignored_removals.append(
                    {
                        "component_id": component_id,
                        "reason": "direct_first_hit_evidence_blocks_component_deletion",
                    }
                )
            else:
                removed_components.add(component_id)

    blocked = np.isin(labels, np.asarray(sorted(removed_components), dtype=np.int64))
    selection[blocked] = False
    reason[blocked] = SELECTION_REASONS["not_selected"]
    before_expansion = selection.copy()
    for _ in range(continuity_rings):
        expanded = expand_across_shared_edges(triangles, selection)
        selection |= expanded & ~blocked
    continuity = selection & ~before_expansion
    reason[continuity] = SELECTION_REASONS["continuity_ring"]

    return selection, reason, ai_code, {
        "method": "face_first_hit_plus_guarded_ai_component_decisions",
        "component_expansion_applied": False,
        "continuity_expansion": "shared_edge_bounded_rings",
        "continuity_rings": int(continuity_rings),
        "direct_first_hit_faces": int(np.count_nonzero(direct)),
        "continuity_faces": int(np.count_nonzero(continuity)),
        "selected_faces": int(np.count_nonzero(selection)),
        "removed_component_ids": sorted(removed_components),
        "protected_component_ids": sorted(protected_components),
        "ignored_ai_removals": ignored_removals,
        "thresholds": {
            "min_first_hit_views": int(min_first_hit_views),
            "min_first_hit_pixels": int(min_first_hit_pixels),
            "ai_remove_confidence": float(ai_remove_confidence),
        },
    }


def expand_across_shared_edges(faces: np.ndarray, selected: np.ndarray) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64)
    active = np.asarray(selected, dtype=bool)
    if active.shape != (triangles.shape[0],):
        raise ValueError("selected must contain one value per face")
    result = active.copy()
    if triangles.shape[0] == 0:
        return result
    face_ids = np.arange(triangles.shape[0], dtype=np.int64)
    edges = np.concatenate(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)])
    )
    edges.sort(axis=1)
    edge_faces = np.tile(face_ids, 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    ordered_edges = edges[order]
    ordered_faces = edge_faces[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(ordered_edges[1:] != ordered_edges[:-1], axis=1))]
    ends = np.r_[starts[1:], ordered_edges.shape[0]]
    for start, end in zip(starts, ends, strict=True):
        incident = ordered_faces[start:end]
        if incident.size > 1 and np.any(active[incident]):
            result[incident] = True
    return result


def component_id_from_candidate(candidate_id: str) -> int:
    prefix = "component_"
    if not candidate_id.startswith(prefix) or not candidate_id[len(prefix) :].isdigit():
        raise ValueError(f"invalid component candidate ID: {candidate_id!r}")
    return int(candidate_id[len(prefix) :])


def _evidence_arrays(
    component_labels: np.ndarray,
    face_first_hit_view_count: np.ndarray,
    face_first_hit_pixel_support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(component_labels, dtype=np.int64)
    view_count = np.asarray(face_first_hit_view_count)
    pixel_support = np.asarray(face_first_hit_pixel_support)
    if labels.ndim != 1:
        raise ValueError("component_labels must be one-dimensional")
    if view_count.shape != labels.shape or pixel_support.shape != labels.shape:
        raise ValueError("visibility evidence must contain one value per face")
    return labels, view_count, pixel_support
