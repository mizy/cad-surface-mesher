from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np


_HARD_MAXIMUM_ENTRIES = 2_000_000
_HARD_MAXIMUM_FACE_IDS = 250_000
_HARD_MAXIMUM_CANDIDATES_PER_CELL = 8_192


def build_proxy_face_index(
    face_uv_min: np.ndarray,
    face_uv_max: np.ndarray,
    face_ids: np.ndarray,
    footprint_min: np.ndarray,
    footprint_max: np.ndarray,
    *,
    maximum_entries: int,
    maximum_candidates_per_cell: int,
) -> dict[str, Any]:
    """Build a bounded uniform 2D index for normal-line proxy queries."""

    count = int(face_ids.size)
    if count > _HARD_MAXIMUM_FACE_IDS:
        return {
            "success": False,
            "reason": "proxy spatial-index face limit exceeded",
            "face_count": count,
        }
    if (
        isinstance(maximum_entries, (bool, np.bool_))
        or not isinstance(maximum_entries, Integral)
        or maximum_entries < 1
        or maximum_entries > _HARD_MAXIMUM_ENTRIES
    ):
        return {
            "success": False,
            "reason": "proxy spatial-index entry limit is invalid",
            "maximum_entries": maximum_entries,
        }
    if (
        isinstance(maximum_candidates_per_cell, (bool, np.bool_))
        or not isinstance(maximum_candidates_per_cell, Integral)
        or maximum_candidates_per_cell < 1
        or maximum_candidates_per_cell > _HARD_MAXIMUM_CANDIDATES_PER_CELL
    ):
        return {
            "success": False,
            "reason": "proxy spatial-index cell candidate limit is invalid",
        }
    resolution = min(256, max(4, int(np.ceil(np.sqrt(max(count, 1) / 8.0)))))
    span = footprint_max - footprint_min
    if np.any(span <= 0.0) or not np.all(np.isfinite(span)):
        return {"success": False, "reason": "proxy footprint bounds are degenerate"}
    cells: dict[int, list[int]] = {}
    entry_count = 0
    for face_id_value in face_ids:
        face_id = int(face_id_value)
        clipped_min = np.maximum(face_uv_min[face_id], footprint_min)
        clipped_max = np.minimum(face_uv_max[face_id], footprint_max)
        if np.any(clipped_min > clipped_max):
            continue
        lower = _cell_coordinates(
            clipped_min,
            footprint_min,
            span,
            resolution,
        )
        upper = _cell_coordinates(
            clipped_max,
            footprint_min,
            span,
            resolution,
        )
        face_entry_count = int(upper[0] - lower[0] + 1) * int(
            upper[1] - lower[1] + 1
        )
        if face_entry_count > maximum_entries - entry_count:
            return {
                "success": False,
                "reason": "proxy spatial-index entry limit exceeded",
                "entry_count": entry_count,
                "next_face_entry_count": face_entry_count,
            }
        for row in range(int(lower[1]), int(upper[1]) + 1):
            for column in range(int(lower[0]), int(upper[0]) + 1):
                cell_key = row * resolution + column
                cell = cells.setdefault(cell_key, [])
                if len(cell) >= maximum_candidates_per_cell:
                    return {
                        "success": False,
                        "reason": "proxy spatial-index cell candidate limit exceeded",
                        "cell_key": cell_key,
                        "candidate_count": len(cell) + 1,
                    }
                cell.append(face_id)
                entry_count += 1
    return {
        "success": True,
        "cells": {
            key: np.asarray(sorted(set(values)), dtype=np.int64)
            for key, values in cells.items()
        },
        "origin": footprint_min.copy(),
        "span": span,
        "resolution": resolution,
        "entry_count": entry_count,
    }


def query_proxy_face_index(
    index: dict[str, Any], point: np.ndarray, *, maximum_candidates: int
) -> dict[str, Any]:
    if (
        isinstance(maximum_candidates, (bool, np.bool_))
        or not isinstance(maximum_candidates, Integral)
        or maximum_candidates < 1
        or maximum_candidates > _HARD_MAXIMUM_CANDIDATES_PER_CELL
    ):
        return {
            "success": False,
            "reason": "proxy query candidate face limit is invalid",
        }
    coordinates = _cell_coordinates(
        point,
        index["origin"],
        index["span"],
        int(index["resolution"]),
    )
    key = int(coordinates[1]) * int(index["resolution"]) + int(coordinates[0])
    candidates = index["cells"].get(key, np.empty(0, dtype=np.int64))
    if candidates.size > maximum_candidates:
        return {
            "success": False,
            "reason": "proxy query candidate face limit exceeded",
            "candidate_count": int(candidates.size),
            "maximum_candidates": int(maximum_candidates),
        }
    return {"success": True, "face_ids": candidates}


def label_proxy_face_components(
    faces: np.ndarray, face_ids: np.ndarray
) -> np.ndarray:
    """Label edge-connected components within the local footprint candidates."""

    local_faces = np.asarray(faces, dtype=np.int64)[face_ids]
    parent = np.arange(local_faces.shape[0], dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    edge_owner: dict[tuple[int, int], int] = {}
    for local_id, face in enumerate(local_faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            first_id, second_id = int(first), int(second)
            edge = (
                (first_id, second_id)
                if first_id < second_id
                else (second_id, first_id)
            )
            previous = edge_owner.get(edge)
            if previous is None:
                edge_owner[edge] = local_id
                continue
            left, right = find(local_id), find(previous)
            if left != right:
                parent[max(left, right)] = min(left, right)
    roots = np.asarray([find(index) for index in range(local_faces.shape[0])], dtype=np.int64)
    unique = {root: index for index, root in enumerate(sorted(set(roots.tolist())))}
    return np.asarray([unique[int(root)] for root in roots], dtype=np.int64)


def _cell_coordinates(
    point: np.ndarray,
    origin: np.ndarray,
    span: np.ndarray,
    resolution: int,
) -> np.ndarray:
    normalized = (np.asarray(point, dtype=np.float64) - origin) / span
    return np.clip((normalized * resolution).astype(np.int64), 0, resolution - 1)
