"""Topology-aware triangle self-intersection checks for arbitrary surface meshes."""

from __future__ import annotations

from typing import Any

import numpy as np

from deterministic_hole_fill import classify_triangle_pair_contact


def bounded_triangle_self_intersections(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    focus_face_ids: np.ndarray | list[int] | None = None,
    max_candidate_pairs: int = 2_000_000,
    max_reported_pairs: int | None = 200,
) -> dict[str, Any]:
    """Audit every broad-phase pair with the repository's robust contact classifier."""

    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
    from vtkmodules.vtkCommonDataModel import (
        vtkCellArray,
        vtkPolyData,
        vtkStaticCellLocator,
    )

    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.all(np.isfinite(points)):
        raise ValueError("points must be a finite (N, 3) array")
    if (
        faces.ndim != 2
        or faces.shape[1:] != (3,)
        or np.any((faces < 0) | (faces >= points.shape[0]))
    ):
        raise ValueError("faces must be a valid (M, 3) array")
    if max_candidate_pairs < 1:
        raise ValueError("max_candidate_pairs must be positive")
    if max_reported_pairs is not None and max_reported_pairs < 0:
        raise ValueError("max_reported_pairs must be nonnegative or None")
    if focus_face_ids is None:
        left_ids = np.arange(faces.shape[0], dtype=np.int64)
        scope = "all_faces"
    else:
        left_ids = np.unique(np.asarray(focus_face_ids, dtype=np.int64).reshape(-1))
        if np.any((left_ids < 0) | (left_ids >= faces.shape[0])):
            raise ValueError("focus_face_ids contains an out-of-range face ID")
        scope = "focused_faces"
    if faces.shape[0] == 0:
        return _result(scope, left_ids, 0, 0, 0, [], max_candidate_pairs)

    poly = _vtk_polydata(
        points,
        faces,
        vtkPoints,
        vtkCellArray,
        vtkPolyData,
        numpy_to_vtk,
        numpy_to_vtkIdTypeArray,
    )
    locator = vtkStaticCellLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()
    triangles = points[faces]
    bounds_min = triangles.min(axis=1)
    bounds_max = triangles.max(axis=1)
    tolerance = max(float(np.linalg.norm(np.ptp(points, axis=0))) * 1.0e-12, 1.0e-15)
    candidates = vtkIdList()
    tested = 0
    intersections = 0
    allowed_contacts = 0
    reported: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for left_value in left_ids:
        left = int(left_value)
        bounds = [
            float(bounds_min[left, 0] - tolerance),
            float(bounds_max[left, 0] + tolerance),
            float(bounds_min[left, 1] - tolerance),
            float(bounds_max[left, 1] + tolerance),
            float(bounds_min[left, 2] - tolerance),
            float(bounds_max[left, 2] + tolerance),
        ]
        candidates.Reset()
        locator.FindCellsWithinBounds(bounds, candidates)
        for candidate_index in range(candidates.GetNumberOfIds()):
            right = int(candidates.GetId(candidate_index))
            if right == left:
                continue
            pair = (min(left, right), max(left, right))
            if focus_face_ids is None:
                if right < left:
                    continue
            else:
                if pair in seen:
                    continue
                seen.add(pair)
            if not _aabb_overlap(
                bounds_min[left],
                bounds_max[left],
                bounds_min[right],
                bounds_max[right],
                tolerance,
            ):
                continue
            if tested >= max_candidate_pairs:
                return {
                    **_result(
                        scope,
                        left_ids,
                        tested,
                        intersections,
                        allowed_contacts,
                        reported,
                        max_candidate_pairs,
                    ),
                    "status": "incomplete_candidate_pair_limit_exceeded",
                    "passed": False,
                    "truncated": True,
                    "failure_reason": "self-intersection candidate-pair limit exceeded",
                }
            tested += 1
            contact = classify_triangle_pair_contact(
                points,
                faces[left],
                faces[right],
                geometric_tolerance=tolerance,
            )
            classification = contact.get("classification")
            if classification == "allowed_topological_contact":
                allowed_contacts += 1
            elif classification == "proper_intersection":
                intersections += 1
                if max_reported_pairs is None or len(reported) < max_reported_pairs:
                    reported.append([pair[0], pair[1]])
    return _result(
        scope,
        left_ids,
        tested,
        intersections,
        allowed_contacts,
        reported,
        max_candidate_pairs,
    )

def _vtk_polydata(
    points: np.ndarray,
    faces: np.ndarray,
    vtk_points_type: Any,
    vtk_cells_type: Any,
    vtk_poly_type: Any,
    numpy_to_vtk: Any,
    numpy_to_vtk_ids: Any,
) -> Any:
    vtk_points = vtk_points_type()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    packed = np.column_stack(
        (np.full(faces.shape[0], 3, dtype=np.int64), faces)
    ).ravel()
    cells = vtk_cells_type()
    cells.ImportLegacyFormat(numpy_to_vtk_ids(packed, deep=True))
    poly = vtk_poly_type()
    poly.SetPoints(vtk_points)
    poly.SetPolys(cells)
    return poly


def _aabb_overlap(
    left_min: np.ndarray,
    left_max: np.ndarray,
    right_min: np.ndarray,
    right_max: np.ndarray,
    tolerance: float,
) -> bool:
    return bool(
        np.all(left_max + tolerance >= right_min)
        and np.all(right_max + tolerance >= left_min)
    )


def _result(
    scope: str,
    focus: np.ndarray,
    tested: int,
    intersections: int,
    allowed: int,
    reported: list[list[int]],
    limit: int,
) -> dict[str, Any]:
    return {
        "method": "vtk_static_cell_locator_plus_robust_triangle_contact",
        "scope": scope,
        "focus_face_count": int(focus.size) if scope == "focused_faces" else None,
        "status": "computed",
        "passed": intersections == 0,
        "intersection_pairs": int(intersections),
        "candidate_pairs_tested": int(tested),
        "ignored_topological_contacts": int(allowed),
        "max_candidate_pairs": int(limit),
        "reported_pairs": reported,
        "truncated": intersections > len(reported),
        "adjacency_policy": "only contact confined to a shared topological feature is allowed",
    }
