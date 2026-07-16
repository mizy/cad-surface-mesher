from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from mesh_io import read_surface, triangle_faces
from mesh_metrics import mesh_report
from source_projected_closure import REJECTION_BITS, rollback_reported_self_intersections


def exact_self_intersection_report(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    first_only: bool = False,
    max_reported_pairs: int = 100_000,
) -> dict[str, Any]:
    """Certify triangle self-intersections with libigl's exact CGAL predicates."""
    vertices = np.ascontiguousarray(points, dtype=np.float64)
    triangles = np.ascontiguousarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if max_reported_pairs <= 0:
        raise ValueError("max_reported_pairs must be positive")

    try:
        from igl.copyleft import cgal
    except ImportError as exc:
        return {
            "method": "libigl_cgal_exact_self_intersection",
            "status": "dependency_missing",
            "passed": False,
            "intersection_pairs": None,
            "reported_pairs": [],
            "failure_reason": f"libigl with CGAL support is required: {exc}",
        }

    result = cgal.remesh_self_intersections(
        vertices,
        triangles,
        detect_only=True,
        first_only=bool(first_only),
        stitch_all=False,
        slow_and_more_precise_rounding=True,
        cutoff=max_reported_pairs,
    )
    pairs = np.asarray(result[2], dtype=np.int64).reshape(-1, 2)
    reported = pairs[:max_reported_pairs]
    return {
        "method": "libigl_cgal_exact_self_intersection",
        "kernel": "CGAL exact predicates via igl.copyleft.cgal.remesh_self_intersections",
        "status": "computed",
        "passed": pairs.shape[0] == 0,
        "intersection_pairs": int(pairs.shape[0]),
        "reported_pairs": reported.tolist(),
        "reported_pair_limit": int(max_reported_pairs),
        "reported_pairs_truncated": bool(reported.shape[0] < pairs.shape[0]),
        "slow_and_more_precise_rounding": True,
        "failure_reason": None if pairs.shape[0] == 0 else "exact intersecting face pairs found",
    }


def repair_closure_proxy_self_intersections(
    points: np.ndarray,
    faces: np.ndarray,
    voxel_pitch: float,
    *,
    max_certification_rounds: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    del voxel_pitch, max_certification_rounds
    baseline = np.asarray(points, dtype=np.float64).copy()
    certification = exact_self_intersection_report(baseline, faces)
    return baseline, {
        "method": "libigl_cgal_exact_proxy_certification_no_geometry_mutation",
        "initial_self_intersection": certification,
        "rounds": [],
        "final_self_intersection": certification,
        "passed": certification.get("status") == "computed" and certification.get("passed") is True,
        "connectivity_changed": False,
        "moved_vertices": 0,
        "displacement": _percentiles(np.zeros(0, dtype=np.float64)),
        "failure_reason": (
            None
            if certification.get("passed") is True
            else "implicit closure proxy must be regenerated rather than heuristically nudged"
        ),
    }


def component_self_intersection_report(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    largest_component_pair_limit: int = 100_000_000,
    other_component_pair_limit: int = 10_000_000,
) -> dict[str, Any]:
    del largest_component_pair_limit, other_component_pair_limit
    return exact_self_intersection_report(points, faces)


def repair_projected_self_intersections(
    points: np.ndarray,
    proxy_points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, dict[str, np.ndarray]],
    *,
    certified_proxy_report: dict[str, Any] | None = None,
    max_rounds: int = 16,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    current = np.asarray(points, dtype=np.float64).copy()
    current_provenance = provenance
    rounds = []
    for round_index in range(1, max_rounds + 1):
        intersections = projected_delta_self_intersection_report(
            current,
            faces,
            current_provenance,
            certified_proxy_report,
        )
        if intersections.get("passed") or not intersections.get("reported_pairs"):
            intersections["projection_rollback_rounds"] = rounds
            return current, current_provenance, intersections
        current, current_provenance, rollback = rollback_reported_self_intersections(
            current,
            proxy_points,
            faces,
            current_provenance,
            intersections,
            # Direct face rollback removes the bulk of projection collisions
            # with minimal loss of source detail.  If a transition keeps
            # folding after three certification rounds, also restore its
            # one-ring neighborhood so a new crease is not created directly
            # beside the previous intersection.
            vertex_rings=0 if round_index <= 3 else 1,
        )
        rounds.append({"round": round_index, **rollback})
        if rollback["rolled_back_vertices"] == 0:
            intersections["projection_rollback_rounds"] = rounds
            intersections["failure_reason"] = (
                "reported intersections remain entirely within the closure proxy fallback geometry"
            )
            return current, current_provenance, intersections
    final = projected_delta_self_intersection_report(
        current,
        faces,
        current_provenance,
        certified_proxy_report,
    )
    final["projection_rollback_rounds"] = rounds
    if not final.get("passed"):
        final["failure_reason"] = "self-intersections remain after bounded projection rollback"
    return current, current_provenance, final


def projected_delta_self_intersection_report(
    points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, dict[str, np.ndarray]],
    certified_proxy_report: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline_passed = bool(
        certified_proxy_report
        and certified_proxy_report.get("status") == "computed"
        and certified_proxy_report.get("passed") is True
    )
    if not baseline_passed:
        return {
            "method": "certified_proxy_plus_projected_delta_intersection",
            "status": "baseline_proxy_not_certified",
            "passed": False,
            "reported_pairs": [],
            "failure_reason": "closure proxy must be globally intersection-free before delta validation",
        }
    applied = np.asarray(provenance["point_data"]["source_projection_applied"]).astype(bool)
    focus = np.flatnonzero(np.any(applied[faces], axis=1))
    if focus.size == 0:
        return {
            "method": "libigl_cgal_exact_projected_candidate_certification",
            "status": "computed",
            "passed": True,
            "intersection_pairs": 0,
            "reported_pairs": [],
            "changed_face_count": 0,
            "proof": "candidate equals the globally certified closure proxy",
        }
    row = exact_self_intersection_report(points, faces)
    return {
        **row,
        "method": "libigl_cgal_exact_projected_candidate_certification",
        "changed_face_count": int(focus.size),
        "scan_scope": "all_faces",
        "baseline_proxy_certified": True,
        "proof": (
            "complete candidate mesh certified with libigl/CGAL exact predicates; "
            "reported face pairs drive projection rollback"
        ),
    }


def chunked_self_intersection_report(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    focus_chunk_size: int = 25_000,
    max_candidate_pairs_per_chunk: int = 2_000_000,
) -> dict[str, Any]:
    del max_candidate_pairs_per_chunk
    report = exact_self_intersection_report(points, faces, max_reported_pairs=100_000)
    return {
        **report,
        "compatibility_entrypoint": "chunked_self_intersection_report",
        "requested_focus_chunk_size": int(focus_chunk_size),
        "scope": "all_faces_exact_cgal_certification",
    }


def roundtrip_validation(
    path: Path,
    expected_points: np.ndarray,
    expected_faces: np.ndarray,
) -> dict[str, Any]:
    try:
        raw = _raw_polydata(path)
        raw_points = np.asarray(raw.points, dtype=np.float64)
        raw_faces = triangle_faces(raw).astype(np.int64, copy=False)
        normalized = read_surface(path)
        normalized_points = np.asarray(normalized.points, dtype=np.float64)
        normalized_faces = triangle_faces(normalized).astype(np.int64, copy=False)
        raw_metrics = mesh_report(raw_points, raw_faces)
        normalized_metrics = mesh_report(normalized_points, normalized_faces)
    except Exception as exc:
        return {
            "status": "failed",
            "passed": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
    raw_exact = (
        raw_points.shape == expected_points.shape
        and np.array_equal(raw_faces, expected_faces)
        and np.allclose(raw_points, expected_points, rtol=0.0, atol=1e-12)
    )
    raw_passed = watertight_metrics(raw_metrics)
    normalized_passed = watertight_metrics(normalized_metrics)
    topology_survived_clean = (
        normalized_points.shape[0] == expected_points.shape[0]
        and normalized_faces.shape == expected_faces.shape
        and normalized_passed
    )
    passed = raw_exact and raw_passed and topology_survived_clean
    return {
        "method": "raw_vtp_and_normalized_read_surface_roundtrip",
        "status": "computed",
        "passed": passed,
        "raw_geometry_exact": raw_exact,
        "raw_watertight": raw_passed,
        "normalized_watertight": normalized_passed,
        "normalized_point_count_preserved": normalized_points.shape[0] == expected_points.shape[0],
        "normalized_face_count_preserved": normalized_faces.shape == expected_faces.shape,
        "raw": topology_summary(raw_metrics),
        "normalized": topology_summary(normalized_metrics),
        "failure_reason": None if passed else "saved mesh changed topology or manifoldness during roundtrip",
    }


def update_projection_report(
    report: dict[str, Any],
    points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, dict[str, np.ndarray]],
    self_intersection: dict[str, Any],
    post_write: dict[str, Any],
) -> dict[str, Any]:
    result = dict(report)
    applied = np.asarray(provenance["point_data"]["source_projection_applied"]).astype(bool)
    fallback = np.asarray(provenance["point_data"]["source_projection_fallback"]).astype(bool)
    rejection = np.asarray(provenance["point_data"]["source_projection_rejection_mask"])
    projection = dict(result.get("projection", {}))
    projection.update({
        "vertices": int(applied.size),
        "source_projected": int(np.count_nonzero(applied)),
        "explicit_proxy_fallback": int(np.count_nonzero(fallback)),
        "unclassified": int(np.count_nonzero(~(applied | fallback))),
        "fallback_without_reason": int(np.count_nonzero(fallback & (rejection == 0))),
        "source_projected_ratio": float(np.count_nonzero(applied) / max(applied.size, 1)),
        "rejection_counts": {name: int(np.count_nonzero(rejection & bit)) for name, bit in REJECTION_BITS.items()},
    })
    result["projection"] = projection
    result["quality"] = {**result.get("quality", {}), "projected": mesh_report(points, faces)}
    comparisons = dict(result.get("comparisons", {}))
    comparisons["projection_mapping"] = {
        "status": "computed",
        "source_projected": projection["source_projected"],
        "explicit_proxy_fallback": projection["explicit_proxy_fallback"],
        "unclassified": projection["unclassified"],
        "fallback_without_reason": projection["fallback_without_reason"],
    }
    comparisons["self_intersection"] = self_intersection
    comparisons["post_write_validation"] = post_write
    result["comparisons"] = comparisons
    result["self_intersection"] = self_intersection
    result["post_write_validation"] = post_write
    return result


def _raw_polydata(path: Path) -> pv.PolyData:
    mesh = pv.read(path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    return mesh.triangulate()


def watertight_metrics(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["topology"]["boundary_edges"] == 0
        and metrics["topology"]["non_manifold_edges"] == 0
        and metrics["topology"]["non_manifold_vertices"] == 0
        and metrics["topology"]["inconsistent_winding_edges"] == 0
        and metrics["quality"]["degenerate_faces"] == 0
        and metrics["volume"]["reliable"]
    )


def topology_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "points": metrics["points"],
        "triangles": metrics["triangles"],
        "boundary_edges": metrics["topology"]["boundary_edges"],
        "non_manifold_edges": metrics["topology"]["non_manifold_edges"],
        "non_manifold_vertices": metrics["topology"]["non_manifold_vertices"],
        "inconsistent_winding_edges": metrics["topology"]["inconsistent_winding_edges"],
        "degenerate_faces": metrics["quality"]["degenerate_faces"],
        "volume_reliable": metrics["volume"]["reliable"],
    }


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"min": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }
