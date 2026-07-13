from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from source_primary_patch_contract import (
    PatchCandidate,
    PatchDelta,
    RegionId,
    finalize_patch_candidate,
    rejected_patch_candidate,
)
from source_primary_patch_geometry import analyze_source_boundary, triangle_quality
from source_primary_patch_inputs import read_bounded_boundary_loop, validate_finite_config
from source_primary_patch_continuity import audit_boundary_normal_continuity
from source_primary_loop_zipper_geometry import triangulate_ordered_loop_bridge


@dataclass(frozen=True)
class LoopBridgeConfig:
    min_loop_normal_abs_dot: float = 0.25
    max_centroid_distance_ratio: float = 1.0
    max_correspondence_distance_ratio: float = 1.0
    max_boundary_vertices: int = 1024
    phase_candidates: int = 12
    min_boundary_normal_dot: float = 0.25
    max_aspect_ratio: float = 100.0
    max_intersection_candidate_pairs: int = 250_000


# @entry
def build_paired_loop_zipper_candidate(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_triangle_index: np.ndarray,
    first_boundary_loop: np.ndarray | list[int],
    second_boundary_loop: np.ndarray | list[int],
    *,
    first_region_id: RegionId,
    second_region_id: RegionId,
    first_oriented_normal: np.ndarray | None = None,
    second_oriented_normal: np.ndarray | None = None,
    config: LoopBridgeConfig | None = None,
) -> PatchCandidate:
    """Build one ordered, self-intersection-free annular source-boundary bridge."""

    cfg = config if config is not None else LoopBridgeConfig()
    if not isinstance(cfg, LoopBridgeConfig):
        return _config_rejection(
            source_points,
            source_faces,
            source_triangle_index,
            "config must be LoopBridgeConfig",
        )
    config_error = _validate_config(cfg)
    if config_error is not None:
        return _config_rejection(
            source_points, source_faces, source_triangle_index, config_error, cfg
        )
    first_raw, first_error = read_bounded_boundary_loop(
        first_boundary_loop, maximum_vertices=cfg.max_boundary_vertices
    )
    second_raw, second_error = read_bounded_boundary_loop(
        second_boundary_loop, maximum_vertices=cfg.max_boundary_vertices
    )
    if first_error or second_error:
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="paired_loop_zipper",
            failure_reason_codes=("paired_loop_zipper_boundary_invalid",),
            diagnostics={"first": first_error, "second": second_error},
        )
    first = analyze_source_boundary(
        source_points,
        source_faces,
        source_triangle_index,
        first_raw,
        region_id=first_region_id,
        boundary_role="paired_loop",
        expected_oriented_normal=first_oriented_normal,
    )
    if not first["success"]:
        return _analysis_rejection(
            source_points, source_faces, source_triangle_index, first, "first"
        )
    second = analyze_source_boundary(
        source_points,
        source_faces,
        source_triangle_index,
        second_raw,
        region_id=second_region_id,
        boundary_role="paired_loop",
        expected_oriented_normal=second_oriented_normal,
    )
    if not second["success"]:
        return _analysis_rejection(
            source_points, source_faces, source_triangle_index, second, "second"
        )
    shared = np.intersect1d(first["loop"], second["loop"])
    if shared.size:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            first,
            second,
            "paired_loop_zipper_boundaries_overlap",
            shared_vertex_ids=shared[:100].astype(int).tolist(),
        )
    points = np.asarray(source_points, dtype=np.float64)
    characteristic_length = max(
        float(first["frame"].footprint_diagonal),
        float(second["frame"].footprint_diagonal),
        first["frame"].geometric_tolerance,
    )
    centroid_distance = float(
        np.linalg.norm(points[first["loop"]].mean(axis=0) - points[second["loop"]].mean(axis=0))
    )
    distance_limit = cfg.max_centroid_distance_ratio * characteristic_length
    if centroid_distance > distance_limit:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            first,
            second,
            "paired_loop_zipper_centroid_distance_exceeded",
            centroid_distance=centroid_distance,
            threshold=distance_limit,
        )
    bridge = triangulate_ordered_loop_bridge(
        points,
        np.asarray(source_faces, dtype=np.int64),
        first,
        second,
        characteristic_length=characteristic_length,
        config=cfg,
    )
    if not bridge["success"]:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            first,
            second,
            bridge["failure_reason_codes"][0],
            bridge=bridge["diagnostics"],
        )
    appended_faces = bridge["faces"]
    quality = triangle_quality(points, appended_faces)
    continuity = audit_boundary_normal_continuity(
        points,
        np.asarray(source_faces, dtype=np.int64),
        appended_faces,
        (first["mapping"], second["mapping"]),
        cfg.min_boundary_normal_dot,
    )
    if quality["maximum_aspect_ratio"] > cfg.max_aspect_ratio or not continuity["passed"]:
        reasons = []
        if quality["maximum_aspect_ratio"] > cfg.max_aspect_ratio:
            reasons.append("paired_loop_zipper_aspect_ratio_exceeded")
        if not continuity["passed"]:
            reasons.append("paired_loop_zipper_boundary_normal_discontinuous")
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="paired_loop_zipper",
            failure_reason_codes=reasons,
            normal=_bridge_boundary_normal(first, second),
            curvature=_combined_curvature(first, second),
            boundary_mapping=(first["mapping"], second["mapping"]),
            diagnostics={
                "bridge": bridge["diagnostics"],
                "quality": quality,
                "continuity": continuity,
            },
        )
    delta = PatchDelta(
        appended_points=np.empty((0, 3), dtype=np.float64),
        appended_faces=appended_faces,
        point_provenance={
            "patch_method": np.empty(0, dtype="U32"),
            "region_id": np.empty(0, dtype="U128"),
        },
        face_provenance={
            "patch_method": np.full(appended_faces.shape[0], "paired_loop_zipper", dtype="U32"),
            "first_region_id": np.full(
                appended_faces.shape[0], first["mapping"].region_id, dtype="U128"
            ),
            "second_region_id": np.full(
                appended_faces.shape[0], second["mapping"].region_id, dtype="U128"
            ),
            "source_triangle_index": np.full(appended_faces.shape[0], -1, dtype=np.int64),
            "source_geometry_consumed": np.zeros(appended_faces.shape[0], dtype=np.uint8),
            "proxy_geometry_consumed": np.zeros(appended_faces.shape[0], dtype=np.uint8),
        },
    )
    return finalize_patch_candidate(
        source_points,
        source_faces,
        source_triangle_index,
        method="paired_loop_zipper",
        delta=delta,
        boundary_mapping=(first["mapping"], second["mapping"]),
        normal=_bridge_boundary_normal(first, second),
        curvature=_combined_curvature(first, second),
        diagnostics={
            "stage": "source_primary_paired_loop_zipper",
            "method": bridge["diagnostics"]["method"],
            "first_boundary": first["diagnostics"],
            "second_boundary": second["diagnostics"],
            "bridge": bridge["diagnostics"],
            "quality": quality,
            "boundary_continuity": continuity,
            "source_points_modified": False,
            "source_faces_modified": False,
            "boundary_edges_split": False,
        },
    )


def _combined_curvature(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    reliable = bool(
        first["curvature"].get("reliable")
        and second["curvature"].get("reliable")
    )
    return {
        "status": "computed" if reliable else "underdetermined",
        "method": "paired_boundary_source_one_ring",
        "reliable": reliable,
        "first_boundary": first["curvature"],
        "second_boundary": second["curvature"],
    }


def _analysis_rejection(points, faces, source_ids, analysis, role):
    return rejected_patch_candidate(
        points,
        faces,
        source_ids,
        method="paired_loop_zipper",
        failure_reason_codes=analysis["failure_reason_codes"],
        diagnostics={"boundary_role": role, **analysis["diagnostics"]},
    )


def _reject(points, faces, source_ids, first, second, code, **diagnostics):
    return rejected_patch_candidate(
        points,
        faces,
        source_ids,
        method="paired_loop_zipper",
        failure_reason_codes=(code,),
        normal=_bridge_boundary_normal(first, second),
        curvature=_combined_curvature(first, second),
        boundary_mapping=(first["mapping"], second["mapping"]),
        diagnostics=diagnostics,
    )


def _bridge_boundary_normal(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    first_normal = np.asarray(first["normal"]["oriented_normal"], dtype=np.float64)
    second_normal = np.asarray(second["normal"]["oriented_normal"], dtype=np.float64)
    if np.dot(first_normal, second_normal) < 0.0:
        second_normal = -second_normal
    value = first_normal + second_normal
    length = float(np.linalg.norm(value))
    value = value / length if length > 1e-30 else first_normal
    return {
        "status": "computed",
        "method": "paired_boundary_reference",
        "oriented_normal": value.tolist(),
        "orientation_reliable": True,
        "first_boundary": first["normal"],
        "second_boundary": second["normal"],
    }


def _config_rejection(points, faces, source_ids, message, config=None):
    diagnostics = {"message": message}
    if config is not None:
        diagnostics["config"] = asdict(config)
    return rejected_patch_candidate(
        points,
        faces,
        source_ids,
        method="paired_loop_zipper",
        failure_reason_codes=("paired_loop_zipper_config_invalid",),
        diagnostics=diagnostics,
    )


def _validate_config(config: LoopBridgeConfig) -> str | None:
    scalar_error = validate_finite_config(
        config,
        LoopBridgeConfig,
        integer_fields=frozenset(
            {
                "max_boundary_vertices",
                "phase_candidates",
                "max_intersection_candidate_pairs",
            }
        ),
    )
    if scalar_error is not None:
        return scalar_error
    if not 0.0 <= config.min_loop_normal_abs_dot <= 1.0:
        return "min_loop_normal_abs_dot must be within [0, 1]"
    if config.max_centroid_distance_ratio <= 0.0 or config.max_correspondence_distance_ratio <= 0.0:
        return "distance ratios must be positive"
    if min(
        config.max_boundary_vertices,
        config.phase_candidates,
        config.max_intersection_candidate_pairs,
    ) < 1:
        return "boundary, phase, and intersection limits must be positive"
    if config.max_boundary_vertices < 3:
        return "max_boundary_vertices must permit a triangle"
    if not -1.0 <= config.min_boundary_normal_dot <= 1.0:
        return "min_boundary_normal_dot must be within [-1, 1]"
    if config.max_aspect_ratio <= 1.0:
        return "max_aspect_ratio must be greater than one"
    return None
