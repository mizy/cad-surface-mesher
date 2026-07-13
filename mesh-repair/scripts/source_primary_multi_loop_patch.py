from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from source_primary_multi_loop_geometry import (
    build_multi_loop_common_plane,
    resolve_multi_loop_nesting,
    triangulate_multi_loop_annulus,
)
from source_primary_patch_contract import (
    PatchCandidate,
    PatchDelta,
    RegionId,
    finalize_patch_candidate,
    rejected_patch_candidate,
)
from source_primary_patch_geometry import analyze_source_boundary
from source_primary_patch_inputs import (
    read_bounded_boundary_loop,
    validate_finite_config,
)
from source_primary_quality import audit_source_primary_patch


@dataclass(frozen=True)
class MultiLoopPatchConfig:
    max_boundary_vertices: int = 2048
    max_total_boundary_vertices: int = 4096
    max_plane_deviation_ratio: float = 1.0e-8
    min_loop_normal_dot: float = 0.95


# @entry
def build_multi_loop_patch_candidate(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_triangle_index: np.ndarray,
    boundary_loops: Sequence[np.ndarray | list[int]],
    *,
    region_id: RegionId,
    oriented_normal: np.ndarray | None = None,
    config: MultiLoopPatchConfig | None = None,
) -> PatchCandidate:
    """Fill one planar outer-minus-inner region without changing either source ring."""

    cfg = config if config is not None else MultiLoopPatchConfig()
    config_error = _validate_config(cfg)
    if config_error is not None:
        return _reject_without_analysis(
            source_points,
            source_faces,
            source_triangle_index,
            "multi_loop_patch_config_invalid",
            message=config_error,
            config=asdict(cfg) if isinstance(cfg, MultiLoopPatchConfig) else None,
        )
    try:
        raw_loops = list(boundary_loops)
    except (TypeError, ValueError) as exc:
        return _reject_without_analysis(
            source_points,
            source_faces,
            source_triangle_index,
            "multi_loop_boundary_invalid",
            message=f"{type(exc).__name__}: {exc}",
        )
    if len(raw_loops) != 2:
        return _reject_without_analysis(
            source_points,
            source_faces,
            source_triangle_index,
            "multi_loop_requires_exactly_two_nested_loops",
            loop_count=len(raw_loops),
        )

    parsed_loops = []
    for index, value in enumerate(raw_loops):
        loop, error = read_bounded_boundary_loop(
            value,
            maximum_vertices=cfg.max_boundary_vertices,
        )
        if error is not None or loop is None:
            return _reject_without_analysis(
                source_points,
                source_faces,
                source_triangle_index,
                "multi_loop_boundary_invalid",
                boundary_index=index,
                message=error,
            )
        parsed_loops.append(loop)
    total_vertices = sum(int(loop.size) for loop in parsed_loops)
    if total_vertices > cfg.max_total_boundary_vertices:
        return _reject_without_analysis(
            source_points,
            source_faces,
            source_triangle_index,
            "multi_loop_boundary_vertex_limit_exceeded",
            actual=total_vertices,
            threshold=cfg.max_total_boundary_vertices,
        )

    analyses = []
    for index, loop in enumerate(parsed_loops):
        analysis = analyze_source_boundary(
            source_points,
            source_faces,
            source_triangle_index,
            loop,
            region_id=region_id,
            boundary_role="paired_loop",
            expected_oriented_normal=oriented_normal,
        )
        if not analysis["success"]:
            return _reject_without_analysis(
                source_points,
                source_faces,
                source_triangle_index,
                str(analysis["failure_reason_codes"][0]),
                boundary_index=index,
                boundary_analysis=analysis.get("diagnostics", {}),
            )
        analyses.append(analysis)

    if np.intersect1d(analyses[0]["loop"], analyses[1]["loop"]).size:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            analyses,
            "multi_loop_boundaries_share_source_vertices",
        )
    source_normals = np.asarray(
        [analysis["normal"]["oriented_normal"] for analysis in analyses],
        dtype=np.float64,
    )
    normal_dot = float(np.dot(source_normals[0], source_normals[1]))
    if normal_dot < cfg.min_loop_normal_dot:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            analyses,
            "multi_loop_boundary_normal_conflict",
            actual=normal_dot,
            threshold=cfg.min_loop_normal_dot,
        )

    points = np.asarray(source_points, dtype=np.float64)
    plane = build_multi_loop_common_plane(
        points,
        analyses,
        source_normals,
        cfg.max_plane_deviation_ratio,
    )
    if not plane["success"]:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            analyses,
            str(plane["reason_code"]),
            plane=plane["diagnostics"],
        )
    relationship = resolve_multi_loop_nesting(points, analyses, plane)
    if not relationship["success"]:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            analyses,
            str(relationship["reason_code"]),
            relationship=relationship["diagnostics"],
        )
    outer_index = int(relationship["outer_index"])
    inner_index = 1 - outer_index
    ordered_analyses = [analyses[outer_index], analyses[inner_index]]
    loop_uv = relationship["loop_uv"]
    triangulation = triangulate_multi_loop_annulus(
        ordered_analyses,
        loop_uv[outer_index],
        loop_uv[inner_index],
        float(plane["tolerance"]),
    )
    if not triangulation["success"]:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            ordered_analyses,
            str(triangulation["reason_code"]),
            triangulation=triangulation["diagnostics"],
        )

    appended_faces = triangulation["faces"]
    mappings = tuple(analysis["mapping"] for analysis in ordered_analyses)
    normal = _paired_normal(ordered_analyses)
    expected_normals = np.repeat(
        np.asarray(normal["oriented_normal"], dtype=np.float64)[None, :],
        appended_faces.shape[0],
        axis=0,
    )
    quality = audit_source_primary_patch(
        points,
        np.asarray(source_faces, dtype=np.int64),
        np.empty((0, 3), dtype=np.float64),
        appended_faces,
        tuple(
            np.asarray(mapping.source_vertex_ids, dtype=np.int64)
            for mapping in mappings
        ),
        expected_face_normals=expected_normals,
    )
    if not quality["passed"]:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            ordered_analyses,
            *(str(code) for code in quality["reason_codes"]),
            triangulation=triangulation["diagnostics"],
            quality=quality,
        )

    curvature = _paired_curvature(ordered_analyses)
    delta = PatchDelta(
        appended_points=np.empty((0, 3), dtype=np.float64),
        appended_faces=appended_faces,
        point_provenance={
            "patch_method": np.empty(0, dtype="U32"),
            "region_id": np.empty(0, dtype="U128"),
        },
        face_provenance={
            "patch_method": np.full(
                appended_faces.shape[0], "paired_loop_zipper", dtype="U32"
            ),
            "first_region_id": np.full(
                appended_faces.shape[0], mappings[0].region_id, dtype="U128"
            ),
            "second_region_id": np.full(
                appended_faces.shape[0], mappings[1].region_id, dtype="U128"
            ),
            "source_triangle_index": np.full(
                appended_faces.shape[0], -1, dtype=np.int64
            ),
            "source_geometry_consumed": np.zeros(
                appended_faces.shape[0], dtype=np.uint8
            ),
            "proxy_geometry_consumed": np.zeros(
                appended_faces.shape[0], dtype=np.uint8
            ),
        },
    )
    return finalize_patch_candidate(
        source_points,
        source_faces,
        source_triangle_index,
        method="paired_loop_zipper",
        delta=delta,
        boundary_mapping=mappings,
        normal=normal,
        curvature=curvature,
        diagnostics={
            "stage": "source_primary_multi_loop_patch",
            "method": "source_fixed_planar_delaunay_annulus",
            "region_id": str(mappings[0].region_id),
            "outer_boundary": ordered_analyses[0]["diagnostics"],
            "inner_island_boundary": ordered_analyses[1]["diagnostics"],
            "plane": plane["diagnostics"],
            "relationship": relationship["diagnostics"],
            "triangulation": triangulation["diagnostics"],
            "quality": quality,
            "source_points_modified": False,
            "source_faces_modified": False,
            "boundary_edges_split": False,
            "inner_island_filled": False,
        },
    )


def _paired_normal(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    normals = np.asarray(
        [analysis["normal"]["oriented_normal"] for analysis in analyses],
        dtype=np.float64,
    )
    if float(np.dot(normals[0], normals[1])) < 0.0:
        normals[1] *= -1.0
    weighted = normals.sum(axis=0)
    length = float(np.linalg.norm(weighted))
    reference = weighted / length if length > 1.0e-30 else normals[0]
    return {
        "status": "computed",
        "method": "paired_immutable_source_boundary_normal_field",
        "oriented_normal": reference.tolist(),
        "orientation_reliable": bool(np.all(np.isfinite(normals)) and length > 0.0),
        "first_boundary": analyses[0]["normal"],
        "second_boundary": analyses[1]["normal"],
    }


def _paired_curvature(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    reliable = bool(
        analyses[0]["curvature"].get("reliable")
        and analyses[1]["curvature"].get("reliable")
    )
    return {
        "status": "computed" if reliable else "underdetermined",
        "method": "paired_boundary_source_one_ring",
        "reliable": reliable,
        "first_boundary": analyses[0]["curvature"],
        "second_boundary": analyses[1]["curvature"],
    }


def _validate_config(config: Any) -> str | None:
    error = validate_finite_config(
        config,
        MultiLoopPatchConfig,
        integer_fields=frozenset(
            {"max_boundary_vertices", "max_total_boundary_vertices"}
        ),
    )
    if error is not None:
        return error
    if config.max_boundary_vertices < 3 or config.max_total_boundary_vertices < 6:
        return "boundary vertex limits are too small"
    if not 0.0 < config.max_plane_deviation_ratio <= 1.0e-4:
        return "max_plane_deviation_ratio must be within (0, 1e-4]"
    if not 0.8 <= config.min_loop_normal_dot <= 1.0:
        return "min_loop_normal_dot must be within [0.8, 1]"
    return None


def _reject_without_analysis(points, faces, source_ids, code, **diagnostics):
    return rejected_patch_candidate(
        points,
        faces,
        source_ids,
        method="paired_loop_zipper",
        failure_reason_codes=(code,),
        diagnostics=diagnostics,
    )


def _reject(points, faces, source_ids, analyses, *codes, **diagnostics):
    valid_codes = tuple(code for code in codes if code) or (
        "multi_loop_patch_rejected",
    )
    return rejected_patch_candidate(
        points,
        faces,
        source_ids,
        method="paired_loop_zipper",
        failure_reason_codes=valid_codes,
        normal={
            "status": "computed",
            "method": "paired_boundary_reference",
            "orientation_reliable": True,
            "first_boundary": analyses[0]["normal"],
            "second_boundary": analyses[1]["normal"],
        },
        curvature=_paired_curvature(analyses),
        boundary_mapping=tuple(analysis["mapping"] for analysis in analyses),
        diagnostics=diagnostics,
    )
