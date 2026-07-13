from __future__ import annotations

from dataclasses import asdict, dataclass
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
from source_primary_patch_parameterization import world_from_local
from source_primary_slit_triangulation import triangulate_source_fixed_slit_bridge
from source_primary_surface_lift import (
    evaluate_thin_plate_surface,
    fit_thin_plate_surface,
)


@dataclass(frozen=True)
class SlitPatchConfig:
    min_length_width_ratio: float = 3.0
    weld_width_length_ratio: float = 1.0e-6
    max_projection_nonplanarity_ratio: float = 0.35
    max_boundary_vertices: int = 512
    max_appended_points: int = 4096
    interior_spacing_width_ratio: float = 0.75
    min_boundary_normal_dot: float = 0.25
    max_aspect_ratio: float = 25.0


# @entry
def build_slit_patch_candidate(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_triangle_index: np.ndarray,
    boundary_loop: np.ndarray | list[int],
    *,
    region_id: RegionId,
    oriented_normal: np.ndarray | None = None,
    config: SlitPatchConfig | None = None,
) -> PatchCandidate:
    """Bridge a finite-width slit; reject welds that require source rewrites."""

    cfg = config if config is not None else SlitPatchConfig()
    if not isinstance(cfg, SlitPatchConfig):
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="slit_bridge",
            failure_reason_codes=("slit_patch_config_invalid",),
            diagnostics={"message": "config must be SlitPatchConfig"},
        )
    config_error = _validate_config(cfg)
    if config_error is not None:
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="slit_bridge",
            failure_reason_codes=("slit_patch_config_invalid",),
            diagnostics={"message": config_error, "config": asdict(cfg)},
        )
    raw_loop, loop_error = read_bounded_boundary_loop(
        boundary_loop, maximum_vertices=cfg.max_boundary_vertices
    )
    if loop_error is not None:
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="slit_bridge",
            failure_reason_codes=("slit_boundary_invalid",),
            diagnostics={
                "message": loop_error,
                "max_boundary_vertices": int(cfg.max_boundary_vertices),
            },
        )
    analysis = analyze_source_boundary(
        source_points,
        source_faces,
        source_triangle_index,
        raw_loop,
        region_id=region_id,
        expected_oriented_normal=oriented_normal,
    )
    if not analysis["success"]:
        collapse_codes = {
            "boundary_newell_normal_degenerate",
            "boundary_loop_zero_length_edge",
            "boundary_perimeter_degenerate",
            "boundary_footprint_degenerate",
            "boundary_orientation_frame_conflict",
            "boundary_projection_self_intersects",
            "boundary_local_frame_degenerate",
        }
        requires_weld = bool(collapse_codes.intersection(analysis["failure_reason_codes"]))
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="slit_weld" if requires_weld else "slit_bridge",
            failure_reason_codes=(
                "slit_weld_requires_source_connectivity_edit"
                if requires_weld
                else analysis["failure_reason_codes"][0],
            ),
            diagnostics={
                **analysis["diagnostics"],
                "source_mutation_forbidden": True,
            },
        )
    try:
        shape = _slit_shape(analysis["frame"].boundary_uv)
    except np.linalg.LinAlgError as exc:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            analysis,
            "slit_shape_svd_failed",
            message=str(exc),
        )
    if analysis["diagnostics"]["planarity_ratio"] > cfg.max_projection_nonplanarity_ratio:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            analysis,
            "slit_projection_domain_folded",
            planarity_ratio=analysis["diagnostics"]["planarity_ratio"],
            threshold=cfg.max_projection_nonplanarity_ratio,
        )
    if shape["length_width_ratio"] < cfg.min_length_width_ratio:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            analysis,
            "slit_geometry_not_narrow",
            shape=shape,
            threshold=cfg.min_length_width_ratio,
        )
    if shape["width_length_ratio"] <= cfg.weld_width_length_ratio:
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="slit_weld",
            failure_reason_codes=("slit_weld_requires_source_connectivity_edit",),
            normal=analysis["normal"],
            curvature=analysis["curvature"],
            boundary_mapping=(analysis["mapping"],),
            diagnostics={
                "shape": shape,
                "weld_width_length_ratio": cfg.weld_width_length_ratio,
                "source_points_modified": False,
                "source_faces_modified": False,
                "reason": (
                    "distinct source point IDs cannot be welded while source face connectivity "
                    "and source_triangle_index remain immutable"
                ),
            },
        )
    triangulation = triangulate_source_fixed_slit_bridge(
        analysis["frame"].boundary_uv,
        analysis["loop"],
        source_point_count=np.asarray(source_points).shape[0],
        tolerance=analysis["frame"].geometric_tolerance,
        maximum_appended_points=cfg.max_appended_points,
        spacing_width_ratio=cfg.interior_spacing_width_ratio,
    )
    if not triangulation["success"]:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            analysis,
            triangulation["failure_reason_codes"][0],
            shape=shape,
            triangulation=triangulation["diagnostics"],
        )
    if (
        triangulation["diagnostics"]["maximum_projected_aspect_ratio"]
        > cfg.max_aspect_ratio
    ):
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            analysis,
            "slit_bridge_projected_aspect_ratio_exceeded",
            shape=shape,
            triangulation=triangulation["diagnostics"],
            threshold=cfg.max_aspect_ratio,
        )
    depth_model = fit_thin_plate_surface(
        analysis["frame"].boundary_uv,
        analysis["frame"].boundary_depth,
        smoothing=1.0e-10,
    )
    if not depth_model["success"]:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            analysis,
            "slit_bridge_depth_interpolation_failed",
            interpolation=depth_model["diagnostics"],
        )
    depth = evaluate_thin_plate_surface(depth_model, triangulation["appended_uv"])
    appended_points = world_from_local(
        analysis["frame"], triangulation["appended_uv"], depth
    )
    appended_faces = triangulation["faces"]
    all_points = np.vstack([np.asarray(source_points, dtype=np.float64), appended_points])
    quality = triangle_quality(all_points, appended_faces)
    continuity = audit_boundary_normal_continuity(
        all_points,
        np.asarray(source_faces, dtype=np.int64),
        appended_faces,
        (analysis["mapping"],),
        cfg.min_boundary_normal_dot,
    )
    if quality["maximum_aspect_ratio"] > cfg.max_aspect_ratio or not continuity["passed"]:
        reasons = []
        if quality["maximum_aspect_ratio"] > cfg.max_aspect_ratio:
            reasons.append("slit_bridge_aspect_ratio_exceeded")
        if not continuity["passed"]:
            reasons.append("slit_bridge_boundary_normal_discontinuous")
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="slit_bridge",
            failure_reason_codes=reasons,
            normal=analysis["normal"],
            curvature=analysis["curvature"],
            boundary_mapping=(analysis["mapping"],),
            diagnostics={
                "shape": shape,
                "triangulation": triangulation["diagnostics"],
                "quality": quality,
                "boundary_continuity": continuity,
                "depth_interpolation": depth_model["diagnostics"],
                "immutable_boundary_edge_split_forbidden": True,
            },
        )
    delta = PatchDelta(
        appended_points=appended_points,
        appended_faces=appended_faces,
        point_provenance={
            "patch_method": np.full(appended_points.shape[0], "slit_bridge", dtype="U32"),
            "region_id": np.full(
                appended_points.shape[0], analysis["mapping"].region_id, dtype="U128"
            ),
            "uv": triangulation["appended_uv"],
            "normal_offset": depth,
            "placement": np.full(
                appended_points.shape[0],
                "thin_plate_boundary_depth_interpolation",
                dtype="U64",
            ),
        },
        face_provenance={
            "patch_method": np.full(appended_faces.shape[0], "slit_bridge", dtype="U32"),
            "region_id": np.full(
                appended_faces.shape[0], analysis["mapping"].region_id, dtype="U128"
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
        method="slit_bridge",
        delta=delta,
        boundary_mapping=(analysis["mapping"],),
        normal=analysis["normal"],
        curvature=analysis["curvature"],
        diagnostics={
            "stage": "source_primary_slit_patch",
            "method": "fixed_source_kernel_fan_thin_plate_bridge",
            "shape": shape,
            "boundary": analysis["diagnostics"],
            "triangulation": triangulation["diagnostics"],
            "depth_interpolation": depth_model["diagnostics"],
            "quality": quality,
            "boundary_continuity": continuity,
            "source_points_modified": False,
            "source_faces_modified": False,
        },
    )


def _slit_shape(boundary_uv: np.ndarray) -> dict[str, float]:
    centered = np.asarray(boundary_uv) - np.mean(boundary_uv, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ vh.T
    spans = np.ptp(coordinates, axis=0)
    length = float(max(spans))
    width = float(min(spans))
    return {
        "length": length,
        "width": width,
        "length_width_ratio": length / max(width, 1e-30),
        "width_length_ratio": width / max(length, 1e-30),
    }


def _reject(points, faces, source_ids, analysis, code, **diagnostics):
    return rejected_patch_candidate(
        points,
        faces,
        source_ids,
        method="slit_bridge",
        failure_reason_codes=(code,),
        normal=analysis["normal"],
        curvature=analysis["curvature"],
        boundary_mapping=(analysis["mapping"],),
        diagnostics=diagnostics,
    )


def _validate_config(config: SlitPatchConfig) -> str | None:
    scalar_error = validate_finite_config(
        config,
        SlitPatchConfig,
        integer_fields=frozenset({"max_boundary_vertices", "max_appended_points"}),
    )
    if scalar_error is not None:
        return scalar_error
    if config.min_length_width_ratio <= 1.0:
        return "min_length_width_ratio must be greater than one"
    if not 0.0 <= config.weld_width_length_ratio < 1.0:
        return "weld_width_length_ratio must be within [0, 1)"
    if not 0.0 <= config.max_projection_nonplanarity_ratio <= 1.0:
        return "max_projection_nonplanarity_ratio must be within [0, 1]"
    if config.max_boundary_vertices < 3 or not 1 <= config.max_appended_points <= 100_000:
        return "boundary and appended point limits are invalid"
    if not 0.1 <= config.interior_spacing_width_ratio <= 2.0:
        return "interior_spacing_width_ratio must be within [0.1, 2]"
    if not -1.0 <= config.min_boundary_normal_dot <= 1.0:
        return "min_boundary_normal_dot must be within [-1, 1]"
    if config.max_aspect_ratio <= 1.0:
        return "max_aspect_ratio must be greater than one"
    return None
