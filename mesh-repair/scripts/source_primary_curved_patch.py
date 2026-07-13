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
from source_primary_patch_geometry import (
    BoundaryFrame,
    analyze_source_boundary,
    triangle_quality,
    world_from_local,
)
from source_primary_patch_inputs import read_bounded_boundary_loop, validate_finite_config
from source_primary_patch_continuity import audit_boundary_normal_continuity
from source_primary_planar_patch import triangulate_constrained_polygon
from source_primary_proxy_sampling import (
    ProxyDepthSamplingConfig,
    sample_proxy_depths_in_footprint,
)
from source_primary_surface_lift import (
    build_boundary_lift_controls,
    evaluate_thin_plate_surface,
    fit_thin_plate_surface,
    limit_surface_controls,
    validate_lift_depth,
)
@dataclass(frozen=True)
class CurvedPatchConfig:
    max_projection_nonplanarity_ratio: float = 0.35
    steiner_rounds: int = 2
    max_boundary_vertices: int = 256
    max_appended_points: int = 100_000
    normal_control_offset_ratio: float = 0.08
    min_normal_graph_component: float = 0.15
    minimum_normal_control_coverage: float = 0.75
    maximum_normal_control_gap_ratio: float = 0.25
    thin_plate_smoothing: float = 1.0e-8
    max_thin_plate_controls: int = 512
    proxy_shape_weight: float = 0.35
    max_absolute_depth_ratio: float = 0.45
    max_thin_plate_overshoot_ratio: float = 0.20
    min_boundary_normal_dot: float = 0.25
    max_aspect_ratio: float = 300.0
# @entry
def build_curved_patch_candidate(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_triangle_index: np.ndarray,
    boundary_loop: np.ndarray | list[int],
    *,
    region_id: RegionId,
    oriented_normal: np.ndarray | None = None,
    closure_proxy_points: np.ndarray | None = None,
    closure_proxy_faces: np.ndarray | None = None,
    closure_proxy_triangle_index: np.ndarray | None = None,
    closure_proxy_component_id: np.ndarray | None = None,
    config: CurvedPatchConfig | None = None,
    proxy_config: ProxyDepthSamplingConfig | None = None,
) -> PatchCandidate:
    """Lift an interior-only triangulation from source normal/curvature evidence."""

    cfg = config if config is not None else CurvedPatchConfig()
    if not isinstance(cfg, CurvedPatchConfig):
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="curved_conformal_patch",
            failure_reason_codes=("curved_patch_config_invalid",),
            diagnostics={"message": "config must be CurvedPatchConfig"},
        )
    config_error = _validate_config(cfg)
    if config_error is not None:
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="curved_conformal_patch",
            failure_reason_codes=("curved_patch_config_invalid",),
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
            method="curved_conformal_patch",
            failure_reason_codes=("curved_patch_boundary_invalid",),
            diagnostics={"message": loop_error},
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
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="curved_conformal_patch",
            failure_reason_codes=analysis["failure_reason_codes"],
            diagnostics=analysis["diagnostics"],
        )
    frame: BoundaryFrame = analysis["frame"]
    planarity = float(analysis["diagnostics"]["planarity_ratio"])
    if planarity > cfg.max_projection_nonplanarity_ratio:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            analysis,
            "curved_patch_projection_domain_folded",
            threshold=cfg.max_projection_nonplanarity_ratio,
        )
    if analysis["curvature"].get("reliable") is not True:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            analysis,
            "curved_patch_curvature_evidence_unreliable",
            curvature=analysis["curvature"],
        )
    triangulation = triangulate_constrained_polygon(
        frame.boundary_uv,
        analysis["loop"],
        source_point_count=np.asarray(source_points).shape[0],
        steiner_rounds=cfg.steiner_rounds,
        max_appended_points=cfg.max_appended_points,
        tolerance=frame.geometric_tolerance,
    )
    if not triangulation["success"]:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            analysis,
            triangulation["failure_reason_codes"][0],
            triangulation=triangulation["diagnostics"],
        )
    appended_uv = triangulation["appended_uv"]
    controls = build_boundary_lift_controls(
        analysis,
        normal_control_offset_ratio=cfg.normal_control_offset_ratio,
        minimum_normal_graph_component=cfg.min_normal_graph_component,
        minimum_normal_control_coverage=cfg.minimum_normal_control_coverage,
        maximum_normal_control_gap_ratio=cfg.maximum_normal_control_gap_ratio,
        maximum_controls=cfg.max_thin_plate_controls,
    )
    if not controls["success"]:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            analysis,
            controls["failure_reason_codes"][0],
            lift_controls=controls["diagnostics"],
        )
    base_model = fit_thin_plate_surface(
        controls["uv"], controls["depth"], cfg.thin_plate_smoothing
    )
    if not base_model["success"]:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            analysis,
            "curved_patch_thin_plate_solve_failed",
            thin_plate=base_model["diagnostics"],
        )
    base_depth = evaluate_thin_plate_surface(base_model, appended_uv)
    proxy_provenance: dict[str, Any] = {
        "used": False,
        "role": "not_used",
        "geometry_consumed": False,
    }
    proxy_diagnostics: dict[str, Any] = {"status": "not_requested"}
    final_controls_uv = controls["uv"]
    final_controls_depth = controls["depth"]
    proxy_presence = any(
        value is not None
        for value in (
            closure_proxy_points,
            closure_proxy_faces,
            closure_proxy_triangle_index,
            closure_proxy_component_id,
        )
    )
    if proxy_presence:
        if closure_proxy_points is None or closure_proxy_faces is None:
            return _reject(
                source_points,
                source_faces,
                source_triangle_index,
                analysis,
                "curved_patch_proxy_input_incomplete",
            )
        if analysis["normal"].get("external_orientation_strongly_consistent") is not True:
            return _reject(
                source_points,
                source_faces,
                source_triangle_index,
                analysis,
                "curved_patch_proxy_requires_oriented_hole_normal",
            )
        proxy = sample_proxy_depths_in_footprint(
            closure_proxy_points,
            closure_proxy_faces,
            sample_uv=appended_uv,
            footprint_uv=frame.boundary_uv,
            frame_center=frame.center,
            frame_u_axis=frame.u_axis,
            frame_v_axis=frame.v_axis,
            oriented_normal=frame.normal,
            footprint_diagonal=frame.footprint_diagonal,
            proxy_triangle_index=closure_proxy_triangle_index,
            proxy_component_id=closure_proxy_component_id,
            config=proxy_config,
        )
        proxy_provenance = proxy["provenance"]
        proxy_diagnostics = proxy["diagnostics"]
        if not proxy["success"]:
            return _reject(
                source_points,
                source_faces,
                source_triangle_index,
                analysis,
                proxy["failure_reason_codes"][0],
                proxy_provenance=proxy_provenance,
                proxy_sampling=proxy_diagnostics,
            )
        selected = proxy["sample_indices"]
        blended = (
            (1.0 - cfg.proxy_shape_weight) * base_depth[selected]
            + cfg.proxy_shape_weight * proxy["signed_depth"]
        )
        optional_limit = cfg.max_thin_plate_controls - final_controls_uv.shape[0]
        if optional_limit < 1:
            return _reject(
                source_points,
                source_faces,
                source_triangle_index,
                analysis,
                "curved_patch_proxy_control_limit_exceeded",
                proxy_provenance=proxy_provenance,
            )
        proxy_uv, proxy_depth, retained_rows = limit_surface_controls(
            proxy["sample_uv"], blended, optional_limit
        )
        final_controls_uv = np.vstack([final_controls_uv, proxy_uv])
        final_controls_depth = np.concatenate([final_controls_depth, proxy_depth])
        proxy_provenance = {
            **proxy_provenance,
            "sampled_evidence_count": int(proxy["sample_uv"].shape[0]),
            "shape_control_evidence_row_indices": retained_rows,
            "shape_control_sample_indices": proxy["sample_indices"][retained_rows],
            "shape_reference_control_count": int(retained_rows.size),
        }
        proxy_diagnostics = {
            **proxy_diagnostics,
            "retained_shape_controls": int(proxy_uv.shape[0]),
        }
    model = fit_thin_plate_surface(
        final_controls_uv, final_controls_depth, cfg.thin_plate_smoothing
    )
    if not model["success"]:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            analysis,
            "curved_patch_thin_plate_solve_failed",
            proxy_provenance=proxy_provenance,
            thin_plate=model["diagnostics"],
        )
    depth = evaluate_thin_plate_surface(model, appended_uv)
    depth_gate = validate_lift_depth(
        depth,
        final_controls_depth,
        frame,
        max_absolute_depth_ratio=cfg.max_absolute_depth_ratio,
        max_overshoot_ratio=cfg.max_thin_plate_overshoot_ratio,
    )
    if not depth_gate["passed"]:
        return _reject(
            source_points,
            source_faces,
            source_triangle_index,
            analysis,
            depth_gate["reason_code"],
            proxy_provenance=proxy_provenance,
            depth_gate=depth_gate,
        )
    appended_points = world_from_local(frame, appended_uv, depth)
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
            reasons.append("curved_patch_aspect_ratio_exceeded")
        if not continuity["passed"]:
            reasons.append("curved_patch_boundary_normal_discontinuous")
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="curved_conformal_patch",
            failure_reason_codes=reasons,
            normal=analysis["normal"],
            curvature=analysis["curvature"],
            boundary_mapping=(analysis["mapping"],),
            proxy_provenance=proxy_provenance,
            diagnostics={
                "quality": quality,
                "boundary_continuity": continuity,
                "proxy_sampling": proxy_diagnostics,
            },
        )
    delta = PatchDelta(
        appended_points=appended_points,
        appended_faces=appended_faces,
        point_provenance={
            "patch_method": np.full(
                appended_points.shape[0], "curved_conformal_patch", dtype="U32"
            ),
            "region_id": np.full(
                appended_points.shape[0], analysis["mapping"].region_id, dtype="U128"
            ),
            "uv": appended_uv,
            "normal_offset": depth,
            "placement": np.full(
                appended_points.shape[0], "thin_plate_lift", dtype="U32"
            ),
        },
        face_provenance={
            "patch_method": np.full(
                appended_faces.shape[0], "curved_conformal_patch", dtype="U32"
            ),
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
        method="curved_conformal_patch",
        delta=delta,
        boundary_mapping=(analysis["mapping"],),
        normal=analysis["normal"],
        curvature=analysis["curvature"],
        proxy_provenance=proxy_provenance,
        diagnostics={
            "stage": "source_primary_curved_patch",
            "method": "boundary_normal_curvature_thin_plate_lift",
            "boundary": analysis["diagnostics"],
            "triangulation": triangulation["diagnostics"],
            "lift_controls": controls["diagnostics"],
            "thin_plate": model["diagnostics"],
            "depth_gate": depth_gate,
            "quality": quality,
            "boundary_continuity": continuity,
            "proxy_sampling": proxy_diagnostics,
            "source_points_modified": False,
            "source_faces_modified": False,
        },
    )
def _reject(
    points: np.ndarray,
    faces: np.ndarray,
    source_ids: np.ndarray,
    analysis: dict[str, Any],
    code: str,
    *,
    proxy_provenance: dict[str, Any] | None = None,
    **diagnostics: Any,
) -> PatchCandidate:
    return rejected_patch_candidate(
        points,
        faces,
        source_ids,
        method="curved_conformal_patch",
        failure_reason_codes=(code,),
        normal=analysis["normal"],
        curvature=analysis["curvature"],
        boundary_mapping=(analysis["mapping"],),
        proxy_provenance=proxy_provenance,
        diagnostics={"boundary": analysis["diagnostics"], **diagnostics},
    )
def _validate_config(config: CurvedPatchConfig) -> str | None:
    scalar_error = validate_finite_config(
        config,
        CurvedPatchConfig,
        integer_fields=frozenset(
            {
                "steiner_rounds",
                "max_boundary_vertices",
                "max_appended_points",
                "max_thin_plate_controls",
            }
        ),
    )
    if scalar_error is not None:
        return scalar_error
    if not 0.0 < config.max_projection_nonplanarity_ratio <= 1.0:
        return "max_projection_nonplanarity_ratio must be within (0, 1]"
    if (
        not 1 <= config.steiner_rounds <= 3
        or config.max_boundary_vertices < 3
        or config.max_appended_points < 1
    ):
        return "Steiner limits are invalid"
    if not 0.0 < config.normal_control_offset_ratio < 0.5:
        return "normal_control_offset_ratio must be within (0, 0.5)"
    if not 0.0 < config.min_normal_graph_component <= 1.0:
        return "min_normal_graph_component must be within (0, 1]"
    if not 0.0 < config.minimum_normal_control_coverage <= 1.0:
        return "minimum_normal_control_coverage must be within (0, 1]"
    if not 0.0 <= config.maximum_normal_control_gap_ratio < 1.0:
        return "maximum_normal_control_gap_ratio must be within [0, 1)"
    if config.thin_plate_smoothing < 0.0 or config.max_thin_plate_controls < 6:
        return "thin-plate smoothing and control limits are invalid"
    if not 0.0 <= config.proxy_shape_weight <= 1.0:
        return "proxy_shape_weight must be within [0, 1]"
    if config.max_absolute_depth_ratio <= 0.0 or config.max_thin_plate_overshoot_ratio < 0.0:
        return "depth limits must be non-negative with a positive absolute limit"
    if not -1.0 <= config.min_boundary_normal_dot <= 1.0:
        return "min_boundary_normal_dot must be within [-1, 1]"
    if config.max_aspect_ratio <= 1.0:
        return "max_aspect_ratio must be greater than one"
    return None
