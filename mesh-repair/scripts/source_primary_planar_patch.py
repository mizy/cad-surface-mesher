from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from source_primary_constrained_triangulation import triangulate_constrained_polygon
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


@dataclass(frozen=True)
class PlanarPatchConfig:
    max_planarity_ratio: float = 0.02
    max_dimensionless_curvature: float = 0.05
    max_boundary_normal_turn_radians: float = 0.10
    steiner_rounds: int = 0
    max_boundary_vertices: int = 512
    max_appended_points: int = 100_000
    min_face_normal_dot: float = 0.05
    min_boundary_normal_dot: float = 0.25
    max_aspect_ratio: float = 250.0


# @entry
def build_planar_patch_candidate(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_triangle_index: np.ndarray,
    boundary_loop: np.ndarray | list[int],
    *,
    region_id: RegionId,
    oriented_normal: np.ndarray | None = None,
    config: PlanarPatchConfig | None = None,
) -> PatchCandidate:
    """Build a constrained planar cap without altering any source array."""

    cfg = config if config is not None else PlanarPatchConfig()
    if not isinstance(cfg, PlanarPatchConfig):
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="planar_cap",
            failure_reason_codes=("planar_patch_config_invalid",),
            diagnostics={"message": "config must be PlanarPatchConfig"},
        )
    config_error = _validate_config(cfg)
    if config_error is not None:
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="planar_cap",
            failure_reason_codes=("planar_patch_config_invalid",),
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
            method="planar_cap",
            failure_reason_codes=("planar_patch_boundary_invalid",),
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
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="planar_cap",
            failure_reason_codes=analysis["failure_reason_codes"],
            diagnostics=analysis["diagnostics"],
        )
    frame: BoundaryFrame = analysis["frame"]
    planarity_ratio = float(analysis["diagnostics"]["planarity_ratio"])
    if planarity_ratio > cfg.max_planarity_ratio:
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="planar_cap",
            failure_reason_codes=("planar_patch_planarity_exceeded",),
            normal=analysis["normal"],
            curvature=analysis["curvature"],
            boundary_mapping=(analysis["mapping"],),
            diagnostics={
                **analysis["diagnostics"],
                "threshold": cfg.max_planarity_ratio,
            },
        )
    curvature_gate = _planar_curvature_gate(analysis, cfg)
    if not curvature_gate["passed"]:
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="planar_cap",
            failure_reason_codes=(curvature_gate["reason_code"],),
            normal=analysis["normal"],
            curvature=analysis["curvature"],
            boundary_mapping=(analysis["mapping"],),
            diagnostics={
                **analysis["diagnostics"],
                "curvature_gate": curvature_gate,
            },
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
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="planar_cap",
            failure_reason_codes=triangulation["failure_reason_codes"],
            normal=analysis["normal"],
            curvature=analysis["curvature"],
            boundary_mapping=(analysis["mapping"],),
            diagnostics={
                "boundary": analysis["diagnostics"],
                "triangulation": triangulation["diagnostics"],
            },
        )
    appended_uv = triangulation["appended_uv"]
    appended_points = world_from_local(frame, appended_uv, 0.0)
    appended_faces = triangulation["faces"]
    quality = triangle_quality(
        np.vstack([np.asarray(source_points, dtype=np.float64), appended_points]),
        appended_faces,
    )
    normal_gate = _face_normal_gate(
        np.vstack([np.asarray(source_points, dtype=np.float64), appended_points]),
        appended_faces,
        frame.normal,
        cfg.min_face_normal_dot,
    )
    continuity = audit_boundary_normal_continuity(
        np.vstack([np.asarray(source_points, dtype=np.float64), appended_points]),
        np.asarray(source_faces, dtype=np.int64),
        appended_faces,
        (analysis["mapping"],),
        cfg.min_boundary_normal_dot,
    )
    if (
        quality["maximum_aspect_ratio"] > cfg.max_aspect_ratio
        or not normal_gate["passed"]
        or not continuity["passed"]
    ):
        reasons = []
        if quality["maximum_aspect_ratio"] > cfg.max_aspect_ratio:
            reasons.append("planar_patch_aspect_ratio_exceeded")
        if not normal_gate["passed"]:
            reasons.append("planar_patch_face_normal_inconsistent")
        if not continuity["passed"]:
            reasons.append("planar_patch_boundary_normal_discontinuous")
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method="planar_cap",
            failure_reason_codes=reasons,
            normal=analysis["normal"],
            curvature=analysis["curvature"],
            boundary_mapping=(analysis["mapping"],),
            diagnostics={
                "boundary": analysis["diagnostics"],
                "triangulation": triangulation["diagnostics"],
                "quality": quality,
                "normal_gate": normal_gate,
                "boundary_continuity": continuity,
            },
        )
    delta = PatchDelta(
        appended_points=appended_points,
        appended_faces=appended_faces,
        point_provenance={
            "patch_method": np.full(appended_points.shape[0], "planar_cap", dtype="U32"),
            "region_id": np.full(
                appended_points.shape[0], analysis["mapping"].region_id, dtype="U128"
            ),
            "uv": appended_uv,
            "normal_offset": np.zeros(appended_points.shape[0], dtype=np.float64),
            "placement": np.full(
                appended_points.shape[0], "fitted_plane", dtype="U32"
            ),
        },
        face_provenance={
            "patch_method": np.full(appended_faces.shape[0], "planar_cap", dtype="U32"),
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
        method="planar_cap",
        delta=delta,
        boundary_mapping=(analysis["mapping"],),
        normal=analysis["normal"],
        curvature=analysis["curvature"],
        diagnostics={
            "stage": "source_primary_planar_patch",
            "method": "local_plane_constrained_ear_clipping_with_steiner_refinement",
            "boundary": analysis["diagnostics"],
            "triangulation": triangulation["diagnostics"],
            "quality": quality,
            "normal_gate": normal_gate,
            "boundary_continuity": continuity,
            "curvature_gate": curvature_gate,
            "source_points_modified": False,
            "source_faces_modified": False,
        },
    )


def _face_normal_gate(
    points: np.ndarray,
    faces: np.ndarray,
    expected: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    triangles = points[faces]
    raw = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(raw, axis=1)
    normals = np.divide(raw, lengths[:, None], out=np.zeros_like(raw), where=lengths[:, None] > 0.0)
    dots = normals @ expected
    return {
        "passed": bool(dots.size and np.min(dots) >= threshold),
        "minimum_dot": float(dots.min()) if dots.size else None,
        "mean_dot": float(dots.mean()) if dots.size else None,
        "threshold": float(threshold),
    }


def _validate_config(config: PlanarPatchConfig) -> str | None:
    scalar_error = validate_finite_config(
        config,
        PlanarPatchConfig,
        integer_fields=frozenset(
            {"steiner_rounds", "max_boundary_vertices", "max_appended_points"}
        ),
    )
    if scalar_error is not None:
        return scalar_error
    if not 0.0 <= config.max_planarity_ratio <= 1.0:
        return "max_planarity_ratio must be within [0, 1]"
    if config.max_dimensionless_curvature < 0.0:
        return "max_dimensionless_curvature must be non-negative"
    if not 0.0 <= config.max_boundary_normal_turn_radians <= np.pi:
        return "max_boundary_normal_turn_radians must be within [0, pi]"
    if not 0 <= config.steiner_rounds <= 3:
        return "steiner_rounds must be between 0 and 3"
    if config.max_boundary_vertices < 3 or config.max_appended_points < 0:
        return "point limits must be non-negative and permit one triangle"
    if config.steiner_rounds and config.max_appended_points < 1:
        return "Steiner refinement requires a positive appended-point limit"
    if not -1.0 <= config.min_face_normal_dot <= 1.0:
        return "min_face_normal_dot must be within [-1, 1]"
    if not -1.0 <= config.min_boundary_normal_dot <= 1.0:
        return "min_boundary_normal_dot must be within [-1, 1]"
    if config.max_aspect_ratio <= 1.0:
        return "max_aspect_ratio must be greater than one"
    return None


def _planar_curvature_gate(
    analysis: dict[str, Any], config: PlanarPatchConfig
) -> dict[str, Any]:
    curvature = analysis["curvature"]
    principal = np.asarray(curvature.get("principal_curvatures", []), dtype=np.float64)
    normal_turn = float(curvature.get("normal_turn_radians_max", np.inf))
    dimensionless = (
        float(np.max(np.abs(principal), initial=0.0))
        * float(analysis["frame"].footprint_diagonal)
        if principal.shape == (2,) and np.all(np.isfinite(principal))
        else np.inf
    )
    reason = None
    if not np.isfinite(dimensionless) or not np.isfinite(normal_turn):
        reason = "planar_patch_curvature_unreliable"
    elif (
        dimensionless > config.max_dimensionless_curvature
        or normal_turn > config.max_boundary_normal_turn_radians
    ):
        reason = "planar_patch_curvature_exceeded"
    return {
        "passed": reason is None,
        "reason_code": reason,
        "dimensionless_max_abs_curvature": dimensionless,
        "boundary_normal_turn_radians_max": normal_turn,
        "thresholds": {
            "dimensionless_max_abs_curvature": config.max_dimensionless_curvature,
            "boundary_normal_turn_radians_max": config.max_boundary_normal_turn_radians,
        },
    }
