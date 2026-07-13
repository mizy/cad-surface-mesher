from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from source_primary_patch_parameterization import points_in_polygon
from source_primary_proxy_queries import (
    barycentric_2d,
    cluster_depth_hits,
    footprint_diagonal_from_uv,
    unit_vector,
    validate_proxy_sampling_config,
    validate_proxy_sampling_inputs,
)
from source_primary_proxy_spatial_index import (
    build_proxy_face_index,
    label_proxy_face_components,
    query_proxy_face_index,
)


@dataclass(frozen=True)
class ProxyDepthSamplingConfig:
    min_signed_depth_ratio: float = -0.50
    max_signed_depth_ratio: float = 0.50
    depth_cluster_tolerance_ratio: float = 1.0e-4
    barycentric_tolerance: float = 1.0e-10
    minimum_normal_abs_dot: float = 0.05
    minimum_sample_coverage: float = 0.50
    max_samples: int = 100_000
    max_hits_per_sample: int = 256
    max_footprint_candidate_faces: int = 250_000
    max_spatial_index_entries: int = 2_000_000
    max_query_candidate_faces: int = 8_192


# @entry
def sample_proxy_depths_in_footprint(
    proxy_points: np.ndarray,
    proxy_faces: np.ndarray,
    *,
    sample_uv: np.ndarray,
    footprint_uv: np.ndarray,
    frame_center: np.ndarray,
    frame_u_axis: np.ndarray,
    frame_v_axis: np.ndarray,
    oriented_normal: np.ndarray,
    footprint_diagonal: float,
    proxy_triangle_index: np.ndarray | None = None,
    proxy_component_id: np.ndarray | None = None,
    config: ProxyDepthSamplingConfig | None = None,
) -> dict[str, Any]:
    """Sample scalar proxy depth on normal-lines strictly inside a hole footprint."""

    cfg = config if config is not None else ProxyDepthSamplingConfig()
    if not isinstance(cfg, ProxyDepthSamplingConfig):
        return _failure(
            "proxy_depth_config_invalid",
            message="config must be ProxyDepthSamplingConfig",
        )
    config_error = validate_proxy_sampling_config(cfg, ProxyDepthSamplingConfig)
    if config_error is not None:
        return _failure(
            "proxy_depth_config_invalid", message=config_error, config=asdict(cfg)
        )
    input_error = validate_proxy_sampling_inputs(
        proxy_points,
        proxy_faces,
        sample_uv,
        footprint_uv,
        frame_center,
        frame_u_axis,
        frame_v_axis,
        oriented_normal,
        footprint_diagonal,
        proxy_triangle_index,
        proxy_component_id,
        cfg,
    )
    if input_error is not None:
        return _failure(input_error[0], message=input_error[1])
    points = np.asarray(proxy_points, dtype=np.float64)
    faces = np.asarray(proxy_faces, dtype=np.int64)
    queries = np.asarray(sample_uv, dtype=np.float64).reshape(-1, 2)
    footprint = np.asarray(footprint_uv, dtype=np.float64).reshape(-1, 2)
    center = np.asarray(frame_center, dtype=np.float64)
    u_axis = unit_vector(np.asarray(frame_u_axis, dtype=np.float64))
    v_axis = unit_vector(np.asarray(frame_v_axis, dtype=np.float64))
    normal = unit_vector(np.asarray(oriented_normal, dtype=np.float64))
    if u_axis is None or v_axis is None or normal is None:
        return _failure("proxy_frame_invalid", message="validated frame lost unit axes")
    internal_scale = footprint_diagonal_from_uv(footprint)
    if internal_scale is None:
        return _failure(
            "proxy_footprint_invalid", message="validated footprint lost finite scale"
        )
    scale = internal_scale
    tolerance = max(scale * np.finfo(np.float64).eps * 512.0, 1e-15)
    strictly_inside = points_in_polygon(
        queries, footprint, tolerance=tolerance, include_boundary=False
    )
    if not np.all(strictly_inside):
        return _failure(
            "proxy_sample_outside_true_footprint",
            outside_sample_indices=np.flatnonzero(~strictly_inside)[:100].astype(int).tolist(),
            outside_sample_count=int(np.count_nonzero(~strictly_inside)),
        )
    basis = np.column_stack((u_axis, v_axis, normal))
    local_points = (points - center) @ basis
    triangles = local_points[faces]
    projected = triangles[:, :, :2]
    face_uv_min = projected.min(axis=1) - tolerance
    face_uv_max = projected.max(axis=1) + tolerance
    raw_normals = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    normal_lengths = np.linalg.norm(raw_normals, axis=1)
    face_normal_dot = np.divide(
        raw_normals @ normal,
        normal_lengths,
        out=np.zeros(faces.shape[0], dtype=np.float64),
        where=normal_lengths > 0.0,
    )
    triangle_ids = (
        np.arange(faces.shape[0], dtype=np.int64)
        if proxy_triangle_index is None
        else np.asarray(proxy_triangle_index, dtype=np.int64)
    )
    component_ids = (
        np.full(faces.shape[0], -1, dtype=np.int64)
        if proxy_component_id is None
        else np.asarray(proxy_component_id, dtype=np.int64)
    )
    depth_min = cfg.min_signed_depth_ratio * scale
    depth_max = cfg.max_signed_depth_ratio * scale
    cluster_tolerance = max(cfg.depth_cluster_tolerance_ratio * scale, tolerance)
    footprint_min = footprint.min(axis=0) - tolerance
    footprint_max = footprint.max(axis=0) + tolerance
    footprint_face_mask = (
        np.all(face_uv_max >= footprint_min, axis=1)
        & np.all(face_uv_min <= footprint_max, axis=1)
        & (triangles[:, :, 2].max(axis=1) >= depth_min)
        & (triangles[:, :, 2].min(axis=1) <= depth_max)
    )
    footprint_face_ids = np.flatnonzero(footprint_face_mask)
    if footprint_face_ids.size > cfg.max_footprint_candidate_faces:
        return _failure(
            "proxy_footprint_candidate_face_limit_exceeded",
            candidate_face_count=int(footprint_face_ids.size),
            max_candidate_faces=int(cfg.max_footprint_candidate_faces),
        )
    footprint_min_exact = footprint.min(axis=0)
    footprint_max_exact = footprint.max(axis=0)
    spatial_index = build_proxy_face_index(
        face_uv_min,
        face_uv_max,
        footprint_face_ids,
        footprint_min_exact,
        footprint_max_exact,
        maximum_entries=cfg.max_spatial_index_entries,
        maximum_candidates_per_cell=cfg.max_query_candidate_faces,
    )
    if not spatial_index["success"]:
        return _failure(
            "proxy_footprint_spatial_index_limit_exceeded",
            **{key: value for key, value in spatial_index.items() if key != "success"},
        )
    local_component_by_face = np.full(faces.shape[0], -1, dtype=np.int64)
    local_component_by_face[footprint_face_ids] = label_proxy_face_components(
        faces, footprint_face_ids
    )
    area_tolerance = max(scale * scale * np.finfo(np.float64).eps * 512.0, 1e-30)
    selected_rows: list[dict[str, Any]] = []
    missing_queries: list[int] = []
    ambiguous_queries: list[dict[str, Any]] = []
    winding_incompatible_queries: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        indexed_result = query_proxy_face_index(
            spatial_index,
            query,
            maximum_candidates=cfg.max_query_candidate_faces,
        )
        if not indexed_result["success"]:
            return _failure(
                "proxy_query_candidate_face_limit_exceeded",
                sample_index=query_index,
                **{key: value for key, value in indexed_result.items() if key != "success"},
            )
        indexed = indexed_result["face_ids"]
        bbox_mask = np.all(face_uv_min[indexed] <= query, axis=1) & np.all(
            query <= face_uv_max[indexed], axis=1
        )
        candidates = indexed[bbox_mask]
        hits: list[dict[str, Any]] = []
        negative_normal_hits = 0
        for face_id in candidates:
            barycentric = barycentric_2d(
                query,
                projected[int(face_id)],
                cfg.barycentric_tolerance,
                area_tolerance,
            )
            if barycentric is None:
                continue
            depth = float(np.dot(barycentric, triangles[int(face_id), :, 2]))
            if not depth_min <= depth <= depth_max:
                continue
            normal_dot = float(face_normal_dot[int(face_id)])
            if normal_dot < cfg.minimum_normal_abs_dot:
                negative_normal_hits += int(
                    normal_dot <= -cfg.minimum_normal_abs_dot
                )
                continue
            hits.append(
                {
                    "face_id": int(face_id),
                    "proxy_triangle_index": int(triangle_ids[int(face_id)]),
                    "component_id": int(component_ids[int(face_id)]),
                    "local_component_id": int(local_component_by_face[int(face_id)]),
                    "depth": depth,
                    "barycentric": barycentric,
                    "normal_dot": normal_dot,
                }
            )
            if len(hits) > cfg.max_hits_per_sample:
                break
        if negative_normal_hits:
            winding_incompatible_queries.append(
                {
                    "sample_index": query_index,
                    "negative_normal_hit_count": negative_normal_hits,
                }
            )
        if not hits:
            missing_queries.append(query_index)
            continue
        if len(hits) > cfg.max_hits_per_sample:
            ambiguous_queries.append(
                {
                    "sample_index": query_index,
                    "reason": "hit_limit_exceeded",
                    "hit_count": len(hits),
                }
            )
            continue
        clusters = cluster_depth_hits(hits, cluster_tolerance)
        if len(clusters) != 1:
            ambiguous_queries.append(
                {
                    "sample_index": query_index,
                    "reason": "multiple_depth_layers",
                    "layer_depths": [
                        float(np.mean([row["depth"] for row in cluster]))
                        for cluster in clusters
                    ],
                }
            )
            continue
        cluster = clusters[0]
        known_components = {
            row["component_id"] for row in cluster if row["component_id"] >= 0
        }
        local_components = {row["local_component_id"] for row in cluster}
        if len(known_components) > 1:
            ambiguous_queries.append(
                {
                    "sample_index": query_index,
                    "reason": "multiple_proxy_components",
                    "component_ids": sorted(known_components),
                }
            )
            continue
        if known_components and any(row["component_id"] < 0 for row in cluster):
            ambiguous_queries.append(
                {
                    "sample_index": query_index,
                    "reason": "mixed_labeled_unlabeled_proxy_components",
                }
            )
            continue
        if len(local_components) > 1:
            ambiguous_queries.append(
                {
                    "sample_index": query_index,
                    "reason": "multiple_local_proxy_components",
                    "local_component_ids": sorted(local_components),
                }
            )
            continue
        selected = min(
            cluster,
            key=lambda row: (
                -float(row["normal_dot"]),
                int(row["proxy_triangle_index"]),
            ),
        )
        selected_rows.append({"sample_index": query_index, **selected})
    if winding_incompatible_queries:
        return _failure(
            "proxy_winding_incompatible",
            incompatible_sample_count=len(winding_incompatible_queries),
            incompatible_samples=winding_incompatible_queries[:100],
            minimum_required_normal_dot=float(cfg.minimum_normal_abs_dot),
        )
    if ambiguous_queries:
        return _failure(
            "proxy_depth_multilayer_or_component_ambiguous",
            ambiguous_sample_count=len(ambiguous_queries),
            ambiguous_samples=ambiguous_queries[:100],
            missing_sample_count=len(missing_queries),
        )
    coverage = len(selected_rows) / max(queries.shape[0], 1)
    if coverage < cfg.minimum_sample_coverage:
        return _failure(
            "proxy_depth_sample_coverage_insufficient",
            coverage=float(coverage),
            threshold=float(cfg.minimum_sample_coverage),
            missing_sample_indices=missing_queries[:100],
        )
    selected_rows.sort(key=lambda row: int(row["sample_index"]))
    selected_local_components = {row["local_component_id"] for row in selected_rows}
    selected_known_components = {
        row["component_id"] for row in selected_rows if row["component_id"] >= 0
    }
    selected_has_unknown_component = any(
        row["component_id"] < 0 for row in selected_rows
    )
    if (
        len(selected_local_components) != 1
        or len(selected_known_components) > 1
        or (selected_known_components and selected_has_unknown_component)
    ):
        return _failure(
            "proxy_depth_global_component_ambiguous",
            local_component_ids=sorted(selected_local_components),
            proxy_component_ids=sorted(selected_known_components),
            contains_unlabeled_component=selected_has_unknown_component,
        )
    sample_indices = np.asarray(
        [row["sample_index"] for row in selected_rows], dtype=np.int64
    )
    selected_normal_dot = np.asarray(
        [row["normal_dot"] for row in selected_rows], dtype=np.float64
    )
    if np.any(selected_normal_dot < cfg.minimum_normal_abs_dot):
        return _failure(
            "proxy_winding_incompatible",
            minimum_normal_dot=float(selected_normal_dot.min(initial=np.inf)),
            minimum_required_normal_dot=float(cfg.minimum_normal_abs_dot),
        )
    signed_depth = np.asarray(
        [row["depth"] for row in selected_rows], dtype=np.float64
    )
    provenance = {
        "used": True,
        "role": "local_normal_depth_reference_only",
        "geometry_consumed": False,
        "shape_reference_used": True,
        "proxy_point_count": int(points.shape[0]),
        "proxy_face_count": int(faces.shape[0]),
        "triangle_index_supplied": proxy_triangle_index is not None,
        "component_labels_supplied": proxy_component_id is not None,
        "proxy_points_sha256": _array_sha256(points),
        "proxy_faces_sha256": _array_sha256(faces),
        "sampling_method": "oriented_normal_line_triangle_barycentric_depth",
        "footprint_policy": "strict_polygon_interior_no_dilation",
        "sample_indices": sample_indices,
        "sample_uv": queries[sample_indices],
        "signed_depth": signed_depth,
        "proxy_triangle_index": np.asarray(
            [row["proxy_triangle_index"] for row in selected_rows], dtype=np.int64
        ),
        "proxy_component_id": np.asarray(
            [row["component_id"] for row in selected_rows], dtype=np.int64
        ),
        "proxy_local_component_id": np.asarray(
            [row["local_component_id"] for row in selected_rows], dtype=np.int64
        ),
        "barycentric": np.asarray(
            [row["barycentric"] for row in selected_rows], dtype=np.float64
        ),
        "normal_dot": selected_normal_dot,
        "minimum_required_normal_dot": float(cfg.minimum_normal_abs_dot),
        "minimum_normal_dot": float(selected_normal_dot.min()),
        "maximum_normal_dot": float(selected_normal_dot.max()),
        "negative_normal_count": int(np.count_nonzero(selected_normal_dot < 0.0)),
        "oriented_normal": normal.tolist(),
        "requested_sample_count": int(queries.shape[0]),
        "selected_sample_count": len(selected_rows),
        "coverage": float(coverage),
        "missing_sample_indices": tuple(int(value) for value in missing_queries),
        "signed_depth_min": float(signed_depth.min()),
        "signed_depth_max": float(signed_depth.max()),
        "footprint_diagonal": float(scale),
    }
    return {
        "success": True,
        "failure_reason_codes": [],
        "sample_indices": sample_indices,
        "sample_uv": queries[sample_indices].copy(),
        "signed_depth": signed_depth,
        "provenance": provenance,
        "diagnostics": {
            "stage": "source_primary_proxy_depth_sampling",
            "coverage": float(coverage),
            "missing_sample_count": len(missing_queries),
            "depth_limits": [float(depth_min), float(depth_max)],
            "depth_cluster_tolerance": float(cluster_tolerance),
            "footprint_diagonal": float(scale),
            "spatial_index": {
                "resolution": int(spatial_index["resolution"]),
                "entry_count": int(spatial_index["entry_count"]),
            },
            "config": asdict(cfg),
        },
    }


def _failure(code: str, **diagnostics: Any) -> dict[str, Any]:
    return {
        "success": False,
        "failure_reason_codes": [code],
        "sample_indices": np.empty(0, dtype=np.int64),
        "sample_uv": np.empty((0, 2), dtype=np.float64),
        "signed_depth": np.empty(0, dtype=np.float64),
        "provenance": {
            "used": True,
            "role": "local_normal_depth_reference_only",
            "geometry_consumed": False,
            "status": "rejected",
            "failure_reason_code": code,
        },
        "diagnostics": diagnostics,
    }


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()
