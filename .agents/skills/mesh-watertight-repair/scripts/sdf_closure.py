from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import trimesh
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import marching_cubes

from sealed_exterior_atlas import build_sealed_exterior_volume_grid


DEFAULT_BYTES_PER_VOXEL_BUDGET = 32
DEFAULT_QUERY_CHUNK = 250_000


@dataclass(frozen=True)
class SdfClosureConfig:
    pitch: float
    seal_radius_voxels: int = 1
    band_voxels: float = 6.0
    smoothing_sigma_voxels: float = 0.5
    max_memory_gb: float = 4.0
    max_projection_distance: float = 0.0
    closest_point_chunk: int = DEFAULT_QUERY_CHUNK


def build_tsdf_closure(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    config: SdfClosureConfig,
    artifact_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a watertight zero surface from a flood-signed narrow-band SDF.

    The far-field flood owns the sign.  Point-to-triangle distance owns the
    narrow-band magnitude.  Input winding is therefore never used to decide
    inside versus outside.
    """

    source_points, source_faces = _validate_mesh(points, faces)
    _validate_config(config)
    memory = sdf_memory_preflight(
        source_points,
        pitch=config.pitch,
        seal_radius_voxels=config.seal_radius_voxels,
        max_memory_gb=config.max_memory_gb,
    )
    if not memory["passed"]:
        raise MemoryError(memory["failure_reason"])

    filled_grid, exterior_volume = build_sealed_exterior_volume_grid(
        source_points,
        source_faces,
        pitch=config.pitch,
        seal_radius_voxels=config.seal_radius_voxels,
        max_projection_distance=config.max_projection_distance,
    )
    solid = np.asarray(filled_grid.matrix, dtype=bool)
    if not np.any(solid) or np.all(solid):
        raise ValueError("sealed exterior volume must contain both solid and outside voxels")
    actual_peak_bytes = int(solid.size * DEFAULT_BYTES_PER_VOXEL_BUDGET)
    memory["actual_shape"] = [int(value) for value in solid.shape]
    memory["actual_voxels"] = int(solid.size)
    memory["actual_peak_bytes_estimate"] = actual_peak_bytes
    if actual_peak_bytes > int(memory["budget_bytes"]):
        raise MemoryError(
            "SDF actual-grid memory estimate exceeds the explicit budget: "
            f"estimated={actual_peak_bytes} bytes budget={memory['budget_bytes']} bytes"
        )

    # EDT supplies a complete, monotone fallback field.  The narrow band is
    # then replaced by exact closest-triangle distances, which gives the zero
    # surface sub-voxel source information without trusting source winding.
    outside_distance = distance_transform_edt(~solid, sampling=config.pitch)
    inside_distance = distance_transform_edt(solid, sampling=config.pitch)
    phi = outside_distance - inside_distance
    del outside_distance, inside_distance

    band_world = float(config.band_voxels * config.pitch)
    narrow_band = np.abs(phi) <= band_world
    exact_distance_report = _replace_narrow_band_magnitude(
        phi,
        solid,
        narrow_band,
        filled_grid.transform,
        source_points,
        source_faces,
        chunk_size=config.closest_point_chunk,
    )
    np.clip(phi, -band_world, band_world, out=phi)
    phi = phi.astype(np.float32, copy=False)

    smoothed, smoothing_report = bounded_sdf_smoothing(
        phi,
        narrow_band,
        pitch=config.pitch,
        sigma_voxels=config.smoothing_sigma_voxels,
        max_zero_shift_voxels=0.5,
    )
    mesh_points, mesh_faces = extract_zero_surface(smoothed, filled_grid.transform)

    output = trimesh.Trimesh(vertices=mesh_points, faces=mesh_faces, process=False)
    output.remove_unreferenced_vertices()
    try:
        output.fix_normals()
    except Exception:
        pass
    mesh_points = np.asarray(output.vertices, dtype=np.float64)
    mesh_faces = np.asarray(output.faces, dtype=np.int64)

    artifact = None
    if artifact_path is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact_path,
            schema=np.asarray("implicit_field/v1"),
            sdf=smoothed.astype(np.float32, copy=False),
            solid=solid.astype(np.uint8, copy=False),
            transform=np.asarray(filled_grid.transform, dtype=np.float64),
            pitch=np.asarray(config.pitch, dtype=np.float64),
            band_voxels=np.asarray(config.band_voxels, dtype=np.float64),
            smoothing_sigma_voxels=np.asarray(
                config.smoothing_sigma_voxels, dtype=np.float64
            ),
        )
        artifact = str(artifact_path)

    report = {
        "schema": "signed_distance_closure/v1",
        "method": "sealed_far_field_sign_plus_exact_narrow_band_triangle_distance",
        "sign_source": "sealed_exterior_six_connected_far_field_flood",
        "distance_source": "closest_point_on_source_triangle_in_narrow_band",
        "pitch": float(config.pitch),
        "band_voxels": float(config.band_voxels),
        "band_world": band_world,
        "matrix_shape": [int(value) for value in solid.shape],
        "solid_voxels": int(np.count_nonzero(solid)),
        "outside_voxels": int(np.count_nonzero(~solid)),
        "narrow_band_voxels": int(np.count_nonzero(narrow_band)),
        "memory_preflight": memory,
        "exact_distance": exact_distance_report,
        "smoothing": smoothing_report,
        "exterior_volume": exterior_volume,
        "artifact": artifact,
        "output": {
            "points": int(mesh_points.shape[0]),
            "triangles": int(mesh_faces.shape[0]),
            "trimesh_watertight": bool(output.is_watertight),
        },
    }
    return mesh_points, mesh_faces, report


def sdf_memory_preflight(
    points: np.ndarray,
    *,
    pitch: float,
    seal_radius_voxels: int,
    max_memory_gb: float,
) -> dict[str, Any]:
    source = np.asarray(points, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or source.shape[0] == 0:
        raise ValueError("points must have shape (point_count, 3) and be non-empty")
    if not np.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("pitch must be finite and positive")
    if not np.isfinite(max_memory_gb) or max_memory_gb <= 0.0:
        raise ValueError("max_memory_gb must be finite and positive")
    padding = max(2, int(seal_radius_voxels) + 2)
    extents = np.ptp(source, axis=0)
    base_shape = np.maximum(np.ceil(extents / pitch).astype(np.int64) + 3, 3)
    estimated_shape = base_shape + 2 * padding
    voxels = int(np.prod(estimated_shape, dtype=np.int64))
    estimated_bytes = int(voxels * DEFAULT_BYTES_PER_VOXEL_BUDGET)
    budget_bytes = int(max_memory_gb * 1024**3)
    passed = estimated_bytes <= budget_bytes
    return {
        "passed": passed,
        "estimated_shape": [int(value) for value in estimated_shape],
        "estimated_voxels": voxels,
        "bytes_per_voxel_budget": DEFAULT_BYTES_PER_VOXEL_BUDGET,
        "estimated_peak_bytes": estimated_bytes,
        "budget_bytes": budget_bytes,
        "failure_reason": (
            None
            if passed
            else (
                "SDF memory preflight exceeds the explicit budget: "
                f"estimated={estimated_bytes} bytes budget={budget_bytes} bytes"
            )
        ),
    }


def bounded_sdf_smoothing(
    phi: np.ndarray,
    narrow_band: np.ndarray,
    *,
    pitch: float,
    sigma_voxels: float,
    max_zero_shift_voxels: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    field = np.asarray(phi, dtype=np.float32)
    band = np.asarray(narrow_band, dtype=bool)
    if sigma_voxels <= 0.0:
        return field.copy(), {
            "requested_sigma_voxels": float(sigma_voxels),
            "applied": False,
            "fallback": False,
            "reason": "smoothing_disabled",
            "max_field_delta": 0.0,
            "max_allowed_field_delta": float(max_zero_shift_voxels * pitch),
        }
    trial = gaussian_filter(field, sigma=float(sigma_voxels), mode="nearest")
    trial = np.where(band, trial, field).astype(np.float32, copy=False)
    near_zero = band & (np.abs(field) <= 2.0 * pitch)
    delta = np.abs(trial - field)
    max_delta = float(np.max(delta[near_zero])) if np.any(near_zero) else 0.0
    limit = float(max_zero_shift_voxels * pitch)
    fallback = max_delta > limit
    return (field.copy() if fallback else trial), {
        "requested_sigma_voxels": float(sigma_voxels),
        "applied": not fallback,
        "fallback": fallback,
        "reason": "bounded_field_delta_exceeded" if fallback else "bounded_smoothing_applied",
        "max_field_delta": max_delta,
        "max_allowed_field_delta": limit,
        "zero_shift_bound_method": "near_zero_field_delta_upper_bound",
    }


def extract_zero_surface(
    phi: np.ndarray,
    transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(phi, dtype=np.float32)
    if float(np.min(field)) >= 0.0 or float(np.max(field)) <= 0.0:
        raise ValueError("signed distance field does not bracket the zero isosurface")
    vertices, faces, _normals, _values = marching_cubes(
        field,
        level=0.0,
        allow_degenerate=False,
    )
    world = trimesh.transform_points(vertices, np.asarray(transform, dtype=np.float64))
    return np.asarray(world, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _replace_narrow_band_magnitude(
    phi: np.ndarray,
    solid: np.ndarray,
    narrow_band: np.ndarray,
    transform: np.ndarray,
    points: np.ndarray,
    faces: np.ndarray,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    indices = np.argwhere(narrow_band)
    if indices.size == 0:
        raise ValueError("signed distance field has no narrow-band voxels")
    if chunk_size <= 0:
        raise ValueError("closest-point chunk size must be positive")
    packed = np.empty((faces.shape[0], 4), dtype=np.int64)
    packed[:, 0] = 3
    packed[:, 1:] = faces
    surface = pv.PolyData(points, packed.ravel())
    maximum = 0.0
    for start in range(0, indices.shape[0], chunk_size):
        stop = min(start + chunk_size, indices.shape[0])
        chunk_indices = indices[start:stop]
        queries = trimesh.transform_points(chunk_indices, transform)
        _cell_ids, closest = surface.find_closest_cell(
            queries,
            return_closest_point=True,
        )
        distances = np.linalg.norm(queries - np.asarray(closest), axis=1)
        signs = np.where(solid[tuple(chunk_indices.T)], -1.0, 1.0)
        phi[tuple(chunk_indices.T)] = distances * signs
        if distances.size:
            maximum = max(maximum, float(np.max(distances)))
    return {
        "method": "pyvista_static_cell_locator_batch_closest_point",
        "query_voxels": int(indices.shape[0]),
        "chunk_size": int(chunk_size),
        "maximum_exact_distance": maximum,
    }


def _validate_mesh(
    points: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_points = np.asarray(points, dtype=np.float64)
    source_faces = np.asarray(faces, dtype=np.int64)
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("points must have shape (point_count, 3)")
    if source_faces.ndim != 2 or source_faces.shape[1] != 3:
        raise ValueError("faces must have shape (face_count, 3)")
    if source_points.shape[0] == 0 or source_faces.shape[0] == 0:
        raise ValueError("mesh must be non-empty")
    if not np.all(np.isfinite(source_points)):
        raise ValueError("points must contain only finite coordinates")
    if int(source_faces.min()) < 0 or int(source_faces.max()) >= source_points.shape[0]:
        raise ValueError("faces contain an invalid point index")
    return source_points, source_faces


def _validate_config(config: SdfClosureConfig) -> None:
    if not np.isfinite(config.pitch) or config.pitch <= 0.0:
        raise ValueError("pitch must be finite and positive")
    if config.seal_radius_voxels < 1:
        raise ValueError("seal_radius_voxels must be at least one")
    if not np.isfinite(config.band_voxels) or config.band_voxels < 2.0:
        raise ValueError("band_voxels must be finite and at least two")
    if not np.isfinite(config.smoothing_sigma_voxels) or config.smoothing_sigma_voxels < 0.0:
        raise ValueError("smoothing_sigma_voxels must be finite and non-negative")
    if not np.isfinite(config.max_projection_distance) or config.max_projection_distance < 0.0:
        raise ValueError("max_projection_distance must be finite and non-negative")
