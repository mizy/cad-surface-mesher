from __future__ import annotations

from typing import Any

import numpy as np
import trimesh
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_propagation,
    distance_transform_edt,
    generate_binary_structure,
)


DEFAULT_GRID_SIZE = 128
DEFAULT_SEAL_RADIUS_VOXELS = 1
DEFAULT_SURFACE_BAND_VOXELS = 1.5
FACE_SAMPLE_COUNT = 7


def build_sealed_exterior_atlas(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    seal_radius_voxels: int = DEFAULT_SEAL_RADIUS_VOXELS,
    surface_band_voxels: float = DEFAULT_SURFACE_BAND_VOXELS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return evidence that each source face touches far-field-connected space.

    The polygon soup is voxelized at a bbox-normalized resolution. A small
    morphological shell thickening seals sub-grid cracks before a six-connected
    flood is propagated inward from the padded grid boundary. The thickening is
    retained only in this evidence grid: eroding it before the flood would reopen
    the very panel seams that the atlas must ignore. Source triangles are sampled
    against the resulting outside-distance band; neither points nor faces are
    modified.
    """
    source_points, source_faces = _validated_polygon_soup(points, faces)
    _validate_thresholds(grid_size, seal_radius_voxels, surface_band_voxels)

    bounds_min = np.min(source_points, axis=0)
    bounds_max = np.max(source_points, axis=0)
    extents = bounds_max - bounds_min
    max_extent = float(np.max(extents))
    if not np.isfinite(max_extent) or max_extent <= 0.0:
        raise ValueError("cannot build an exterior atlas from a zero-size mesh")

    pitch = max_extent / float(grid_size)
    source = trimesh.Trimesh(vertices=source_points, faces=source_faces, process=False)
    voxel_grid = source.voxelized(pitch=pitch)
    source_shell = np.asarray(voxel_grid.matrix, dtype=bool)

    padding = max(
        2,
        int(seal_radius_voxels) + int(np.ceil(surface_band_voxels)) + 1,
    )
    shell, sealed_shell, outside, filled, far_field_seeds = _sealed_outside_fill(
        source_shell,
        seal_radius_voxels=seal_radius_voxels,
        padding=padding,
    )
    free = ~sealed_shell

    samples = _face_samples(source_points[source_faces])
    unpadded_indices = np.asarray(
        voxel_grid.points_to_indices(samples.reshape(-1, 3)),
        dtype=np.int64,
    )
    padded_indices = unpadded_indices + padding
    clamped_indices = np.clip(
        padded_indices,
        np.zeros(3, dtype=np.int64),
        np.asarray(outside.shape, dtype=np.int64) - 1,
    )
    clamped_sample_count = int(
        np.count_nonzero(np.any(clamped_indices != padded_indices, axis=1))
    )

    distance_to_outside = distance_transform_edt(~outside)
    sample_distances = distance_to_outside[tuple(clamped_indices.T)].reshape(
        source_faces.shape[0],
        FACE_SAMPLE_COUNT,
    )
    face_distances = np.min(sample_distances, axis=1)
    exterior_mask = face_distances <= float(surface_band_voxels)

    original_shell_in_padded_grid = shell
    added_by_sealing = sealed_shell & ~original_shell_in_padded_grid
    removed_by_sealing = original_shell_in_padded_grid & ~sealed_shell
    enclosed_empty = filled & ~sealed_shell
    face_count = int(source_faces.shape[0])
    exterior_face_count = int(np.count_nonzero(exterior_mask))
    report = {
        "schema": "sealed_exterior_atlas/v1",
        "method": "bbox_normalized_voxel_thickened_seal_six_connected_far_field_flood",
        "role": "evidence_only",
        "geometry_modified": False,
        "input": {
            "point_count": int(source_points.shape[0]),
            "face_count": face_count,
            "bounds_min": [float(value) for value in bounds_min],
            "bounds_max": [float(value) for value in bounds_max],
            "bbox_extents": [float(value) for value in extents],
            "bbox_max_extent": max_extent,
        },
        "thresholds": {
            "grid_size_longest_axis": int(grid_size),
            "pitch": float(pitch),
            "seal_radius_voxels": int(seal_radius_voxels),
            "seal_radius_world": float(seal_radius_voxels * pitch),
            "surface_band_voxels": float(surface_band_voxels),
            "surface_band_world": float(surface_band_voxels * pitch),
            "padding_voxels": int(padding),
            "flood_connectivity": 6,
            "face_samples_per_triangle": FACE_SAMPLE_COUNT,
        },
        "voxel_statistics": {
            "unpacked_matrix_shape": [int(value) for value in source_shell.shape],
            "padded_matrix_shape": [int(value) for value in shell.shape],
            "unsealed_shell_voxels": int(np.count_nonzero(shell)),
            "sealed_shell_voxels": int(np.count_nonzero(sealed_shell)),
            "voxels_added_by_sealing": int(np.count_nonzero(added_by_sealing)),
            "voxels_removed_by_sealing": int(np.count_nonzero(removed_by_sealing)),
            "far_field_seed_voxels": int(np.count_nonzero(far_field_seeds)),
            "outside_voxels": int(np.count_nonzero(outside)),
            "filled_or_shell_voxels": int(np.count_nonzero(filled)),
            "enclosed_empty_voxels": int(np.count_nonzero(enclosed_empty)),
            "free_voxels_after_sealing": int(np.count_nonzero(free)),
        },
        "face_evidence": {
            "exterior_face_count": exterior_face_count,
            "interior_or_occluded_face_count": face_count - exterior_face_count,
            "exterior_face_ratio": float(exterior_face_count / max(face_count, 1)),
            "minimum_distance_to_outside_voxels": float(np.min(face_distances)),
            "maximum_distance_to_outside_voxels": float(np.max(face_distances)),
            "clamped_sample_count": clamped_sample_count,
        },
        "output": {
            "face_count": face_count,
            "mask_dtype": "bool",
        },
    }
    return exterior_mask, report


def build_sealed_exterior_volume_grid(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    pitch: float,
    seal_radius_voxels: int = DEFAULT_SEAL_RADIUS_VOXELS,
    max_projection_distance: float = 0.0,
) -> tuple[trimesh.voxel.VoxelGrid, dict[str, Any]]:
    """Build the solid region enclosed by a locally sealed polygon soup.

    Unlike :meth:`trimesh.voxel.VoxelGrid.fill`, this construction deliberately
    seals sub-grid cracks *before* propagating outside space from a padded far
    field.  It therefore yields an exterior solid instead of a watertight tube
    around every still-leaky input panel.

    The returned grid retains padding so marching cubes has a guaranteed empty
    layer around the solid.  Its transform is shifted with that padding, keeping
    all generated points in the input mesh's world coordinates.
    """
    source_points, source_faces = _validated_polygon_soup(points, faces)
    if not np.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("pitch must be finite and positive")
    if not np.isfinite(max_projection_distance) or max_projection_distance < 0.0:
        raise ValueError("max_projection_distance must be finite and non-negative")
    if isinstance(seal_radius_voxels, bool) or not isinstance(
        seal_radius_voxels,
        (int, np.integer),
    ):
        raise ValueError("seal_radius_voxels must be an integer")
    if seal_radius_voxels < 1:
        raise ValueError("exterior-volume construction requires seal_radius_voxels >= 1")

    source = trimesh.Trimesh(
        vertices=source_points,
        faces=source_faces,
        process=False,
    )
    source_grid = source.voxelized(pitch=float(pitch))
    source_shell = np.asarray(source_grid.matrix, dtype=bool)
    padding = max(2, int(seal_radius_voxels) + 2)
    shell, sealed_shell, outside, filled, far_field_seeds = _sealed_outside_fill(
        source_shell,
        seal_radius_voxels=int(seal_radius_voxels),
        padding=padding,
    )

    transform = np.asarray(source_grid.transform, dtype=np.float64).copy()
    transform[:3, 3] -= transform[:3, :3] @ np.full(3, padding, dtype=np.float64)
    restored_filled = binary_erosion(
        filled,
        structure=_euclidean_ball(int(seal_radius_voxels)),
        border_value=0,
    )
    if not np.any(restored_filled):
        raise ValueError("sealed exterior volume vanished while restoring dilation offset")
    core_radius_voxels = projection_erosion_radius_voxels(
        max_projection_distance,
        pitch,
    )
    erosion_core = (
        restored_filled
        if core_radius_voxels == 0
        else binary_erosion(
            restored_filled,
            structure=_euclidean_ball(core_radius_voxels),
            border_value=0,
        )
    )
    core_mesh_volume = None
    core_mesh_watertight = False
    core_mesh_triangles = 0
    if np.any(erosion_core):
        core_grid = trimesh.voxel.VoxelGrid(erosion_core, transform=transform)
        core_mesh = core_grid.marching_cubes
        core_mesh.apply_transform(core_grid.transform)
        core_mesh.remove_unreferenced_vertices()
        try:
            core_mesh.fix_normals()
        except Exception:
            pass
        core_mesh_volume = float(abs(core_mesh.volume))
        core_mesh_watertight = bool(core_mesh.is_watertight)
        core_mesh_triangles = int(core_mesh.faces.shape[0])
    filled_grid = trimesh.voxel.VoxelGrid(restored_filled, transform=transform)
    enclosed_empty = restored_filled & ~shell
    report = {
        "schema": "sealed_exterior_volume/v1",
        "method": "voxel_shell_dilation_six_connected_far_field_flood",
        "pitch": float(pitch),
        "seal_radius_voxels": int(seal_radius_voxels),
        "seal_radius_world": float(seal_radius_voxels * pitch),
        "padding_voxels": int(padding),
        "unpacked_matrix_shape": [int(value) for value in source_shell.shape],
        "padded_matrix_shape": [int(value) for value in filled.shape],
        "source_shell_voxels": int(np.count_nonzero(shell)),
        "sealed_shell_voxels": int(np.count_nonzero(sealed_shell)),
        "voxels_added_by_sealing": int(np.count_nonzero(sealed_shell & ~shell)),
        "far_field_seed_voxels": int(np.count_nonzero(far_field_seeds)),
        "outside_voxels": int(np.count_nonzero(outside)),
        "flood_filled_or_shell_voxels_before_offset_restore": int(
            np.count_nonzero(filled)
        ),
        "filled_or_shell_voxels": int(np.count_nonzero(restored_filled)),
        "voxels_removed_by_offset_restore": int(
            np.count_nonzero(filled & ~restored_filled)
        ),
        "surface_offset_restore": "erode_filled_solid_by_seal_radius",
        "enclosed_empty_voxels": int(np.count_nonzero(enclosed_empty)),
        "estimated_filled_volume": float(
            np.count_nonzero(restored_filled) * pitch**3
        ),
        "estimated_enclosed_empty_volume": float(
            np.count_nonzero(enclosed_empty) * pitch**3
        ),
        "projection_erosion_core": {
            "method": "binary_erosion_by_ceil_projection_distance_over_pitch",
            "max_projection_distance": float(max_projection_distance),
            "erosion_radius_voxels": int(core_radius_voxels),
            "erosion_radius_world": float(core_radius_voxels * pitch),
            "filled_voxels": int(np.count_nonzero(erosion_core)),
            "estimated_volume": float(np.count_nonzero(erosion_core) * pitch**3),
            "mesh_volume_method": "marching_cubes_signed_abs_volume",
            "mesh_signed_abs_volume": core_mesh_volume,
            "mesh_watertight": core_mesh_watertight,
            "mesh_triangles": core_mesh_triangles,
            "nonempty": bool(np.any(erosion_core)),
        },
    }
    return filled_grid, report


def projection_erosion_radius_voxels(
    max_projection_distance: float,
    pitch: float,
) -> int:
    """Return the smallest voxel erosion radius covering a world distance."""
    if not np.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("pitch must be finite and positive")
    if not np.isfinite(max_projection_distance) or max_projection_distance < 0.0:
        raise ValueError("max_projection_distance must be finite and non-negative")
    ratio = float(max_projection_distance) / float(pitch)
    roundoff = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(ratio))
    return int(np.ceil(max(0.0, ratio - roundoff)))


def _sealed_outside_fill(
    source_shell: np.ndarray,
    *,
    seal_radius_voxels: int,
    padding: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Seal a voxel shell and partition its complement from the far field."""
    source_shell = np.asarray(source_shell, dtype=bool)
    if source_shell.ndim != 3 or not all(size > 0 for size in source_shell.shape):
        raise ValueError("source_shell must be a non-empty rank-3 array")
    if seal_radius_voxels < 0:
        raise ValueError("seal_radius_voxels must be non-negative")
    if padding <= seal_radius_voxels:
        raise ValueError("padding must exceed seal_radius_voxels")

    shell = np.pad(
        source_shell,
        int(padding),
        mode="constant",
        constant_values=False,
    )
    sealed_shell = binary_dilation(
        shell,
        structure=_euclidean_ball(int(seal_radius_voxels)),
    )
    free = ~sealed_shell
    far_field_seeds = _array_boundary_mask(free.shape) & free
    outside = binary_propagation(
        far_field_seeds,
        structure=generate_binary_structure(3, 1),
        mask=free,
    )
    filled = ~outside
    return shell, sealed_shell, outside, filled, far_field_seeds


def _validated_polygon_soup(
    points: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_points = np.asarray(points, dtype=np.float64)
    source_faces = np.asarray(faces, dtype=np.int64)
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if source_faces.ndim != 2 or source_faces.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if source_points.shape[0] == 0 or source_faces.shape[0] == 0:
        raise ValueError("polygon soup must contain points and triangular faces")
    if not np.all(np.isfinite(source_points)):
        raise ValueError("points must contain only finite coordinates")
    if np.any(source_faces < 0) or np.any(source_faces >= source_points.shape[0]):
        raise ValueError("faces contain an out-of-range point index")
    return source_points, source_faces


def _validate_thresholds(
    grid_size: int,
    seal_radius_voxels: int,
    surface_band_voxels: float,
) -> None:
    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)):
        raise ValueError("grid_size must be an integer")
    if grid_size < 16:
        raise ValueError("grid_size must be at least 16")
    if isinstance(seal_radius_voxels, bool) or not isinstance(
        seal_radius_voxels,
        (int, np.integer),
    ):
        raise ValueError("seal_radius_voxels must be an integer")
    if seal_radius_voxels < 0:
        raise ValueError("seal_radius_voxels must be non-negative")
    if not np.isfinite(surface_band_voxels) or surface_band_voxels < 0.0:
        raise ValueError("surface_band_voxels must be finite and non-negative")


def _euclidean_ball(radius: int) -> np.ndarray:
    if radius == 0:
        return np.ones((1, 1, 1), dtype=bool)
    coordinate = np.arange(-radius, radius + 1, dtype=np.int64)
    x, y, z = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    return x * x + y * y + z * z <= radius * radius


def _array_boundary_mask(shape: tuple[int, ...]) -> np.ndarray:
    boundary = np.zeros(shape, dtype=bool)
    boundary[0, :, :] = True
    boundary[-1, :, :] = True
    boundary[:, 0, :] = True
    boundary[:, -1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True
    return boundary


def _face_samples(triangles: np.ndarray) -> np.ndarray:
    vertices = triangles
    edge_midpoints = np.stack(
        (
            0.5 * (vertices[:, 0] + vertices[:, 1]),
            0.5 * (vertices[:, 1] + vertices[:, 2]),
            0.5 * (vertices[:, 2] + vertices[:, 0]),
        ),
        axis=1,
    )
    centroids = np.mean(vertices, axis=1, keepdims=True)
    return np.concatenate((centroids, vertices, edge_midpoints), axis=1)
