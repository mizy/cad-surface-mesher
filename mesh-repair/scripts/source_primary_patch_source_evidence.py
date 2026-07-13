from __future__ import annotations

from typing import Any, Mapping, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from source_primary_patch_contract import BoundaryMapping, PatchDelta


def validate_source_normal_evidence(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    mappings: tuple[BoundaryMapping, ...],
    normal: Mapping[str, Any],
    method: str,
    delta: PatchDelta,
) -> list[str]:
    """Bind reported orientation and chart evidence back to immutable geometry."""

    if method == "paired_loop_zipper":
        return _validate_paired_patch_normal(source_points, normal, delta)
    if len(mappings) != 1:
        return [f"{method} source-normal evidence requires exactly one mapped boundary"]
    mapping = mappings[0]
    loop = np.asarray(mapping.source_vertex_ids, dtype=np.int64)
    face_ids = np.asarray(mapping.source_edge_face_ids, dtype=np.int64)
    if (
        loop.size < 3
        or np.min(loop) < 0
        or np.max(loop) >= source_points.shape[0]
        or face_ids.shape != (loop.size,)
        or np.min(face_ids) < 0
        or np.max(face_ids) >= source_faces.shape[0]
    ):
        return ["source-normal evidence cannot be bound to invalid boundary mappings"]
    boundary = np.asarray(source_points, dtype=np.float64)[loop]
    lengths = np.linalg.norm(np.roll(boundary, -1, axis=0) - boundary, axis=1)
    perimeter = float(lengths.sum())
    if not np.isfinite(perimeter) or perimeter <= 0.0:
        return ["source-normal boundary perimeter is degenerate"]
    center = np.sum(
        0.5 * (boundary + np.roll(boundary, -1, axis=0)) * lengths[:, None], axis=0
    ) / perimeter
    centered = boundary - center
    newell = np.sum(np.cross(centered, np.roll(centered, -1, axis=0)), axis=0)
    patch_normal = _unit(-newell)
    try:
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return ["source-normal boundary PCA failed"]
    if patch_normal is None:
        return ["source-normal boundary Newell normal is degenerate"]
    pca_normal = axes[-1]
    if float(np.dot(pca_normal, patch_normal)) < 0.0:
        pca_normal = -pca_normal
    adjacent_normals, adjacent_areas = _face_normals_and_double_areas(
        np.asarray(source_points, dtype=np.float64),
        np.asarray(source_faces, dtype=np.int64)[face_ids],
    )
    adjacent = _unit(np.sum(adjacent_normals * adjacent_areas[:, None], axis=0))
    if adjacent is None:
        return ["source-normal adjacent face field is degenerate"]
    frame_normal = _unit(patch_normal + pca_normal + adjacent)
    if frame_normal is None:
        return ["source-normal parameterization consensus is degenerate"]
    u_axis = _deterministic_tangent(axes[0], frame_normal)
    if u_axis is None:
        return ["source-normal parameterization tangent is degenerate"]
    expected = {
        "oriented_normal": frame_normal,
        "parameterization_normal": frame_normal,
        "parameterization_origin": center,
        "parameterization_u_axis": u_axis,
        "parameterization_v_axis": np.cross(frame_normal, u_axis),
        "boundary_induced_patch_normal": patch_normal,
        "pca_normal": pca_normal,
        "adjacent_area_weighted_normal": adjacent,
    }
    errors: list[str] = []
    scale = max(float(np.linalg.norm(np.ptp(boundary, axis=0))), 1.0)
    for name, expected_value in expected.items():
        actual = _read_vector(normal.get(name))
        absolute = scale * np.finfo(np.float64).eps * 4096.0 if name.endswith("origin") else 1e-6
        if actual is None or not np.allclose(actual, expected_value, rtol=1e-6, atol=absolute):
            errors.append(f"normal evidence {name} does not match immutable source geometry")
    supplied = normal.get("external_orientation_supplied") is True
    expected_method = (
        "external_checked_source_winding_newell_pca_area_weighted_consensus"
        if supplied
        else "source_winding_newell_pca_area_weighted_consensus"
    )
    if normal.get("method") != expected_method:
        errors.append("normal evidence method does not match its source-orientation inputs")
    if supplied:
        external = _read_vector(normal.get("external_oriented_normal"))
        threshold = _read_float(normal.get("external_orientation_strong_alignment_threshold"))
        if (
            external is None
            or threshold is None
            or threshold < 0.8
            or float(np.dot(external, frame_normal)) < threshold
        ):
            errors.append("external oriented normal is not strongly source-consistent")
    vertex_normals = _read_numeric(normal.get("boundary_vertex_normals"))
    if (
        vertex_normals is None
        or vertex_normals.shape != (loop.size, 3)
        or not np.all(np.isfinite(vertex_normals))
        or np.any(np.linalg.norm(vertex_normals, axis=1) < 1.0 - 1.0e-6)
    ):
        errors.append("boundary vertex-normal evidence is incomplete")
    return errors


def _validate_paired_patch_normal(
    source_points: np.ndarray,
    normal: Mapping[str, Any],
    delta: PatchDelta,
) -> list[str]:
    if normal.get("method") not in {
        "paired_boundary_reference",
        "paired_immutable_source_boundary_normal_field",
    }:
        return ["paired-loop normal evidence method is invalid"]
    first_row = normal.get("first_boundary")
    second_row = normal.get("second_boundary")
    first = _read_vector(
        first_row.get("oriented_normal") if isinstance(first_row, Mapping) else None
    )
    second = _read_vector(
        second_row.get("oriented_normal") if isinstance(second_row, Mapping) else None
    )
    oriented = _read_vector(normal.get("oriented_normal"))
    if first is None or second is None or oriented is None:
        return ["paired-loop immutable source normal evidence is incomplete"]
    if float(np.dot(first, second)) < 0.0:
        second = -second
    reference = _unit(first + second)
    if reference is None or not np.allclose(
        oriented, reference, rtol=1.0e-6, atol=1.0e-8
    ):
        return ["paired-loop oriented normal does not match source boundaries"]
    return []


def _face_normals_and_double_areas(
    points: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    triangles = points[faces]
    raw = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(raw, axis=1)
    normals = np.divide(
        raw, lengths[:, None], out=np.zeros_like(raw), where=lengths[:, None] > 0.0
    )
    return normals, lengths


def _deterministic_tangent(candidate: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    tangent = candidate - float(np.dot(candidate, normal)) * normal
    tangent = _unit(tangent)
    if tangent is None:
        axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
        tangent = _unit(axis - float(np.dot(axis, normal)) * normal)
    if tangent is None:
        return None
    pivot = int(np.argmax(np.abs(tangent)))
    return -tangent if tangent[pivot] < 0.0 else tangent


def _unit(value: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(value))
    return None if not np.isfinite(length) or length <= 1e-30 else value / length


def _read_vector(value: Any) -> np.ndarray | None:
    array = _read_numeric(value)
    return array if array is not None and array.shape == (3,) and np.all(np.isfinite(array)) else None


def _read_numeric(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            return None
        return array.astype(np.float64, copy=False)
    except (OverflowError, TypeError, ValueError):
        return None


def _read_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None
