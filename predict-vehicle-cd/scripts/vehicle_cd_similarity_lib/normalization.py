from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pyvista as pv

from .constants import (
    DEFAULT_TARGET_LENGTH_M,
    NORMALIZED_LENGTH_RANGE,
    PCA_AXIS_RATIO_THRESHOLD,
    REALISTIC_LENGTH_RANGE_M,
)
from .types import NormalizedVehicle


def bounds_dict(points: np.ndarray) -> dict[str, list[float]]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return {
        "min": [float(v) for v in mins],
        "max": [float(v) for v in maxs],
        "extent": [float(v) for v in maxs - mins],
    }


def axis_warning(extents: np.ndarray) -> str | None:
    ordered = np.sort(extents)[::-1]
    if ordered[1] <= 1e-12 or ordered[2] <= 1e-12:
        return "Degenerate mesh extents; axis detection may be unreliable."
    if ordered[0] / ordered[1] < 1.25:
        return "Length and width spans are close; extent-based length axis is low-confidence."
    if ordered[1] / ordered[2] < 1.10:
        return "Width and height spans are close; extent-based height axis is low-confidence."
    return None


def canonicalize_vehicle_axes(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    spans = np.ptp(points, axis=0).astype(np.float64)
    order = np.argsort(spans)[::-1].astype(int)
    ordered_spans = spans[order]
    info: dict[str, Any] = {
        "method": "extent_order",
        "extent_order_raw_axes": [int(v) for v in order],
        "warning": axis_warning(spans),
    }

    if ordered_spans[1] > 1e-12 and ordered_spans[0] / ordered_spans[1] < PCA_AXIS_RATIO_THRESHOLD:
        height_axis = int(order[2])
        horizontal_axes = [axis for axis in range(3) if axis != height_axis]
        horizontal = points[:, horizontal_axes]
        centered_horizontal = horizontal - horizontal.mean(axis=0)
        cov = np.cov(centered_horizontal, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        pca_order = np.argsort(eigvals)[::-1]
        length_vec = eigvecs[:, pca_order[0]]
        width_vec = eigvecs[:, pca_order[1]]
        canonical = np.column_stack(
            (
                centered_horizontal @ length_vec,
                centered_horizontal @ width_vec,
                points[:, height_axis],
            )
        )
        info.update(
            {
                "method": "horizontal_pca",
                "height_raw_axis": height_axis,
                "horizontal_raw_axes": [int(v) for v in horizontal_axes],
                "pca_length_vector": [float(v) for v in length_vec],
                "pca_width_vector": [float(v) for v in width_vec],
                "warning": "Length and width spans are close; used horizontal PCA to align length/width.",
            }
        )
        return canonical, info

    return points[:, order].copy(), info


def should_flip_z(canonical_points: np.ndarray) -> tuple[bool, dict[str, float]]:
    z = canonical_points[:, 2]
    height = float(np.ptp(z))
    if height <= 1e-12:
        return False, {"bottom_fraction": 0.0, "top_fraction": 0.0, "confidence": 0.0}

    z_min = float(z.min())
    z_max = float(z.max())
    band = max(height * 0.08, 1e-12)
    bottom_fraction = float(np.mean(z <= z_min + band))
    top_fraction = float(np.mean(z >= z_max - band))
    confidence = abs(top_fraction - bottom_fraction)
    return top_fraction > bottom_fraction * 1.4 and confidence > 0.03, {
        "bottom_fraction": bottom_fraction,
        "top_fraction": top_fraction,
        "confidence": confidence,
    }


def detect_length_scale(length: float, target_length: float, *, force_target_length: bool = False) -> dict[str, Any]:
    if length <= 1e-12:
        raise ValueError("Mesh has zero length after axis normalization")
    if target_length <= 0:
        raise ValueError("--target-length must be positive")

    if force_target_length:
        scale = target_length / length
        return {
            "detected_unit": "forced_target_length",
            "unit_scale_to_m": None,
            "uniform_scale": scale,
            "length_m": target_length,
            "reason": f"Forced length {length:.3f} to target length {target_length:.3f} m.",
        }

    min_real, max_real = REALISTIC_LENGTH_RANGE_M
    for unit, scale in (("m", 1.0), ("mm", 0.001), ("cm", 0.01)):
        scaled_length = length * scale
        if min_real <= scaled_length <= max_real:
            return {
                "detected_unit": unit,
                "unit_scale_to_m": scale,
                "uniform_scale": scale,
                "length_m": scaled_length,
                "reason": f"Length {scaled_length:.3f} m falls in realistic vehicle range.",
            }

    if NORMALIZED_LENGTH_RANGE[0] <= length <= NORMALIZED_LENGTH_RANGE[1]:
        scale = target_length / length
        return {
            "detected_unit": "unitless_normalized",
            "unit_scale_to_m": None,
            "uniform_scale": scale,
            "length_m": target_length,
            "reason": f"Length {length:.3f} looks normalized; scaled to target length.",
        }

    scale = target_length / length
    return {
        "detected_unit": "unknown_scaled_to_target",
        "unit_scale_to_m": None,
        "uniform_scale": scale,
        "length_m": target_length,
        "reason": f"Length {length:.3f} did not match m/mm/cm ranges; scaled to target length.",
    }


def parse_target_dimensions(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected L,W,H in meters, for example 4.8,1.875,1.46")
    try:
        dimensions = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Target dimensions must be numeric meters") from exc
    if any(dimension <= 0 for dimension in dimensions):
        raise argparse.ArgumentTypeError("Target dimensions must be positive")
    return dimensions  # type: ignore[return-value]


def normalize_vehicle_mesh(
    mesh: pv.PolyData,
    *,
    rotate_180_z: bool = False,
    target_length: float = DEFAULT_TARGET_LENGTH_M,
    force_target_length: bool = False,
    target_dimensions: tuple[float, float, float] | None = None,
) -> NormalizedVehicle:
    out = mesh.copy(deep=True)
    points = np.asarray(out.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.size == 0:
        raise ValueError("Mesh points must be an N x 3 array")

    raw_bounds = bounds_dict(points)
    canonical, axis_info = canonicalize_vehicle_axes(points)

    z_flipped, z_flip_stats = should_flip_z(canonical)
    if z_flipped:
        canonical[:, 2] *= -1.0

    length_before_scale = float(np.ptp(canonical[:, 0]))
    if target_dimensions is not None:
        current_extents = np.ptp(canonical, axis=0)
        if np.any(current_extents <= 1e-12):
            raise ValueError("Mesh has zero extent on at least one axis after axis normalization")
        scale_vector = np.asarray(target_dimensions, dtype=np.float64) / current_extents
        canonical *= scale_vector
        scale_info = {
            "detected_unit": "forced_target_dimensions",
            "unit_scale_to_m": None,
            "uniform_scale": None,
            "scale_vector_xyz": [float(v) for v in scale_vector],
            "length_m": float(target_dimensions[0]),
            "reason": (
                "Forced extents to target dimensions "
                f"{target_dimensions[0]:.3f} x {target_dimensions[1]:.3f} x {target_dimensions[2]:.3f} m."
            ),
        }
    else:
        scale_info = detect_length_scale(length_before_scale, target_length, force_target_length=force_target_length)
        canonical *= float(scale_info["uniform_scale"])

    if rotate_180_z:
        canonical[:, 0] *= -1.0
        canonical[:, 1] *= -1.0

    canonical[:, 0] -= 0.5 * (float(canonical[:, 0].min()) + float(canonical[:, 0].max()))
    canonical[:, 1] -= 0.5 * (float(canonical[:, 1].min()) + float(canonical[:, 1].max()))
    canonical[:, 2] -= float(canonical[:, 2].min())

    out.points = canonical.astype(np.float32)
    final_points = np.asarray(out.points, dtype=np.float64)
    info = {
        "target_orientation": {"length": "X", "width": "Y", "height": "Z", "front": "-X", "top": "+Z"},
        "raw_bounds": raw_bounds,
        "axis_order_raw_to_xyz": axis_info["extent_order_raw_axes"],
        "axis_order_note": "new X/Y/Z coordinates are raw axes at these indices when axis_method is extent_order",
        "axis_method": axis_info["method"],
        "axis_warning": axis_info["warning"],
        "axis_details": {
            key: value
            for key, value in axis_info.items()
            if key not in {"method", "extent_order_raw_axes", "warning"}
        },
        "z_flipped": bool(z_flipped),
        "z_flip_stats": z_flip_stats,
        "length_before_scale": length_before_scale,
        "detected_unit": scale_info["detected_unit"],
        "unit_scale_to_m": scale_info["unit_scale_to_m"],
        "uniform_scale": float(scale_info["uniform_scale"]) if scale_info["uniform_scale"] is not None else None,
        "scale_vector_xyz": scale_info.get("scale_vector_xyz"),
        "target_dimensions_m": [float(v) for v in target_dimensions] if target_dimensions is not None else None,
        "length_m": float(scale_info["length_m"]),
        "scale_reason": scale_info["reason"],
        "rotate_180_z": bool(rotate_180_z),
        "ground_aligned_z_min": 0.0,
        "final_bounds": bounds_dict(final_points),
    }
    return NormalizedVehicle(mesh=out, info=info)
