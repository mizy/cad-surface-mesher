from __future__ import annotations

from dataclasses import fields, is_dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np


def validate_finite_config(
    config: Any,
    expected_type: type,
    *,
    integer_fields: frozenset[str] = frozenset(),
) -> str | None:
    """Reject mistyped, boolean, complex, and non-finite config values."""

    if not isinstance(config, expected_type) or not is_dataclass(config):
        return f"config must be {expected_type.__name__}"
    for field in fields(config):
        value = getattr(config, field.name)
        if field.name in integer_fields:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                return f"{field.name} must be an integer"
        elif isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            return f"{field.name} must be a real number"
        try:
            numeric = float(value)
        except (OverflowError, TypeError, ValueError):
            return f"{field.name} must be finite"
        if not np.isfinite(numeric):
            return f"{field.name} must be finite"
    return None


def read_bounded_real(
    value: Any,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> tuple[float | None, str | None]:
    """Read a finite scalar without invoking unsafe NumPy coercion paths."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None, f"{name} must be a real number"
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None, f"{name} must be finite"
    if not np.isfinite(result) or not minimum <= result <= maximum:
        return None, f"{name} must be within [{minimum}, {maximum}]"
    return result, None


def validate_boundary_cycle_graph(
    faces: np.ndarray, loop: np.ndarray
) -> tuple[str | None, dict[str, Any]]:
    """Require each loop vertex to have only its two cycle neighbors in the boundary graph."""

    directed = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    incident = np.isin(directed[:, 0], loop) | np.isin(directed[:, 1], loop)
    canonical = np.sort(directed[incident], axis=1)
    unique_edges, counts = np.unique(canonical, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    for index, vertex_id in enumerate(loop):
        rows = boundary_edges[np.any(boundary_edges == vertex_id, axis=1)]
        neighbors = set(int(value) for value in rows.reshape(-1) if value != vertex_id)
        expected = {
            int(loop[index - 1]),
            int(loop[(index + 1) % loop.size]),
        }
        diagnostics = {
            "vertex_id": int(vertex_id),
            "boundary_degree": int(rows.shape[0]),
            "boundary_neighbors": sorted(neighbors),
            "expected_neighbors": sorted(expected),
        }
        if rows.shape[0] != 2:
            return "boundary_loop_vertex_degree_not_two", diagnostics
        if neighbors != expected:
            return "boundary_loop_vertex_neighbors_mismatch", diagnostics
    return None, {}


def geometric_tolerance(points: np.ndarray) -> float:
    """Scale a local tolerance from loop extent and coordinate precision only."""

    extent = float(np.ptp(points, axis=0).max())
    coordinate_ulp = float(np.max(np.abs(np.spacing(points)), initial=0.0))
    return max(
        extent * np.finfo(np.float64).eps * 512.0,
        coordinate_ulp * 8.0,
        np.finfo(np.float64).tiny,
    )


def read_bounded_boundary_loop(
    value: Any,
    *,
    maximum_vertices: int,
) -> tuple[np.ndarray | None, str | None]:
    """Read an untrusted loop without silently coercing fractional point IDs."""

    if (
        isinstance(maximum_vertices, (bool, np.bool_))
        or not isinstance(maximum_vertices, Integral)
        or maximum_vertices < 3
    ):
        return None, "maximum_vertices must be an integer of at least three"
    try:
        raw = np.asarray(value)
    except (OverflowError, TypeError, ValueError) as exc:
        return None, f"boundary loop cannot be read: {exc}"
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        return None, "boundary loop must be a one-dimensional integer array"
    if raw.size > maximum_vertices:
        return None, f"boundary loop has {raw.size} vertices; limit is {maximum_vertices}"
    int64 = np.iinfo(np.int64)
    if raw.size and (np.min(raw) < int64.min or np.max(raw) > int64.max):
        return None, "boundary loop point IDs exceed the supported integer range"
    return raw.astype(np.int64, copy=False), None
