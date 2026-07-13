from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from source_primary_patch_validation import validate_patch_candidate


PATCH_METHODS = frozenset(
    {
        "planar_cap",
        "curved_conformal_patch",
        "paired_loop_zipper",
        "slit_bridge",
        "slit_weld",
    }
)
RegionId = str | int
MAX_REGION_ID_LENGTH = 128


@dataclass(frozen=True)
class BoundaryMapping:
    """Identity mapping from one oriented source boundary into a patch delta."""

    region_id: RegionId
    source_vertex_ids: tuple[int, ...]
    candidate_vertex_ids: tuple[int, ...]
    source_edge_face_ids: tuple[int, ...]
    source_triangle_indices: tuple[int, ...]
    closed: bool = True
    orientation: str = "source_face_winding_induced"

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", normalize_region_id(self.region_id))
        for name in (
            "source_vertex_ids",
            "candidate_vertex_ids",
            "source_edge_face_ids",
            "source_triangle_indices",
        ):
            raw = np.asarray(getattr(self, name))
            if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
                raise TypeError(f"{name} must be a one-dimensional integer sequence")
            object.__setattr__(self, name, tuple(int(value) for value in raw))
        if not isinstance(self.closed, (bool, np.bool_)):
            raise TypeError("closed must be boolean")
        object.__setattr__(self, "closed", bool(self.closed))


@dataclass(frozen=True)
class PatchDelta:
    """Append-only geometry; source points and faces are deliberately absent."""

    appended_points: np.ndarray
    appended_faces: np.ndarray
    point_provenance: Mapping[str, Any]
    face_provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        raw_points = np.asarray(self.appended_points)
        raw_faces = np.asarray(self.appended_faces)
        if not np.issubdtype(raw_points.dtype, np.number) or np.iscomplexobj(raw_points):
            raise TypeError("appended_points must contain numeric values")
        if not np.issubdtype(raw_faces.dtype, np.integer):
            raise TypeError("appended_faces must contain integer point IDs")
        object.__setattr__(
            self, "appended_points", _owned_array(self.appended_points, np.float64, (-1, 3))
        )
        object.__setattr__(
            self, "appended_faces", _owned_array(self.appended_faces, np.int64, (-1, 3))
        )
        object.__setattr__(self, "point_provenance", _owned_mapping(self.point_provenance))
        object.__setattr__(self, "face_provenance", _owned_mapping(self.face_provenance))


@dataclass(frozen=True)
class PatchCandidate:
    """Uniform result contract shared by every source-primary local repairer."""

    status: str
    method: str
    delta: PatchDelta
    boundary_mapping: tuple[BoundaryMapping, ...]
    normal: Mapping[str, Any]
    curvature: Mapping[str, Any]
    source_provenance: Mapping[str, Any]
    proxy_provenance: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    failure_reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.delta, PatchDelta):
            raise TypeError("delta must be PatchDelta")
        if any(not isinstance(item, BoundaryMapping) for item in self.boundary_mapping):
            raise TypeError("boundary_mapping must contain BoundaryMapping values")
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "method", str(self.method))
        object.__setattr__(self, "boundary_mapping", tuple(self.boundary_mapping))
        for name in (
            "normal",
            "curvature",
            "source_provenance",
            "proxy_provenance",
            "diagnostics",
        ):
            object.__setattr__(self, name, _owned_mapping(getattr(self, name)))
        object.__setattr__(
            self,
            "failure_reason_codes",
            tuple(str(code) for code in self.failure_reason_codes),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "status": self.status,
                "method": self.method,
                "delta": {
                    "appended_points": self.delta.appended_points,
                    "appended_faces": self.delta.appended_faces,
                    "point_provenance": self.delta.point_provenance,
                    "face_provenance": self.delta.face_provenance,
                },
                "boundary_mapping": [mapping.__dict__ for mapping in self.boundary_mapping],
                "normal": self.normal,
                "curvature": self.curvature,
                "source_provenance": self.source_provenance,
                "proxy_provenance": self.proxy_provenance,
                "diagnostics": self.diagnostics,
                "failure_reason_codes": self.failure_reason_codes,
            }
        )


def empty_patch_delta() -> PatchDelta:
    return PatchDelta(
        appended_points=np.empty((0, 3), dtype=np.float64),
        appended_faces=np.empty((0, 3), dtype=np.int64),
        point_provenance={},
        face_provenance={},
    )


def normalize_region_id(value: RegionId) -> str:
    """Preserve the durable inventory ID while rejecting unsafe artifact keys."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError("region_id must be a durable string or integer, not boolean")
    if isinstance(value, (int, np.integer)):
        if int(value) < 0:
            raise ValueError("numeric region_id must be non-negative")
        text = str(int(value))
    elif isinstance(value, str):
        text = value
    else:
        raise TypeError("region_id must be a durable string or integer")
    if (
        not text
        or text != text.strip()
        or len(text) > MAX_REGION_ID_LENGTH
        or not text.isascii()
        or any(not (character.isalnum() or character in "_-.") for character in text)
    ):
        raise ValueError("region_id must be a non-empty, artifact-safe ASCII identifier")
    return text


def build_source_provenance(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_triangle_index: np.ndarray,
) -> dict[str, Any]:
    errors = validate_source_arrays(source_points, source_faces, source_triangle_index)
    if errors:
        raise ValueError("; ".join(errors))
    points = np.ascontiguousarray(source_points)
    faces = np.ascontiguousarray(source_faces)
    triangle_index = np.ascontiguousarray(source_triangle_index)
    return {
        "point_count": int(points.shape[0]),
        "face_count": int(faces.shape[0]),
        "source_triangle_index_count": int(triangle_index.shape[0]),
        "points_sha256": _array_sha256(points),
        "faces_sha256": _array_sha256(faces),
        "source_triangle_index_sha256": _array_sha256(triangle_index),
        "source_points_unchanged": True,
        "source_faces_unchanged": True,
        "source_triangle_index_unchanged": True,
        "mutation_policy": "append_only_patch_delta",
    }


def validate_source_arrays(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_triangle_index: np.ndarray,
) -> list[str]:
    errors: list[str] = []
    try:
        points = np.asarray(source_points)
        faces = np.asarray(source_faces)
        triangle_index = np.asarray(source_triangle_index)
    except (TypeError, ValueError) as exc:
        return [f"source arrays cannot be read: {exc}"]
    if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] < 3:
        errors.append("source_points must have shape (N, 3) with N >= 3")
    elif (
        not np.issubdtype(points.dtype, np.number)
        or np.iscomplexobj(points)
        or not np.all(np.isfinite(points))
    ):
        errors.append("source_points must contain only finite numeric values")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        errors.append("source_faces must have shape (M, 3)")
    elif not np.issubdtype(faces.dtype, np.integer):
        errors.append("source_faces must contain integer point IDs")
    elif faces.size and (
        np.min(faces) < 0 or points.ndim != 2 or np.max(faces) >= points.shape[0]
    ):
        errors.append("source_faces contain out-of-range point IDs")
    elif faces.size and np.any(
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    ):
        errors.append("source_faces contain repeated point IDs")
    face_count = faces.shape[0] if faces.ndim >= 1 else -1
    if triangle_index.ndim != 1 or triangle_index.shape[0] != face_count:
        errors.append("source_triangle_index must contain one value per source face")
    elif not np.issubdtype(triangle_index.dtype, np.integer):
        errors.append("source_triangle_index must contain integer values")
    return errors


# @entry
def finalize_patch_candidate(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_triangle_index: np.ndarray,
    *,
    method: str,
    delta: PatchDelta,
    boundary_mapping: Sequence[BoundaryMapping],
    normal: Mapping[str, Any],
    curvature: Mapping[str, Any],
    proxy_provenance: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> PatchCandidate:
    source_errors = validate_source_arrays(
        source_points, source_faces, source_triangle_index
    )
    if source_errors:
        return rejected_patch_candidate(
            source_points,
            source_faces,
            source_triangle_index,
            method=method,
            failure_reason_codes=("patch_source_arrays_invalid",),
            diagnostics={"source_validation_errors": source_errors},
        )
    source_provenance = build_source_provenance(source_points, source_faces, source_triangle_index)
    candidate = PatchCandidate(
        status="candidate",
        method=method,
        delta=delta,
        boundary_mapping=tuple(boundary_mapping),
        normal=normal,
        curvature=curvature,
        source_provenance=source_provenance,
        proxy_provenance=(
            proxy_provenance
            or {"used": False, "role": "not_used", "geometry_consumed": False}
        ),
        diagnostics=diagnostics or {},
    )
    errors = validate_patch_candidate(
        source_points, source_faces, source_triangle_index, candidate
    )
    if not errors:
        return candidate
    return rejected_patch_candidate(
        source_points,
        source_faces,
        source_triangle_index,
        method=method,
        failure_reason_codes=("patch_candidate_contract_invalid",),
        normal=normal,
        curvature=curvature,
        boundary_mapping=boundary_mapping,
        proxy_provenance=proxy_provenance,
        diagnostics={
            **dict(diagnostics or {}),
            "contract_validation_errors": errors,
        },
    )


def rejected_patch_candidate(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_triangle_index: np.ndarray,
    *,
    method: str,
    failure_reason_codes: Sequence[str],
    normal: Mapping[str, Any] | None = None,
    curvature: Mapping[str, Any] | None = None,
    boundary_mapping: Sequence[BoundaryMapping] = (),
    proxy_provenance: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> PatchCandidate:
    codes = tuple(str(code) for code in failure_reason_codes if str(code))
    if not codes:
        codes = ("patch_candidate_rejected_without_reason",)
    try:
        source_provenance = build_source_provenance(
            source_points, source_faces, source_triangle_index
        )
    except (TypeError, ValueError) as exc:
        source_provenance = {
            "valid": False,
            "validation_error": str(exc),
            "source_points_unchanged": True,
            "source_faces_unchanged": True,
            "source_triangle_index_unchanged": True,
            "mutation_policy": "append_only_patch_delta",
        }
    return PatchCandidate(
        status="rejected",
        method=method,
        delta=empty_patch_delta(),
        boundary_mapping=tuple(boundary_mapping),
        normal=normal or {"status": "unavailable"},
        curvature=curvature or {"status": "unavailable"},
        source_provenance=source_provenance,
        proxy_provenance=_safe_rejected_proxy_provenance(proxy_provenance),
        diagnostics=diagnostics or {},
        failure_reason_codes=codes,
    )


def _owned_array(value: Any, dtype: np.dtype[Any], shape: tuple[int, int]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1:] != shape[1:]:
        raise ValueError(f"array must have shape (N, {shape[1]}), got {raw.shape}")
    array = np.asarray(raw, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _safe_rejected_proxy_provenance(
    provenance: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    default = {"used": False, "role": "not_used", "geometry_consumed": False}
    if provenance is None:
        return default
    from source_primary_patch_provenance_validation import validate_proxy_provenance

    try:
        errors = validate_proxy_provenance(provenance)
    except (AttributeError, TypeError, ValueError, OverflowError):
        errors = ["proxy provenance could not be validated"]
    if errors:
        return {
            **default,
            "status": "discarded_invalid_provenance",
            "validation_errors": tuple(errors),
        }
    return provenance


def _owned_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for name, value in dict(values).items():
        if isinstance(value, np.ndarray):
            copied = value.copy()
            copied.setflags(write=False)
            result[str(name)] = copied
        elif isinstance(value, Mapping):
            result[str(name)] = _owned_mapping(value)
        elif isinstance(value, (list, tuple)):
            result[str(name)] = tuple(_owned_value(item) for item in value)
        else:
            result[str(name)] = value
    return MappingProxyType(result)


def _owned_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.setflags(write=False)
        return result
    if isinstance(value, Mapping):
        return _owned_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_owned_value(item) for item in value)
    return value


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
