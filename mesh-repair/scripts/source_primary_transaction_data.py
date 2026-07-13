from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from source_primary_patch_contract import PatchCandidate, RegionId, normalize_region_id
from source_primary_quality_geometry import compare_prefix_bytes


PATCH_METHOD_CODES = {
    "planar_cap": 1,
    "curved_conformal_patch": 2,
    "paired_loop_zipper": 3,
    "slit_bridge": 4,
}
FACE_ORIGIN_BY_METHOD = {
    "planar_cap": 3,
    "curved_conformal_patch": 1,
    "paired_loop_zipper": 2,
    "slit_bridge": 2,
}


def initialize_transaction_state(
    points: np.ndarray,
    faces: np.ndarray,
    source_triangle_index: np.ndarray,
    source_cell_data: Mapping[str, np.ndarray] | None,
    source_point_data: Mapping[str, np.ndarray] | None,
) -> dict[str, Any]:
    points = np.asarray(points)
    faces = np.asarray(faces)
    source_index = np.asarray(source_triangle_index)
    message = _validate_source_arrays(points, faces, source_index)
    cell_data = {
        name: np.asarray(values) for name, values in (source_cell_data or {}).items()
    }
    point_data = {
        name: np.asarray(values) for name, values in (source_point_data or {}).items()
    }
    if message is None:
        message = _validate_data_rows(cell_data, faces.shape[0], "source cell-data")
    if message is None:
        message = _validate_data_rows(point_data, points.shape[0], "source point-data")
    supplied_index = cell_data.get("source_triangle_index")
    if (
        message is None
        and supplied_index is not None
        and not compare_prefix_bytes(source_index, supplied_index)
    ):
        message = (
            "source_cell_data.source_triangle_index conflicts with the canonical input"
        )
    if message is None:
        message = _validate_reserved_source_fields(cell_data, point_data)
    canonical_cells = {name: values.copy() for name, values in cell_data.items()}
    canonical_points = {name: values.copy() for name, values in point_data.items()}
    canonical_cells["source_triangle_index"] = source_index.copy()
    _add_default_field(
        canonical_cells,
        "fusion_region_id",
        faces.shape[0],
        -1,
        dtype=np.int64,
    )
    _add_default_field(canonical_cells, "patch_method_code", faces.shape[0], 0)
    _add_default_field(canonical_cells, "proxy_triangle_index", faces.shape[0], -1)
    _add_default_field(canonical_cells, "face_origin", faces.shape[0], 0)
    _add_default_field(
        canonical_points,
        "fusion_region_id",
        points.shape[0],
        -1,
        dtype=np.int64,
    )
    _add_default_field(canonical_points, "patch_method_code", points.shape[0], 0)
    _add_default_field(canonical_points, "point_origin", points.shape[0], 0)
    state = {
        "points": points.copy(),
        "faces": faces.copy(),
        "cell_data": canonical_cells,
        "point_data": canonical_points,
    }
    return {"success": message is None, "state": state, "failure_reason": message}


def map_candidate_faces(
    candidate: PatchCandidate,
    *,
    source_point_count: int,
    current_point_count: int,
) -> np.ndarray:
    faces = np.asarray(candidate.delta.appended_faces, dtype=np.int64).copy()
    generated = faces >= source_point_count
    faces[generated] += current_point_count - source_point_count
    return faces


def append_candidate_delta(
    current: Mapping[str, Any],
    candidate: PatchCandidate,
    mapped_faces: np.ndarray,
    *,
    region_ids: Sequence[RegionId],
    fusion_region_id: int,
) -> dict[str, Any]:
    added_points = np.asarray(
        candidate.delta.appended_points, dtype=current["points"].dtype
    )
    face_dtype = current["faces"].dtype
    if mapped_faces.size and int(mapped_faces.max()) > int(np.iinfo(face_dtype).max):
        raise ValueError("mapped patch point ID exceeds source face dtype")
    added_faces = mapped_faces.astype(face_dtype, copy=False)
    normalized_region_ids = tuple(normalize_region_id(value) for value in region_ids)
    if not normalized_region_ids:
        raise ValueError("a patch delta must own at least one region")
    if len(set(normalized_region_ids)) != 1:
        raise ValueError("one patch delta must own one composite canonical region")
    fusion_region_id = validate_fusion_region_id(fusion_region_id)
    point_explicit = dict(candidate.delta.point_provenance)
    face_explicit = dict(candidate.delta.face_provenance)
    method_code = PATCH_METHOD_CODES[candidate.method]
    point_region = _region_array(
        current["point_data"]["fusion_region_id"],
        added_points.shape[0],
        fusion_region_id,
    )
    face_region = _region_array(
        current["cell_data"]["fusion_region_id"],
        added_faces.shape[0],
        fusion_region_id,
    )
    _set_reserved_field(point_explicit, "fusion_region_id", point_region)
    _set_reserved_field(
        point_explicit,
        "patch_method_code",
        np.full(added_points.shape[0], method_code, dtype=np.int16),
    )
    _set_reserved_field(
        point_explicit,
        "point_origin",
        np.ones(added_points.shape[0], dtype=np.int16),
    )
    _set_reserved_field(face_explicit, "fusion_region_id", face_region)
    _set_reserved_field(
        face_explicit,
        "patch_method_code",
        np.full(added_faces.shape[0], method_code, dtype=np.int16),
    )
    face_explicit.setdefault(
        "proxy_triangle_index", np.full(added_faces.shape[0], -1, dtype=np.int64)
    )
    _set_reserved_field(
        face_explicit,
        "face_origin",
        np.full(
            added_faces.shape[0],
            FACE_ORIGIN_BY_METHOD[candidate.method],
            dtype=np.int16,
        ),
    )
    if not np.all(np.asarray(face_explicit.get("source_triangle_index", [])) == -1):
        raise ValueError("generated patch faces must use source_triangle_index=-1")
    face_explicit["source_triangle_index"] = np.full(
        added_faces.shape[0],
        -1,
        dtype=current["cell_data"]["source_triangle_index"].dtype,
    )
    point_data = _extend_data(
        current["point_data"],
        point_explicit,
        current["points"].shape[0],
        added_points.shape[0],
    )
    cell_data = _extend_data(
        current["cell_data"],
        face_explicit,
        current["faces"].shape[0],
        added_faces.shape[0],
    )
    return {
        "points": np.concatenate([current["points"], added_points], axis=0),
        "faces": np.concatenate([current["faces"], added_faces], axis=0),
        "point_data": point_data,
        "cell_data": cell_data,
    }


def audit_data_prefix(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    point_changes = _changed_prefix_fields(
        baseline["point_data"], candidate["point_data"]
    )
    cell_changes = _changed_prefix_fields(baseline["cell_data"], candidate["cell_data"])
    reason_codes = []
    if point_changes:
        reason_codes.append("source_point_data_prefix_changed")
    if cell_changes:
        reason_codes.append("source_cell_data_prefix_changed")
    return {
        "passed": not reason_codes,
        "changed_point_fields": point_changes,
        "changed_cell_fields": cell_changes,
        "reason_codes": reason_codes,
    }


def audit_transaction_prefix(
    current: Mapping[str, Any],
    trial: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove a later append did not alter source or earlier committed rows."""

    points_equal = compare_prefix_bytes(current["points"], trial["points"])
    faces_equal = compare_prefix_bytes(current["faces"], trial["faces"])
    data = audit_data_prefix(current, trial)
    reason_codes = list(data["reason_codes"])
    if not points_equal:
        reason_codes.append("retained_point_prefix_changed")
    if not faces_equal:
        reason_codes.append("retained_face_prefix_changed")
    return {
        "passed": not reason_codes,
        "retained_point_prefix_bitwise_equal": points_equal,
        "retained_face_prefix_bitwise_equal": faces_equal,
        "changed_point_fields": data["changed_point_fields"],
        "changed_cell_fields": data["changed_cell_fields"],
        "reason_codes": reason_codes,
    }


def transaction_state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for role, values in (("points", state["points"]), ("faces", state["faces"])):
        _update_array_hash(digest, role, np.asarray(values))
    for role in ("point_data", "cell_data"):
        for name in sorted(state[role]):
            _update_array_hash(digest, f"{role}.{name}", np.asarray(state[role][name]))
    return digest.hexdigest()


def copy_transaction_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "points": state["points"].copy(),
        "faces": state["faces"].copy(),
        "point_data": {
            name: values.copy() for name, values in state["point_data"].items()
        },
        "cell_data": {
            name: values.copy() for name, values in state["cell_data"].items()
        },
    }


def _extend_data(
    existing: Mapping[str, np.ndarray],
    explicit: Mapping[str, Any],
    old_count: int,
    new_count: int,
) -> dict[str, np.ndarray]:
    explicit_arrays = {
        str(name): np.asarray(values)
        for name, values in explicit.items()
        if np.asarray(values).ndim > 0
    }
    result = {}
    for name in sorted(set(existing) | set(explicit_arrays)):
        old_values = existing.get(name)
        new_values = explicit_arrays.get(name)
        if old_values is None:
            assert new_values is not None
            if new_values.shape[0] != new_count:
                raise ValueError(
                    f"generated data field {name} must have one row per appended item"
                )
            old_values = _default_array(
                name, old_count, new_values.dtype, new_values.shape[1:]
            )
        if new_values is None:
            new_values = _default_array(
                name, new_count, old_values.dtype, old_values.shape[1:]
            )
        if new_values.shape != (new_count, *old_values.shape[1:]):
            raise ValueError(
                f"generated data field {name} shape does not match its existing schema"
            )
        if new_values.dtype != old_values.dtype:
            new_values = _safe_cast(new_values, old_values.dtype, name)
        result[name] = np.concatenate([old_values, new_values], axis=0)
    return result


def _default_array(
    name: str,
    count: int,
    dtype: np.dtype[Any],
    trailing_shape: tuple[int, ...],
) -> np.ndarray:
    if np.issubdtype(dtype, np.str_):
        value: Any = "source" if name in {"patch_method", "placement"} else ""
    elif "region_id" in name or "triangle_index" in name:
        value = -1
    else:
        value = 0
    return np.full((count, *trailing_shape), value, dtype=dtype)


def _validate_source_arrays(
    points: np.ndarray, faces: np.ndarray, source_index: np.ndarray
) -> str | None:
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or not np.issubdtype(points.dtype, np.floating)
    ):
        return "source_points must be a floating (N, 3) array"
    if not np.all(np.isfinite(points)):
        return "source_points must contain only finite coordinates"
    if (
        faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not np.issubdtype(faces.dtype, np.integer)
    ):
        return "source_faces must be an integer (M, 3) array"
    if faces.size and (np.any(faces < 0) or np.any(faces >= points.shape[0])):
        return "source_faces contains an out-of-range point ID"
    if source_index.shape != (faces.shape[0],) or not np.issubdtype(
        source_index.dtype, np.signedinteger
    ):
        return "source_triangle_index must be a signed integer array with one item per source face"
    return None


def _validate_data_rows(
    values: Mapping[str, np.ndarray], count: int, role: str
) -> str | None:
    invalid = [
        name
        for name, array in values.items()
        if array.ndim == 0 or array.shape[0] != count
    ]
    return (
        f"every {role} field must contain {count} rows: {invalid}" if invalid else None
    )


def _add_default_field(
    values: dict[str, np.ndarray],
    name: str,
    count: int,
    value: Any,
    *,
    dtype: np.dtype[Any] | type[Any] = np.int32,
) -> None:
    values.setdefault(name, np.full(count, value, dtype=dtype))


def _changed_prefix_fields(
    baseline: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
) -> list[str]:
    return [
        name
        for name, values in baseline.items()
        if name not in candidate or not compare_prefix_bytes(values, candidate[name])
    ]


def _region_array(
    existing: np.ndarray,
    count: int,
    fusion_region_id: int,
) -> np.ndarray:
    dtype = np.asarray(existing).dtype
    if dtype != np.dtype(np.int64):
        raise ValueError("fusion_region_id source schema must use int64")
    return np.full(count, validate_fusion_region_id(fusion_region_id), dtype=np.int64)


def _set_reserved_field(
    explicit: dict[str, Any], name: str, expected: np.ndarray
) -> None:
    supplied = explicit.get(name)
    if supplied is not None:
        supplied_array = np.asarray(supplied)
        if supplied_array.shape != expected.shape or not np.array_equal(
            supplied_array.astype(expected.dtype, copy=False), expected
        ):
            raise ValueError(
                f"generated reserved provenance field {name} conflicts with transaction authority"
            )
    explicit[name] = expected


def _validate_reserved_source_fields(
    cell_data: Mapping[str, np.ndarray],
    point_data: Mapping[str, np.ndarray],
) -> str | None:
    expected = (
        (cell_data, "fusion_region_id", -1),
        (cell_data, "patch_method_code", 0),
        (cell_data, "proxy_triangle_index", -1),
        (cell_data, "face_origin", 0),
        (point_data, "fusion_region_id", -1),
        (point_data, "patch_method_code", 0),
        (point_data, "point_origin", 0),
    )
    invalid_region_schema = [
        role
        for role, values in (("cell", cell_data), ("point", point_data))
        if "fusion_region_id" in values
        and np.asarray(values["fusion_region_id"]).dtype != np.dtype(np.int64)
    ]
    if invalid_region_schema:
        return (
            "reserved fusion_region_id source fields must use int64: "
            f"{invalid_region_schema}"
        )
    conflicts = [
        name
        for values, name, sentinel in expected
        if name in values
        and not np.all(np.asarray(values[name]).astype(str) == str(sentinel))
    ]
    return (
        f"reserved source provenance fields contain non-source values: {conflicts}"
        if conflicts
        else None
    )


def validate_fusion_region_id(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("fusion_region_id must be a non-negative integer")
    token = int(value)
    if token < 0 or token > np.iinfo(np.int64).max:
        raise ValueError("fusion_region_id must fit the non-negative int64 range")
    return token


def _safe_cast(values: np.ndarray, dtype: np.dtype[Any], name: str) -> np.ndarray:
    try:
        cast = values.astype(dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"generated data field {name} cannot be represented by its source dtype"
        ) from exc
    if np.issubdtype(dtype, np.str_):
        if not np.array_equal(cast.astype(str), values.astype(str)):
            raise ValueError(
                f"generated data field {name} would be truncated by its source dtype"
            )
    elif np.issubdtype(values.dtype, np.number) and np.issubdtype(dtype, np.number):
        if not np.allclose(
            cast.astype(values.dtype), values, rtol=0.0, atol=0.0, equal_nan=True
        ):
            raise ValueError(
                f"generated data field {name} would lose values in its source dtype"
            )
    elif not np.array_equal(cast.astype(str), values.astype(str)):
        raise ValueError(
            f"generated data field {name} cannot be safely cast to its source dtype"
        )
    return cast


def _update_array_hash(digest: Any, name: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    if array.dtype.hasobject:
        payload = json.dumps(
            array.tolist(), sort_keys=True, ensure_ascii=False, default=str
        )
        digest.update(payload.encode("utf-8"))
    else:
        digest.update(array.tobytes())
