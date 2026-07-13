from __future__ import annotations

from typing import Any, Mapping, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from source_primary_patch_contract import BoundaryMapping, PatchDelta


def validate_delta_provenance(
    delta: PatchDelta,
    point_count: int,
    face_count: int,
    method: str,
    mappings: tuple[BoundaryMapping, ...],
) -> list[str]:
    errors: list[str] = []
    for role, provenance, expected_count in (
        ("point", delta.point_provenance, point_count),
        ("face", delta.face_provenance, face_count),
    ):
        for name, value in provenance.items():
            array = _read_array(value)
            if array is None:
                errors.append(f"{role} provenance {name!r} is not a readable array")
            elif array.ndim > 0 and array.shape[0] != expected_count:
                errors.append(
                    f"{role} provenance {name!r} must contain {expected_count} rows"
                )
    required_point = {"patch_method", "region_id"}
    required_face = {
        "patch_method",
        "source_triangle_index",
        "source_geometry_consumed",
        "proxy_geometry_consumed",
    }
    required_face.add("region_id" if method != "paired_loop_zipper" else "first_region_id")
    if method == "paired_loop_zipper":
        required_face.add("second_region_id")
    missing_point = sorted(required_point.difference(delta.point_provenance))
    missing_face = sorted(required_face.difference(delta.face_provenance))
    if missing_point:
        errors.append(f"point provenance is missing required fields: {missing_point}")
    if missing_face:
        errors.append(f"face provenance is missing required fields: {missing_face}")
    source_ids = _read_array(delta.face_provenance.get("source_triangle_index", []))
    if (
        source_ids is None
        or source_ids.shape != (face_count,)
        or not np.issubdtype(source_ids.dtype, np.integer)
        or np.issubdtype(source_ids.dtype, np.bool_)
        or np.any(source_ids != -1)
    ):
        errors.append("generated face source_triangle_index must be -1 for every face")
    for role, provenance, count in (
        ("point", delta.point_provenance, point_count),
        ("face", delta.face_provenance, face_count),
    ):
        patch_method = _read_string_array(provenance.get("patch_method"))
        if patch_method is None or patch_method.shape != (count,) or np.any(
            patch_method != method
        ):
            errors.append(f"{role} patch_method provenance does not match candidate method")
        region_ids = _read_string_array(provenance.get("region_id"))
        if region_ids is not None:
            allowed = {str(mapping.region_id) for mapping in mappings}
            if region_ids.shape != (count,) or any(value not in allowed for value in region_ids):
                errors.append(f"{role} region_id provenance is not a mapped boundary ID")
    if method == "paired_loop_zipper" and len(mappings) != 2:
        errors.append("paired_loop_zipper provenance requires exactly two mapped boundaries")
    elif method == "paired_loop_zipper":
        expected_region_ids = (
            str(mappings[0].region_id),
            str(mappings[1].region_id),
        )
        for name, region_id in zip(
            ("first_region_id", "second_region_id"),
            expected_region_ids,
            strict=True,
        ):
            region_values = _read_string_array(delta.face_provenance.get(name))
            if (
                region_values is None
                or region_values.shape != (face_count,)
                or np.any(region_values != region_id)
            ):
                errors.append(f"face {name} provenance does not match its mapped boundary")
    elif point_count and method in {"planar_cap", "curved_conformal_patch", "slit_bridge"}:
        for name in ("uv", "normal_offset", "placement"):
            if name not in delta.point_provenance:
                errors.append(f"point provenance is missing required field {name!r}")
        uv = _read_numeric_array(delta.point_provenance.get("uv"))
        offset = _read_numeric_array(delta.point_provenance.get("normal_offset"))
        placement = _read_string_array(delta.point_provenance.get("placement"))
        if uv is None or uv.shape != (point_count, 2) or not np.all(np.isfinite(uv)):
            errors.append("point uv provenance must contain one finite 2-vector per point")
        if offset is None or offset.shape != (point_count,) or not np.all(np.isfinite(offset)):
            errors.append("point normal_offset provenance must be finite and row-aligned")
        expected_placement = {
            "planar_cap": "fitted_plane",
            "curved_conformal_patch": "thin_plate_lift",
            "slit_bridge": "thin_plate_boundary_depth_interpolation",
        }[method]
        if placement is None or placement.shape != (point_count,) or np.any(
            placement != expected_placement
        ):
            errors.append("point placement provenance does not match the patch method")
    for name in ("source_geometry_consumed", "proxy_geometry_consumed"):
        consumed_values = _read_array(delta.face_provenance.get(name))
        if (
            consumed_values is None
            or consumed_values.shape != (face_count,)
            or not (
                np.issubdtype(consumed_values.dtype, np.bool_)
                or np.issubdtype(consumed_values.dtype, np.integer)
            )
            or np.any(consumed_values != 0)
        ):
            errors.append(f"generated face {name} must be false for every face")
    expected_normals = delta.face_provenance.get("expected_face_normals")
    if expected_normals is not None:
        normal_values = _read_numeric_array(expected_normals)
        if (
            normal_values is None
            or normal_values.shape != (face_count, 3)
            or not np.all(np.isfinite(normal_values))
        ):
            errors.append("expected_face_normals provenance must contain finite 3-vectors")
    return errors


def validate_normal_evidence(normal: Mapping[str, Any], method: str) -> list[str]:
    value = _read_numeric_array(normal.get("oriented_normal"))
    if value is None or value.shape != (3,) or not np.all(np.isfinite(value)):
        return ["normal evidence must include one finite oriented_normal"]
    if not np.isclose(np.linalg.norm(value), 1.0, rtol=1e-6, atol=1e-8):
        return ["oriented_normal must be a unit vector"]
    if normal.get("orientation_reliable") is not True:
        return ["normal orientation evidence is not reliable"]
    evidence_method = normal.get("method")
    if not isinstance(evidence_method, str) or not evidence_method:
        return ["normal evidence must identify its computation method"]
    if method == "paired_loop_zipper":
        return []
    parameterization = _read_numeric_array(normal.get("parameterization_normal"))
    origin = _read_numeric_array(normal.get("parameterization_origin"))
    u_axis = _read_numeric_array(normal.get("parameterization_u_axis"))
    v_axis = _read_numeric_array(normal.get("parameterization_v_axis"))
    if (
        parameterization is None
        or parameterization.shape != (3,)
        or not np.all(np.isfinite(parameterization))
        or not np.isclose(np.linalg.norm(parameterization), 1.0, rtol=1e-6, atol=1e-8)
    ):
        return ["normal evidence must include a unit parameterization_normal"]
    if origin is None or origin.shape != (3,) or not np.all(np.isfinite(origin)):
        return ["normal evidence must include a finite parameterization_origin"]
    if any(
        axis is None
        or axis.shape != (3,)
        or not np.all(np.isfinite(axis))
        or not np.isclose(np.linalg.norm(axis), 1.0, rtol=1e-6, atol=1e-8)
        for axis in (u_axis, v_axis)
    ):
        return ["normal evidence must include unit parameterization tangent axes"]
    if (
        max(
            abs(float(np.dot(u_axis, v_axis))),
            abs(float(np.dot(u_axis, parameterization))),
            abs(float(np.dot(v_axis, parameterization))),
        )
        > 1.0e-6
        or float(np.dot(np.cross(u_axis, v_axis), parameterization)) < 1.0 - 1.0e-6
    ):
        return ["normal parameterization axes must be right-handed and orthonormal"]
    if float(np.dot(value, parameterization)) < 1.0 - 1.0e-6:
        return ["oriented_normal and parameterization_normal must share one direction"]
    if normal.get("chart_uses_external_orientation") is not False:
        return ["external normal evidence cannot rotate the local parameterization chart"]
    supplied = normal.get("external_orientation_supplied")
    strong = normal.get("external_orientation_strongly_consistent")
    if not isinstance(supplied, (bool, np.bool_)) or not isinstance(strong, (bool, np.bool_)):
        return ["normal evidence must state external orientation use and consistency"]
    if bool(supplied) != bool(strong):
        return ["supplied external orientation evidence must be strongly source-consistent"]
    return []


def validate_curvature_evidence(curvature: Mapping[str, Any]) -> list[str]:
    if not curvature or curvature.get("status") not in {"computed", "underdetermined"}:
        return ["curvature evidence must include an explicit status"]
    method = curvature.get("method")
    reliable = curvature.get("reliable")
    if not isinstance(method, str) or not method or not isinstance(
        reliable, (bool, np.bool_)
    ):
        return ["curvature evidence must identify its method and reliability"]
    if curvature["status"] == "computed" and reliable is not True:
        return ["computed curvature evidence must be reliable"]
    if curvature["status"] == "underdetermined" and reliable is not False:
        return ["underdetermined curvature evidence cannot be marked reliable"]
    if method == "paired_boundary_source_one_ring":
        errors: list[str] = []
        for name in ("first_boundary", "second_boundary"):
            nested = curvature.get(name)
            if not isinstance(nested, Mapping):
                errors.append(f"paired curvature evidence is missing {name}")
            else:
                errors.extend(
                    f"{name}: {error}" for error in validate_curvature_evidence(nested)
                )
        return errors
    if method != "source_one_ring_quadratic_height_and_normal_variation":
        return ["curvature evidence method is unsupported"]
    principal = _read_numeric_array(curvature.get("principal_curvatures"))
    hessian = _read_numeric_array(curvature.get("height_hessian"))
    scalars = _read_finite_scalars(
        curvature,
        ("fit_rms", "normal_turn_radians_max", "normal_turn_radians_mean"),
    )
    if (
        principal is None
        or principal.shape != (2,)
        or hessian is None
        or hessian.shape != (2, 2)
        or not np.all(np.isfinite(principal))
        or not np.all(np.isfinite(hessian))
        or not scalars
    ):
        return ["curvature evidence must include finite fit, turn, and Hessian values"]
    return []


def validate_proxy_provenance(
    provenance: Mapping[str, Any],
    *,
    consumable: bool = False,
    candidate_method: str | None = None,
    candidate_normal: Mapping[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    used = provenance.get("used")
    if not isinstance(used, (bool, np.bool_)):
        errors.append("proxy provenance must state whether reference evidence was used")
    if provenance.get("geometry_consumed") is not False:
        errors.append("closure proxy geometry may only be reference evidence")
    if consumable and used is False and provenance.get("role") != "not_used":
        errors.append("unused closure proxy provenance must have role not_used")
    if consumable and used is True:
        if candidate_method != "curved_conformal_patch":
            errors.append("closure proxy evidence is only valid for curved_conformal_patch")
        errors.extend(_validate_used_proxy_provenance(provenance))
        proxy_normal = _read_numeric_array(provenance.get("oriented_normal"))
        patch_normal = _read_numeric_array(
            candidate_normal.get("oriented_normal") if candidate_normal else None
        )
        if (
            proxy_normal is None
            or patch_normal is None
            or proxy_normal.shape != (3,)
            or patch_normal.shape != (3,)
            or not np.allclose(proxy_normal, patch_normal, rtol=1e-6, atol=1e-8)
        ):
            errors.append("closure proxy sampling normal must equal the oriented hole normal")
    forbidden = {"points", "faces", "vertices", "triangles", "proxy_points", "proxy_faces"}
    pending: list[Any] = [provenance]
    while pending:
        current = pending.pop()
        if isinstance(current, np.ndarray):
            if current.dtype == object:
                errors.append("proxy provenance cannot contain object arrays")
                break
            continue
        if isinstance(current, (list, tuple)):
            pending.extend(current)
            continue
        if not isinstance(current, Mapping):
            continue
        if forbidden.intersection(current):
            errors.append("proxy provenance must not carry consumable proxy geometry")
            break
        if "geometry_consumed" in current and current["geometry_consumed"] is not False:
            errors.append("nested proxy provenance cannot consume proxy geometry")
            break
        pending.extend(current.values())
    return errors


def _validate_used_proxy_provenance(provenance: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
        "role": "local_normal_depth_reference_only",
        "footprint_policy": "strict_polygon_interior_no_dilation",
        "sampling_method": "oriented_normal_line_triangle_barycentric_depth",
        "shape_reference_used": True,
    }
    for name, value in expected.items():
        if provenance.get(name) != value:
            errors.append(f"used closure proxy provenance has invalid {name}")
    requested = _read_nonnegative_int(provenance.get("requested_sample_count"))
    selected = _read_nonnegative_int(provenance.get("selected_sample_count"))
    if requested is None or selected is None or not 0 < selected <= requested:
        errors.append("used closure proxy sample counts are invalid")
        return errors
    arrays = {
        "sample_indices": ((selected,), False),
        "sample_uv": ((selected, 2), True),
        "signed_depth": ((selected,), True),
        "proxy_triangle_index": ((selected,), False),
        "proxy_component_id": ((selected,), False),
        "proxy_local_component_id": ((selected,), False),
        "barycentric": ((selected, 3), True),
        "normal_dot": ((selected,), True),
    }
    read: dict[str, np.ndarray] = {}
    for name, (shape, numeric) in arrays.items():
        value = _read_numeric_array(provenance.get(name)) if numeric else _read_array(
            provenance.get(name)
        )
        if (
            value is None
            or value.shape != shape
            or (numeric and not np.all(np.isfinite(value)))
            or (not numeric and not np.issubdtype(value.dtype, np.integer))
        ):
            errors.append(f"used closure proxy provenance {name} is invalid")
        else:
            read[name] = value
    barycentric = read.get("barycentric")
    if barycentric is not None and (
        np.any(barycentric < -1.0e-8)
        or not np.allclose(barycentric.sum(axis=1), 1.0, rtol=1e-8, atol=1e-8)
    ):
        errors.append("used closure proxy barycentric provenance is inconsistent")
    sample_indices = read.get("sample_indices")
    if sample_indices is not None and (
        np.min(sample_indices) < 0
        or np.max(sample_indices) >= requested
        or np.unique(sample_indices).size != selected
    ):
        errors.append("used closure proxy sample indices are invalid")
    normal_dot = read.get("normal_dot")
    minimum_normal_dot = _read_finite_float(
        provenance.get("minimum_required_normal_dot")
    )
    if (
        normal_dot is not None
        and (
            minimum_normal_dot is None
            or minimum_normal_dot <= 0.0
            or minimum_normal_dot > 1.0
            or np.any(normal_dot < minimum_normal_dot)
            or np.any(normal_dot > 1.0 + 1.0e-8)
        )
    ):
        errors.append("used closure proxy normal-dot provenance is not outward")
    if normal_dot is not None and (
        provenance.get("negative_normal_count")
        != int(np.count_nonzero(normal_dot < 0.0))
        or not np.isclose(
            provenance.get("minimum_normal_dot", np.nan),
            normal_dot.min(),
            rtol=0.0,
            atol=1.0e-12,
        )
        or not np.isclose(
            provenance.get("maximum_normal_dot", np.nan),
            normal_dot.max(),
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        errors.append("used closure proxy normal-dot summary is inconsistent")
    triangle_ids = read.get("proxy_triangle_index")
    local_components = read.get("proxy_local_component_id")
    components = read.get("proxy_component_id")
    coverage = _read_finite_float(provenance.get("coverage"))
    if coverage is None or not np.isclose(coverage, selected / requested, rtol=1e-12, atol=1e-12):
        errors.append("used closure proxy coverage is inconsistent with sample counts")
    for name in ("proxy_points_sha256", "proxy_faces_sha256"):
        if not _is_sha256(provenance.get(name)):
            errors.append(f"used closure proxy provenance {name} is invalid")
    point_count = _read_nonnegative_int(provenance.get("proxy_point_count"))
    face_count = _read_nonnegative_int(provenance.get("proxy_face_count"))
    if point_count is None or point_count < 3 or face_count is None or face_count < 1:
        errors.append("used closure proxy geometry counts are invalid")
    if triangle_ids is not None and face_count is not None and (
        np.min(triangle_ids) < 0 or np.max(triangle_ids) >= face_count
    ):
        errors.append("used closure proxy triangle indices are out of range")
    if local_components is not None and np.min(local_components) < 0:
        errors.append("used closure proxy local component IDs must be non-negative")
    labels_supplied = provenance.get("component_labels_supplied")
    if not isinstance(labels_supplied, (bool, np.bool_)) or (
        components is not None
        and (
            (labels_supplied and np.min(components) < 0)
            or (not labels_supplied and np.any(components != -1))
        )
    ):
        errors.append("used closure proxy component label provenance is inconsistent")
    oriented = _read_numeric_array(provenance.get("oriented_normal"))
    if (
        oriented is None
        or oriented.shape != (3,)
        or not np.all(np.isfinite(oriented))
        or not np.isclose(np.linalg.norm(oriented), 1.0, rtol=1e-6, atol=1e-8)
    ):
        errors.append("used closure proxy oriented-normal provenance is invalid")
    errors.extend(_validate_retained_proxy_controls(provenance, read, selected))
    return errors


def _validate_retained_proxy_controls(
    provenance: Mapping[str, Any], read: Mapping[str, np.ndarray], selected: int
) -> list[str]:
    rows = _read_array(provenance.get("shape_control_evidence_row_indices"))
    samples = _read_array(provenance.get("shape_control_sample_indices"))
    count = _read_nonnegative_int(provenance.get("shape_reference_control_count"))
    evidence_count = _read_nonnegative_int(provenance.get("sampled_evidence_count"))
    if (
        rows is None
        or rows.ndim != 1
        or not np.issubdtype(rows.dtype, np.integer)
        or samples is None
        or samples.shape != rows.shape
        or count != rows.size
        or evidence_count != selected
        or rows.size < 1
        or np.min(rows) < 0
        or np.max(rows) >= selected
        or np.unique(rows).size != rows.size
    ):
        return ["retained closure proxy shape-control provenance is invalid"]
    requested_samples = read.get("sample_indices")
    if requested_samples is None or not np.array_equal(samples, requested_samples[rows]):
        return ["retained closure proxy controls do not match sampled evidence rows"]
    return []


def _read_array(value: Any) -> np.ndarray | None:
    try:
        return np.asarray(value)
    except (OverflowError, TypeError, ValueError):
        return None


def _read_numeric_array(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            return None
        return array.astype(np.float64, copy=False)
    except (OverflowError, TypeError, ValueError):
        return None


def _read_string_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value).astype(str)
    except (OverflowError, TypeError, ValueError):
        return None


def _read_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        return None
    result = int(value)
    return result if result >= 0 else None


def _read_finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _read_finite_scalars(values: Mapping[str, Any], names: tuple[str, ...]) -> bool:
    return all(_read_finite_float(values.get(name)) is not None for name in names)


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
