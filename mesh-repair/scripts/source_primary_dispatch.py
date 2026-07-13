from __future__ import annotations

from functools import partial
from typing import Any, Callable, Mapping

import numpy as np

from mesh_metrics import edge_topology, face_component_labels
from source_primary_curved_patch import build_curved_patch_candidate
from source_primary_loop_bridge import build_paired_loop_zipper_candidate
from source_primary_patch_contract import PatchCandidate, rejected_patch_candidate
from source_primary_planar_patch import build_planar_patch_candidate
from source_primary_slit_patch import build_slit_patch_candidate
from source_primary_transaction_jobs import PatchTransactionJob


SUPPORTED_DISPATCH_METHODS = {
    "planar_cap",
    "curved_conformal_patch",
    "paired_loop_zipper",
    "slit_bridge",
}


# @entry
def build_source_primary_dispatch(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_triangle_index: np.ndarray,
    inventory: Mapping[str, Any],
    *,
    closure_proxy_points: np.ndarray,
    closure_proxy_faces: np.ndarray,
) -> dict[str, Any]:
    """Assign every inventory region exactly once to a candidate transaction."""

    regions = sorted(inventory.get("regions", []), key=lambda row: row["region_id"])
    loop_rows = {row["loop_id"]: row for row in inventory.get("loops", [])}
    region_id_map = [
        {"region_id": str(region["region_id"]), "fusion_region_id": index}
        for index, region in enumerate(regions)
    ]
    fusion_by_region = {
        str(region["region_id"]): index for index, region in enumerate(regions)
    }
    _, proxy_edge_faces = edge_topology(np.asarray(closure_proxy_faces, dtype=np.int64))
    proxy_components = face_component_labels(
        np.asarray(closure_proxy_faces).shape[0], proxy_edge_faces
    )
    proxy_triangle_index = np.arange(
        np.asarray(closure_proxy_faces).shape[0], dtype=np.int64
    )
    jobs = []
    dispatch_rows = []
    for region in regions:
        region_id = str(region["region_id"])
        method, builder, reasons = _region_builder(
            np.asarray(source_points),
            np.asarray(source_faces),
            np.asarray(source_triangle_index),
            region,
            loop_rows,
            np.asarray(closure_proxy_points),
            np.asarray(closure_proxy_faces),
            proxy_triangle_index,
            proxy_components,
        )
        component_ids = list(region.get("component_ids", []))
        component_id = component_ids[0] if len(component_ids) == 1 else None
        jobs.append(
            PatchTransactionJob(
                region_ids=(region_id,),
                patch_method=method,
                build_candidate=builder,
                component_id=component_id,
            )
        )
        dispatch_rows.append(
            {
                "region_id": region_id,
                "fusion_region_id": fusion_by_region[region_id],
                "classification": region.get("classification"),
                "recommended_operator": region.get("recommended_operator"),
                "patch_method": method,
                "assignment_status": "assigned_rejected_candidate"
                if reasons
                else "assigned_candidate_builder",
                "reason_codes": reasons,
                "component_id": component_id,
                "loop_ids": list(region.get("loop_ids", [])),
            }
        )
    return {
        "jobs": jobs,
        "required_region_ids": tuple(row["region_id"] for row in region_id_map),
        "fusion_region_id_by_region": fusion_by_region,
        "region_id_map": region_id_map,
        "method_dispatch": {
            "status": "computed",
            "complete_inventory_assignment": len(jobs) == len(regions),
            "inventory_region_count": len(regions),
            "assigned_transaction_count": len(jobs),
            "rows": dispatch_rows,
            "supported_methods": sorted(SUPPORTED_DISPATCH_METHODS),
        },
    }


def _region_builder(
    points: np.ndarray,
    faces: np.ndarray,
    source_ids: np.ndarray,
    region: Mapping[str, Any],
    loop_rows: Mapping[str, Mapping[str, Any]],
    proxy_points: np.ndarray,
    proxy_faces: np.ndarray,
    proxy_triangle_index: np.ndarray,
    proxy_components: np.ndarray,
) -> tuple[str, Callable[[], PatchCandidate], list[str]]:
    region_id = str(region["region_id"])
    blockers = sorted(set(str(code) for code in region.get("blocking_reason_codes", [])))
    if region.get("patch_eligible") is not True:
        reasons = blockers or ["inventory_region_not_patch_eligible"]
        return _rejected_builder(
            points, faces, source_ids, region, "planar_cap", reasons
        )
    loops = [
        np.asarray(
            loop_rows[str(loop_id)].get("source_winding_vertex_ids", []),
            dtype=np.int64,
        )
        for loop_id in region.get("loop_ids", [])
        if str(loop_id) in loop_rows
    ]
    oriented = _oriented_normal(region)
    operator = region.get("recommended_operator")
    if operator == "planar_cap" and len(loops) == 1:
        return (
            "planar_cap",
            partial(
                build_planar_patch_candidate,
                points,
                faces,
                source_ids,
                loops[0],
                region_id=region_id,
                oriented_normal=oriented,
            ),
            [],
        )
    if operator == "curved_conformal_patch" and len(loops) == 1:
        return (
            "curved_conformal_patch",
            partial(
                build_curved_patch_candidate,
                points,
                faces,
                source_ids,
                loops[0],
                region_id=region_id,
                oriented_normal=oriented,
                closure_proxy_points=proxy_points,
                closure_proxy_faces=proxy_faces,
                closure_proxy_triangle_index=proxy_triangle_index,
                closure_proxy_component_id=proxy_components,
            ),
            [],
        )
    if operator in {"paired_loop_zipper", "fixed_boundary_seam_bridge"} and len(loops) == 2:
        normals = _loop_normals(region, loop_rows)
        return (
            "paired_loop_zipper",
            partial(
                build_paired_loop_zipper_candidate,
                points,
                faces,
                source_ids,
                loops[0],
                loops[1],
                first_region_id=region_id,
                second_region_id=region_id,
                first_oriented_normal=normals[0],
                second_oriented_normal=normals[1],
            ),
            [],
        )
    if operator == "fixed_boundary_slit_bridge" and len(loops) == 1:
        return (
            "slit_bridge",
            partial(
                build_slit_patch_candidate,
                points,
                faces,
                source_ids,
                loops[0],
                region_id=region_id,
                oriented_normal=oriented,
            ),
            [],
        )
    if operator == "multi_loop_constrained_patch" and len(loops) == 2:
        try:
            from source_primary_multi_loop_patch import build_multi_loop_patch_candidate

            return (
                "paired_loop_zipper",
                partial(
                    build_multi_loop_patch_candidate,
                    points,
                    faces,
                    source_ids,
                    loops,
                    region_id=region_id,
                    oriented_normal=oriented,
                ),
                [],
            )
        except (ImportError, AttributeError) as exc:
            return _rejected_builder(
                points,
                faces,
                source_ids,
                region,
                "paired_loop_zipper",
                ["multi_loop_constrained_patch_unavailable", type(exc).__name__],
            )
    return _rejected_builder(
        points,
        faces,
        source_ids,
        region,
        "planar_cap",
        ["inventory_operator_not_supported_by_dispatch"],
    )


def _rejected_builder(
    points: np.ndarray,
    faces: np.ndarray,
    source_ids: np.ndarray,
    region: Mapping[str, Any],
    method: str,
    reasons: list[str],
) -> tuple[str, Callable[[], PatchCandidate], list[str]]:
    def build() -> PatchCandidate:
        return rejected_patch_candidate(
            points,
            faces,
            source_ids,
            method=method,
            failure_reason_codes=reasons,
            diagnostics={
                "region_id": str(region["region_id"]),
                "classification": region.get("classification"),
                "recommended_operator": region.get("recommended_operator"),
                "inventory_blocking_reason_codes": list(
                    region.get("blocking_reason_codes", [])
                ),
            },
        )

    return method, build, reasons


def _oriented_normal(region: Mapping[str, Any]) -> np.ndarray | None:
    oriented = region.get("normals", {}).get("oriented", {})
    if oriented.get("resolved") is not True:
        return None
    value = np.asarray(oriented.get("normal"), dtype=np.float64)
    return value if value.shape == (3,) and np.all(np.isfinite(value)) else None


def _loop_normals(
    region: Mapping[str, Any], loop_rows: Mapping[str, Mapping[str, Any]]
) -> tuple[np.ndarray | None, np.ndarray | None]:
    result = []
    for loop_id in region.get("loop_ids", [])[:2]:
        row = loop_rows.get(str(loop_id), {})
        result.append(_oriented_normal(row.get("geometry", {})))
    while len(result) < 2:
        result.append(None)
    return result[0], result[1]
