from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from mesh_io import read_surface, triangle_faces
from mesh_metrics import face_component_labels, mesh_report
from mesh_vertex_topology import non_manifold_vertex_ids


ISSUE_BOUNDARY = 1
ISSUE_NON_MANIFOLD = 2
ISSUE_INCONSISTENT_WINDING = 4
ISSUE_DEGENERATE_FACE = 8
ISSUE_NON_MANIFOLD_VERTEX = 16
ISSUE_TYPE_NAMES = {
    ISSUE_BOUNDARY: "boundary_edge",
    ISSUE_NON_MANIFOLD: "non_manifold_edge",
    ISSUE_INCONSISTENT_WINDING: "inconsistent_winding_edge",
    ISSUE_DEGENERATE_FACE: "degenerate_face",
    ISSUE_NON_MANIFOLD_VERTEX: "non_manifold_vertex",
}


def directed_edge_tables(
    faces: np.ndarray,
) -> tuple[dict[tuple[int, int], list[int]], dict[tuple[int, int], list[int]]]:
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    edge_directions: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_id, face in enumerate(faces):
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            left_i, right_i = int(left), int(right)
            edge = (left_i, right_i) if left_i < right_i else (right_i, left_i)
            edge_faces[edge].append(face_id)
            edge_directions[edge].append(1 if (left_i, right_i) == edge else -1)
    return dict(edge_faces), dict(edge_directions)


def degenerate_face_mask(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = points[faces]
    edge_0 = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    edge_1 = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    edge_2 = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    minimum_edge = np.minimum.reduce((edge_0, edge_1, edge_2))
    area_epsilon = max(float(np.nanmedian(areas)) * 1e-12, 1e-18)
    return (areas <= area_epsilon) | (minimum_edge <= 1e-15)


def connected_edge_region_ids(edges: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge in sorted(edges):
        adjacency[edge[0]].append(edge)
        adjacency[edge[1]].append(edge)

    unvisited = set(edges)
    region_by_edge: dict[tuple[int, int], int] = {}
    region_id = 0
    while unvisited:
        seed = min(unvisited)
        queue = deque([seed])
        unvisited.remove(seed)
        while queue:
            edge = queue.popleft()
            region_by_edge[edge] = region_id
            for vertex_id in edge:
                for neighbor in adjacency[vertex_id]:
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        queue.append(neighbor)
        region_id += 1
    return region_by_edge


def component_arrays(
    face_count: int,
    edge_faces: dict[tuple[int, int], list[int]],
) -> dict[str, np.ndarray]:
    labels = face_component_labels(face_count, edge_faces)
    counts = np.bincount(labels) if labels.size else np.zeros(0, dtype=np.int64)
    order = np.argsort(-counts, kind="stable")
    rank_by_component = np.empty(counts.size, dtype=np.int64)
    rank_by_component[order] = np.arange(counts.size, dtype=np.int64)
    return {
        "watertight_audit_component_id": labels,
        "watertight_audit_component_rank": rank_by_component[labels],
        "watertight_audit_component_face_count": counts[labels],
        "watertight_audit_component_is_largest": (rank_by_component[labels] == 0).astype(np.uint8),
    }


def analyze_geometry(points: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    edge_faces, edge_directions = directed_edge_tables(faces)
    boundary_edges = sorted(edge for edge, face_ids in edge_faces.items() if len(face_ids) == 1)
    non_manifold_edges = sorted(edge for edge, face_ids in edge_faces.items() if len(face_ids) > 2)
    non_manifold_vertices = non_manifold_vertex_ids(faces, edge_faces)
    inconsistent_edges = sorted(
        edge
        for edge, directions in edge_directions.items()
        if len(directions) == 2 and directions[0] == directions[1]
    )
    degenerate = degenerate_face_mask(points, faces)

    face_issue_mask = np.zeros(faces.shape[0], dtype=np.uint8)
    face_boundary_count = np.zeros(faces.shape[0], dtype=np.uint8)
    face_non_manifold_count = np.zeros(faces.shape[0], dtype=np.uint8)
    face_winding_count = np.zeros(faces.shape[0], dtype=np.uint8)
    face_non_manifold_vertex_count = np.isin(faces, non_manifold_vertices).sum(
        axis=1, dtype=np.uint8
    )
    for issue_code, edges, counts in (
        (ISSUE_BOUNDARY, boundary_edges, face_boundary_count),
        (ISSUE_NON_MANIFOLD, non_manifold_edges, face_non_manifold_count),
        (ISSUE_INCONSISTENT_WINDING, inconsistent_edges, face_winding_count),
    ):
        for edge in edges:
            for face_id in edge_faces[edge]:
                counts[face_id] += 1
                face_issue_mask[face_id] |= issue_code
    face_issue_mask[degenerate] |= ISSUE_DEGENERATE_FACE
    face_issue_mask[face_non_manifold_vertex_count > 0] |= ISSUE_NON_MANIFOLD_VERTEX

    regions = {
        ISSUE_BOUNDARY: connected_edge_region_ids(boundary_edges),
        ISSUE_NON_MANIFOLD: connected_edge_region_ids(non_manifold_edges),
        ISSUE_INCONSISTENT_WINDING: connected_edge_region_ids(inconsistent_edges),
    }
    cell_arrays = {
        "watertight_audit_face_id": np.arange(faces.shape[0], dtype=np.int64),
        "watertight_audit_issue_mask": face_issue_mask,
        "watertight_audit_has_issue": (face_issue_mask != 0).astype(np.uint8),
        "watertight_audit_boundary_edge_count": face_boundary_count,
        "watertight_audit_non_manifold_edge_count": face_non_manifold_count,
        "watertight_audit_inconsistent_winding_edge_count": face_winding_count,
        "watertight_audit_degenerate_face": degenerate.astype(np.uint8),
        "watertight_audit_non_manifold_vertex_count": face_non_manifold_vertex_count,
        **component_arrays(faces.shape[0], edge_faces),
    }
    point_arrays = {
        "watertight_audit_source_point_id": np.arange(points.shape[0], dtype=np.int64),
        "watertight_audit_non_manifold_vertex": np.isin(
            np.arange(points.shape[0], dtype=np.int64), non_manifold_vertices
        ).astype(np.uint8),
    }
    metrics = mesh_report(points, faces)
    topology_watertight = (
        not boundary_edges
        and not non_manifold_edges
        and non_manifold_vertices.size == 0
    )
    engineering_watertight = bool(
        topology_watertight and not inconsistent_edges and not np.any(degenerate)
    )
    summary = {
        "points": int(points.shape[0]),
        "triangles": int(faces.shape[0]),
        "topology_watertight": topology_watertight,
        "engineering_watertight": engineering_watertight,
        "boundary_edges": len(boundary_edges),
        "boundary_regions": len(set(regions[ISSUE_BOUNDARY].values())),
        "non_manifold_edges": len(non_manifold_edges),
        "non_manifold_regions": len(set(regions[ISSUE_NON_MANIFOLD].values())),
        "non_manifold_vertices": int(non_manifold_vertices.size),
        "inconsistent_winding_edges": len(inconsistent_edges),
        "inconsistent_winding_regions": len(set(regions[ISSUE_INCONSISTENT_WINDING].values())),
        "degenerate_faces": int(np.count_nonzero(degenerate)),
        "affected_faces": int(np.count_nonzero(face_issue_mask)),
        "connected_components": int(metrics["topology"]["components"]["count"]),
        "closed_manifold": bool(metrics["topology"]["closed_manifold"]),
    }
    return {
        "edge_faces": edge_faces,
        "edges": {
            ISSUE_BOUNDARY: boundary_edges,
            ISSUE_NON_MANIFOLD: non_manifold_edges,
            ISSUE_INCONSISTENT_WINDING: inconsistent_edges,
        },
        "edge_regions": regions,
        "cell_arrays": cell_arrays,
        "point_arrays": point_arrays,
        "non_manifold_vertex_ids": non_manifold_vertices,
        "issue_face_ids": np.flatnonzero(face_issue_mask).astype(np.int64),
        "metrics": metrics,
        "summary": summary,
    }


def write_line_marker_vtp(
    path: Path,
    points: np.ndarray,
    edge_faces: dict[tuple[int, int], list[int]],
    typed_edges: list[tuple[int, tuple[int, int]]],
    edge_regions: dict[int, dict[tuple[int, int], int]],
) -> None:
    typed_edges = sorted(typed_edges, key=lambda item: (item[0], item[1]))
    source_point_ids = np.asarray(
        sorted({vertex_id for _, edge in typed_edges for vertex_id in edge}),
        dtype=np.int64,
    )
    point_map = {int(source_id): compact_id for compact_id, source_id in enumerate(source_point_ids)}
    lines = np.empty((len(typed_edges), 3), dtype=np.int64)
    if typed_edges:
        lines[:, 0] = 2
        lines[:, 1:] = np.asarray(
            [[point_map[edge[0]], point_map[edge[1]]] for _, edge in typed_edges],
            dtype=np.int64,
        )

    marker = pv.PolyData()
    marker.points = points[source_point_ids] if source_point_ids.size else np.empty((0, 3), dtype=np.float64)
    marker.lines = lines.ravel()
    marker.point_data["watertight_audit_source_point_id"] = source_point_ids

    issue_types = np.asarray([issue_type for issue_type, _ in typed_edges], dtype=np.uint8)
    incidences = np.asarray([len(edge_faces[edge]) for _, edge in typed_edges], dtype=np.int32)
    marker.cell_data["watertight_audit_issue_type"] = issue_types
    marker.cell_data["watertight_audit_issue_region_id"] = np.asarray(
        [edge_regions[issue_type][edge] for issue_type, edge in typed_edges],
        dtype=np.int32,
    )
    marker.cell_data["watertight_audit_edge_incidence"] = incidences
    marker.cell_data["watertight_audit_adjacent_face_count"] = incidences
    marker.cell_data["watertight_audit_adjacent_face_first"] = np.asarray(
        [edge_faces[edge][0] for _, edge in typed_edges],
        dtype=np.int64,
    )
    marker.cell_data["watertight_audit_source_point_id_0"] = np.asarray(
        [edge[0] for _, edge in typed_edges],
        dtype=np.int64,
    )
    marker.cell_data["watertight_audit_source_point_id_1"] = np.asarray(
        [edge[1] for _, edge in typed_edges],
        dtype=np.int64,
    )
    marker.cell_data["watertight_audit_edge_length"] = np.asarray(
        [float(np.linalg.norm(points[edge[1]] - points[edge[0]])) for _, edge in typed_edges],
        dtype=np.float64,
    )
    marker.save(path)


def write_issue_faces_vtp(
    path: Path,
    points: np.ndarray,
    faces: np.ndarray,
    issue_face_ids: np.ndarray,
    source_cell_arrays: dict[str, np.ndarray],
) -> None:
    selected_faces = faces[issue_face_ids]
    source_point_ids = np.unique(selected_faces.ravel()) if selected_faces.size else np.zeros(0, dtype=np.int64)
    point_map = np.full(points.shape[0], -1, dtype=np.int64)
    point_map[source_point_ids] = np.arange(source_point_ids.size, dtype=np.int64)
    compact_faces = point_map[selected_faces] if selected_faces.size else np.empty((0, 3), dtype=np.int64)
    packed = np.empty((compact_faces.shape[0], 4), dtype=np.int64)
    if compact_faces.size:
        packed[:, 0] = 3
        packed[:, 1:] = compact_faces

    marker = pv.PolyData()
    marker.points = points[source_point_ids] if source_point_ids.size else np.empty((0, 3), dtype=np.float64)
    marker.faces = packed.ravel()
    marker.point_data["watertight_audit_source_point_id"] = source_point_ids
    for name, values in source_cell_arrays.items():
        marker.cell_data[name] = np.asarray(values)[issue_face_ids]
    marker.save(path)


def write_point_marker_vtp(
    path: Path,
    points: np.ndarray,
    source_point_ids: np.ndarray,
) -> None:
    source_point_ids = np.asarray(source_point_ids, dtype=np.int64)
    marker = pv.PolyData(points[source_point_ids])
    marker.verts = np.column_stack(
        (
            np.ones(source_point_ids.size, dtype=np.int64),
            np.arange(source_point_ids.size, dtype=np.int64),
        )
    ).ravel()
    marker.point_data["watertight_audit_source_point_id"] = source_point_ids
    marker.point_data["watertight_audit_non_manifold_vertex"] = np.ones(
        source_point_ids.size, dtype=np.uint8
    )
    marker.save(path)


def write_annotated_surface_vtp(
    path: Path,
    surface: pv.PolyData,
    cell_arrays: dict[str, np.ndarray],
    point_arrays: dict[str, np.ndarray],
) -> None:
    annotated = surface.copy(deep=True)
    for name, values in cell_arrays.items():
        annotated.cell_data[name] = values
    for name, values in point_arrays.items():
        annotated.point_data[name] = values
    annotated.save(path)


def audit_surface(role: str, input_path: Path, output_dir: Path) -> dict[str, Any]:
    surface = read_surface(input_path)
    points = np.asarray(surface.points)
    faces = triangle_faces(surface)
    analysis = analyze_geometry(points, faces)

    artifacts = {
        "annotated_surface_vtp": output_dir / f"{role}_surface_with_issue_arrays.vtp",
        "all_issue_edges_vtp": output_dir / f"{role}_watertight_issue_edges.vtp",
        "boundary_edges_vtp": output_dir / f"{role}_boundary_edges.vtp",
        "non_manifold_edges_vtp": output_dir / f"{role}_non_manifold_edges.vtp",
        "non_manifold_vertices_vtp": output_dir / f"{role}_non_manifold_vertices.vtp",
        "inconsistent_winding_edges_vtp": output_dir / f"{role}_inconsistent_winding_edges.vtp",
        "issue_faces_vtp": output_dir / f"{role}_watertight_issue_faces.vtp",
    }
    write_annotated_surface_vtp(
        artifacts["annotated_surface_vtp"],
        surface,
        analysis["cell_arrays"],
        analysis["point_arrays"],
    )
    typed_edges = [
        (issue_type, edge)
        for issue_type in (ISSUE_BOUNDARY, ISSUE_NON_MANIFOLD, ISSUE_INCONSISTENT_WINDING)
        for edge in analysis["edges"][issue_type]
    ]
    write_line_marker_vtp(
        artifacts["all_issue_edges_vtp"],
        points,
        analysis["edge_faces"],
        typed_edges,
        analysis["edge_regions"],
    )
    for issue_type, key in (
        (ISSUE_BOUNDARY, "boundary_edges_vtp"),
        (ISSUE_NON_MANIFOLD, "non_manifold_edges_vtp"),
        (ISSUE_INCONSISTENT_WINDING, "inconsistent_winding_edges_vtp"),
    ):
        write_line_marker_vtp(
            artifacts[key],
            points,
            analysis["edge_faces"],
            [(issue_type, edge) for edge in analysis["edges"][issue_type]],
            analysis["edge_regions"],
        )
    write_issue_faces_vtp(
        artifacts["issue_faces_vtp"],
        points,
        faces,
        analysis["issue_face_ids"],
        analysis["cell_arrays"],
    )
    write_point_marker_vtp(
        artifacts["non_manifold_vertices_vtp"],
        points,
        analysis["non_manifold_vertex_ids"],
    )

    return {
        "role": role,
        "input_path": str(input_path.resolve()),
        "input_kind": "mesh",
        "output_kind": "diagnostic_mesh",
        "normalization": {
            "method": "surface_extract_triangulate_clean",
            "decimated": False,
            "geometry_resolution": "full_normalized_triangle_surface",
            "source_point_arrays_preserved": sorted(surface.point_data.keys()),
            "source_cell_arrays_preserved": sorted(
                name for name in surface.cell_data.keys() if not name.startswith("watertight_audit_")
            ),
        },
        "summary": analysis["summary"],
        "metrics": analysis["metrics"],
        "artifacts": {key: str(path.resolve()) for key, path in artifacts.items()},
    }


def comparison_summary(original: dict[str, Any], processed: dict[str, Any]) -> dict[str, Any]:
    compared_metrics = (
        "boundary_edges",
        "boundary_regions",
        "non_manifold_edges",
        "non_manifold_regions",
        "non_manifold_vertices",
        "inconsistent_winding_edges",
        "inconsistent_winding_regions",
        "degenerate_faces",
        "affected_faces",
        "connected_components",
    )
    values = {}
    for name in compared_metrics:
        before = int(original["summary"][name])
        after = int(processed["summary"][name])
        values[name] = {
            "original": before,
            "processed": after,
            "delta_processed_minus_original": after - before,
            "reduction": before - after,
            "reduction_ratio": (float((before - after) / before) if before else None),
        }
    return {
        "original_topology_watertight": original["summary"]["topology_watertight"],
        "processed_topology_watertight": processed["summary"]["topology_watertight"],
        "original_engineering_watertight": original["summary"]["engineering_watertight"],
        "processed_engineering_watertight": processed["summary"]["engineering_watertight"],
        "metrics": values,
    }


def generate_issue_report(
    original_path: Path,
    processed_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "watertight_issue_report.json"
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"report already exists; pass --overwrite to replace it: {report_path}")

    original = audit_surface("original", original_path, output_dir)
    processed = audit_surface("processed", processed_path, output_dir)
    report = {
        "schema_version": 1,
        "report_type": "watertight_issue_vtp_comparison",
        "target": "watertight-exterior-shell",
        "input_kind": "mesh",
        "output_kind": "diagnostic_mesh",
        "repair_domain": "none_diagnostic_only",
        "issue_type_codes": {str(code): name for code, name in ISSUE_TYPE_NAMES.items()},
        "array_contract": {
            "edge_cell_issue_type": "watertight_audit_issue_type",
            "edge_cell_region_id": "watertight_audit_issue_region_id",
            "edge_cell_incidence": "watertight_audit_edge_incidence",
            "surface_cell_issue_bitmask": "watertight_audit_issue_mask",
            "surface_point_non_manifold_vertex": "watertight_audit_non_manifold_vertex",
            "surface_cell_component_id": "watertight_audit_component_id",
            "surface_cell_component_rank": "watertight_audit_component_rank",
            "issue_mask_bits": {str(code): name for code, name in ISSUE_TYPE_NAMES.items()},
        },
        "watertight_definition": {
            "topology_watertight": (
                "boundary_edges == 0 and non_manifold_edges == 0 and non_manifold_vertices == 0"
            ),
            "engineering_watertight": (
                "topology_watertight and inconsistent_winding_edges == 0 and degenerate_faces == 0"
            ),
            "component_count": (
                "connected components are labeled for diagnosis; their count is not a watertightness gate"
            ),
        },
        "checks": {
            "boundary_edges": "computed",
            "non_manifold_edges": "computed",
            "non_manifold_vertices": "computed",
            "inconsistent_winding_edges": "computed",
            "degenerate_faces": "computed",
            "connected_components": "computed",
            "self_intersections": "not_checked",
            "functional_opening_semantics": "not_checked",
        },
        "original": original,
        "processed": processed,
        "comparison": comparison_summary(original, processed),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {**report, "report_path": str(report_path.resolve())}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write full-resolution annotated surfaces plus edge and point VTP markers for "
            "watertightness defects in an original and processed mesh."
        )
    )
    parser.add_argument("original_mesh", type=Path)
    parser.add_argument("processed_mesh", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate_issue_report(
        args.original_mesh,
        args.processed_mesh,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps({"report_path": result["report_path"], "comparison": result["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
