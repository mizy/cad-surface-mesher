from __future__ import annotations

from pathlib import Path
from typing import Any


def build_tessellation_report(
    *,
    np: Any,
    input_path: Path,
    cad_format: str,
    controls: Any,
    imported_entities: list[dict[str, int]],
    entity_names: dict[str, list[dict[str, Any]]],
    entity_counts: dict[str, int],
    mesh: dict[str, Any],
    mesh_path: Path,
    report_path: Path,
    debug_msh_path: Path | None,
    applied_options: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    points = mesh["points"]
    triangles = mesh["triangles"]
    edge_stats, edge_to_faces = edge_topology(triangles)
    quality = triangle_quality(np, points, triangles)
    components = connected_components(triangles.shape[0], edge_to_faces)
    surface_summary = build_surface_summary(mesh, entity_names)
    volume_summary = build_volume_summary(mesh, entity_names)
    limitations = provenance_limitations(mesh, controls)
    named_surfaces = count_named(entity_names.get("2", []))
    named_volumes = count_named(entity_names.get("3", []))
    controls_report = controls_dict(controls, applied_options)

    report = {
        "input": {
            "path": str(input_path),
            "format": cad_format,
            "file_size_bytes": input_path.stat().st_size,
        },
        "controls": controls_report,
        "import_summary": {
            "highest_dimension": highest_dimension(entity_counts),
            "imported_entities": imported_entities,
            "entity_counts": entity_counts,
            "labels_imported": bool(controls.import_labels),
            "named_surface_count": named_surfaces,
            "named_volume_count": named_volumes,
        },
        "outputs": {
            "surface_mesh_vtp": str(mesh_path),
            "report": str(report_path),
            "debug_msh": str(debug_msh_path) if debug_msh_path else None,
        },
        "mesh_metrics": {
            "points": int(points.shape[0]),
            "triangles": int(triangles.shape[0]),
            "bounds": bounds_dict(points),
            "surface_area": quality["surface_area"],
        },
        "topology": {
            **edge_stats,
            "components": components,
            "degenerate_faces": quality["degenerate_faces"],
        },
        "quality": {
            "area": quality["area"],
            "aspect_ratio": quality["aspect_ratio"],
        },
        "provenance": {
            "cell_arrays": ["gmsh_surface_tag", "gmsh_parent_volume_tag", "gmsh_element_tag"],
            "point_arrays": ["gmsh_node_tag"],
            "surface_summary": surface_summary,
            "volume_summary": volume_summary,
            "limitations": limitations,
        },
        "gates": {
            "non_empty": bool(points.shape[0] > 0 and triangles.shape[0] > 0),
            "all_triangles": bool(mesh["all_triangles"]),
            "report_complete": True,
            "surface_mesh_written": mesh_path.exists(),
        },
        "warnings": warnings,
    }
    if named_surfaces == 0 and named_volumes == 0:
        report["warnings"].append("No imported surface or volume names were available from Gmsh/OCC.")
    return report


def controls_dict(controls: Any, applied_options: dict[str, Any]) -> dict[str, Any]:
    return {
        "occ_target_unit": controls.occ_target_unit,
        "mesh_size": controls.mesh_size,
        "mesh_size_min": controls.mesh_size_min,
        "mesh_size_max": controls.mesh_size_max,
        "angle_deg": controls.angle_deg,
        "mesh_size_from_curvature": max(1, int(round(360.0 / controls.angle_deg))) if controls.angle_deg else None,
        "chord": controls.chord,
        "import_labels": controls.import_labels,
        "save_debug_msh": controls.save_debug_msh,
        "applied_gmsh_options": applied_options,
    }


def bounds_dict(points: Any) -> dict[str, Any]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = maxs - mins
    return {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "extents": extents.tolist(),
        "length_x": float(extents[0]),
        "width_y": float(extents[1]),
        "height_z": float(extents[2]),
    }


def edge_topology(faces: Any) -> tuple[dict[str, int], dict[tuple[int, int], list[int]]]:
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, (a, b, c) in enumerate(faces):
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted((int(edge[0]), int(edge[1]))))
            edge_to_faces.setdefault(key, []).append(face_index)
    counts = [len(face_ids) for face_ids in edge_to_faces.values()]
    return {
        "edges": len(counts),
        "boundary_edges": sum(1 for count in counts if count == 1),
        "manifold_edges": sum(1 for count in counts if count == 2),
        "non_manifold_edges": sum(1 for count in counts if count > 2),
    }, edge_to_faces


def connected_components(face_count: int, edge_to_faces: dict[tuple[int, int], list[int]]) -> dict[str, Any]:
    parent = list(range(face_count))
    rank = [0] * face_count

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if rank[root_left] < rank[root_right]:
            parent[root_left] = root_right
        elif rank[root_left] > rank[root_right]:
            parent[root_right] = root_left
        else:
            parent[root_right] = root_left
            rank[root_left] += 1

    for face_ids in edge_to_faces.values():
        for face_id in face_ids[1:]:
            union(face_ids[0], face_id)
    counts: dict[int, int] = {}
    for face_index in range(face_count):
        root = find(face_index)
        counts[root] = counts.get(root, 0) + 1
    sizes = sorted(counts.values(), reverse=True)
    return {"count": len(sizes), "largest_faces": sizes[:10]}


def triangle_quality(np: Any, points: Any, faces: Any) -> dict[str, Any]:
    triangles = points[faces]
    e0 = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    e1 = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    e2 = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    max_edge = np.maximum.reduce([e0, e1, e2])
    min_edge = np.minimum.reduce([e0, e1, e2])
    min_altitude = np.divide(2.0 * areas, max_edge, out=np.zeros_like(areas), where=max_edge > 1e-15)
    aspect = np.divide(max_edge, min_altitude, out=np.full_like(max_edge, np.inf), where=min_altitude > 1e-15)
    finite_aspect = aspect[np.isfinite(aspect)]
    eps_area = max(float(np.nanmedian(areas)) * 1e-12, 1e-18) if areas.size else 1e-18
    return {
        "surface_area": float(areas.sum()),
        "degenerate_faces": int(np.count_nonzero((areas <= eps_area) | (min_edge <= 1e-15))),
        "area": percentile_summary(np, areas),
        "aspect_ratio": percentile_summary(np, finite_aspect),
    }


def percentile_summary(np: Any, values: Any) -> dict[str, Any]:
    if values.size == 0:
        return {"min": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def build_surface_summary(mesh: dict[str, Any], entity_names: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    names = {row["tag"]: row["name"] for row in entity_names.get("2", [])}
    return [
        {
            "surface_tag": tag,
            "name": names.get(tag),
            "parent_volume_tag": summary["parent_volume_tag"],
            "triangle_count": summary["triangle_count"],
        }
        for tag, summary in sorted(mesh["surface_summaries"].items())
    ]


def build_volume_summary(mesh: dict[str, Any], entity_names: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    names = {row["tag"]: row["name"] for row in entity_names.get("3", [])}
    surface_counts: dict[int, int] = {}
    triangle_counts: dict[int, int] = {}
    for summary in mesh["surface_summaries"].values():
        volume_tag = int(summary["parent_volume_tag"])
        surface_counts[volume_tag] = surface_counts.get(volume_tag, 0) + 1
        triangle_counts[volume_tag] = triangle_counts.get(volume_tag, 0) + int(summary["triangle_count"])
    return [
        {
            "volume_tag": tag,
            "name": names.get(tag),
            "surface_count": surface_counts[tag],
            "triangle_count": triangle_counts[tag],
        }
        for tag in sorted(surface_counts)
    ]


def provenance_limitations(mesh: dict[str, Any], controls: Any) -> list[str]:
    limitations = [
        "Gmsh entity tags are import-session identifiers, not guaranteed persistent source CAD face/body IDs.",
        "Assembly hierarchy, instance transforms, colors, layers, and materials are not preserved in this first exporter.",
        "VTP is the primary output because STL/OBJ would drop the cell-level provenance arrays.",
    ]
    if controls.chord is not None:
        limitations.append(
            "--chord is applied as a best-effort Gmsh/OCC/STL deflection hint when supported; it is not a certified Hausdorff bound."
        )
    if any(int(tag) < 0 for tag in mesh["parent_volume_tags"].tolist()):
        limitations.append("Some triangles have gmsh_parent_volume_tag=-1 because no unique upward volume adjacency was available.")
    return limitations


def highest_dimension(entity_counts: dict[str, int]) -> int | None:
    dims = [int(dim) for dim, count in entity_counts.items() if count > 0]
    return max(dims) if dims else None


def count_named(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("name"))
