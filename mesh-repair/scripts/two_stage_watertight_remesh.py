#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from mesh_io import compact_mesh, grid_shape, read_surface, save_depth_preview, triangle_faces, write_vtp
from mesh_metrics import bbox_drift_from_reports, mesh_report, silhouette_drift_from_meshes
from html_report import write_html_report
from repair_report import two_stage_repair_report


@dataclass(frozen=True)
class ViewSpec:
    name: str
    project_axes: tuple[int, int]
    depth_vector: tuple[float, float, float]


VIEWS = [
    ViewSpec("front_minus_x", (1, 2), (1.0, 0.0, 0.0)),
    ViewSpec("rear_plus_x", (1, 2), (-1.0, 0.0, 0.0)),
    ViewSpec("left_minus_y", (0, 2), (0.0, 1.0, 0.0)),
    ViewSpec("right_plus_y", (0, 2), (0.0, -1.0, 0.0)),
    ViewSpec("top_plus_z", (0, 1), (0.0, 0.0, -1.0)),
    ViewSpec("bottom_minus_z", (0, 1), (0.0, 0.0, 1.0)),
]

DEFAULT_REMOVE_NAME_REGEX = (
    r"(^|[_:])int(ernal)?([_:]|$)|internal|interior|hidden|inside|cavity|"
    r"carinternal|seat|dashboard|centerconsole|steeringwheel"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract an exterior wall candidate from a mesh-only assembly, "
            "then voxel-remesh that candidate into a watertight surface."
        )
    )
    parser.add_argument("input", type=Path, help="Input mesh: VTP, STL, OBJ, GLB, GLTF, or VTK.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--group-source-gltf",
        type=Path,
        help=(
            "Optional GLTF/GLB source with geometry names. When its flattened triangle count "
            "matches the input, non-target groups are removed before visibility extraction."
        ),
    )
    parser.add_argument("--target-name", default="external-flow-skin")
    parser.add_argument(
        "--remove-name-regex",
        default=DEFAULT_REMOVE_NAME_REGEX,
        help="Case-insensitive group/geometry-name regex removed before visibility extraction.",
    )
    parser.add_argument("--visibility-grid", type=int, default=900)
    parser.add_argument("--depth-tolerance", type=float, default=0.02)
    parser.add_argument("--dilate-rings", type=int, default=1)
    parser.add_argument("--voxel-pitch", type=float, default=0.04)
    parser.add_argument("--preview-size", type=int, default=900)
    parser.add_argument("--skip-previews", action="store_true")
    return parser.parse_args()


def gltf_group_keep_mask(
    gltf_path: Path,
    expected_faces: int,
    remove_name_regex: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    scene = trimesh.load(gltf_path, force="scene")
    remove_pattern = re.compile(remove_name_regex, re.IGNORECASE)
    keep = np.ones(expected_faces, dtype=bool)
    cursor = 0
    removed_groups = []
    kept_groups = []
    for name, geometry in scene.geometry.items():
        face_count = int(len(geometry.faces))
        start = cursor
        end = cursor + face_count
        cursor = end
        if end > expected_faces:
            raise ValueError("GLTF flattened face count exceeds input face count")
        if remove_pattern.search(name):
            keep[start:end] = False
            removed_groups.append({"name": name, "triangles": face_count})
        else:
            kept_groups.append({"name": name, "triangles": face_count})
    if cursor != expected_faces:
        raise ValueError(
            f"GLTF flattened face count {cursor} does not match input face count {expected_faces}"
        )
    return keep, group_filter_report(gltf_path, remove_pattern, removed_groups, kept_groups, expected_faces)


def group_filter_report(
    gltf_path: Path,
    remove_pattern: re.Pattern[str],
    removed_groups: list[dict[str, Any]],
    kept_groups: list[dict[str, Any]],
    expected_faces: int,
) -> dict[str, Any]:
    removed_triangles = sum(int(row["triangles"]) for row in removed_groups)
    return {
        "method": "gltf_geometry_name_filter_by_flattened_face_ranges",
        "source": str(gltf_path),
        "assumption": "GLTF geometry flatten order matches the input mesh triangle order.",
        "remove_pattern": remove_pattern.pattern,
        "removed_groups": len(removed_groups),
        "kept_groups": len(kept_groups),
        "removed_triangles": removed_triangles,
        "removed_triangle_ratio": float(removed_triangles / expected_faces),
        "largest_removed_groups": largest_groups(removed_groups),
        "largest_kept_groups": largest_groups(kept_groups),
    }


def largest_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(groups, key=lambda row: row["triangles"], reverse=True)[:20]


def visible_faces_from_view(
    centroids: np.ndarray,
    view: ViewSpec,
    *,
    grid_size: int,
    depth_tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    coords = centroids[:, view.project_axes]
    rows, cols, mins, spans = grid_shape(coords, grid_size)
    uv = np.clip((coords - mins) / spans, 0.0, 1.0)
    col = np.minimum((uv[:, 0] * cols).astype(np.int64), cols - 1)
    row = np.minimum((uv[:, 1] * rows).astype(np.int64), rows - 1)
    linear = row * cols + col

    depth_vector = np.asarray(view.depth_vector, dtype=np.float64)
    depth = centroids @ depth_vector
    min_depth = np.full(rows * cols, np.inf, dtype=np.float64)
    np.minimum.at(min_depth, linear, depth)

    visible = depth <= min_depth[linear] + depth_tolerance
    return visible, {
        "view": view.name,
        "grid": {"rows": rows, "cols": cols},
        "visible_faces": int(np.count_nonzero(visible)),
        "occupied_pixels": int(np.count_nonzero(np.isfinite(min_depth))),
    }


def exterior_visibility_mask(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    grid_size: int,
    depth_tolerance: float,
    dilate_rings: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    centroids = points[faces].mean(axis=1)
    visible_score = np.zeros(faces.shape[0], dtype=np.int16)
    view_reports = []
    for view in VIEWS:
        visible, view_report = visible_faces_from_view(
            centroids,
            view,
            grid_size=grid_size,
            depth_tolerance=depth_tolerance,
        )
        visible_score[visible] += 1
        view_reports.append(view_report)

    selected = visible_score > 0
    for _ in range(max(0, dilate_rings)):
        selected = dilate_by_vertices(faces, selected, points.shape[0])
    return selected, visible_score, view_reports


def dilate_by_vertices(faces: np.ndarray, selected: np.ndarray, point_count: int) -> np.ndarray:
    vertex_selected = np.zeros(point_count, dtype=bool)
    vertex_selected[faces[selected].ravel()] = True
    return np.any(vertex_selected[faces], axis=1)


def voxel_watertight_remesh(points: np.ndarray, faces: np.ndarray, pitch: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    voxel = source.voxelized(pitch=pitch)
    filled = voxel.fill()
    remeshed = filled.marching_cubes
    remeshed.apply_transform(filled.transform)
    remeshed.remove_unreferenced_vertices()
    try:
        remeshed.fix_normals()
    except Exception:
        pass
    report = {
        "pitch": pitch,
        "source_trimesh_watertight": bool(source.is_watertight),
        "shell_voxels": voxel_count(voxel),
        "filled_voxels": voxel_count(filled),
        "output_trimesh_watertight": bool(remeshed.is_watertight),
    }
    return (
        np.asarray(remeshed.vertices, dtype=np.float64),
        np.asarray(remeshed.faces, dtype=np.int64),
        report,
    )


def voxel_count(voxel: Any) -> int | None:
    try:
        return int(voxel.encoding.sparse_indices.shape[0])
    except Exception:
        return None


def prepare_group_candidate(
    args: argparse.Namespace,
    points: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    candidate_indices = np.arange(faces.shape[0], dtype=np.int64)
    if not args.group_source_gltf:
        return candidate_indices, points, faces, None

    group_keep, group_filter = gltf_group_keep_mask(
        args.group_source_gltf,
        faces.shape[0],
        args.remove_name_regex,
    )
    candidate_indices = np.flatnonzero(group_keep)
    group_points, group_faces = compact_mesh(points, faces, candidate_indices)
    group_path = args.output_dir / "stage0_group_filtered.vtp"
    write_vtp(
        group_path,
        group_points,
        group_faces,
        {"source_triangle_index": candidate_indices.astype(np.int64)},
    )
    group_filter["output_vtp"] = str(group_path)
    return candidate_indices, group_points, group_faces, group_filter


def build_report(
    args: argparse.Namespace,
    stage_reports: dict[str, dict[str, Any]],
    extraction: dict[str, Any],
    remesh: dict[str, Any],
    outputs: dict[str, Any],
    group_filter: dict[str, Any] | None,
    visual_drift: dict[str, Any],
) -> dict[str, Any]:
    stage2 = stage_reports["stage2_watertight_remesh"]
    stage2_topology = stage2["topology"]
    stage2_quality = stage2["quality"]
    closed_manifold_pass = (
        stage2_topology["boundary_edges"] == 0
        and stage2_topology["non_manifold_edges"] == 0
        and stage2_quality["degenerate_faces"] == 0
    )
    single_volume_pass = closed_manifold_pass and stage2["volume"]["reliable"]
    stage1_drift = bbox_drift_from_reports(stage_reports["original"], stage_reports["stage1_exterior_candidate"])
    stage2_drift = bbox_drift_from_reports(stage_reports["stage1_exterior_candidate"], stage2)
    return {
        "input": {"path": str(args.input), "kind": "mesh", "group_metadata_available": group_filter is not None},
        "output_contract": {
            "input_kind": "mesh",
            "output_kind": "watertight_mesh",
            "repair_domain": "mesh_domain_voxel_remesh",
            "cad_output": {"supported": False, "reason": "this prototype does not perform reverse CAD fitting"},
        },
        "target": {"name": args.target_name, "opening_policy": "unresolved openings are capped by coarse voxel remesh"},
        "parameters": parameters_report(args),
        "limitations": limitations(group_filter),
        "group_filter": group_filter,
        "stages": {
            "original": stage_reports["original"],
            "stage1_exterior_candidate": {**stage_reports["stage1_exterior_candidate"], "extraction": extraction},
            "stage2_watertight_remesh": {**stage2, "remesh": remesh},
        },
        "repair_report": two_stage_repair_report(stage_reports, extraction, group_filter, remesh),
        "comparisons": {
            "stage1_vs_original": stage1_drift,
            "stage2_vs_stage1": stage2_drift,
            "stage2_silhouette_vs_stage1": visual_drift,
        },
        "gates": gates_report(
            extraction,
            stage2_topology,
            stage2_quality,
            stage2,
            closed_manifold_pass,
            single_volume_pass,
            stage2_drift,
            visual_drift,
        ),
        "outputs": outputs,
    }


def parameters_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "visibility_grid": args.visibility_grid,
        "depth_tolerance": args.depth_tolerance,
        "dilate_rings": args.dilate_rings,
        "voxel_pitch": args.voxel_pitch,
        "remove_name_regex": args.remove_name_regex,
    }


def limitations(group_filter: dict[str, Any] | None) -> list[str]:
    result = [
        "Target-specific functional openings and through-holes are not classified individually.",
        "Voxel remesh is a watertight coarse envelope, not a curvature-preserving final CFD surface.",
        "Stage2 visual preservation is checked by shared-projection silhouette drift; high drift rejects voxel output as a final geometry.",
    ]
    if group_filter:
        result.append("Group filtering assumes the GLTF geometry flatten order matches the input mesh triangle order.")
    else:
        result.extend([
            "No CAD/GLTF group tree was provided for this run.",
            "Stage 1 uses mesh-only multi-view outermost visibility instead of group hide/show.",
        ])
    return result


def gates_report(
    extraction: dict[str, Any],
    topology: dict[str, Any],
    quality: dict[str, Any],
    stage2: dict[str, Any],
    closed_manifold_pass: bool,
    single_volume_pass: bool,
    drift: dict[str, Any],
    visual_drift: dict[str, Any],
) -> dict[str, Any]:
    bbox_pass = drift["max_ratio"] <= 0.002
    silhouette_pass = visual_drift["summary"]["changed_ratio_max"] <= 0.05
    automated_geometry_pass = closed_manifold_pass and single_volume_pass and bbox_pass and silhouette_pass
    return {
        "stage1_removed_internal_or_hidden_triangle_ratio": extraction["removed_triangle_ratio"],
        "watertight_topology_pass": closed_manifold_pass,
        "single_reliable_volume_pass": single_volume_pass,
        "stage2_boundary_edges_zero": topology["boundary_edges"] == 0,
        "stage2_non_manifold_edges_zero": topology["non_manifold_edges"] == 0,
        "stage2_degenerate_faces_zero": quality["degenerate_faces"] == 0,
        "stage2_volume_reliable": stage2["volume"]["reliable"],
        "stage2_bbox_drift_within_default_0_002": bbox_pass,
        "stage2_silhouette_changed_ratio_max": visual_drift["summary"]["changed_ratio_max"],
        "stage2_silhouette_drift_within_default_0_05": silhouette_pass,
        "automated_geometry_preservation_pass": automated_geometry_pass,
        "engineering_pass": False,
        "engineering_pass_reason": (
            "This prototype only proves watertight topology. Final engineering pass still requires "
            "self-intersection checks, visual opening-policy review, and drift within target tolerance. "
            "High silhouette drift means the voxel output is a closure proxy, not an accepted final geometry."
        ),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_mesh = read_surface(args.input)
    points = np.asarray(source_mesh.points, dtype=np.float64)
    faces = triangle_faces(source_mesh)
    original_report = mesh_report(points, faces)
    candidate_indices, group_points, group_faces, group_filter = prepare_group_candidate(args, points, faces)

    selected, visible_score, view_reports = exterior_visibility_mask(
        group_points,
        group_faces,
        grid_size=args.visibility_grid,
        depth_tolerance=args.depth_tolerance,
        dilate_rings=args.dilate_rings,
    )
    local_stage1_indices = np.flatnonzero(selected)
    stage1_indices = candidate_indices[local_stage1_indices]
    stage1_points, stage1_faces = compact_mesh(points, faces, stage1_indices)
    stage1_path = write_stage1(args, stage1_points, stage1_faces, stage1_indices, visible_score, local_stage1_indices)
    stage1_report = mesh_report(stage1_points, stage1_faces)

    stage2_points, stage2_faces, remesh_report = voxel_watertight_remesh(stage1_points, stage1_faces, args.voxel_pitch)
    stage2_path = args.output_dir / "stage2_watertight_surface.vtp"
    write_vtp(stage2_path, stage2_points, stage2_faces)
    stage2_report = mesh_report(stage2_points, stage2_faces)
    visual_drift = silhouette_drift_from_meshes(
        stage1_points,
        stage1_faces,
        stage2_points,
        stage2_faces,
        [VIEWS[0], VIEWS[2], VIEWS[4]],
        max_size=min(args.visibility_grid, 720),
    )

    previews = write_previews(args, stage1_points, stage1_faces, stage2_points, stage2_faces)
    extraction = extraction_report(view_reports, visible_score, stage1_indices, faces)
    outputs = outputs_report(args, stage1_path, stage2_path, previews)
    report = build_report(
        args,
        {
            "original": original_report,
            "stage1_exterior_candidate": stage1_report,
            "stage2_watertight_remesh": stage2_report,
        },
        extraction,
        remesh_report,
        outputs,
        group_filter,
        visual_drift,
    )
    html_path = args.output_dir / "two_stage_report.html"
    report["outputs"]["html_report"] = str(html_path)
    report_path = args.output_dir / "two_stage_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_html_report(report, html_path, "Two Stage Watertight Mesh Report")
    print_summary(report_path, stage1_report, stage2_report, report)
    return 0


def write_stage1(
    args: argparse.Namespace,
    points: np.ndarray,
    faces: np.ndarray,
    source_indices: np.ndarray,
    visible_score: np.ndarray,
    local_indices: np.ndarray,
) -> Path:
    path = args.output_dir / "stage1_exterior_candidate.vtp"
    write_vtp(
        path,
        points,
        faces,
        {
            "source_triangle_index": source_indices.astype(np.int64),
            "visible_view_count": visible_score[local_indices].astype(np.int16),
        },
    )
    return path


def write_previews(
    args: argparse.Namespace,
    stage1_points: np.ndarray,
    stage1_faces: np.ndarray,
    stage2_points: np.ndarray,
    stage2_faces: np.ndarray,
) -> list[str]:
    if args.skip_previews:
        return []
    visual_dir = args.output_dir / "visual"
    previews = save_depth_preview(visual_dir, "stage1", stage1_points, stage1_faces, VIEWS, size=args.preview_size)
    previews.extend(save_depth_preview(visual_dir, "stage2", stage2_points, stage2_faces, VIEWS, size=args.preview_size))
    return previews


def extraction_report(
    view_reports: list[dict[str, Any]],
    visible_score: np.ndarray,
    stage1_indices: np.ndarray,
    original_faces: np.ndarray,
) -> dict[str, Any]:
    return {
        "method": "group_name_filter_then_six_view_centroid_zbuffer",
        "view_reports": view_reports,
        "visible_before_dilation_triangles": int(np.count_nonzero(visible_score > 0)),
        "selected_after_dilation_triangles": int(stage1_indices.size),
        "removed_triangles": int(original_faces.shape[0] - stage1_indices.size),
        "removed_triangle_ratio": float(1.0 - stage1_indices.size / original_faces.shape[0]),
    }


def outputs_report(args: argparse.Namespace, stage1_path: Path, stage2_path: Path, previews: list[str]) -> dict[str, Any]:
    return {
        "stage1_exterior_candidate_vtp": str(stage1_path),
        "stage2_watertight_surface_vtp": str(stage2_path),
        "report_json": str(args.output_dir / "two_stage_report.json"),
        "previews": previews,
    }


def print_summary(
    report_path: Path,
    stage1_report: dict[str, Any],
    stage2_report: dict[str, Any],
    report: dict[str, Any],
) -> None:
    print(json.dumps({
        "report": str(report_path),
        "stage1_triangles": stage1_report["triangles"],
        "stage2_triangles": stage2_report["triangles"],
        "stage2_boundary_edges": stage2_report["topology"]["boundary_edges"],
        "stage2_non_manifold_edges": stage2_report["topology"]["non_manifold_edges"],
        "engineering_pass": report["gates"]["engineering_pass"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
