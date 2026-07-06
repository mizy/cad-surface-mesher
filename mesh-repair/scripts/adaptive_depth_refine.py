#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mesh_io import grid_shape, read_surface, triangle_faces, write_vtp
from mesh_metrics import mesh_report
from repair_report import adaptive_refinement_change_report


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a depth-map-driven adaptive refinement field on source exterior mesh."
    )
    parser.add_argument("source_mesh", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=900)
    parser.add_argument("--gradient-percentile", type=float, default=96.0)
    parser.add_argument("--include-face-jump", action="store_true")
    parser.add_argument("--disable-silhouette", action="store_true")
    parser.add_argument("--base-size", type=float, default=0.03)
    parser.add_argument("--transition-size", type=float, default=0.015)
    parser.add_argument("--fine-size", type=float, default=0.008)
    parser.add_argument("--fine-score-percentile", type=float, default=75.0)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--max-output-faces", type=int, default=900_000)
    return parser.parse_args()


def zbuffer_face_map(
    centroids: np.ndarray,
    view: ViewSpec,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = centroids[:, view.project_axes]
    rows, cols, mins, spans = grid_shape(coords, grid_size)
    uv = np.clip((coords - mins) / spans, 0.0, 1.0)
    col = np.minimum((uv[:, 0] * cols).astype(np.int64), cols - 1)
    row = np.minimum((uv[:, 1] * rows).astype(np.int64), rows - 1)
    linear = row * cols + col
    depth = centroids @ np.asarray(view.depth_vector, dtype=np.float64)

    order = np.lexsort((depth, linear))
    sorted_linear = linear[order]
    first = np.r_[True, sorted_linear[1:] != sorted_linear[:-1]]
    visible_faces = order[first]
    visible_linear = sorted_linear[first]

    depth_grid = np.full(rows * cols, np.nan, dtype=np.float64)
    face_grid = np.full(rows * cols, -1, dtype=np.int64)
    depth_grid[visible_linear] = depth[visible_faces]
    face_grid[visible_linear] = visible_faces
    valid = face_grid >= 0
    return depth_grid.reshape(rows, cols), face_grid.reshape(rows, cols), valid.reshape(rows, cols)


def critical_pixels(
    depth: np.ndarray,
    face_id: np.ndarray,
    valid: np.ndarray,
    percentile: float,
    include_face_jump: bool,
    include_silhouette: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    grad = np.zeros(depth.shape, dtype=np.float64)
    deltas = []
    add_gradient(depth[:, 1:] - depth[:, :-1], valid[:, 1:] & valid[:, :-1], grad[:, 1:], grad[:, :-1], deltas)
    add_gradient(depth[1:, :] - depth[:-1, :], valid[1:, :] & valid[:-1, :], grad[1:, :], grad[:-1, :], deltas)
    threshold = float(np.percentile(np.concatenate(deltas), percentile)) if deltas else np.inf
    depth_edges = grad >= threshold

    silhouette = silhouette_pixels(valid) if include_silhouette else np.zeros(valid.shape, dtype=bool)

    face_jump = face_jump_pixels(face_id, valid, grad) if include_face_jump else np.zeros(valid.shape, dtype=bool)

    critical = valid & (depth_edges | silhouette | face_jump)
    return critical, {
        "gradient_threshold": threshold,
        "critical_pixels": int(np.count_nonzero(critical)),
        "valid_pixels": int(np.count_nonzero(valid)),
        "include_face_jump": include_face_jump,
        "include_silhouette": include_silhouette,
    }


def silhouette_pixels(valid: np.ndarray) -> np.ndarray:
    silhouette = np.zeros(valid.shape, dtype=bool)
    mark_mask_transition(valid[:, 1:] != valid[:, :-1], silhouette[:, 1:], silhouette[:, :-1])
    mark_mask_transition(valid[1:, :] != valid[:-1, :], silhouette[1:, :], silhouette[:-1, :])
    return silhouette


def face_jump_pixels(face_id: np.ndarray, valid: np.ndarray, grad: np.ndarray) -> np.ndarray:
    face_jump = np.zeros(valid.shape, dtype=bool)
    horizontal_jump = (face_id[:, 1:] != face_id[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    vertical_jump = (face_id[1:, :] != face_id[:-1, :]) & valid[1:, :] & valid[:-1, :]
    mark_mask_transition(horizontal_jump & (grad[:, 1:] > 0.0), face_jump[:, 1:], face_jump[:, :-1])
    mark_mask_transition(vertical_jump & (grad[1:, :] > 0.0), face_jump[1:, :], face_jump[:-1, :])
    return face_jump


def add_gradient(
    delta: np.ndarray,
    pair_valid: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    deltas: list[np.ndarray],
) -> None:
    values = np.abs(delta[pair_valid])
    if values.size:
        deltas.append(values)
    local = np.zeros(delta.shape, dtype=np.float64)
    local[pair_valid] = np.abs(delta[pair_valid])
    np.maximum(left, local, out=left)
    np.maximum(right, local, out=right)


def mark_mask_transition(mask: np.ndarray, left: np.ndarray, right: np.ndarray) -> None:
    left |= mask
    right |= mask


def score_faces(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    grid_size: int,
    gradient_percentile: float,
    include_face_jump: bool,
    include_silhouette: bool,
    visual_dir: Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    centroids = points[faces].mean(axis=1)
    hits = np.zeros(faces.shape[0], dtype=np.int64)
    view_reports = []
    visual_dir.mkdir(parents=True, exist_ok=True)
    for view in VIEWS:
        depth, face_id, valid = zbuffer_face_map(centroids, view, grid_size)
        critical, stats = critical_pixels(
            depth,
            face_id,
            valid,
            gradient_percentile,
            include_face_jump,
            include_silhouette,
        )
        critical_faces = face_id[critical]
        critical_faces = critical_faces[critical_faces >= 0]
        np.add.at(hits, critical_faces, 1)
        image_path = visual_dir / f"{view.name}_critical.png"
        save_critical_image(image_path, depth, valid, critical)
        view_reports.append({"view": view.name, "image": str(image_path), **stats})
    return normalize_hits(hits), view_reports


def normalize_hits(hits: np.ndarray) -> np.ndarray:
    if hits.max(initial=0) == 0:
        return np.zeros(hits.shape[0], dtype=np.float64)
    return hits.astype(np.float64) / float(hits.max())


def save_critical_image(path: Path, depth: np.ndarray, valid: np.ndarray, critical: np.ndarray) -> None:
    image = np.zeros((*depth.shape, 3), dtype=np.uint8)
    image[~valid] = (255, 255, 255)
    if np.any(valid):
        values = depth[valid]
        denom = max(float(values.max() - values.min()), 1e-12)
        shade = 220 - ((values - values.min()) / denom * 160).astype(np.uint8)
        image[valid] = np.column_stack((shade, shade, shade))
    image[critical] = (255, 64, 32)
    Image.fromarray(image).save(path)


def target_sizes(
    score: np.ndarray,
    *,
    base_size: float,
    transition_size: float,
    fine_size: float,
    fine_score_percentile: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    sizes = np.full(score.shape[0], base_size, dtype=np.float64)
    nonzero = score[score > 0.0]
    fine_threshold = float(np.percentile(nonzero, fine_score_percentile)) if nonzero.size else np.inf
    sizes[score > 0.0] = transition_size
    sizes[score >= fine_threshold] = fine_size
    return sizes, {
        "base_size": base_size,
        "transition_size": transition_size,
        "fine_size": fine_size,
        "fine_score_threshold": fine_threshold,
        "base_faces": int(np.count_nonzero(sizes == base_size)),
        "transition_faces": int(np.count_nonzero(sizes == transition_size)),
        "fine_faces": int(np.count_nonzero(sizes == fine_size)),
    }


def adaptive_edge_refine(
    points: np.ndarray,
    faces: np.ndarray,
    sizes: np.ndarray,
    score: np.ndarray,
    source_ids: np.ndarray,
    *,
    max_iterations: int,
    max_output_faces: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    current_points = points.copy()
    current_faces = faces.copy()
    face_sizes = sizes.copy()
    face_scores = score.copy()
    face_sources = source_ids.copy()
    reports = []
    for iteration in range(max_iterations):
        candidate_edges, max_ratio = violating_edges(current_points, current_faces, face_sizes)
        selected_edges, stopped_by_cap = limit_edges(candidate_edges, len(current_faces), max_output_faces)
        reports.append({
            "iteration": iteration,
            "candidate_edges": len(candidate_edges),
            "split_edges": len(selected_edges),
            "max_edge_ratio": max_ratio,
            "stopped_by_face_cap": stopped_by_cap,
        })
        if not selected_edges:
            break
        current_points, current_faces, face_sizes, face_scores, face_sources = split_edges(
            current_points,
            current_faces,
            face_sizes,
            face_scores,
            face_sources,
            selected_edges,
        )
    return current_points, current_faces, {
        "target_size": face_sizes,
        "depth_refine_score": face_scores,
        "source_triangle_index": face_sources,
    }, reports


def violating_edges(
    points: np.ndarray,
    faces: np.ndarray,
    sizes: np.ndarray,
) -> tuple[set[tuple[int, int]], float]:
    selected = set()
    max_ratio = 0.0
    for face, target in zip(faces, sizes):
        edges = ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
        lengths = [float(np.linalg.norm(points[a] - points[b])) for a, b in edges]
        ratio = max(lengths) / max(float(target), 1e-12)
        max_ratio = max(max_ratio, ratio)
        if ratio > 1.0:
            selected.add(edge_key(*edges[int(np.argmax(lengths))]))
    return selected, max_ratio


def limit_edges(
    edges: set[tuple[int, int]],
    face_count: int,
    max_faces: int,
) -> tuple[set[tuple[int, int]], bool]:
    if face_count + len(edges) * 2 <= max_faces:
        return edges, False
    allowed = max(0, (max_faces - face_count) // 2)
    return set(list(edges)[:allowed]), True


def split_edges(
    points: np.ndarray,
    faces: np.ndarray,
    sizes: np.ndarray,
    scores: np.ndarray,
    sources: np.ndarray,
    selected_edges: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    point_list = points.tolist()
    midpoint_index: dict[tuple[int, int], int] = {}
    for edge in selected_edges:
        a, b = edge
        midpoint_index[edge] = len(point_list)
        point_list.append(((points[a] + points[b]) * 0.5).tolist())
    new_points = np.asarray(point_list, dtype=np.float64)

    new_faces = []
    new_sizes = []
    new_scores = []
    new_sources = []
    for face, size, score, source in zip(faces, sizes, scores, sources):
        children = split_face(face, new_points, midpoint_index)
        for child in children:
            new_faces.append(child)
            new_sizes.append(size)
            new_scores.append(score)
            new_sources.append(source)
    return (
        new_points,
        np.asarray(new_faces, dtype=np.int64),
        np.asarray(new_sizes, dtype=np.float64),
        np.asarray(new_scores, dtype=np.float64),
        np.asarray(new_sources, dtype=np.int64),
    )


def split_face(
    face: np.ndarray,
    points: np.ndarray,
    midpoint_index: dict[tuple[int, int], int],
) -> list[list[int]]:
    a, b, c = [int(v) for v in face]
    mids = {
        "ab": midpoint_index.get(edge_key(a, b)),
        "bc": midpoint_index.get(edge_key(b, c)),
        "ca": midpoint_index.get(edge_key(c, a)),
    }
    raw = split_face_raw(a, b, c, mids)
    normal = np.cross(points[b] - points[a], points[c] - points[a])
    return [orient(child, points, normal) for child in raw]


def split_face_raw(a: int, b: int, c: int, mids: dict[str, int | None]) -> list[list[int]]:
    ab, bc, ca = mids["ab"], mids["bc"], mids["ca"]
    if ab is not None and bc is not None and ca is not None:
        return [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
    if ab is not None and bc is not None:
        return [[b, bc, ab], [a, ab, c], [ab, bc, c]]
    if bc is not None and ca is not None:
        return [[c, ca, bc], [b, bc, a], [bc, ca, a]]
    if ca is not None and ab is not None:
        return [[a, ab, ca], [c, ca, b], [ca, ab, b]]
    if ab is not None:
        return [[a, ab, c], [ab, b, c]]
    if bc is not None:
        return [[b, bc, a], [bc, c, a]]
    if ca is not None:
        return [[c, ca, b], [ca, a, b]]
    return [[a, b, c]]


def orient(face: list[int], points: np.ndarray, normal: np.ndarray) -> list[int]:
    p0, p1, p2 = points[face]
    child_normal = np.cross(p1 - p0, p2 - p0)
    if float(np.dot(child_normal, normal)) < 0.0:
        return [face[0], face[2], face[1]]
    return face


def edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def source_ids_from_mesh(mesh: Any, face_count: int) -> np.ndarray:
    if "source_triangle_index" in mesh.cell_data:
        return np.asarray(mesh.cell_data["source_triangle_index"], dtype=np.int64)
    return np.arange(face_count, dtype=np.int64)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh = read_surface(args.source_mesh)
    points = np.asarray(mesh.points, dtype=np.float64)
    faces = triangle_faces(mesh)
    source_ids = source_ids_from_mesh(mesh, faces.shape[0])

    score, view_reports = score_faces(
        points,
        faces,
        grid_size=args.grid_size,
        gradient_percentile=args.gradient_percentile,
        include_face_jump=args.include_face_jump,
        include_silhouette=not args.disable_silhouette,
        visual_dir=args.output_dir / "visual",
    )
    sizes, size_report = target_sizes(
        score,
        base_size=args.base_size,
        transition_size=args.transition_size,
        fine_size=args.fine_size,
        fine_score_percentile=args.fine_score_percentile,
    )
    field_path = args.output_dir / "refinement_field.vtp"
    write_vtp(field_path, points, faces, {"depth_refine_score": score, "target_size": sizes, "source_triangle_index": source_ids})

    refined_points, refined_faces, cell_data, refine_reports = adaptive_edge_refine(
        points,
        faces,
        sizes,
        score,
        source_ids,
        max_iterations=args.max_iterations,
        max_output_faces=args.max_output_faces,
    )
    refined_path = args.output_dir / "adaptive_refined_source.vtp"
    write_vtp(refined_path, refined_points, refined_faces, cell_data)
    report = build_report(args, points, faces, refined_points, refined_faces, view_reports, size_report, refine_reports)
    report["outputs"] = {"refinement_field_vtp": str(field_path), "adaptive_refined_source_vtp": str(refined_path)}
    report_path = args.output_dir / "adaptive_refinement_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary(report_path, report), indent=2))
    return 0


def build_report(
    args: argparse.Namespace,
    points: np.ndarray,
    faces: np.ndarray,
    refined_points: np.ndarray,
    refined_faces: np.ndarray,
    view_reports: list[dict[str, Any]],
    size_report: dict[str, Any],
    refine_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    source_metrics = mesh_report(points, faces)
    refined_metrics = mesh_report(refined_points, refined_faces)
    return {
        "input": str(args.source_mesh),
        "method": "depth_face_id_buffers_to_source_face_size_field_then_conforming_edge_split",
        "limitations": [
            "This prototype refines source triangles but does not coarsen already-dense non-critical source regions.",
            "The output preserves source-surface detail but does not perform watertight sealing by itself.",
        ],
        "parameters": vars(args) | {"output_dir": str(args.output_dir), "source_mesh": str(args.source_mesh)},
        "view_reports": view_reports,
        "size_field": size_report,
        "refinement_iterations": refine_reports,
        "source_metrics": source_metrics,
        "refined_metrics": refined_metrics,
        "change_report": adaptive_refinement_change_report(source_metrics, refined_metrics, size_report),
    }


def summary(report_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["refined_metrics"]
    return {
        "report": str(report_path),
        "source_triangles": report["source_metrics"]["triangles"],
        "refined_triangles": metrics["triangles"],
        "fine_faces": report["size_field"]["fine_faces"],
        "transition_faces": report["size_field"]["transition_faces"],
        "boundary_edges": metrics["topology"]["boundary_edges"],
        "non_manifold_edges": metrics["topology"]["non_manifold_edges"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
