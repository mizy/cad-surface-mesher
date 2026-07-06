from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import rasterize_side, render_mesh_artifacts, save_artifacts
from .constants import DEFAULT_WEIGHTS
from .mesh_io import read_mesh
from .normalization import normalize_vehicle_mesh
from .scoring import confidence_from_score, score_library, softmax
from .utils import read_json, resolve_path, write_json


def build_library(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest = read_json(manifest_path)
    manifest_dir = manifest_path.parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cars = []
    for car in manifest.get("cars", []):
        car_id = car["id"]
        mesh_path = resolve_path(car["mesh"], base=manifest_dir)
        target_dimensions = car.get("target_dimensions_m") or args.target_dimensions
        if target_dimensions is not None:
            if len(target_dimensions) != 3 or any(float(v) <= 0 for v in target_dimensions):
                raise ValueError(f"{car_id} target_dimensions_m must be three positive meter values")
            target_dimensions = tuple(float(v) for v in target_dimensions)
        artifacts, _normalized = render_mesh_artifacts(
            mesh_path,
            width=args.width,
            height=args.height,
            target_length=args.target_length,
            force_target_length=args.force_target_length,
            target_dimensions=target_dimensions,
        )
        artifact_paths = save_artifacts(artifacts, output_dir, car_id)
        record = {
            "id": car_id,
            "display_name": car.get("display_name", car_id),
            "category": car.get("category", "unknown"),
            "cd": float(car["cd"]),
            "cd_source": car.get("cd_source", ""),
            "cd_confidence": car.get("cd_confidence", "seed"),
            "artifacts": artifact_paths,
        }
        for key in ("mesh_source", "mesh_license", "mesh_author", "mesh_note", "target_dimensions_m", "dimensions_source"):
            if key in car:
                record[key] = car[key]
        cars.append(record)

    library = {
        "version": 2,
        "view": "side_-y",
        "image_width": args.width,
        "image_height": args.height,
        "score_weights": DEFAULT_WEIGHTS,
        "same_category_bonus": 0.05,
        "notes": manifest.get("notes", ""),
        "cars": cars,
    }
    library_path = output_dir / "reference_cars.json"
    write_json(library_path, library)
    print(str(library_path))
    return 0


def predict(args: argparse.Namespace) -> int:
    mesh_path = Path(args.mesh).resolve()
    library_path = Path(args.library).resolve()
    library = read_json(library_path)
    library_dir = library_path.parent
    width = int(library.get("image_width", args.width))
    height = int(library.get("image_height", args.height))

    mesh = read_mesh(mesh_path)
    variants = []
    for orientation, rotate_180_z in (("front_as_normalized", False), ("front_rotated_180_z", True)):
        normalized = normalize_vehicle_mesh(
            mesh,
            rotate_180_z=rotate_180_z,
            target_length=args.target_length,
            force_target_length=args.force_target_length,
            target_dimensions=args.target_dimensions,
        )
        artifacts = rasterize_side(normalized.mesh, width=width, height=height)
        scores = score_library(artifacts, library, library_dir, query_category=args.category)
        variants.append((orientation, normalized, artifacts, scores))

    orientation, normalized_query, query_artifacts, scores = max(
        variants,
        key=lambda item: item[3][0]["score"] if item[3] else -1.0,
    )
    if not scores:
        raise ValueError("Reference library is empty")

    top_k = scores[: max(1, min(args.top_k, len(scores)))]
    distances = [max(0.0, 1.0 - item["score"]) for item in top_k]
    logits = [args.temperature / (distance + args.epsilon) for distance in distances]
    weights = softmax(logits)
    cd = sum(weight * item["cd"] for weight, item in zip(weights, top_k))

    for item, weight, distance in zip(top_k, weights, distances):
        item["weight"] = float(weight)
        item["distance"] = float(distance)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    artifact_paths = None
    normalized_mesh_path = None
    if output_dir:
        artifact_paths = save_artifacts(query_artifacts, output_dir, "query")
        artifact_paths = {k: str(output_dir / v) for k, v in artifact_paths.items()}
        if not args.no_normalized_mesh:
            normalized_mesh_path = output_dir / "query_normalized.vtp"
            normalized_query.mesh.save(normalized_mesh_path)

    result = {
        "mesh": str(mesh_path),
        "library": str(library_path),
        "cd": float(cd),
        "confidence": confidence_from_score(float(top_k[0]["score"])),
        "query_category": args.category,
        "orientation": orientation,
        "normalization": normalized_query.info,
        "top_k": top_k,
        "artifacts": artifact_paths,
        "normalized_mesh": str(normalized_mesh_path) if normalized_mesh_path else None,
        "warning": "Reference Cd values come from heterogeneous public sources and test conditions; inspect top_k[].cd_source before engineering use.",
    }

    text = json.dumps(result, indent=2)
    print(text)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "prediction.json", result)
    return 0
