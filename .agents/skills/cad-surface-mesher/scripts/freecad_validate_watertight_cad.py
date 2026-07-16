#!/usr/bin/env python3
"""Validate a STEP/BREP watertight CAD result inside FreeCADCmd."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import Part


STEP_ENTITIES = (
    "MANIFOLD_SOLID_BREP",
    "BREP_WITH_VOIDS",
    "CLOSED_SHELL",
    "OPEN_SHELL",
    "SHELL_BASED_SURFACE_MODEL",
    "ADVANCED_FACE",
)


def forwarded_arguments() -> list[str] | None:
    raw = sys.argv[1:]
    forwarded = []
    index = 0
    while index < len(raw):
        value = raw[index]
        if value == "--pass":
            if index + 1 >= len(raw):
                raise SystemExit("FreeCAD --pass requires one following argument")
            forwarded.append(raw[index + 1])
            index += 2
            continue
        if value.startswith("--pass="):
            forwarded.append(value.split("=", 1)[1])
        index += 1
    return forwarded or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate physical closure and optional STEP/BREP round-trip drift."
    )
    parser.add_argument("input", type=Path, help="STEP, STP, BREP, or BRP CAD file")
    parser.add_argument(
        "--reference",
        type=Path,
        help="Optional in-memory/native BREP reference for drift comparison",
    )
    parser.add_argument("--expected-solids", type=int)
    parser.add_argument("--max-bbox-drift-mm", type=float, default=1.0e-3)
    parser.add_argument("--max-relative-measure-drift", type=float, default=5.0e-7)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(forwarded_arguments())


def bbox(shape: Part.Shape) -> list[float]:
    box = shape.BoundBox
    return [box.XMin, box.YMin, box.ZMin, box.XMax, box.YMax, box.ZMax]


def check_errors(shape: Part.Shape) -> list[str]:
    try:
        result = shape.check(False)
    except Exception as error:
        lines = str(error).splitlines()
    else:
        if result is None:
            return []
        lines = str(result).splitlines()
    return [line for line in lines if line.strip() and line.strip() != "No error"]


def edge_topology(shape: Part.Shape) -> dict[str, object]:
    physical_free = []
    degenerated_free = []
    nonmanifold = []
    for edge in shape.Edges:
        owner_count = len(shape.ancestorsOfType(edge, Part.Face))
        if owner_count == 1 and edge.Degenerated:
            degenerated_free.append(edge)
        elif owner_count == 1:
            physical_free.append(edge)
        elif owner_count > 2:
            nonmanifold.append(edge)
    return {
        "physical_free_edge_count": len(physical_free),
        "physical_free_edge_length_mm": sum(edge.Length for edge in physical_free),
        "degenerated_free_edge_count": len(degenerated_free),
        "degenerated_free_edge_length_mm": sum(
            edge.Length for edge in degenerated_free
        ),
        "nonmanifold_edge_count": len(nonmanifold),
    }


def shape_stats(shape: Part.Shape) -> dict[str, object]:
    return {
        "shape_type": shape.ShapeType,
        "valid": shape.isValid(),
        "closed": shape.isClosed(),
        "face_count": len(shape.Faces),
        "shell_count": len(shape.Shells),
        "solid_count": len(shape.Solids),
        "area_mm2": shape.Area,
        "volume_mm3": shape.Volume,
        "bbox_mm": bbox(shape),
        "check_errors": check_errors(shape),
        **edge_topology(shape),
    }


def validate_shape(
    shape: Part.Shape, stats: dict[str, object], expected_solids: int | None
) -> list[str]:
    failures = []
    if not stats["valid"]:
        failures.append("shape is invalid")
    if not stats["closed"]:
        failures.append("shape is open")
    if not shape.Solids:
        failures.append("shape contains no solid")
    if expected_solids is not None and len(shape.Solids) != expected_solids:
        failures.append(f"expected {expected_solids} solids, found {len(shape.Solids)}")
    if stats["physical_free_edge_count"] != 0:
        failures.append(f"physical free edges={stats['physical_free_edge_count']}")
    if stats["nonmanifold_edge_count"] != 0:
        failures.append(f"non-manifold edges={stats['nonmanifold_edge_count']}")
    if stats["check_errors"]:
        failures.append(f"shape.check errors={stats['check_errors']}")
    for index, solid in enumerate(shape.Solids):
        if not solid.isValid() or not solid.isClosed() or solid.Volume <= 0.0:
            failures.append(f"solid {index} is invalid, open, or negatively oriented")
    return failures


def compare_shapes(
    reference: Part.Shape,
    target: Part.Shape,
    max_bbox_drift_mm: float,
    max_relative_measure_drift: float,
) -> tuple[dict[str, object], list[str]]:
    bbox_drift = [
        abs(after - before) for before, after in zip(bbox(reference), bbox(target))
    ]
    area_delta = abs(target.Area - reference.Area)
    volume_delta = abs(target.Volume - reference.Volume)
    area_relative_delta = area_delta / max(abs(reference.Area), 1.0e-300)
    volume_relative_delta = volume_delta / max(abs(reference.Volume), 1.0e-300)
    failures = []
    for name, before, after in (
        ("face", len(reference.Faces), len(target.Faces)),
        ("shell", len(reference.Shells), len(target.Shells)),
        ("solid", len(reference.Solids), len(target.Solids)),
    ):
        if before != after:
            failures.append(f"{name} count changed: {before} -> {after}")
    if max(bbox_drift) > max_bbox_drift_mm:
        failures.append(f"bbox drift {max(bbox_drift)} mm exceeds limit")
    if area_relative_delta > max_relative_measure_drift:
        failures.append(f"relative area drift {area_relative_delta} exceeds limit")
    if volume_relative_delta > max_relative_measure_drift:
        failures.append(f"relative volume drift {volume_relative_delta} exceeds limit")
    return {
        "bbox_drift_mm": bbox_drift,
        "max_bbox_drift_mm": max(bbox_drift),
        "area_delta_mm2": area_delta,
        "area_relative_delta": area_relative_delta,
        "volume_delta_mm3": volume_delta,
        "volume_relative_delta": volume_relative_delta,
    }, failures


def step_entity_counts(path: Path) -> dict[str, int] | None:
    if path.suffix.lower() not in {".step", ".stp"}:
        return None
    pattern = re.compile(r"=\s*(" + "|".join(STEP_ENTITIES) + r")\s*\(")
    counts = {name: 0 for name in STEP_ENTITIES}
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = pattern.search(line)
            if match:
                counts[match.group(1)] += 1
    return counts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# @entry
def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    report_path = (
        args.report.expanduser().resolve()
        if args.report
        else input_path.with_suffix(input_path.suffix + ".watertight-validation.json")
    )
    shape = Part.read(str(input_path))
    stats = shape_stats(shape)
    failures = validate_shape(shape, stats, args.expected_solids)
    payload: dict[str, object] = {
        "input": str(input_path),
        "sha256": sha256(input_path),
        "acceptance": {
            "expected_solids": args.expected_solids,
            "max_bbox_drift_mm": args.max_bbox_drift_mm,
            "max_relative_measure_drift": args.max_relative_measure_drift,
        },
        "shape": stats,
        "solids": [shape_stats(solid) for solid in shape.Solids],
        "step_entities": step_entity_counts(input_path),
    }
    if args.reference:
        reference_path = args.reference.expanduser().resolve()
        reference = Part.read(str(reference_path))
        comparison, comparison_failures = compare_shapes(
            reference,
            shape,
            args.max_bbox_drift_mm,
            args.max_relative_measure_drift,
        )
        failures.extend(comparison_failures)
        payload["reference"] = {
            "path": str(reference_path),
            "sha256": sha256(reference_path),
            "shape": shape_stats(reference),
        }
        payload["comparison"] = comparison
    payload["failures"] = failures
    payload["passed"] = not failures
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
