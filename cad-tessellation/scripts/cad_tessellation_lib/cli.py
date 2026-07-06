from __future__ import annotations

import argparse
import sys
from pathlib import Path


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a numeric value") from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("expected a positive value")
    return parsed


def run_tessellate(args: argparse.Namespace) -> int:
    from .gmsh_pipeline import SUPPORTED_SUFFIXES, TessellationControls, TessellationError, tessellate_cad
    from .mesh_pipeline import SUPPORTED_MESH_SUFFIXES, tessellate_mesh

    controls = TessellationControls(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        cad_format=args.format,
        occ_target_unit=args.occ_target_unit,
        mesh_size=args.mesh_size,
        mesh_size_min=args.mesh_size_min,
        mesh_size_max=args.mesh_size_max,
        angle_deg=args.angle_deg,
        chord=args.chord,
        import_labels=args.import_labels,
        save_debug_msh=args.save_debug_msh,
    )
    suffix = controls.input_path.suffix.lower()
    if controls.cad_format != "auto":
        if suffix in SUPPORTED_MESH_SUFFIXES:
            raise TessellationError("--format is only valid for CAD inputs; omit it for mesh inputs.")
        result = tessellate_cad(controls)
    elif suffix in SUPPORTED_SUFFIXES:
        result = tessellate_cad(controls)
    elif suffix in SUPPORTED_MESH_SUFFIXES:
        result = tessellate_mesh(controls)
    else:
        supported = ", ".join(sorted((*SUPPORTED_SUFFIXES, *SUPPORTED_MESH_SUFFIXES)))
        raise TessellationError(f"Cannot infer input format from {controls.input_path.name}; supported suffixes: {supported}")
    print(result.report_path)
    return 0


def run_smoke(args: argparse.Namespace) -> int:
    from .smoke import run_smoke_test

    report = run_smoke_test(Path(args.output_dir), keep_fixture=args.keep_fixture)
    print(report["outputs"]["report"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert CAD or mesh inputs into a triangle-only VTP surface mesh "
            "with a JSON tessellation quality report."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tessellate = sub.add_parser("tessellate", help="Convert one CAD or mesh file into surface_mesh.vtp.")
    tessellate.add_argument(
        "input",
        help="Input CAD or mesh file (.step/.stp/.iges/.igs/.brep/.brp/.stl/.obj/.vtp/.vtk/.glb/.gltf).",
    )
    tessellate.add_argument("--output-dir", required=True, help="Directory for surface_mesh.vtp and tessellation_report.json.")
    tessellate.add_argument("--format", choices=("auto", "step", "iges", "brep"), default="auto", help="CAD input format override.")
    tessellate.add_argument("--occ-target-unit", choices=("auto", "mm", "cm", "m"), default="auto", help="CAD/OCC unit target.")
    tessellate.add_argument("--mesh-size", type=positive_float, help="CAD input only: uniform target mesh size passed to Gmsh.")
    tessellate.add_argument("--mesh-size-min", type=positive_float, help="CAD input only: minimum Gmsh mesh size.")
    tessellate.add_argument("--mesh-size-max", type=positive_float, help="CAD input only: maximum Gmsh mesh size.")
    tessellate.add_argument("--angle-deg", type=positive_float, default=20.0, help="CAD input only: curvature angle control.")
    tessellate.add_argument(
        "--chord",
        type=positive_float,
        help="CAD input only: best-effort OCC chord/deflection hint.",
    )
    labels = tessellate.add_mutually_exclusive_group()
    labels.add_argument("--import-labels", dest="import_labels", action="store_true", default=True)
    labels.add_argument("--no-import-labels", dest="import_labels", action="store_false")
    tessellate.add_argument("--save-debug-msh", action="store_true", help="CAD input only: also save raw Gmsh mesh as surface_mesh.msh.")
    tessellate.set_defaults(func=run_tessellate)

    smoke = sub.add_parser("smoke", help="Generate a synthetic STEP fixture and tessellate it end to end.")
    smoke.add_argument("--output-dir", required=True, help="Directory for the generated fixture and tessellation outputs.")
    smoke.add_argument(
        "--discard-fixture",
        dest="keep_fixture",
        action="store_false",
        default=True,
        help="Remove the generated synthetic STEP fixture after a successful smoke run.",
    )
    smoke.set_defaults(func=run_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # pragma: no cover - keeps CLI failures user-readable
        if exc.__class__.__name__ in {"DependencyError", "TessellationError"}:
            print(str(exc), file=sys.stderr)
            return 2
        raise
