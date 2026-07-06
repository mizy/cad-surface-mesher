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
    from .gmsh_pipeline import TessellationControls, tessellate_cad

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
    result = tessellate_cad(controls)
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
            "Convert STEP, IGES, or BREP CAD files into a triangle-only VTP surface mesh "
            "with a JSON tessellation quality report."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tessellate = sub.add_parser("tessellate", help="Tessellate one CAD file into surface_mesh.vtp.")
    tessellate.add_argument("input", help="Input CAD file (.step, .stp, .iges, .igs, .brep, .brp).")
    tessellate.add_argument("--output-dir", required=True, help="Directory for surface_mesh.vtp and tessellation_report.json.")
    tessellate.add_argument("--format", choices=("auto", "step", "iges", "brep"), default="auto")
    tessellate.add_argument("--occ-target-unit", choices=("auto", "mm", "cm", "m"), default="auto")
    tessellate.add_argument("--mesh-size", type=positive_float, help="Uniform target mesh size passed to Gmsh.")
    tessellate.add_argument("--mesh-size-min", type=positive_float, help="Minimum Gmsh mesh size.")
    tessellate.add_argument("--mesh-size-max", type=positive_float, help="Maximum Gmsh mesh size.")
    tessellate.add_argument("--angle-deg", type=positive_float, default=20.0, help="Curvature angle control in degrees.")
    tessellate.add_argument(
        "--chord",
        type=positive_float,
        help="Best-effort OCC chord/deflection hint. The report records its exact limitation.",
    )
    labels = tessellate.add_mutually_exclusive_group()
    labels.add_argument("--import-labels", dest="import_labels", action="store_true", default=True)
    labels.add_argument("--no-import-labels", dest="import_labels", action="store_false")
    tessellate.add_argument("--save-debug-msh", action="store_true", help="Also save the raw Gmsh mesh as surface_mesh.msh.")
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
