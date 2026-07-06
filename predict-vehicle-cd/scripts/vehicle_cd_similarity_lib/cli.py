from __future__ import annotations

import argparse

from .constants import DEFAULT_HEIGHT, DEFAULT_TARGET_LENGTH_M, DEFAULT_WIDTH, SUPPORTED_FORMATS_TEXT


def parse_target_dimensions(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected L,W,H in meters, for example 4.8,1.875,1.46")
    try:
        dimensions = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Target dimensions must be numeric meters") from exc
    if any(dimension <= 0 for dimension in dimensions):
        raise argparse.ArgumentTypeError("Target dimensions must be positive")
    return dimensions  # type: ignore[return-value]


def run_build_library(args: argparse.Namespace) -> int:
    from .commands import build_library

    return build_library(args)


def run_predict(args: argparse.Namespace) -> int:
    from .commands import predict

    return predict(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Estimate vehicle Cd from side-view depth and normal similarity. Supported inputs: {SUPPORTED_FORMATS_TEXT}."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-library", help="Generate reference side-view artifacts from a manifest.")
    build.add_argument("manifest", help="JSON manifest containing reference cars and mesh paths.")
    build.add_argument("--output-dir", required=True, help="Directory for reference_cars.json and PNG artifacts.")
    build.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    build.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    build.add_argument("--target-length", type=float, default=DEFAULT_TARGET_LENGTH_M)
    build.add_argument(
        "--force-target-length",
        action="store_true",
        help="Uniformly scale every mesh to --target-length even if its size is plausible.",
    )
    build.add_argument(
        "--target-dimensions",
        type=parse_target_dimensions,
        help="Force dimensions to L,W,H meters, for example 4.8,1.875,1.46.",
    )
    build.set_defaults(func=run_build_library)

    pred = sub.add_parser("predict", help="Predict Cd for one 3D mesh.")
    pred.add_argument("mesh", help=f"Path to a 3D vehicle mesh ({SUPPORTED_FORMATS_TEXT}).")
    pred.add_argument("--library", required=True, help="Path to reference_cars.json.")
    pred.add_argument("--output-dir", help="Optional directory for query artifacts and prediction.json.")
    pred.add_argument("--top-k", type=int, default=3)
    pred.add_argument("--category", help="Optional query vehicle category, for example sedan, suv, pickup, van, or mpv.")
    pred.add_argument("--temperature", type=float, default=0.08)
    pred.add_argument("--epsilon", type=float, default=1e-6)
    pred.add_argument("--target-length", type=float, default=DEFAULT_TARGET_LENGTH_M)
    pred.add_argument(
        "--force-target-length",
        action="store_true",
        help="Uniformly scale the mesh to --target-length even if its size is plausible.",
    )
    pred.add_argument(
        "--target-dimensions",
        type=parse_target_dimensions,
        help="Force dimensions to L,W,H meters, for example 4.8,1.875,1.46.",
    )
    pred.add_argument("--no-normalized-mesh", action="store_true", help="Do not write query_normalized.vtp when --output-dir is set.")
    pred.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    pred.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    pred.set_defaults(func=run_predict)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
