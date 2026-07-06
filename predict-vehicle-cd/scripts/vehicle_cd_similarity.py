#!/usr/bin/env python3
from __future__ import annotations

try:
    from vehicle_cd_similarity_lib.cli import main
except ImportError as exc:  # pragma: no cover - exercised by users without deps
    raise SystemExit(
        "Missing dependency. Install numpy, pillow, pyvista, vtk, and optional trimesh, then retry. "
        f"Original import error: {exc}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
