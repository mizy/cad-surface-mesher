from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .gmsh_pipeline import TessellationControls, TessellationError, load_runtime, tessellate_cad


def run_smoke_test(output_dir: Path, *, keep_fixture: bool = True) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    fixture_path = output_dir / "fixtures" / "synthetic_two_body.step"
    make_synthetic_step_fixture(fixture_path)

    result = tessellate_cad(
        TessellationControls(
            input_path=fixture_path,
            output_dir=output_dir / "tessellated",
            cad_format="step",
            occ_target_unit="m",
            mesh_size=0.18,
            angle_deg=18.0,
            chord=0.02,
            import_labels=True,
        )
    )
    validate_smoke_report(result.report)
    if not keep_fixture and fixture_path.exists():
        fixture_path.unlink()
    return result.report


def make_synthetic_step_fixture(path: Path) -> None:
    runtime = load_runtime()
    gmsh = runtime.gmsh
    path.parent.mkdir(parents=True, exist_ok=True)
    initialized = False
    try:
        gmsh.initialize()
        initialized = True
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("synthetic_two_body")
        box = gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1000.0, 550.0, 350.0)
        cylinder = gmsh.model.occ.addCylinder(1450.0, 0.0, 0.0, 0.0, 550.0, 0.0, 220.0)
        gmsh.model.occ.synchronize()
        gmsh.model.setEntityName(3, box, "synthetic_box_body")
        gmsh.model.setEntityName(3, cylinder, "synthetic_cylinder_body")
        for dim, tag in gmsh.model.getEntities(2):
            gmsh.model.setEntityName(dim, tag, f"synthetic_surface_{tag}")
        with suppress_native_stdout():
            gmsh.write(str(path))
    finally:
        if initialized:
            gmsh.finalize()


@contextmanager
def suppress_native_stdout() -> Any:
    saved_stdout = os.dup(1)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            yield
    finally:
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)


def validate_smoke_report(report: dict[str, Any]) -> None:
    gates = report.get("gates", {})
    if not gates.get("non_empty"):
        raise TessellationError("Smoke tessellation produced an empty mesh.")
    if not gates.get("all_triangles"):
        raise TessellationError("Smoke tessellation did not produce a triangle-only mesh.")
    if not Path(report["outputs"]["surface_mesh_vtp"]).exists():
        raise TessellationError("Smoke tessellation did not write surface_mesh.vtp.")
    if not Path(report["outputs"]["report"]).exists():
        raise TessellationError("Smoke tessellation did not write tessellation_report.json.")
