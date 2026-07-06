from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cad_tessellate.py"


def runtime_available() -> tuple[bool, str]:
    failures = []
    for name in ("gmsh", "numpy", "pyvista", "vtk"):
        try:
            importlib.import_module(name)
        except (ImportError, OSError) as exc:
            failures.append(f"{name}: {exc}")
    return not failures, "; ".join(failures)


def mesh_runtime_available() -> tuple[bool, str]:
    failures = []
    for name in ("numpy", "pyvista", "vtk"):
        try:
            importlib.import_module(name)
        except (ImportError, OSError) as exc:
            failures.append(f"{name}: {exc}")
    return not failures, "; ".join(failures)


class CadTessellationSmokeTest(unittest.TestCase):
    def test_help_without_runtime_imports(self) -> None:
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tessellate", result.stdout)
        self.assertIn("smoke", result.stdout)

    def test_generated_step_smoke(self) -> None:
        available, reason = runtime_available()
        if not available:
            self.skipTest(f"CAD tessellation runtime dependencies unavailable: {reason}")
        with tempfile.TemporaryDirectory(prefix="cad-tessellation-smoke-") as tmp:
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "smoke", "--output-dir", tmp],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report_path = Path(tmp) / "tessellated" / "tessellation_report.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["gates"]["non_empty"])
            self.assertTrue(report["gates"]["all_triangles"])
            self.assertTrue(Path(report["outputs"]["surface_mesh_vtp"]).exists())
            self.assertIn("gmsh_surface_tag", report["provenance"]["cell_arrays"])

    def test_vtp_mesh_input_smoke(self) -> None:
        available, reason = mesh_runtime_available()
        if not available:
            self.skipTest(f"Mesh surface runtime dependencies unavailable: {reason}")
        import numpy as np
        import pyvista as pv

        with tempfile.TemporaryDirectory(prefix="mesh-tessellation-smoke-") as tmp:
            root = Path(tmp)
            mesh_path = root / "quad.vtp"
            points = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            )
            pv.PolyData(points, np.array([4, 0, 1, 2, 3])).save(mesh_path)

            result = subprocess.run(
                [sys.executable, str(SCRIPT), "tessellate", str(mesh_path), "--output-dir", str(root / "out")],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report_path = root / "out" / "tessellation_report.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["input"]["kind"], "mesh")
            self.assertEqual(report["mesh_metrics"]["triangles"], 2)
            self.assertTrue(report["gates"]["all_triangles"])
            self.assertIn("source_triangle_index", report["provenance"]["cell_arrays"])


if __name__ == "__main__":
    unittest.main()
