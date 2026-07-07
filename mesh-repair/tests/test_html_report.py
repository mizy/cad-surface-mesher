from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyvista as pv


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import html_report  # noqa: E402


def write_grid_mesh(path: Path, cells_per_side: int = 8) -> int:
    points = []
    for y in range(cells_per_side + 1):
        for x in range(cells_per_side + 1):
            points.append((float(x), float(y), 0.0))

    faces = []
    row = cells_per_side + 1
    for y in range(cells_per_side):
        for x in range(cells_per_side):
            a = y * row + x
            b = a + 1
            c = a + row
            d = c + 1
            faces.extend([3, a, b, d, 3, a, d, c])

    mesh = pv.PolyData(np.asarray(points, dtype=np.float64), np.asarray(faces, dtype=np.int64))
    mesh.save(path)
    return cells_per_side * cells_per_side * 2


class HtmlReportMeshPreviewTest(unittest.TestCase):
    def test_mesh_preview_defaults_to_full_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "source.vtp"
            original_triangles = write_grid_mesh(path)
            mesh = pv.read(path)

            with (
                mock.patch.dict(os.environ, {"CAD_SURFACE_MESHER_VIEWER_TRIANGLES": ""}),
                mock.patch.object(html_report, "read_surface", return_value=mesh),
            ):
                preview = html_report.mesh_preview("dirty input", path)

        self.assertEqual(preview["triangles"], original_triangles)
        self.assertEqual(preview["original_triangles"], original_triangles)
        self.assertTrue(preview["full_resolution"])
        self.assertFalse(preview["downsampled"])
        self.assertIsNone(preview["viewer_triangle_limit"])

    def test_mesh_preview_downsamples_only_when_limit_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "source.vtp"
            original_triangles = write_grid_mesh(path)
            mesh = pv.read(path)

            with (
                mock.patch.dict(os.environ, {"CAD_SURFACE_MESHER_VIEWER_TRIANGLES": "24"}),
                mock.patch.object(html_report, "read_surface", return_value=mesh),
            ):
                preview = html_report.mesh_preview("dirty input", path)

        self.assertLess(preview["triangles"], original_triangles)
        self.assertEqual(preview["original_triangles"], original_triangles)
        self.assertFalse(preview["full_resolution"])
        self.assertTrue(preview["downsampled"])
        self.assertEqual(preview["viewer_triangle_limit"], 24)


if __name__ == "__main__":
    unittest.main()
