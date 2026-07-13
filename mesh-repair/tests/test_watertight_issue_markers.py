from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from mesh_io import write_vtp  # noqa: E402
from watertight_issue_markers import (  # noqa: E402
    ISSUE_INCONSISTENT_WINDING,
    ISSUE_NON_MANIFOLD,
    analyze_geometry,
    generate_issue_report,
    write_line_marker_vtp,
    write_point_marker_vtp,
)


CUBE_POINTS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)
CUBE_FACES = np.asarray(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ],
    dtype=np.int64,
)


class WatertightIssueMarkerTest(unittest.TestCase):
    def test_report_writes_full_surface_and_exact_line_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            original_path = root / "original.vtp"
            processed_path = root / "processed.vtp"
            output_dir = root / "report"
            open_faces = CUBE_FACES[2:]
            write_vtp(
                original_path,
                CUBE_POINTS,
                open_faces,
                {"source_face_id": np.arange(open_faces.shape[0], dtype=np.int64)},
            )
            write_vtp(
                processed_path,
                CUBE_POINTS,
                CUBE_FACES,
                {"source_face_id": np.arange(CUBE_FACES.shape[0], dtype=np.int64)},
            )

            report = generate_issue_report(original_path, processed_path, output_dir)

            self.assertEqual(report["original"]["summary"]["boundary_edges"], 4)
            self.assertEqual(report["original"]["summary"]["boundary_regions"], 1)
            self.assertFalse(report["original"]["summary"]["topology_watertight"])
            self.assertEqual(report["processed"]["summary"]["boundary_edges"], 0)
            self.assertTrue(report["processed"]["summary"]["engineering_watertight"])
            self.assertEqual(report["comparison"]["metrics"]["boundary_edges"]["reduction"], 4)

            original_surface = pv.read(report["original"]["artifacts"]["annotated_surface_vtp"])
            self.assertEqual(original_surface.n_cells, open_faces.shape[0])
            self.assertIn("source_face_id", original_surface.cell_data)
            self.assertIn("watertight_audit_issue_mask", original_surface.cell_data)
            self.assertIn("watertight_audit_component_id", original_surface.cell_data)
            self.assertIn(
                "watertight_audit_non_manifold_vertex", original_surface.point_data
            )

            boundary_marker = pv.read(report["original"]["artifacts"]["boundary_edges_vtp"])
            self.assertEqual(boundary_marker.n_lines, 4)
            self.assertEqual(boundary_marker.n_cells, 4)
            np.testing.assert_array_equal(
                boundary_marker.cell_data["watertight_audit_edge_incidence"],
                np.ones(4, dtype=np.int32),
            )
            np.testing.assert_array_equal(
                np.unique(boundary_marker.cell_data["watertight_audit_issue_region_id"]),
                np.asarray([0]),
            )

            processed_marker = pv.read(report["processed"]["artifacts"]["all_issue_edges_vtp"])
            self.assertEqual(processed_marker.n_lines, 0)
            self.assertIn("watertight_audit_issue_type", processed_marker.cell_data)

            issue_faces = pv.read(report["original"]["artifacts"]["issue_faces_vtp"])
            self.assertEqual(issue_faces.n_cells, 4)
            self.assertLess(issue_faces.n_points, original_surface.n_points + 1)

            persisted = json.loads((output_dir / "watertight_issue_report.json").read_text())
            self.assertEqual(persisted["schema_version"], 1)
            self.assertEqual(persisted["checks"]["self_intersections"], "not_checked")
            self.assertEqual(
                persisted["array_contract"]["edge_cell_issue_type"],
                "watertight_audit_issue_type",
            )

    def test_vertex_only_non_manifold_contact_is_counted_and_marked(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        tetra = np.asarray(
            [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
            dtype=np.int64,
        )
        second = np.asarray(
            [[0, 4, 5], [0, 6, 4], [4, 6, 5], [5, 6, 0]],
            dtype=np.int64,
        )
        analysis = analyze_geometry(points, np.vstack((tetra, second)))

        self.assertEqual(analysis["summary"]["boundary_edges"], 0)
        self.assertEqual(analysis["summary"]["non_manifold_edges"], 0)
        self.assertEqual(analysis["summary"]["non_manifold_vertices"], 1)
        self.assertFalse(analysis["summary"]["topology_watertight"])
        np.testing.assert_array_equal(
            analysis["non_manifold_vertex_ids"], np.asarray([0], dtype=np.int64)
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "non_manifold_vertices.vtp"
            write_point_marker_vtp(
                path,
                points,
                analysis["non_manifold_vertex_ids"],
            )
            marker = pv.read(path)

        self.assertEqual(marker.n_verts, 1)
        self.assertEqual(
            int(marker.point_data["watertight_audit_source_point_id"][0]), 0
        )

    def test_non_manifold_edge_marker_keeps_incidence_and_source_ids(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)
        analysis = analyze_geometry(points, faces)
        edge = (0, 1)

        self.assertEqual(analysis["edges"][ISSUE_NON_MANIFOLD], [edge])
        self.assertEqual(analysis["summary"]["non_manifold_edges"], 1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "non_manifold.vtp"
            write_line_marker_vtp(
                path,
                points,
                analysis["edge_faces"],
                [(ISSUE_NON_MANIFOLD, edge)],
                analysis["edge_regions"],
            )
            marker = pv.read(path)

        self.assertEqual(marker.n_lines, 1)
        self.assertEqual(int(marker.cell_data["watertight_audit_edge_incidence"][0]), 3)
        self.assertEqual(int(marker.cell_data["watertight_audit_source_point_id_0"][0]), 0)
        self.assertEqual(int(marker.cell_data["watertight_audit_source_point_id_1"][0]), 1)

    def test_same_direction_shared_edge_is_marked_as_winding_issue(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [0, 1, 3]], dtype=np.int64)

        analysis = analyze_geometry(points, faces)

        self.assertEqual(analysis["edges"][ISSUE_INCONSISTENT_WINDING], [(0, 1)])
        self.assertEqual(analysis["summary"]["inconsistent_winding_edges"], 1)
        self.assertFalse(analysis["summary"]["engineering_watertight"])


if __name__ == "__main__":
    unittest.main()
