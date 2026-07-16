from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from mesh_metrics import mesh_report, silhouette_drift_from_meshes  # noqa: E402


@dataclass(frozen=True)
class ViewSpec:
    name: str
    project_axes: tuple[int, int]
    depth_vector: tuple[float, float, float]


TOP_VIEW = ViewSpec("plus_z", (0, 1), (0.0, 0.0, -1.0))


def square_mesh(offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [offset, 0.0, 0.0],
            [offset + 1.0, 0.0, 0.0],
            [offset + 1.0, 1.0, 0.0],
            [offset, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return points, np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


def tetrahedron(offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [offset, 0.0, 0.0],
            [offset + 1.0, 0.0, 0.0],
            [offset, 1.0, 0.0],
            [offset, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        dtype=np.int64,
    )
    return points, faces


class MeshMetricsTest(unittest.TestCase):
    def test_identical_meshes_have_zero_silhouette_drift(self) -> None:
        points, faces = square_mesh()
        report = silhouette_drift_from_meshes(
            points, faces, points, faces, [TOP_VIEW], max_size=128
        )
        self.assertEqual(report["summary"]["changed_ratio_max"], 0.0)
        self.assertEqual(report["summary"]["overlap_ratio_min"], 1.0)

    def test_shifted_mesh_reports_silhouette_drift(self) -> None:
        source_points, faces = square_mesh()
        shifted_points, _ = square_mesh(0.25)
        report = silhouette_drift_from_meshes(
            source_points, faces, shifted_points, faces, [TOP_VIEW], max_size=128
        )
        self.assertGreater(report["summary"]["changed_ratio_max"], 0.0)

    def test_two_closed_components_have_reliable_componentwise_volume(self) -> None:
        left_points, left_faces = tetrahedron()
        right_points, right_faces = tetrahedron(2.0)
        points = np.vstack((left_points, right_points))
        faces = np.vstack((left_faces, right_faces + left_points.shape[0]))
        report = mesh_report(points, faces)
        self.assertEqual(report["topology"]["components"]["count"], 2)
        self.assertTrue(report["volume"]["reliable"])
        self.assertTrue(report["volume"]["orientation_consistent"])

    def test_point_touching_closed_components_have_non_manifold_vertex(self) -> None:
        left_points, left_faces = tetrahedron()
        right_points, right_faces = tetrahedron()
        right_points = right_points + np.asarray([0.0, 0.0, -1.0])
        right_points[3] = left_points[0]
        points = np.vstack((left_points, right_points[1:]))
        remap = np.asarray([0, 4, 5, 6], dtype=np.int64)
        faces = np.vstack((left_faces, remap[right_faces]))
        report = mesh_report(points, faces)
        self.assertEqual(report["topology"]["non_manifold_vertices"], 1)
        self.assertFalse(report["topology"]["closed_manifold"])
        self.assertFalse(report["volume"]["reliable"])

    def test_open_boundary_vertex_links_are_paths(self) -> None:
        points, faces = square_mesh()
        report = mesh_report(points, faces)
        self.assertEqual(report["topology"]["boundary_edges"], 4)
        self.assertEqual(report["topology"]["non_manifold_vertices"], 0)


if __name__ == "__main__":
    unittest.main()
