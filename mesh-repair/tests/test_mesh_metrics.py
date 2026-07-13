from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from mesh_metrics import (  # noqa: E402
    mesh_report,
    self_intersection_report,
    silhouette_drift_from_meshes,
)


@dataclass(frozen=True)
class ViewSpec:
    name: str
    project_axes: tuple[int, int]
    depth_vector: tuple[float, float, float]


TOP_VIEW = ViewSpec("plus_z", (0, 1), (0.0, 0.0, -1.0))


def square_mesh(
    x0: float, x1: float, y0: float, y1: float
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [x0, y0, 0.0],
            [x1, y0, 0.0],
            [x1, y1, 0.0],
            [x0, y1, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return points, faces


def grid_square_mesh(divisions: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [column / divisions, row / divisions, 0.0]
            for row in range(divisions + 1)
            for column in range(divisions + 1)
        ],
        dtype=np.float64,
    )
    faces: list[list[int]] = []
    for row in range(divisions):
        for column in range(divisions):
            lower_left = row * (divisions + 1) + column
            lower_right = lower_left + 1
            upper_left = lower_left + divisions + 1
            upper_right = upper_left + 1
            faces.extend(
                [
                    [lower_left, lower_right, upper_right],
                    [lower_left, upper_right, upper_left],
                ]
            )
    return points, np.asarray(faces, dtype=np.int64)


def tetrahedron(offset: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ) + np.asarray(offset, dtype=np.float64)
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        dtype=np.int64,
    )
    return points, faces


class SilhouetteDriftTest(unittest.TestCase):
    def test_identical_meshes_have_zero_silhouette_drift(self) -> None:
        points, faces = square_mesh(0.0, 1.0, 0.0, 1.0)

        drift = silhouette_drift_from_meshes(
            points, faces, points, faces, [TOP_VIEW], max_size=64
        )

        self.assertEqual(drift["summary"]["changed_ratio_max"], 0.0)
        self.assertEqual(drift["summary"]["overlap_ratio_min"], 1.0)

    def test_shifted_mesh_reports_silhouette_drift(self) -> None:
        reference_points, reference_faces = square_mesh(0.0, 1.0, 0.0, 1.0)
        candidate_points, candidate_faces = square_mesh(0.25, 1.25, 0.0, 1.0)

        drift = silhouette_drift_from_meshes(
            reference_points,
            reference_faces,
            candidate_points,
            candidate_faces,
            [TOP_VIEW],
            max_size=64,
        )

        self.assertGreater(drift["summary"]["changed_ratio_max"], 0.0)
        self.assertLess(drift["summary"]["overlap_ratio_min"], 1.0)

    def test_formal_silhouette_is_invariant_to_density_and_face_order(self) -> None:
        coarse_points, coarse_faces = square_mesh(0.0, 1.0, 0.0, 1.0)
        dense_points, dense_faces = grid_square_mesh(8)

        drift = silhouette_drift_from_meshes(
            coarse_points,
            coarse_faces,
            dense_points,
            dense_faces[::-1],
            [TOP_VIEW],
            max_size=64,
        )

        self.assertEqual(
            drift["method"],
            "shared_bbox_six_direction_conservative_solid_triangle_raster",
        )
        self.assertEqual(
            tuple(drift["per_view"]),
            ("minus_x", "plus_x", "minus_y", "plus_y", "minus_z", "plus_z"),
        )
        self.assertEqual(drift["summary"]["changed_ratio_max"], 0.0)
        self.assertEqual(drift["summary"]["overlap_ratio_min"], 1.0)


class MultiComponentVolumeTest(unittest.TestCase):
    def test_two_closed_components_have_reliable_componentwise_volume(self) -> None:
        left_points, left_faces = tetrahedron((0.0, 0.0, 0.0))
        right_points, right_faces = tetrahedron((3.0, 0.0, 0.0))
        points = np.vstack([left_points, right_points])
        faces = np.vstack([left_faces, right_faces + left_points.shape[0]])

        report = mesh_report(points, faces)

        self.assertEqual(report["topology"]["components"]["count"], 2)
        self.assertTrue(report["topology"]["closed_manifold"])
        self.assertTrue(report["volume"]["reliable"])
        self.assertTrue(report["volume"]["orientation_consistent"])
        self.assertAlmostEqual(report["volume"]["signed_abs"], 1.0 / 3.0)

    def test_point_touching_closed_components_have_non_manifold_vertex(self) -> None:
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
        faces = np.asarray(
            [
                [0, 2, 1],
                [0, 1, 3],
                [0, 3, 2],
                [1, 2, 3],
                [0, 4, 5],
                [0, 6, 4],
                [0, 5, 6],
                [4, 6, 5],
            ],
            dtype=np.int64,
        )

        report = mesh_report(points, faces)
        intersections = self_intersection_report(points, faces)

        self.assertEqual(report["topology"]["boundary_edges"], 0)
        self.assertEqual(report["topology"]["non_manifold_edges"], 0)
        self.assertEqual(report["topology"]["non_manifold_vertices"], 1)
        self.assertFalse(report["topology"]["closed_manifold"])
        self.assertFalse(report["volume"]["reliable"])
        self.assertEqual(intersections["intersection_pairs"], 0)

    def test_open_boundary_vertex_links_are_paths(self) -> None:
        points, faces = square_mesh(0.0, 1.0, 0.0, 1.0)

        report = mesh_report(points, faces)

        self.assertEqual(report["topology"]["non_manifold_vertices"], 0)


class SelfIntersectionTest(unittest.TestCase):
    def test_adjacent_triangles_are_not_self_intersections(self) -> None:
        points, faces = square_mesh(0.0, 1.0, 0.0, 1.0)

        report = self_intersection_report(points, faces)

        self.assertTrue(report["passed"])
        self.assertEqual(report["intersection_pairs"], 0)
        self.assertEqual(report["ignored_topological_contacts"], 1)

    def test_shared_vertex_does_not_hide_penetration_away_from_vertex(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.25, 0.25, -1.0],
                [0.25, 0.25, 1.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [0, 3, 4]], dtype=np.int64)

        report = self_intersection_report(points, faces)

        self.assertFalse(report["passed"])
        self.assertEqual(report["intersection_pairs"], 1)
        self.assertEqual(report["candidate_pairs_tested"], 1)
        self.assertEqual(report["reported_pairs"], [[0, 1]])

    def test_shared_edge_does_not_hide_coplanar_area_overlap(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [1, 0, 3]], dtype=np.int64)

        report = self_intersection_report(points, faces)

        self.assertFalse(report["passed"])
        self.assertEqual(report["intersection_pairs"], 1)
        self.assertEqual(report["reported_pairs"], [[0, 1]])

    def test_crossing_non_adjacent_triangles_are_reported(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.2, 0.2, -1.0],
                [0.2, 0.2, 1.0],
                [0.8, 0.2, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)

        report = self_intersection_report(points, faces)

        self.assertFalse(report["passed"])
        self.assertEqual(report["intersection_pairs"], 1)
        self.assertEqual(report["reported_pairs"], [[0, 1]])

    def test_focused_generated_face_is_checked_against_the_complete_mesh(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.2, 0.2, -1.0],
                [0.2, 0.2, 1.0],
                [0.8, 0.2, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
                [3.2, 0.2, -1.0],
                [3.2, 0.2, 1.0],
                [3.8, 0.2, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
            dtype=np.int64,
        )

        report = self_intersection_report(points, faces, focus_face_ids=[3])

        self.assertEqual(report["scope"], "focused_faces")
        self.assertFalse(report["passed"])
        self.assertEqual(report["intersection_pairs"], 1)
        self.assertEqual(report["reported_pairs"], [[2, 3]])

    def test_focused_pairs_are_deduplicated_and_unfocused_intersections_are_ignored(
        self,
    ) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.2, 0.2, -1.0],
                [0.2, 0.2, 1.0],
                [0.8, 0.2, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
                [3.2, 0.2, -1.0],
                [3.2, 0.2, 1.0],
                [3.8, 0.2, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
            dtype=np.int64,
        )

        report = self_intersection_report(points, faces, focus_face_ids=[2, 3])

        self.assertEqual(report["intersection_pairs"], 1)
        self.assertEqual(report["reported_pairs"], [[2, 3]])
        self.assertNotIn([0, 1], report["reported_pairs"])


if __name__ == "__main__":
    unittest.main()
