from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from deterministic_hole_fill import (  # noqa: E402
    run_deterministic_hole_fill,
    triangulate_fixed_boundary_loop,
    validate_local_patch_intersections,
)


def warped_square_annulus() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inner = np.asarray(
        [
            [-1.0, -1.0, 0.00],
            [1.0, -1.0, 0.18],
            [1.0, 1.0, -0.08],
            [-1.0, 1.0, 0.12],
        ],
        dtype=np.float64,
    )
    outer = np.asarray(
        [
            [-2.0, -2.0, 0.00],
            [2.0, -2.0, 0.20],
            [2.0, 2.0, -0.05],
            [-2.0, 2.0, 0.10],
        ],
        dtype=np.float64,
    )
    points = np.vstack([outer, inner])
    faces = []
    for index in range(4):
        following = (index + 1) % 4
        faces.extend(
            [
                [index, following, 4 + following],
                [index, 4 + following, 4 + index],
            ]
        )
    return points, np.asarray(faces, dtype=np.int64), np.arange(4, 8, dtype=np.int64)


def planar_square_annulus_with_crossing_triangle() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, faces, loop = warped_square_annulus()
    points[:, 2] = 0.0
    crossing = np.asarray(
        [
            [0.0, -0.7, -1.0],
            [0.0, 0.7, 1.0],
            [0.6, 0.0, 0.5],
        ],
        dtype=np.float64,
    )
    points = np.vstack([points, crossing])
    faces = np.vstack([faces, [[8, 9, 10]]])
    return points, faces, loop


def warped_round_annulus(vertex_count: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, vertex_count, endpoint=False)
    inner = np.column_stack(
        [
            np.cos(angles),
            np.sin(angles),
            0.25 * np.sin(3.0 * angles) + 0.05 * np.cos(2.0 * angles),
        ]
    )
    outer = np.column_stack(
        [
            2.0 * np.cos(angles),
            2.0 * np.sin(angles),
            0.12 * np.sin(3.0 * angles),
        ]
    )
    points = np.vstack([outer, inner])
    faces = []
    for index in range(vertex_count):
        following = (index + 1) % vertex_count
        faces.extend(
            [
                [index, following, vertex_count + following],
                [index, vertex_count + following, vertex_count + index],
            ]
        )
    loop = np.arange(vertex_count, 2 * vertex_count, dtype=np.int64)
    return points, np.asarray(faces, dtype=np.int64), loop


def directed_edge_occurrences(faces: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    result: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face in faces:
        for index, left_value in enumerate(face):
            left = int(left_value)
            right = int(face[(index + 1) % 3])
            result.setdefault(tuple(sorted((left, right))), []).append((left, right))
    return result


class FixedBoundaryTriangulationTest(unittest.TestCase):
    def test_warped_loop_is_deterministic_and_reuses_only_source_boundary_vertices(self) -> None:
        points, faces, loop = warped_square_annulus()

        first = triangulate_fixed_boundary_loop(points, faces, loop[::-1])
        second = triangulate_fixed_boundary_loop(points, faces, loop)

        self.assertTrue(first["success"], first["diagnostics"])
        self.assertTrue(second["success"], second["diagnostics"])
        np.testing.assert_array_equal(first["generated_faces"], second["generated_faces"])
        self.assertEqual(first["generated_faces"].shape, (2, 3))
        self.assertEqual(set(first["generated_faces"].ravel()), set(loop))
        self.assertGreater(
            first["diagnostics"]["boundary"]["max_plane_deviation_ratio"],
            0.0,
        )
        self.assertTrue(first["diagnostics"]["source_points_modified"] is False)
        self.assertEqual(first["diagnostics"]["new_points_added"], 0)

    def test_multi_vertex_curved_loop_commits_without_flattening_boundary(self) -> None:
        points, faces, loop = warped_round_annulus()
        original_points = points.copy()

        result = run_deterministic_hole_fill(points, faces, loop)

        self.assertTrue(result["committed"], result["diagnostics"])
        self.assertEqual(result["generated_faces"].shape, (loop.size - 2, 3))
        np.testing.assert_array_equal(result["points"], original_points)
        self.assertGreater(result["diagnostics"]["boundary"]["max_plane_deviation_ratio"], 0.10)
        self.assertTrue(result["diagnostics"]["topology"]["passed"])

    def test_invalid_non_boundary_loop_fails_without_faces(self) -> None:
        points, faces, _ = warped_square_annulus()

        result = triangulate_fixed_boundary_loop(points, faces, [0, 1, 2])

        self.assertFalse(result["success"])
        self.assertEqual(result["generated_faces"].shape, (0, 3))
        self.assertEqual(result["failure_reason_codes"], ["loop_edge_is_not_boundary"])


class LocalIntersectionPolicyTest(unittest.TestCase):
    def test_shared_edges_vertices_and_patch_internal_edge_are_allowed(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, -1.0, 0.0],
                [2.0, 0.5, 0.0],
                [0.5, 2.0, 0.0],
                [-1.0, 0.5, 0.0],
            ],
            dtype=np.float64,
        )
        patch_faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        source_faces = np.asarray(
            [[1, 0, 4], [2, 1, 5], [3, 2, 6], [0, 3, 7]],
            dtype=np.int64,
        )

        report = validate_local_patch_intersections(points, source_faces, patch_faces)

        self.assertTrue(report["passed"], report)
        self.assertEqual(report["intersection_pairs"], 0)
        self.assertGreaterEqual(report["ignored_topological_contacts"], 5)

    def test_duplicate_vertex_boundary_touch_is_allowed_without_shared_ids(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, -1.0, 0.0],
            ],
            dtype=np.float64,
        )

        report = validate_local_patch_intersections(
            points,
            np.asarray([[3, 4, 5]], dtype=np.int64),
            np.asarray([[0, 1, 2]], dtype=np.int64),
        )

        self.assertTrue(report["passed"], report)
        self.assertEqual(report["ignored_boundary_contacts"], 1)

    def test_non_coplanar_interior_penetration_is_rejected(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.25, 0.25, -1.0],
                [0.25, 0.25, 1.0],
                [0.75, 0.25, 0.0],
            ],
            dtype=np.float64,
        )

        report = validate_local_patch_intersections(
            points,
            np.asarray([[3, 4, 5]], dtype=np.int64),
            np.asarray([[0, 1, 2]], dtype=np.int64),
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["intersection_pairs"], 1)
        self.assertEqual(report["reported_pairs"][0]["contact"], "non_coplanar_penetration")

    def test_shared_vertex_does_not_hide_penetration_away_from_that_vertex(self) -> None:
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

        report = validate_local_patch_intersections(
            points,
            np.asarray([[0, 3, 4]], dtype=np.int64),
            np.asarray([[0, 1, 2]], dtype=np.int64),
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["intersection_pairs"], 1)

    def test_coplanar_area_overlap_is_rejected(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.1, 0.0],
                [0.8, 0.1, 0.0],
                [0.1, 0.8, 0.0],
            ],
            dtype=np.float64,
        )

        report = validate_local_patch_intersections(
            points,
            np.asarray([[3, 4, 5]], dtype=np.int64),
            np.asarray([[0, 1, 2]], dtype=np.int64),
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["reported_pairs"][0]["contact"], "coplanar_area_overlap")


class DeterministicHoleFillTransactionTest(unittest.TestCase):
    def test_commit_preserves_points_orients_seam_and_appends_provenance(self) -> None:
        points, faces, loop = warped_square_annulus()
        points = points.astype(np.float32)
        faces = faces.astype(np.int32)
        provenance = {
            "face_origin": np.zeros(faces.shape[0], dtype=np.int16),
            "source_triangle_index": np.arange(100, 100 + faces.shape[0], dtype=np.int64),
            "fusion_region_id": np.zeros(faces.shape[0], dtype=np.int32),
            "proxy_weight": np.zeros(faces.shape[0], dtype=np.float32),
        }
        original_points = points.copy()
        original_faces = faces.copy()

        result = run_deterministic_hole_fill(
            points,
            faces,
            loop[::-1],
            provenance,
            fusion_region_id=17,
        )

        self.assertTrue(result["committed"], result["diagnostics"])
        np.testing.assert_array_equal(points, original_points)
        np.testing.assert_array_equal(faces, original_faces)
        np.testing.assert_array_equal(result["points"], original_points)
        self.assertEqual(result["points"].dtype, original_points.dtype)
        self.assertEqual(result["generated_faces"].shape, (2, 3))
        self.assertEqual(result["faces"].shape[0], faces.shape[0] + 2)
        self.assertTrue(result["local_intersection"]["passed"])

        occurrences = directed_edge_occurrences(result["faces"])
        oriented_loop = result["oriented_boundary_loop"]
        for index, left_value in enumerate(oriented_loop):
            edge = tuple(
                sorted((int(left_value), int(oriented_loop[(index + 1) % oriented_loop.size])))
            )
            self.assertEqual(len(occurrences[edge]), 2)
            self.assertEqual(occurrences[edge][0], occurrences[edge][1][::-1])

        generated_provenance = result["generated_provenance"]
        np.testing.assert_array_equal(generated_provenance["face_origin"], [3, 3])
        np.testing.assert_array_equal(generated_provenance["source_triangle_index"], [-1, -1])
        np.testing.assert_array_equal(generated_provenance["fusion_region_id"], [17, 17])
        np.testing.assert_array_equal(generated_provenance["proxy_weight"], [0.0, 0.0])
        self.assertTrue(result["diagnostics"]["topology"]["passed"])

    def test_real_patch_penetration_rolls_back_geometry_and_provenance(self) -> None:
        points, faces, loop = planar_square_annulus_with_crossing_triangle()
        points = points.astype(np.float32)
        faces = faces.astype(np.int32)
        provenance = {
            "face_origin": np.zeros(faces.shape[0], dtype=np.int16),
            "source_triangle_index": np.arange(faces.shape[0], dtype=np.int64),
            "fusion_region_id": np.zeros(faces.shape[0], dtype=np.int32),
        }
        original_points = points.copy()
        original_faces = faces.copy()
        original_provenance = {name: values.copy() for name, values in provenance.items()}

        result = run_deterministic_hole_fill(
            points,
            faces,
            loop,
            provenance,
            fusion_region_id=23,
        )

        self.assertFalse(result["committed"])
        self.assertEqual(result["status"], "rolled_back")
        self.assertEqual(result["failure_reason_codes"], ["local_self_intersection_detected"])
        self.assertEqual(result["generated_faces"].shape, (0, 3))
        self.assertGreater(result["candidate"]["generated_faces"].shape[0], 0)
        self.assertFalse(result["local_intersection"]["passed"])
        np.testing.assert_array_equal(result["points"], original_points)
        np.testing.assert_array_equal(result["faces"], original_faces)
        self.assertEqual(result["points"].dtype, original_points.dtype)
        self.assertEqual(result["faces"].dtype, original_faces.dtype)
        np.testing.assert_array_equal(points, original_points)
        np.testing.assert_array_equal(faces, original_faces)
        for name, values in original_provenance.items():
            np.testing.assert_array_equal(result["provenance"][name], values)
            np.testing.assert_array_equal(provenance[name], values)

    def test_invalid_provenance_rolls_back_before_geometry_work(self) -> None:
        points, faces, loop = warped_square_annulus()
        provenance = {"source_triangle_index": np.arange(faces.shape[0] - 1)}

        result = run_deterministic_hole_fill(points, faces, loop, provenance)

        self.assertFalse(result["committed"])
        self.assertEqual(result["failure_reason_codes"], ["invalid_source_provenance"])
        self.assertEqual(result["candidate"]["generated_faces"].shape, (0, 3))

    def test_invalid_tolerance_fails_closed_without_mutating_source(self) -> None:
        points, faces, loop = warped_square_annulus()
        original_points = points.copy()
        original_faces = faces.copy()

        result = run_deterministic_hole_fill(
            points,
            faces,
            loop,
            geometric_tolerance=0.0,
        )

        self.assertFalse(result["committed"])
        self.assertEqual(result["failure_reason_codes"], ["invalid_geometric_tolerance"])
        np.testing.assert_array_equal(result["points"], original_points)
        np.testing.assert_array_equal(result["faces"], original_faces)


if __name__ == "__main__":
    unittest.main()
