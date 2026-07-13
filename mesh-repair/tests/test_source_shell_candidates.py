from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from source_shell_candidates import (  # noqa: E402
    describe_components,
    face_component_labels,
    merge_kept_components,
    remove_low_risk_faces,
)


class SourceShellCandidatesTest(unittest.TestCase):
    def test_face_components_require_a_shared_edge(self) -> None:
        faces = np.asarray(
            [
                [0, 1, 2],
                [2, 1, 3],
                [2, 4, 5],
                [6, 7, 8],
                [8, 7, 9],
            ],
            dtype=np.int64,
        )

        labels = face_component_labels(faces)

        np.testing.assert_array_equal(labels, [0, 0, 1, 2, 2])

    def test_component_descriptions_include_faces_geometry_and_sources(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [10.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [10.0, 2.0, 0.0],
            ]
        )
        faces = np.asarray([[0, 1, 2], [2, 1, 3], [4, 5, 6]], dtype=np.int64)
        sources = np.asarray([101, 102, 900], dtype=np.int64)

        descriptions = describe_components(points, faces, sources)

        self.assertEqual(len(descriptions), 2)
        self.assertEqual(descriptions[0]["component_id"], 0)
        self.assertEqual(descriptions[0]["face_ids"], [0, 1])
        self.assertEqual(descriptions[0]["face_count"], 2)
        self.assertAlmostEqual(descriptions[0]["surface_area"], 1.0)
        self.assertEqual(descriptions[0]["bbox"]["min"], [0.0, 0.0, 0.0])
        self.assertEqual(descriptions[0]["bbox"]["max"], [1.0, 1.0, 0.0])
        self.assertEqual(descriptions[0]["source_triangle_ids"], [101, 102])
        self.assertEqual(descriptions[1]["face_ids"], [2])
        self.assertAlmostEqual(descriptions[1]["surface_area"], 2.0)

    def test_cleanup_removes_exact_degenerate_and_coordinate_duplicates(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray(
            [
                [0, 1, 2],
                [0, 1, 2],
                [5, 4, 3],
                [0, 0, 1],
                [0, 6, 7],
            ],
            dtype=np.int64,
        )
        sources = np.asarray([10, 11, 12, 13, 14], dtype=np.int64)

        clean_points, clean_faces, clean_sources, report = remove_low_risk_faces(
            points, faces, sources
        )

        np.testing.assert_array_equal(clean_points, points)
        np.testing.assert_array_equal(clean_faces, [[0, 1, 2]])
        np.testing.assert_array_equal(clean_sources, [10])
        self.assertEqual(report["removed_degenerate_face_ids"], [3, 4])
        self.assertEqual(report["removed_duplicate_face_ids"], [1, 2])
        self.assertEqual(report["removed_degenerate_source_triangle_ids"], [13, 14])
        self.assertEqual(report["removed_duplicate_source_triangle_ids"], [11, 12])

    def test_cleanup_keeps_near_degenerate_source_geometry(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1e-18, 0.0]]
        )
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        sources = np.asarray([77], dtype=np.int64)

        _, clean_faces, clean_sources, report = remove_low_risk_faces(
            points, faces, sources
        )

        np.testing.assert_array_equal(clean_faces, faces)
        np.testing.assert_array_equal(clean_sources, sources)
        self.assertEqual(report["removed_face_count"], 0)

    def test_merge_builds_one_mesh_without_coordinate_welding(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [5.0, 1.0, 0.0],
            ]
        )
        faces = np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)
        sources = np.asarray([20, 21, 22], dtype=np.int64)
        labels = face_component_labels(faces)

        merged_points, merged_faces, merged_sources, source_points = (
            merge_kept_components(points, faces, sources, labels, [0, 1])
        )

        np.testing.assert_array_equal(source_points, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(merged_faces, [[0, 1, 2], [3, 4, 5]])
        np.testing.assert_array_equal(merged_sources, [20, 21])
        self.assertEqual(merged_points.shape, (6, 3))
        self.assertEqual(np.count_nonzero(np.all(merged_points == [0.0, 0.0, 0.0], axis=1)), 2)
        self.assertEqual(np.count_nonzero(np.all(merged_points == [1.0, 0.0, 0.0], axis=1)), 2)

    def test_merge_rejects_unknown_component_ids(self) -> None:
        points = np.eye(3)
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        sources = np.asarray([1], dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "unknown component IDs"):
            merge_kept_components(points, faces, sources, np.asarray([0]), [2])

    def test_empty_mesh_is_supported(self) -> None:
        points = np.empty((0, 3), dtype=np.float64)
        faces = np.empty((0, 3), dtype=np.int64)
        sources = np.empty(0, dtype=np.int64)
        labels = face_component_labels(faces)

        self.assertEqual(describe_components(points, faces, sources, labels), [])
        merged = merge_kept_components(points, faces, sources, labels, [])
        self.assertEqual(merged[0].shape, (0, 3))
        self.assertEqual(merged[1].shape, (0, 3))
        self.assertEqual(merged[2].shape, (0,))
        self.assertEqual(merged[3].shape, (0,))


if __name__ == "__main__":
    unittest.main()
