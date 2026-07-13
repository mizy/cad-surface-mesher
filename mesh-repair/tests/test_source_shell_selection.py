from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from source_shell_selection import (  # noqa: E402
    SELECTION_REASONS,
    attach_component_visibility,
    select_ai_candidates,
    select_source_shell_faces,
)


class SourceShellSelectionTest(unittest.TestCase):
    def test_ai_candidates_are_invisible_components_sorted_by_area(self) -> None:
        rows = [
            {"component_id": 0, "face_count": 10, "surface_area": 4.0, "first_hit_face_count": 2},
            {"component_id": 1, "face_count": 5, "surface_area": 2.0, "first_hit_face_count": 0},
            {"component_id": 2, "face_count": 8, "surface_area": 3.0, "first_hit_face_count": 0},
        ]
        selected = select_ai_candidates(rows, max_candidates=1)
        self.assertEqual([row["component_id"] for row in selected], [2])

    def test_visible_face_does_not_restore_whole_component(self) -> None:
        faces = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int64)
        labels = np.asarray([0, 0, 1], dtype=np.int64)
        selected, reasons, _, report = select_source_shell_faces(
            faces,
            labels,
            np.asarray([2, 0, 0]),
            np.asarray([4, 0, 0]),
            [],
            continuity_rings=0,
        )
        np.testing.assert_array_equal(selected, [True, False, False])
        self.assertFalse(report["component_expansion_applied"])
        self.assertEqual(reasons[0], SELECTION_REASONS["direct_first_hit"])

    def test_continuity_is_bounded_and_ai_removal_cannot_delete_visible_face(self) -> None:
        faces = np.asarray(
            [[0, 1, 2], [0, 2, 3], [0, 3, 4], [5, 6, 7]], dtype=np.int64
        )
        labels = np.asarray([0, 0, 0, 1], dtype=np.int64)
        decisions = [
            {
                "candidate_id": "component_000000",
                "decision": "remove_internal",
                "confidence": 0.99,
            },
            {
                "candidate_id": "component_000001",
                "decision": "remove_internal",
                "confidence": 0.99,
            },
        ]
        selected, reasons, _, report = select_source_shell_faces(
            faces,
            labels,
            np.asarray([1, 0, 0, 0]),
            np.asarray([2, 0, 0, 0]),
            decisions,
            continuity_rings=1,
        )
        np.testing.assert_array_equal(selected, [True, True, False, False])
        self.assertEqual(reasons[1], SELECTION_REASONS["continuity_ring"])
        self.assertEqual(report["removed_component_ids"], [1])
        self.assertEqual(report["ignored_ai_removals"][0]["component_id"], 0)

    def test_component_visibility_rows_expose_candidate_statistics(self) -> None:
        descriptions = [
            {"component_id": 0, "face_count": 2, "surface_area": 1.0},
            {"component_id": 1, "face_count": 1, "surface_area": 0.5},
        ]
        rows = attach_component_visibility(
            descriptions,
            np.asarray([0, 0, 1]),
            np.asarray([2, 0, 0]),
            np.asarray([5, 0, 0]),
        )
        self.assertEqual(rows[0]["candidate_id"], "component_000000")
        self.assertEqual(rows[0]["first_hit_face_count"], 1)
        self.assertEqual(rows[1]["first_hit_face_count"], 0)


if __name__ == "__main__":
    unittest.main()
