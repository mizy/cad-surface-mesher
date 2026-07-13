from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from coincident_loop_weld import (  # noqa: E402
    CoincidentLoopWeldConfig,
    weld_coincident_boundary_loops,
)
from hybrid_proxy_geometry import extract_ordered_boundary_loops  # noqa: E402
from mesh_metrics import edge_topology  # noqa: E402


def open_tube_half(
    z_min: float,
    z_max: float,
    ring_count: int,
    *,
    selected_end: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, ring_count, endpoint=False)
    ring_xy = np.column_stack([np.cos(angles), np.sin(angles)])
    points = np.vstack(
        [
            np.column_stack([ring_xy, np.full(ring_count, z_min)]),
            np.column_stack([ring_xy, np.full(ring_count, z_max)]),
        ]
    )
    faces = []
    for index in range(ring_count):
        next_index = (index + 1) % ring_count
        faces.extend(
            [
                [index, next_index, ring_count + next_index],
                [index, ring_count + next_index, ring_count + index],
            ]
        )
    faces_array = np.asarray(faces, dtype=np.int64)
    extraction = extract_ordered_boundary_loops(points, faces_array)
    selected = (
        max(extraction["loops"], key=lambda loop: float(points[loop, 2].mean()))
        if selected_end == "maximum"
        else min(
            extraction["loops"],
            key=lambda loop: float(points[loop, 2].mean()),
        )
    )
    return points, faces_array, selected


def planar_disk(
    ring_count: int,
    *,
    radius: float = 1.0,
    z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, ring_count, endpoint=False)
    boundary = np.column_stack(
        [
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.full(ring_count, z),
        ]
    )
    points = np.vstack([boundary, np.asarray([[0.0, 0.0, z]])])
    faces = np.asarray(
        [[ring_count, index, (index + 1) % ring_count] for index in range(ring_count)],
        dtype=np.int64,
    )
    loop = extract_ordered_boundary_loops(points, faces)["loops"][0]
    return points, faces, loop


def planar_annulus(
    ring_count: int,
    *,
    inner_radius: float = 1.005,
    outer_radius: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, ring_count, endpoint=False)
    inner = np.column_stack(
        [
            inner_radius * np.cos(angles),
            inner_radius * np.sin(angles),
            np.zeros(ring_count),
        ]
    )
    outer = np.column_stack(
        [
            outer_radius * np.cos(angles),
            outer_radius * np.sin(angles),
            np.zeros(ring_count),
        ]
    )
    points = np.vstack([inner, outer])
    faces = []
    for index in range(ring_count):
        next_index = (index + 1) % ring_count
        faces.extend(
            [
                [index, ring_count + index, ring_count + next_index],
                [index, ring_count + next_index, next_index],
            ]
        )
    faces_array = np.asarray(faces, dtype=np.int64)
    loops = extract_ordered_boundary_loops(points, faces_array)["loops"]
    inner_loop = min(
        loops,
        key=lambda loop: float(np.linalg.norm(points[loop, :2], axis=1).mean()),
    )
    return points, faces_array, inner_loop


def failed_intersection_report() -> dict:
    return {
        "method": "vtk_static_cell_locator_triangle_intersection",
        "scope": "focused_faces",
        "focus_face_count": 4,
        "status": "computed",
        "passed": False,
        "intersection_pairs": 1,
        "candidate_pairs_tested": 1,
        "reported_pairs": [[0, 3]],
        "truncated": False,
    }


class CoincidentLoopWeldTest(unittest.TestCase):
    def test_welds_two_open_tube_halves_without_annular_faces(self) -> None:
        source = open_tube_half(-1.0, 0.0, 12, selected_end="maximum")
        target = open_tube_half(0.005, 1.005, 16, selected_end="minimum")
        source_points_before = source[0].copy()
        source_faces_before = source[1].copy()
        target_points_before = target[0].copy()
        target_faces_before = target[1].copy()
        source_indices = np.arange(source[1].shape[0], dtype=np.int64) + 100
        target_indices = np.arange(target[1].shape[0], dtype=np.int64) + 1000

        result = weld_coincident_boundary_loops(
            *source,
            *target,
            config=CoincidentLoopWeldConfig(max_target_displacement=0.06),
            source_triangle_indices=source_indices,
            target_triangle_indices=target_indices,
            region_id=19,
        )

        self.assertTrue(result["success"], result["diagnostics"])
        np.testing.assert_array_equal(source[0], source_points_before)
        np.testing.assert_array_equal(source[1], source_faces_before)
        np.testing.assert_array_equal(target[0], target_points_before)
        np.testing.assert_array_equal(target[1], target_faces_before)
        np.testing.assert_array_equal(
            result["points"][: source[0].shape[0]], source_points_before
        )
        self.assertEqual(result["stitch_face_ids"].size, 0)
        self.assertFalse(
            result["diagnostics"]["contract"]["annular_bridge_generated"]
        )
        self.assertGreater(
            result["source_ring_vertex_ids"].size,
            max(source[2].size, target[2].size),
        )
        seam = result["diagnostics"]["seam_validation"]
        self.assertTrue(seam["edge_incidence_two"])
        self.assertTrue(seam["opposite_directed_edge_pairs"])
        self.assertTrue(seam["one_source_and_one_target_face"])
        self.assertEqual(seam["degenerate_faces"], [])
        self.assertGreaterEqual(
            seam["adjacent_normal_dot_min"], seam["minimum_adjacent_normal_dot"]
        )
        self.assertTrue(result["diagnostics"]["local_self_intersection"]["passed"])
        self.assertLessEqual(
            result["diagnostics"]["target_displacement"]["maximum"], 0.06
        )

        provenance = result["provenance"]
        exactly_one_parent = (provenance["source_face_parent"] >= 0) ^ (
            provenance["target_face_parent"] >= 0
        )
        self.assertTrue(np.all(exactly_one_parent))
        self.assertTrue(np.all(provenance["component_face_parent"] >= 0))
        self.assertTrue(
            np.all(np.isin(provenance["source_triangle_index"], np.r_[source_indices, target_indices]))
        )
        self.assertTrue(np.all(provenance["fusion_region_id"] == 19))

        topology, edge_faces = edge_topology(result["faces"])
        self.assertEqual(topology["non_manifold_edges"], 0)
        for left, right in result["welded_seam_edges"]:
            self.assertEqual(len(edge_faces[tuple(sorted((int(left), int(right))))]), 2)

    def test_welds_adjacent_planar_disk_and_annulus(self) -> None:
        source = planar_disk(12)
        target = planar_annulus(16)

        result = weld_coincident_boundary_loops(
            *source,
            *target,
            config=CoincidentLoopWeldConfig(max_target_displacement=0.06),
        )

        self.assertTrue(result["success"], result["diagnostics"])
        topology, _ = edge_topology(result["faces"])
        self.assertEqual(topology["boundary_edges"], 16)
        self.assertEqual(topology["non_manifold_edges"], 0)
        seam = result["diagnostics"]["seam_validation"]
        self.assertAlmostEqual(seam["adjacent_normal_dot_min"], 1.0)
        self.assertLess(seam["incident_side_dot_max"], -0.99)
        self.assertTrue(result["diagnostics"]["local_self_intersection"]["passed"])

    def test_rejects_same_side_overlapping_surfaces(self) -> None:
        source = planar_disk(8, z=0.0)
        target = planar_disk(8, z=0.01)
        source_before = tuple(array.copy() for array in source)
        target_before = tuple(array.copy() for array in target)

        result = weld_coincident_boundary_loops(
            *source,
            *target,
            config=CoincidentLoopWeldConfig(
                max_target_displacement=0.08,
                min_adjacent_normal_dot=-1.0,
                normal_score_weight=0.0,
                allow_target_face_flip=True,
                check_self_intersections=False,
            ),
        )

        self.assertFalse(result["success"])
        self.assertEqual(
            result["failure_reason_codes"], ["weld_same_side_overlap_detected"]
        )
        self.assertEqual(result["points"].shape, (0, 3))
        self.assertEqual(result["faces"].shape, (0, 3))
        self.assertEqual(result["provenance"]["face_origin"].size, 0)
        self.assertTrue(result["diagnostics"]["rollback"]["applied"])
        json.dumps(result["diagnostics"])
        self.assertGreater(
            result["diagnostics"]["seam_validation"]["incident_side_dot_max"],
            0.99,
        )
        for actual, expected in zip(source, source_before):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(target, target_before):
            np.testing.assert_array_equal(actual, expected)

    def test_rejects_target_ring_beyond_displacement_limit(self) -> None:
        source = open_tube_half(-1.0, 0.0, 12, selected_end="maximum")
        target = open_tube_half(0.5, 1.5, 12, selected_end="minimum")

        result = weld_coincident_boundary_loops(
            *source,
            *target,
            config=CoincidentLoopWeldConfig(max_target_displacement=0.05),
        )

        self.assertFalse(result["success"])
        self.assertEqual(
            result["failure_reason_codes"], ["loop_correspondence_distance_exceeded"]
        )
        self.assertEqual(result["points"].shape, (0, 3))
        self.assertEqual(result["source_face_parent"].size, 0)
        self.assertGreater(
            result["diagnostics"]["correspondence"]["distance_max"], 0.05
        )

    def test_rejects_geometrically_non_simple_boundary_loop(self) -> None:
        source_points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        source_faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        source_loop = np.asarray([0, 1, 2, 3], dtype=np.int64)
        target = planar_disk(8)

        result = weld_coincident_boundary_loops(
            source_points,
            source_faces,
            source_loop,
            *target,
            config=CoincidentLoopWeldConfig(check_self_intersections=False),
        )

        self.assertFalse(result["success"])
        self.assertEqual(
            result["failure_reason_codes"], ["boundary_loop_self_intersection"]
        )
        self.assertEqual(result["points"].shape, (0, 3))
        self.assertEqual(result["provenance"]["component_face_parent"].size, 0)
        self.assertEqual(
            result["diagnostics"]["correspondence"]["segment_pairs"], [[0, 2]]
        )

    def test_focused_self_intersection_failure_rolls_back(self) -> None:
        source = open_tube_half(-1.0, 0.0, 12, selected_end="maximum")
        target = open_tube_half(0.005, 1.005, 12, selected_end="minimum")

        with mock.patch(
            "coincident_loop_weld.self_intersection_report",
            return_value=failed_intersection_report(),
        ) as intersection_check:
            result = weld_coincident_boundary_loops(
                *source,
                *target,
                config=CoincidentLoopWeldConfig(max_target_displacement=0.06),
            )

        self.assertFalse(result["success"])
        self.assertEqual(
            result["failure_reason_codes"],
            ["coincident_weld_local_self_intersection_detected"],
        )
        self.assertEqual(result["faces"].shape, (0, 3))
        self.assertEqual(result["provenance"]["source_triangle_index"].size, 0)
        focus_face_ids = intersection_check.call_args.kwargs["focus_face_ids"]
        self.assertGreater(focus_face_ids.size, 0)
        self.assertTrue(result["diagnostics"]["rollback"]["applied"])


if __name__ == "__main__":
    unittest.main()
