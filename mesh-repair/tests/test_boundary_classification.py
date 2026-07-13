from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from boundary_classification import (  # noqa: E402
    classify_boundary_regions,
    classify_one,
    normalized_boundary_thresholds,
    normalized_component_thresholds,
    order_simple_loop,
)
from mesh_metrics import edge_topology, face_component_labels  # noqa: E402
from repair_inventory import connected_edge_regions  # noqa: E402


def boundary_inputs(
    points: np.ndarray,
    faces: np.ndarray,
) -> tuple[dict[tuple[int, int], list[int]], np.ndarray, list[list[tuple[int, int]]]]:
    _, edge_to_faces = edge_topology(faces)
    components = face_component_labels(faces.shape[0], edge_to_faces)
    regions = connected_edge_regions(
        edge_to_faces,
        lambda count: count == 1,
        face_component_ids=components,
    )
    return edge_to_faces, components, regions


def rectangular_ring(
    inner_half_extent: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    inner_x, inner_y = inner_half_extent
    points = np.asarray(
        [
            [-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0],
            [-inner_x, -inner_y, 0.0], [inner_x, -inner_y, 0.0],
            [inner_x, inner_y, 0.0], [-inner_x, inner_y, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return points, faces


def ring_with_triangle_components(
    inner_half_extent: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    points, faces = rectangular_ring(inner_half_extent)
    point_rows = [row for row in points]
    face_rows = [row for row in faces]
    for index in range(12):
        angle = 2.0 * np.pi * index / 12.0
        center = np.asarray([1.5 * np.cos(angle), 1.5 * np.sin(angle), 0.1])
        offset = len(point_rows)
        point_rows.extend(
            [
                center + [-0.025, -0.02, 0.0],
                center + [0.025, -0.02, 0.0],
                center + [0.0, 0.025, 0.0],
            ]
        )
        face_rows.append([offset, offset + 1, offset + 2])
    return np.asarray(point_rows, dtype=np.float64), np.asarray(face_rows, dtype=np.int64)


def separated_square_patches(count: int) -> tuple[np.ndarray, np.ndarray]:
    point_rows = []
    face_rows = []
    for index in range(count):
        center_x = float(index)
        offset = len(point_rows)
        point_rows.extend(
            [
                [center_x - 0.1, -0.1, 0.0], [center_x + 0.1, -0.1, 0.0],
                [center_x + 0.1, 0.1, 0.0], [center_x - 0.1, 0.1, 0.0],
            ]
        )
        face_rows.extend([[offset, offset + 1, offset + 2], [offset, offset + 2, offset + 3]])
    return np.asarray(point_rows, dtype=np.float64), np.asarray(face_rows, dtype=np.int64)


class BoundaryClassificationTest(unittest.TestCase):
    def test_order_simple_loop_rejects_branched_boundary_graph(self) -> None:
        self.assertEqual(order_simple_loop([(0, 1), (1, 2), (2, 3), (3, 0)])[0], 0)
        self.assertIsNone(order_simple_loop([(0, 1), (1, 2), (1, 3)]))

    def test_near_coincident_component_loops_route_to_zipper(self) -> None:
        lower = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        points = np.vstack([lower, lower + np.asarray([0.0, 0.0, 0.01])])
        faces = np.asarray([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]], dtype=np.int64)
        edge_to_faces, components, regions = boundary_inputs(points, faces)

        classified, summary = classify_boundary_regions(
            points,
            faces,
            regions,
            edge_to_faces,
            components,
            exterior_face_score=np.ones(faces.shape[0]),
        )

        self.assertEqual(summary["loop_pair_count"], 1)
        self.assertEqual({row["classification"] for row in classified}, {"near_coincident_part_seam"})
        self.assertTrue(all(row["operator"] == "loop_pair_zipper" for row in classified))
        self.assertTrue(
            all(np.isclose(row["dimensionless_features"]["pair_gap"], 0.01) for row in classified)
        )
        self.assertTrue(all(row["dimensionless_features"]["pair_gap_bbox_ratio"] > 0.0 for row in classified))
        self.assertIn("pair_max_gap_local_edge_ratio", summary["boundary_thresholds"])

        unpaired, strict_summary = classify_boundary_regions(
            points,
            faces,
            regions,
            edge_to_faces,
            components,
            exterior_face_score=np.ones(faces.shape[0]),
            boundary_thresholds={
                "pair_max_gap_local_edge_ratio": 0.005,
                "pair_gap_bbox_ratio_floor": 0.0,
                "pair_interface_max_gap_local_edge_ratio": 0.005,
                "pair_interface_max_gap_loop_diameter_ratio": 0.0,
            },
        )
        self.assertEqual(strict_summary["loop_pair_count"], 0)
        self.assertNotIn("near_coincident_part_seam", {row["classification"] for row in unpaired})

    def test_concentric_component_interface_routes_to_zipper_when_strict_gap_fails(self) -> None:
        lower = np.asarray(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        upper = lower * np.asarray([0.95, 0.95, 1.0]) + np.asarray([0.0, 0.0, 0.15])
        points = np.vstack([lower, upper])
        faces = np.asarray([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]], dtype=np.int64)
        edge_to_faces, components, regions = boundary_inputs(points, faces)

        classified, summary = classify_boundary_regions(
            points,
            faces,
            regions,
            edge_to_faces,
            components,
            exterior_face_score=np.ones(faces.shape[0]),
            boundary_thresholds={
                "pair_max_gap_local_edge_ratio": 0.0,
                "pair_max_gap_loop_diameter_ratio": 0.0,
                "pair_gap_bbox_ratio_floor": 0.0,
            },
        )

        self.assertEqual(summary["loop_pair_count"], 1)
        self.assertEqual({row["classification"] for row in classified}, {"near_coincident_part_seam"})
        self.assertTrue(
            all(row["pair_evidence"]["pair_mode"] == "concentric_component_interface" for row in classified)
        )

    def test_isolated_missing_triangle_uses_algorithmic_fill_above_micro_hole_scale(self) -> None:
        features = {
            "diameter_bbox_ratio": 0.011,
            "diameter_local_edge_ratio": 1.3,
            "projected_area_bbox_ratio": 3.0e-5,
            "diameter_component_bbox_ratio": 0.2,
            "projected_area_component_bbox_ratio": 0.01,
            "planarity": 0.0,
            "compactness": 0.54,
            "component_boundary_count": 1,
            "pair_gap": None,
            "pair_gap_bbox_ratio": None,
            "pair_gap_loop_diameter_ratio": None,
        }

        result = classify_one(
            {
                "simple_closed_loop": True,
                "ordered_vertex_ids": [0, 1, 2],
                "dimensionless_features": features,
            },
            {
                "exterior_confidence": 0.9,
                "face_ratio": 0.01,
                "strictly_contained": False,
                "bbox_diagonal": 1.0,
            },
            None,
            623,
            453,
            normalized_component_thresholds(None),
            normalized_boundary_thresholds(None),
        )

        self.assertEqual(result["classification"], "small_exterior_hole")
        self.assertEqual(result["detector_reason"], "isolated_missing_polygon_boundary")
        self.assertEqual(result["operator"], "constrained_loop_triangulation")

    def test_only_small_inner_loop_is_automatic_hole(self) -> None:
        points = np.asarray(
            [
                [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
                [-0.004, -0.004, 0.0], [0.004, -0.004, 0.0], [0.004, 0.004, 0.0], [-0.004, 0.004, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray(
            [
                [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
            ],
            dtype=np.int64,
        )
        edge_to_faces, components, regions = boundary_inputs(points, faces)

        classified, _ = classify_boundary_regions(
            points,
            faces,
            regions,
            edge_to_faces,
            components,
            exterior_face_score=np.ones(faces.shape[0]),
        )

        by_class = {row["classification"]: row for row in classified}
        self.assertIn("small_exterior_hole", by_class)
        self.assertTrue(by_class["small_exterior_hole"]["patch_eligible"])
        features = by_class["small_exterior_hole"]["dimensionless_features"]
        self.assertLessEqual(features["diameter_bbox_ratio"], 0.005)
        self.assertLessEqual(features["projected_area_bbox_ratio"], 2.5e-5)
        self.assertAlmostEqual(features["diameter_local_edge_ratio"], np.sqrt(2.0))
        self.assertEqual(features["component_boundary_count"], 2)
        self.assertIsNone(features["pair_gap"])
        self.assertIn("large_opening_or_missing_surface", by_class)
        self.assertTrue(by_class["large_opening_or_missing_surface"]["requires_policy"])

    def test_assembly_part_perimeters_do_not_flood_large_opening_policy(self) -> None:
        points, faces = separated_square_patches(13)
        edge_to_faces, components, regions = boundary_inputs(points, faces)

        classified, summary = classify_boundary_regions(
            points,
            faces,
            regions,
            edge_to_faces,
            components,
            exterior_face_score=np.ones(faces.shape[0]),
        )

        self.assertEqual(summary["route"], "assembly_exterior_reconstruction")
        self.assertEqual(
            {row["classification"] for row in classified},
            {"part_perimeter_or_opening_unknown"},
        )
        self.assertTrue(
            all(row["detector_reason"] == "loop_diameter_spans_component_bbox" for row in classified)
        )
        self.assertTrue(all(not row["requires_policy"] for row in classified))

    def test_compact_interior_loop_remains_large_semantic_opening_in_assembly(self) -> None:
        points, faces = ring_with_triangle_components((0.5, 0.5))
        edge_to_faces, components, regions = boundary_inputs(points, faces)

        classified, summary = classify_boundary_regions(
            points,
            faces,
            regions,
            edge_to_faces,
            components,
            exterior_face_score=np.ones(faces.shape[0]),
        )

        semantic = [row for row in classified if row["classification"] == "large_opening_or_missing_surface"]
        self.assertEqual(summary["route"], "assembly_exterior_reconstruction")
        self.assertEqual(len(semantic), 1)
        self.assertEqual(semantic[0]["component_id"], 0)
        self.assertTrue(semantic[0]["requires_policy"])
        self.assertGreaterEqual(
            semantic[0]["dimensionless_features"]["compactness"],
            summary["boundary_thresholds"]["semantic_min_compactness"],
        )
        self.assertGreaterEqual(
            semantic[0]["dimensionless_features"]["projected_area_bbox_ratio"],
            summary["boundary_thresholds"]["semantic_min_projected_area_bbox_ratio"],
        )

    def test_long_narrow_loop_stays_unknown_instead_of_large_opening(self) -> None:
        points, faces = ring_with_triangle_components((0.8, 0.005))
        edge_to_faces, components, regions = boundary_inputs(points, faces)

        classified, _ = classify_boundary_regions(
            points,
            faces,
            regions,
            edge_to_faces,
            components,
            exterior_face_score=np.ones(faces.shape[0]),
        )

        inner = [
            row
            for row in classified
            if row["component_id"] == 0
            and row["diameter_component_bbox_ratio"] < 0.5
        ]
        self.assertEqual(len(inner), 1)
        self.assertEqual(inner[0]["classification"], "part_perimeter_or_opening_unknown")
        self.assertEqual(inner[0]["detector_reason"], "long_narrow_or_low_compactness_boundary")
        self.assertFalse(inner[0]["requires_policy"])

    def test_many_boundaries_on_one_component_are_policy_conservative_and_configurable(self) -> None:
        points, faces = separated_square_patches(5)
        _, edge_to_faces = edge_topology(faces)
        regions = connected_edge_regions(edge_to_faces, lambda count: count == 1)
        one_component = np.zeros(faces.shape[0], dtype=np.int64)

        classified, summary = classify_boundary_regions(
            points,
            faces,
            regions,
            edge_to_faces,
            one_component,
            exterior_face_score=np.ones(faces.shape[0]),
        )

        self.assertEqual(
            {row["classification"] for row in classified},
            {"part_perimeter_or_opening_unknown"},
        )
        self.assertTrue(
            all(
                row["detector_reason"] == "component_has_too_many_boundaries_for_semantic_opening"
                for row in classified
            )
        )
        self.assertTrue(
            all(row["dimensionless_features"]["component_boundary_count"] == 5 for row in classified)
        )
        self.assertEqual(
            set(summary["boundary_thresholds"]),
            set(summary["boundary_threshold_definitions"]),
        )

        permissive, permissive_summary = classify_boundary_regions(
            points,
            faces,
            regions,
            edge_to_faces,
            one_component,
            exterior_face_score=np.ones(faces.shape[0]),
            boundary_thresholds={"semantic_max_component_boundary_count": 5},
        )
        self.assertEqual(
            {row["classification"] for row in permissive},
            {"large_opening_or_missing_surface"},
        )
        self.assertEqual(permissive_summary["boundary_thresholds"]["semantic_max_component_boundary_count"], 5)

        with self.assertRaisesRegex(ValueError, "unknown boundary threshold"):
            classify_boundary_regions(
                points,
                faces,
                regions,
                edge_to_faces,
                one_component,
                boundary_thresholds={"semantic_typo": 1.0},
            )

    def test_nonplanar_closed_loop_is_not_promoted_to_semantic_opening(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        edge_to_faces, components, regions = boundary_inputs(points, faces)

        classified, summary = classify_boundary_regions(points, faces, regions, edge_to_faces, components)

        self.assertEqual(classified[0]["classification"], "part_perimeter_or_opening_unknown")
        self.assertIn(
            classified[0]["detector_reason"],
            {
                "boundary_nonplanarity_exceeds_semantic_geometry_limit",
                "loop_diameter_spans_component_bbox",
            },
        )
        self.assertGreater(
            classified[0]["dimensionless_features"]["planarity"],
            summary["boundary_thresholds"]["semantic_max_planarity_ratio"],
        )

    def test_vertex_touching_triangles_are_not_mislabeled_as_holes(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
        edge_to_faces, components, regions = boundary_inputs(points, faces)

        classified, _ = classify_boundary_regions(points, faces, regions, edge_to_faces, components)

        self.assertEqual(len(classified), 2)
        self.assertEqual(
            {row["classification"] for row in classified},
            {"part_perimeter_or_opening_unknown"},
        )
        self.assertTrue(all(not row["patch_eligible"] for row in classified))

    def test_component_thresholds_separate_contained_internal_and_flying_fragments(self) -> None:
        cube = np.asarray(
            [
                [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        core_faces = np.asarray(
            [
                [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
            ],
            dtype=np.int64,
        )
        internal = np.asarray([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0]])
        flying = np.asarray([[4.0, 0.0, 0.0], [4.05, 0.0, 0.0], [4.0, 0.05, 0.0]])
        points = np.vstack([cube, internal, flying])
        faces = np.vstack([core_faces, [[8, 9, 10]], [[11, 12, 13]]])
        edge_to_faces, components, regions = boundary_inputs(points, faces)
        thresholds = {
            "internal_max_face_ratio": 0.1,
            "floating_max_face_ratio": 0.1,
            "floating_max_diameter_ratio": 0.05,
            "shell_envelope_min_component_face_ratio": 0.5,
            "sealed_exterior_support_threshold": 0.01,
        }
        _, summary = classify_boundary_regions(
            points,
            faces,
            regions,
            edge_to_faces,
            components,
            exterior_face_score=np.r_[np.ones(core_faces.shape[0]), 0.0, 1.0],
            sealed_exterior_face_mask=np.r_[
                np.ones(core_faces.shape[0], dtype=bool),
                False,
                False,
            ],
            component_thresholds=thresholds,
        )
        internal_evidence = next(
            row for row in summary["component_evidence"] if row["component_id"] == 1
        )
        visible_flying = next(
            row for row in summary["component_evidence"] if row["component_id"] == 2
        )
        self.assertTrue(internal_evidence["direct_and_sealed_internal_consensus"])
        self.assertTrue(internal_evidence["physical_scale_small"])
        self.assertTrue(internal_evidence["automatic_remove_candidate"])
        self.assertEqual(
            internal_evidence["removal_classification"],
            "internal_or_fragment_component_perimeter",
        )
        self.assertTrue(visible_flying["visible_hard_keep"])
        self.assertFalse(visible_flying["automatic_remove_candidate"])
        self.assertIn("multi_view_first_hit", visible_flying["hard_keep_reasons"])

        _, visible_internal = classify_boundary_regions(
            points,
            faces,
            regions,
            edge_to_faces,
            components,
            exterior_face_score=np.ones(faces.shape[0]),
            sealed_exterior_face_mask=np.r_[
                np.ones(core_faces.shape[0], dtype=bool),
                False,
                False,
            ],
            component_thresholds=thresholds,
        )
        visible_internal_evidence = next(
            row for row in visible_internal["component_evidence"] if row["component_id"] == 1
        )
        self.assertEqual(visible_internal_evidence["visibility_label"], "visible_hard_keep")
        self.assertFalse(visible_internal_evidence["automatic_remove_candidate"])
        self.assertTrue(visible_internal_evidence["protected_from_automatic_removal"])

        _, sealed_only = classify_boundary_regions(
            points,
            faces,
            regions,
            edge_to_faces,
            components,
            sealed_exterior_face_mask=np.r_[
                np.ones(core_faces.shape[0], dtype=bool),
                False,
                False,
            ],
            component_thresholds=thresholds,
        )
        sealed_only_internal = next(
            row for row in sealed_only["component_evidence"] if row["component_id"] == 1
        )
        self.assertEqual(sealed_only_internal["visibility_label"], "visibility_unavailable")
        self.assertFalse(sealed_only_internal["automatic_remove_candidate"])
        self.assertIn(
            "visibility_evidence_unavailable",
            sealed_only_internal["hard_keep_reasons"],
        )

    def test_large_low_poly_contained_plate_is_not_treated_as_a_tiny_internal(self) -> None:
        cube = np.asarray(
            [
                [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        plate = np.asarray(
            [[-0.6, -0.6, 0.0], [0.6, -0.6, 0.0], [0.6, 0.6, 0.0], [-0.6, 0.6, 0.0]],
            dtype=np.float64,
        )
        faces = np.asarray(
            [
                [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
                [8, 9, 10], [8, 10, 11],
            ],
            dtype=np.int64,
        )
        points = np.vstack([cube, plate])
        edge_to_faces, components, regions = boundary_inputs(points, faces)

        _, summary = classify_boundary_regions(
            points,
            faces,
            regions,
            edge_to_faces,
            components,
            exterior_face_score=np.r_[np.ones(12), 0.0, 0.0],
            sealed_exterior_face_mask=np.r_[np.ones(12, dtype=bool), False, False],
            component_thresholds={
                "internal_max_face_ratio": 0.2,
                "shell_envelope_min_component_face_ratio": 0.5,
            },
        )

        evidence = next(
            row for row in summary["component_evidence"] if row["component_id"] == 1
        )
        self.assertFalse(evidence["physical_scale_small"])
        self.assertFalse(evidence["automatic_remove_candidate"])
        self.assertTrue(
            {
                "physical_diameter_exceeds_internal_limit",
                "projected_bbox_area_exceeds_internal_limit",
            }
            & set(evidence["hard_keep_reasons"])
        )


if __name__ == "__main__":
    unittest.main()
