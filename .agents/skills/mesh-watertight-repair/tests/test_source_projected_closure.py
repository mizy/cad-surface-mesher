from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from mesh_metrics import mesh_report  # noqa: E402
from source_projected_closure import (  # noqa: E402
    ProjectionThresholds,
    project_closure_to_source,
    rollback_reported_self_intersections,
)


def tetrahedron(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]],
        dtype=np.float64,
    ) * scale
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    return points, faces


def packed_polydata(points: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    packed = np.column_stack([np.full(faces.shape[0], 3), faces]).astype(np.int64)
    return pv.PolyData(points, packed.ravel())


class SourceProjectedClosureTest(unittest.TestCase):
    def test_exact_projection_preserves_watertight_proxy_connectivity(self) -> None:
        proxy_points, proxy_faces = tetrahedron(1.0)
        source_points, source_faces = tetrahedron(2.0)
        thresholds = ProjectionThresholds(
            max_projection_distance=4.0,
            min_signed_normal_dot=0.2,
            source_edge_barycentric_margin=0.01,
            defect_edge_barycentric_margin=0.01,
        )

        points, faces, provenance, report = project_closure_to_source(
            proxy_points,
            proxy_faces,
            source_points,
            source_faces,
            thresholds=thresholds,
        )

        applied = provenance["point_data"]["source_projection_applied"].astype(bool)
        self.assertTrue(np.all(applied), report["projection"])
        self.assertTrue(np.array_equal(faces, proxy_faces))
        self.assertFalse(np.allclose(points, proxy_points))
        self.assertTrue(np.allclose(report["comparisons"]["source_distance"]["residual"]["max"], 0.0))
        self.assertEqual(mesh_report(points, faces)["topology"]["boundary_edges"], 0)
        self.assertEqual(report["comparisons"]["projection_mapping"]["unclassified"], 0)
        self.assertEqual(report["comparisons"]["self_intersection"]["status"], "not_computed")

    def test_distance_and_reliability_fail_to_explicit_proxy_fallback(self) -> None:
        proxy_points, proxy_faces = tetrahedron(1.0)
        source_points, source_faces = tetrahedron(20.0)
        reliable = np.ones(source_faces.shape[0], dtype=bool)
        reliable[0] = False

        points, faces, provenance, report = project_closure_to_source(
            proxy_points,
            proxy_faces,
            source_points,
            source_faces,
            thresholds=ProjectionThresholds(max_projection_distance=0.01),
            reliable_source_face_mask=reliable,
        )

        self.assertTrue(np.array_equal(points, proxy_points))
        self.assertTrue(np.array_equal(faces, proxy_faces))
        self.assertTrue(np.all(provenance["point_data"]["source_projection_fallback"]))
        self.assertEqual(report["projection"]["unclassified"], 0)
        self.assertGreater(report["projection"]["rejection_counts"]["distance_gate_failed"], 0)

    def test_explicit_source_face_reliability_mask_is_blocking(self) -> None:
        proxy_points, proxy_faces = tetrahedron(1.0)
        source_points, source_faces = tetrahedron(2.0)

        points, _, _, report = project_closure_to_source(
            proxy_points,
            proxy_faces,
            source_points,
            source_faces,
            thresholds=ProjectionThresholds(max_projection_distance=4.0),
            reliable_source_face_mask=np.zeros(source_faces.shape[0], dtype=bool),
        )

        self.assertTrue(np.array_equal(points, proxy_points))
        self.assertEqual(
            report["projection"]["rejection_counts"]["source_face_unreliable"],
            proxy_points.shape[0],
        )

    def test_source_edge_gate_rejects_projection_near_any_triangle_edge(self) -> None:
        proxy_points = np.asarray([[0.001, 0.2, 0.1], [0.001, 0.8, 0.1], [0.2, 0.2, 0.1]])
        proxy_faces = np.asarray([[0, 1, 2]])
        source_points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        source_faces = np.asarray([[0, 1, 2]])

        _, _, provenance, report = project_closure_to_source(
            proxy_points,
            proxy_faces,
            source_points,
            source_faces,
            thresholds=ProjectionThresholds(
                max_projection_distance=1.0,
                min_signed_normal_dot=-1.0,
                orientation_vote_min_margin=0.0,
                source_edge_barycentric_margin=0.01,
                defect_edge_barycentric_margin=0.0,
                require_component_consistency=False,
            ),
        )

        applied = provenance["point_data"]["source_projection_applied"].astype(bool)
        self.assertFalse(applied[0])
        self.assertFalse(applied[1])
        self.assertGreaterEqual(report["projection"]["rejection_counts"]["source_edge_or_vertex_gate_failed"], 2)

    def test_opposite_shell_sides_cannot_both_collapse_to_one_source_sheet(self) -> None:
        proxy_points = np.asarray(
            [
                [-0.4, -0.3, -0.05], [0.4, -0.3, -0.05], [0.4, 0.3, -0.05], [-0.4, 0.3, -0.05],
                [-0.4, -0.3, 0.05], [0.4, -0.3, 0.05], [0.4, 0.3, 0.05], [-0.4, 0.3, 0.05],
            ],
            dtype=np.float64,
        )
        proxy_faces = np.asarray(
            [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
            dtype=np.int64,
        )
        source_points = np.asarray([[-3.0, -2.0, 0.0], [3.0, -2.0, 0.0], [0.0, 3.0, 0.0]])
        source_faces = np.asarray([[0, 1, 2]])
        thresholds = ProjectionThresholds(
            max_projection_distance=1.0,
            min_signed_normal_dot=-1.0,
            orientation_vote_min_margin=0.0,
            source_edge_barycentric_margin=0.0,
            defect_edge_barycentric_margin=0.0,
            collision_tolerance_bbox_ratio=1e-7,
            require_component_consistency=False,
        )

        points, faces, provenance, report = project_closure_to_source(
            proxy_points,
            proxy_faces,
            source_points,
            source_faces,
            thresholds=thresholds,
        )

        applied = provenance["point_data"]["source_projection_applied"].astype(bool)
        for bottom, top in zip(range(4), range(4, 8), strict=True):
            self.assertFalse(applied[bottom] and applied[top])
        protected_vertices = (
            report["collision_gate"]["rolled_back_vertices"]
            + report["projection"]["rejection_counts"]["triangle_quality_rollback"]
        )
        self.assertGreater(protected_vertices, 0)
        cleaned = packed_polydata(points, faces).clean()
        clean_faces = np.asarray(cleaned.faces).reshape(-1, 4)[:, 1:]
        clean_report = mesh_report(np.asarray(cleaned.points), clean_faces)
        self.assertEqual(clean_report["topology"]["boundary_edges"], 0)
        self.assertEqual(clean_report["topology"]["non_manifold_edges"], 0)

    def test_nonadjacent_layers_use_collision_fallback_before_clean(self) -> None:
        proxy_points = np.asarray(
            [
                [-0.4, -0.3, -0.05], [0.4, -0.3, -0.05], [0.0, 0.3, -0.05],
                [-0.4, -0.3, 0.05], [0.4, -0.3, 0.05], [0.0, 0.3, 0.05],
            ]
        )
        proxy_faces = np.asarray([[0, 2, 1], [3, 4, 5]])
        source_points = np.asarray([[-3.0, -2.0, 0.0], [3.0, -2.0, 0.0], [0.0, 3.0, 0.0]])
        source_faces = np.asarray([[0, 1, 2]])

        _, _, provenance, report = project_closure_to_source(
            proxy_points,
            proxy_faces,
            source_points,
            source_faces,
            thresholds=ProjectionThresholds(
                max_projection_distance=1.0,
                min_signed_normal_dot=-1.0,
                orientation_vote_min_margin=0.0,
                source_edge_barycentric_margin=0.0,
                defect_edge_barycentric_margin=0.0,
                collision_tolerance_bbox_ratio=1e-7,
                require_component_consistency=False,
            ),
        )

        applied = provenance["point_data"]["source_projection_applied"].astype(bool)
        for lower, upper in zip(range(3), range(3, 6), strict=True):
            self.assertFalse(applied[lower] and applied[upper])
        self.assertGreater(report["collision_gate"]["rolled_back_vertices"], 0)
        self.assertEqual(
            report["projection"]["rejection_counts"]["projected_vertex_collision"],
            report["collision_gate"]["rolled_back_vertices"],
        )

    def test_component_orientation_vote_rejects_ambiguous_opposite_sides(self) -> None:
        proxy_points = np.asarray(
            [
                [-0.4, -0.3, -0.05], [0.4, -0.3, -0.05], [0.0, 0.3, -0.05],
                [-0.4, -0.3, 0.05], [0.4, -0.3, 0.05], [0.0, 0.3, 0.05],
            ]
        )
        proxy_faces = np.asarray([[0, 2, 1], [3, 4, 5]])
        source_points = np.asarray([[-3.0, -2.0, 0.0], [3.0, -2.0, 0.0], [0.0, 3.0, 0.0]])
        source_faces = np.asarray([[0, 1, 2]])

        points, _, provenance, report = project_closure_to_source(
            proxy_points,
            proxy_faces,
            source_points,
            source_faces,
            thresholds=ProjectionThresholds(
                max_projection_distance=1.0,
                source_edge_barycentric_margin=0.0,
                defect_edge_barycentric_margin=0.0,
                require_component_consistency=False,
            ),
        )

        self.assertTrue(np.array_equal(points, proxy_points))
        self.assertFalse(np.any(provenance["point_data"]["source_projection_applied"]))
        self.assertEqual(
            report["projection"]["rejection_counts"]["normal_orientation_ambiguous"],
            proxy_points.shape[0],
        )

    def test_component_spatial_gate_prevents_one_proxy_face_crossing_components(self) -> None:
        proxy_points = np.asarray([[-0.8, 0.0, 0.1], [0.8, 0.0, 0.1], [0.0, 0.8, 0.1]])
        proxy_faces = np.asarray([[0, 1, 2]])
        source_points = np.asarray(
            [[-1.5, -1.0, 0.0], [0.0, -1.0, 0.0], [-0.75, 1.5, 0.0], [0.0, -1.0, 0.0], [1.5, -1.0, 0.0], [0.75, 1.5, 0.0]]
        )
        source_faces = np.asarray([[0, 1, 2], [3, 4, 5]])

        _, _, _, report = project_closure_to_source(
            proxy_points,
            proxy_faces,
            source_points,
            source_faces,
            thresholds=ProjectionThresholds(
                max_projection_distance=1.0,
                min_signed_normal_dot=-1.0,
                orientation_vote_min_margin=0.0,
                source_edge_barycentric_margin=0.0,
                defect_edge_barycentric_margin=0.0,
            ),
        )

        self.assertEqual(report["component_gate"]["inconsistent_proxy_faces"], 1)
        self.assertGreater(report["projection"]["rejection_counts"]["source_component_spatial_mismatch"], 0)

    def test_triangle_collapse_rolls_projection_back_to_proxy(self) -> None:
        proxy_points, proxy_faces = tetrahedron(1.0)
        source_points = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
        source_faces = np.asarray([[0, 1, 2]])

        points, _, provenance, report = project_closure_to_source(
            proxy_points,
            proxy_faces,
            source_points,
            source_faces,
            thresholds=ProjectionThresholds(
                max_projection_distance=10.0,
                min_signed_normal_dot=-1.0,
                orientation_vote_min_margin=0.0,
                source_edge_barycentric_margin=0.0,
                defect_edge_barycentric_margin=0.0,
                collision_tolerance_bbox_ratio=0.0,
                require_component_consistency=False,
            ),
        )

        self.assertTrue(np.array_equal(points, proxy_points))
        self.assertFalse(np.any(provenance["point_data"]["source_projection_applied"]))
        self.assertGreater(report["projection"]["rejection_counts"]["triangle_quality_rollback"], 0)

    def test_reported_intersection_helper_rolls_back_and_marks_provenance(self) -> None:
        proxy_points, faces = tetrahedron(1.0)
        points = proxy_points + 0.1
        applied = np.ones(proxy_points.shape[0], dtype=np.uint8)
        provenance = {
            "point_data": {
                "source_projection_applied": applied,
                "source_projection_fallback": 1 - applied,
                "source_projection_rejection_mask": np.zeros(proxy_points.shape[0], dtype=np.uint16),
            },
            "cell_data": {
                "source_projected_vertex_count": np.full(faces.shape[0], 3, dtype=np.uint8),
                "source_projected_vertex_fraction": np.ones(faces.shape[0], dtype=np.float32),
            },
        }

        result, updated, report = rollback_reported_self_intersections(
            points,
            proxy_points,
            faces,
            provenance,
            {"status": "computed", "reported_pairs": [[0, 1]]},
        )

        touched = np.unique(faces[[0, 1]].ravel())
        self.assertTrue(np.array_equal(result[touched], proxy_points[touched]))
        self.assertTrue(np.all(updated["point_data"]["source_projection_fallback"][touched]))
        self.assertEqual(report["rolled_back_vertices"], touched.size)
        self.assertTrue(report["requires_global_recheck"])

    def test_reported_intersection_helper_can_restore_a_transition_ring(self) -> None:
        proxy_points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [2, 1, 3], [2, 3, 4]], dtype=np.int64)
        points = proxy_points + np.asarray([0.0, 0.0, 0.1])
        applied = np.ones(proxy_points.shape[0], dtype=np.uint8)
        provenance = {
            "point_data": {
                "source_projection_applied": applied,
                "source_projection_fallback": 1 - applied,
                "source_projection_rejection_mask": np.zeros(
                    proxy_points.shape[0], dtype=np.uint16
                ),
            },
            "cell_data": {
                "source_projected_vertex_count": np.full(
                    faces.shape[0], 3, dtype=np.uint8
                ),
                "source_projected_vertex_fraction": np.ones(
                    faces.shape[0], dtype=np.float32
                ),
            },
        }

        result, _, report = rollback_reported_self_intersections(
            points,
            proxy_points,
            faces,
            provenance,
            {"status": "computed", "reported_pairs": [[0, 1]]},
            vertex_rings=1,
        )

        np.testing.assert_array_equal(result, proxy_points)
        self.assertEqual(report["intersection_core_vertices"], 4)
        self.assertEqual(report["rolled_back_vertices"], 5)
        self.assertEqual(report["rollback_vertex_rings"], 1)


if __name__ == "__main__":
    unittest.main()
