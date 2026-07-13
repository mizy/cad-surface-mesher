from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from mesh_io import write_vtp  # noqa: E402
from mesh_metrics import mesh_report  # noqa: E402
from source_preserving_repair import build_decision  # noqa: E402
from source_projected_contract import (  # noqa: E402
    build_projection_gates,
    projection_comparisons,
    volume_drift,
    watertight_shell_policy,
)
from source_projected_validation import (  # noqa: E402
    chunked_self_intersection_report,
    projected_delta_self_intersection_report,
    repair_closure_proxy_self_intersections,
    roundtrip_validation,
)


def tetrahedron(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]],
        dtype=np.float64,
    ) * scale
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    return points, faces


def projection_report(vertex_count: int, self_intersection: dict | None = None) -> dict:
    return {
        "thresholds": {"resolved_max_projection_distance": 0.1},
        "sealed_exterior_construction": {
            "schema": "sealed_exterior_volume/v1",
            "method": "voxel_shell_dilation_six_connected_far_field_flood",
            "pitch": 0.05,
            "surface_offset_restore": "erode_filled_solid_by_seal_radius",
            "filled_or_shell_voxels": 8000,
            "estimated_filled_volume": 1.0,
            "projection_erosion_core": {
                "method": "binary_erosion_by_ceil_projection_distance_over_pitch",
                "max_projection_distance": 0.1,
                "erosion_radius_voxels": 2,
                "erosion_radius_world": 0.1,
                "filled_voxels": 800,
                "estimated_volume": 0.1,
                "mesh_volume_method": "marching_cubes_signed_abs_volume",
                "mesh_signed_abs_volume": 0.1,
                "mesh_watertight": True,
                "mesh_triangles": 100,
                "nonempty": True,
            },
        },
        "projection": {
            "vertices": vertex_count,
            "source_projected": vertex_count,
            "explicit_proxy_fallback": 0,
            "unclassified": 0,
            "fallback_without_reason": 0,
            "distance_before": {
                "min": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "max": 0.0,
            },
            "source_distance_after_for_projected": {
                "min": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "max": 0.0,
            },
        },
        "six_view_depth_regression": {
            "status": "computed",
            "passed": True,
            "regression_pixels": 0,
        },
        "comparisons": {
            "projection_topology": {
                "status": "computed",
                "faces_equal": True,
                "point_count_equal": True,
            },
            "projection_mapping": {
                "status": "computed",
                "source_projected": vertex_count,
                "explicit_proxy_fallback": 0,
                "unclassified": 0,
            },
            "self_intersection": self_intersection
            or {"status": "computed", "passed": True, "intersection_pairs": 0},
        },
    }


class SourceProjectedContractTest(unittest.TestCase):
    def test_projected_candidate_is_the_only_accepted_path(self) -> None:
        points, faces = tetrahedron()
        metrics = mesh_report(points, faces)
        report = projection_report(points.shape[0])
        comparisons = projection_comparisons(
            metrics,
            metrics,
            metrics,
            report,
            {
                "method": "test",
                "summary": {"changed_ratio_max": 0.0, "overlap_ratio_min": 1.0},
                "per_view": {},
            },
            {"status": "computed", "passed": True},
        )
        gates = build_projection_gates(
            metrics,
            comparisons,
            [],
            "projected.vtp",
            True,
            "proxy.vtp",
            report,
            watertight_shell_policy("watertight-exterior-shell"),
        )

        decision = build_decision(gates, "projected.vtp")

        self.assertEqual(decision["status"], "accepted")
        self.assertEqual(decision["final_output_path"], "projected.vtp")
        self.assertFalse(gates["opening_policy_resolved"]["required"])

    def test_proxy_volume_change_is_diagnostic_not_acceptance_truth(self) -> None:
        proxy_points, faces = tetrahedron()
        projected_points = proxy_points * 0.8
        proxy_metrics = mesh_report(proxy_points, faces)
        projected_metrics = mesh_report(projected_points, faces)
        report = projection_report(projected_points.shape[0])
        comparisons = projection_comparisons(
            projected_metrics,
            proxy_metrics,
            projected_metrics,
            report,
            {
                "method": "test",
                "summary": {"changed_ratio_max": 0.0, "overlap_ratio_min": 1.0},
                "per_view": {},
            },
            {"status": "computed", "passed": True},
        )
        gates = build_projection_gates(
            projected_metrics,
            comparisons,
            [],
            "projected.vtp",
            True,
            "proxy.vtp",
            report,
            watertight_shell_policy("watertight-exterior-shell"),
        )

        # Scaling by 0.8 changes volume by 1 - 0.8**3 = 48.8%, well past the
        # removed 30% proxy-relative threshold.  The projected mesh is the
        # source-faithful geometry in this fixture; the proxy has no authority
        # to veto it.
        self.assertGreater(comparisons["volume_drift"]["max_relative_abs"], 0.30)
        self.assertIsNone(comparisons["volume_drift"]["threshold"])
        self.assertEqual(comparisons["volume_drift"]["role"], "diagnostic_only")
        self.assertFalse(gates["proxy_to_projected_volume_change"]["required"])
        self.assertEqual(build_decision(gates, "projected.vtp")["status"], "accepted")

    def test_volume_diagnostic_requires_reliable_corresponding_components(self) -> None:
        points, faces = tetrahedron()
        metrics = mesh_report(points, faces)
        missing = volume_drift(metrics, {**metrics, "volume": {"reliable": False}})

        self.assertEqual(missing["status"], "unavailable")
        self.assertEqual(missing["role"], "diagnostic_only")

    def test_bbox_drift_uses_reported_closure_resolution(self) -> None:
        source_points, faces = tetrahedron()
        candidate_points = source_points + np.asarray([0.006, 0.0, 0.0])
        source_metrics = mesh_report(source_points, faces)
        candidate_metrics = mesh_report(candidate_points, faces)
        report = projection_report(candidate_points.shape[0])
        report["thresholds"]["resolved_max_projection_distance"] = 0.024
        report["resolution_context"] = {
            "closure_pitch": 0.008,
            "projection_distance_in_closure_pitches": 3.0,
        }

        comparisons = projection_comparisons(
            source_metrics,
            source_metrics,
            candidate_metrics,
            report,
            {"summary": {"changed_ratio_max": 0.0, "overlap_ratio_min": 1.0}},
            {"status": "computed", "passed": True},
        )
        bbox = comparisons["bbox_drift"]

        self.assertAlmostEqual(bbox["max_ratio"], 0.003)
        self.assertAlmostEqual(bbox["base_threshold_ratio"], 0.002)
        self.assertAlmostEqual(bbox["resolution_threshold_ratio"], 0.004)
        self.assertAlmostEqual(bbox["effective_threshold_ratio"], 0.004)
        self.assertEqual(bbox["resolution_source"], "reported_closure_pitch")
        self.assertFalse(bbox["hard_cap_applied"])
        self.assertTrue(bbox["passed"])

    def test_bbox_resolution_allowance_never_exceeds_hard_cap(self) -> None:
        source_points, faces = tetrahedron()
        candidate_points = source_points + np.asarray([0.012, 0.0, 0.0])
        source_metrics = mesh_report(source_points, faces)
        candidate_metrics = mesh_report(candidate_points, faces)
        report = projection_report(candidate_points.shape[0])
        report["thresholds"]["resolved_max_projection_distance"] = 0.12
        report["resolution_context"] = {
            "closure_pitch": 0.04,
            "projection_distance_in_closure_pitches": 3.0,
        }

        comparisons = projection_comparisons(
            source_metrics,
            source_metrics,
            candidate_metrics,
            report,
            {"summary": {"changed_ratio_max": 0.0, "overlap_ratio_min": 1.0}},
            {"status": "computed", "passed": True},
        )
        bbox = comparisons["bbox_drift"]

        self.assertAlmostEqual(bbox["max_ratio"], 0.006)
        self.assertAlmostEqual(bbox["effective_threshold_ratio"], 0.005)
        self.assertTrue(bbox["hard_cap_applied"])
        self.assertFalse(bbox["passed"])

    def test_bbox_drift_keeps_base_threshold_without_resolution_evidence(self) -> None:
        source_points, faces = tetrahedron()
        candidate_points = source_points + np.asarray([0.006, 0.0, 0.0])
        source_metrics = mesh_report(source_points, faces)
        candidate_metrics = mesh_report(candidate_points, faces)
        report = projection_report(candidate_points.shape[0])
        report["thresholds"] = {}

        comparisons = projection_comparisons(
            source_metrics,
            source_metrics,
            candidate_metrics,
            report,
            {"summary": {"changed_ratio_max": 0.0, "overlap_ratio_min": 1.0}},
            {"status": "computed", "passed": True},
        )
        bbox = comparisons["bbox_drift"]

        self.assertEqual(bbox["resolution_source"], "unavailable")
        self.assertAlmostEqual(bbox["effective_threshold_ratio"], 0.002)
        self.assertFalse(bbox["passed"])

    def test_incomplete_self_intersection_check_rejects(self) -> None:
        points, faces = tetrahedron()
        metrics = mesh_report(points, faces)
        report = projection_report(
            points.shape[0],
            {"status": "incomplete_candidate_pair_limit_exceeded", "passed": False},
        )
        comparisons = projection_comparisons(
            metrics,
            metrics,
            metrics,
            report,
            {"summary": {"changed_ratio_max": 0.0, "overlap_ratio_min": 1.0}},
            {"status": "computed", "passed": True},
        )
        gates = build_projection_gates(
            metrics,
            comparisons,
            [],
            "projected.vtp",
            True,
            "proxy.vtp",
            report,
            watertight_shell_policy("watertight-exterior-shell"),
        )

        self.assertFalse(gates["self_intersection_free"]["passed"])
        self.assertEqual(build_decision(gates, "projected.vtp")["status"], "rejected")

    def test_projected_volume_below_resolution_erosion_core_rejects(self) -> None:
        points, faces = tetrahedron()
        metrics = mesh_report(points, faces)
        report = projection_report(points.shape[0])
        candidate_volume = float(metrics["volume"]["signed_abs"])
        construction = report["sealed_exterior_construction"]
        construction["filled_or_shell_voxels"] = 40_000
        construction["estimated_filled_volume"] = 5.0
        construction["projection_erosion_core"]["filled_voxels"] = 30_000
        construction["projection_erosion_core"]["estimated_volume"] = 3.75
        construction["projection_erosion_core"]["mesh_signed_abs_volume"] = 3.75
        self.assertLess(candidate_volume, 3.75)
        comparisons = projection_comparisons(
            metrics,
            metrics,
            metrics,
            report,
            {
                "summary": {"changed_ratio_max": 0.0, "overlap_ratio_min": 1.0},
                "per_view": {},
            },
            {"status": "computed", "passed": True},
        )
        gates = build_projection_gates(
            metrics,
            comparisons,
            [],
            "projected.vtp",
            True,
            "proxy.vtp",
            report,
            watertight_shell_policy("watertight-exterior-shell"),
        )

        occupancy = comparisons["sealed_exterior_occupancy"]
        self.assertEqual(occupancy["status"], "computed")
        self.assertFalse(occupancy["passed"])
        self.assertFalse(gates["sealed_exterior_erosion_core_preserved"]["passed"])
        decision = build_decision(gates, "projected.vtp")
        self.assertEqual(decision["status"], "rejected")
        self.assertIn("sealed_exterior_core_failed", decision["reason_codes"])

    def test_missing_or_mismatched_erosion_core_fails_closed(self) -> None:
        points, faces = tetrahedron()
        metrics = mesh_report(points, faces)
        report = projection_report(points.shape[0])
        report["sealed_exterior_construction"]["projection_erosion_core"][
            "erosion_radius_voxels"
        ] = 1
        comparisons = projection_comparisons(
            metrics,
            metrics,
            metrics,
            report,
            {
                "summary": {"changed_ratio_max": 0.0, "overlap_ratio_min": 1.0},
                "per_view": {},
            },
            {"status": "computed", "passed": True},
        )

        occupancy = comparisons["sealed_exterior_occupancy"]
        self.assertEqual(occupancy["status"], "required_metric_missing")
        self.assertFalse(occupancy["passed"])
        self.assertIn("does not match", occupancy["failure_reason"])

    def test_roundtrip_and_chunked_intersection_validation(self) -> None:
        points, faces = tetrahedron()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.vtp"
            write_vtp(path, points, faces)
            roundtrip = roundtrip_validation(path, points, faces)

        intersections = chunked_self_intersection_report(points, faces, focus_chunk_size=2)
        repaired, proxy_repair = repair_closure_proxy_self_intersections(points, faces, 0.1)

        self.assertTrue(roundtrip["passed"], roundtrip)
        self.assertTrue(intersections["passed"], intersections)
        self.assertEqual(intersections["status"], "computed")
        self.assertTrue(proxy_repair["passed"], proxy_repair)
        self.assertTrue(np.array_equal(repaired, points))

    def test_broad_projection_uses_complete_intersection_scan(self) -> None:
        points, faces = tetrahedron()
        provenance = {
            "point_data": {
                # Three projected vertices touch every tetrahedron face.  A
                # focused scan would retain a needless pair-de-duplication set.
                "source_projection_applied": np.asarray([1, 1, 1, 0], dtype=np.uint8),
            }
        }
        computed = {
            "status": "computed",
            "passed": True,
            "intersection_pairs": 0,
            "reported_pairs": [],
        }
        with patch(
            "source_projected_validation.self_intersection_report",
            return_value=computed,
        ) as scanner:
            report = projected_delta_self_intersection_report(
                points,
                faces,
                provenance,
                {"status": "computed", "passed": True},
            )

        self.assertIsNone(scanner.call_args.kwargs["focus_face_ids"])
        self.assertEqual(report["scan_scope"], "all_faces")
        self.assertIn("complete candidate mesh", report["proof"])


if __name__ == "__main__":
    unittest.main()
