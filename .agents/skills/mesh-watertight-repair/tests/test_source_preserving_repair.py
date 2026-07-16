from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from mesh_metrics import mesh_report  # noqa: E402
from source_preserving_repair import (  # noqa: E402
    accepted_outputs,
    array_sha256,
    build_decision,
    build_gates,
    build_ignored_outputs,
    policy_resolution_gate,
    run_deterministic_repair,
)


def dirty_triangle_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 1, 2], [0, 0, 0]], dtype=np.int64)
    return points, faces, np.asarray([10, 11, 12], dtype=np.int64)


def tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    return points, faces


def passing_comparisons() -> dict:
    return {
        "source_distance": {"status": "computed", "passed": True, "max": 0.0, "threshold": 1.0},
        "bbox_drift": {"status": "computed", "passed": True, "max_ratio": 0.0, "threshold": 0.01},
        "silhouette_drift": {"status": "computed", "passed": True, "changed_ratio_max": 0.0, "threshold": 0.01},
        "self_intersection": {"status": "computed", "passed": True, "intersection_pairs": 0},
    }


class SourcePreservingRepairTest(unittest.TestCase):
    def test_deterministic_cleanup_removes_only_degenerate_and_exact_duplicates(self) -> None:
        points, faces, source_ids = dirty_triangle_mesh()
        repaired_points, repaired_faces, repaired_sources, passes = run_deterministic_repair(
            points, faces, source_ids
        )
        self.assertEqual(repaired_faces.shape, (1, 3))
        self.assertEqual(repaired_sources.tolist(), [10])
        self.assertEqual(repaired_points.shape, (3, 3))
        self.assertEqual([row["status"] for row in passes], ["applied", "applied"])

    def test_non_manifold_vertex_is_a_required_gate(self) -> None:
        points, faces = tetrahedron()
        metrics = mesh_report(points, faces)
        metrics["topology"]["non_manifold_vertices"] = 1
        gates = build_gates(
            metrics,
            passing_comparisons(),
            {"truncated": False},
            [],
            [],
            "candidate.vtp",
            "proxy.vtp",
            [],
        )
        self.assertFalse(gates["non_manifold_vertices_zero"]["passed"])
        self.assertTrue(gates["non_manifold_vertices_zero"]["required"])

    def test_policy_gate_distinguishes_pending_and_rejected(self) -> None:
        pending = policy_resolution_gate(pending_count=1, rejected_count=0, packet_truncated=False)
        rejected = policy_resolution_gate(pending_count=0, rejected_count=1, packet_truncated=False)
        self.assertIn("policy_review_pending", pending["reason_codes"])
        self.assertIn("policy_region_rejected", rejected["reason_codes"])

    def test_build_decision_accepts_only_when_every_required_gate_passes(self) -> None:
        accepted = build_decision({"a": {"required": True, "passed": True}}, "candidate.vtp")
        rejected = build_decision({"a": {"required": True, "passed": False}}, "candidate.vtp")
        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(accepted["final_output_path"], "candidate.vtp")
        self.assertEqual(rejected["status"], "rejected")
        self.assertIsNone(rejected["final_output_path"])

    def test_rejected_output_labels_source_fallback_as_not_engineering_ready(self) -> None:
        result = accepted_outputs(
            {"source_preserving_candidate_vtp": "source.vtp"},
            {"status": "rejected"},
            "projected.vtp",
            "projected.vtp",
            source_fallback_path="source.vtp",
        )
        self.assertFalse(result["mesh_result"]["engineering_ready"])
        self.assertEqual(result["source_fallback_vtp"], "source.vtp")
        self.assertIsNone(result["accepted_mesh_vtp"])

    def test_ignored_outputs_mark_proxy_and_stale_meshes_unsafe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            stale = output_dir / "stale.vtp"
            stale.write_text("stale", encoding="utf-8")
            rows = build_ignored_outputs(
                closure_proxy_path=str(output_dir / "proxy.vtp"),
                stage1_path=str(output_dir / "stage1.vtp"),
                candidate_path=str(output_dir / "source.vtp"),
                previews=[],
                group_filter=None,
                output_dir=output_dir,
            )
        kinds = {row["kind"] for row in rows}
        self.assertIn("closure_proxy", kinds)
        self.assertIn("stale_output_file", kinds)
        self.assertTrue(all(not row["safe_for_acceptance"] for row in rows))

    def test_array_hash_binds_dtype_shape_and_values(self) -> None:
        values = np.asarray([1, 2, 3], dtype=np.int64)
        self.assertEqual(array_sha256(values), array_sha256(values.copy()))
        self.assertNotEqual(array_sha256(values), array_sha256(values.astype(np.int32)))


if __name__ == "__main__":
    unittest.main()
