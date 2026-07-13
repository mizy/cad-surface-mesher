from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from mesh_metrics import mesh_report  # noqa: E402
from repair_inventory import build_inventory  # noqa: E402
from repair_policy import build_policy_packet, resolve_policy_decisions  # noqa: E402
from repair_report import requested_capabilities  # noqa: E402
from source_preserving_repair import (  # noqa: E402
    accepted_outputs,
    build_comparisons,
    build_decision,
    build_gates,
    build_ignored_outputs,
    build_patch_regions,
    build_unhandled_items,
    classify_patch_region,
    policy_resolution_gate,
    prune_classified_internal_components,
    run_deterministic_repair,
)
from two_stage_contract import hybrid_unhandled_items  # noqa: E402


def dirty_triangle_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 1, 2], [0, 0, 0]], dtype=np.int64)
    source_indices = np.asarray([10, 11, 12], dtype=np.int64)
    return points, faces, source_indices


def open_cube_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed cube walls and floor with one bbox-scale missing top face."""
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return points, faces, np.arange(faces.shape[0], dtype=np.int64)


def mark_first_gap_as_semantic_opening(inventory: dict) -> None:
    item = inventory["gap_regions"]["items"][0]
    item["classification"] = "pending_policy"
    item["detector_reason"] = "semantic_opening_candidate"
    item["requires_policy"] = True
    item["policy_reason_source"] = "unit test explicit semantic opening classifier"


class SourcePreservingRepairContractTest(unittest.TestCase):
    def test_non_manifold_vertex_is_an_explicit_required_gate(self) -> None:
        metrics = {
            "topology": {
                "boundary_edges": 0,
                "non_manifold_edges": 0,
                "non_manifold_vertices": 1,
                "inconsistent_winding_edges": 0,
                "components": {"count": 1},
            },
            "quality": {"degenerate_faces": 0},
            "volume": {"reliable": False, "orientation_consistent": False},
        }
        comparisons = {
            "source_distance": {"status": "computed", "passed": True, "max": 0.0},
            "bbox_drift": {"status": "computed", "passed": True, "max_ratio": 0.0},
            "silhouette_drift": {
                "status": "computed",
                "passed": True,
                "changed_ratio_max": 0.0,
            },
            "self_intersection": {
                "status": "computed",
                "passed": True,
                "intersection_pairs": 0,
            },
        }

        gates = build_gates(
            metrics,
            comparisons,
            {"items": [], "truncated": False},
            [],
            [],
            "candidate.vtp",
            "proxy.vtp",
            [],
        )

        self.assertFalse(gates["non_manifold_vertices_zero"]["passed"])
        self.assertTrue(gates["non_manifold_vertices_zero"]["required"])

    def test_policy_gate_distinguishes_pending_from_explicit_reject(self) -> None:
        pending = policy_resolution_gate(
            pending_count=2,
            rejected_count=0,
            packet_truncated=False,
        )
        rejected = policy_resolution_gate(
            pending_count=0,
            rejected_count=3,
            packet_truncated=False,
        )

        self.assertFalse(pending["passed"])
        self.assertEqual(pending["reason_codes"], ["policy_review_pending"])
        self.assertFalse(rejected["passed"])
        self.assertEqual(rejected["reason_codes"], ["policy_region_rejected"])
        self.assertEqual(rejected["rejected_count"], 3)
        self.assertEqual(
            classify_patch_region(
                "gap_regions",
                {"requires_policy": True},
                {"status": "decided", "decision": "reject"},
            ),
            ("reject_region", "reject_patch", ["policy_region_rejected"]),
        )

    def test_coincident_weld_satisfies_source_zipper_artifact_contract(self) -> None:
        gates = build_gates(
            {"topology": {"components": {"count": 1}}, "quality": {}, "volume": {}},
            {},
            {"items": []},
            [],
            [],
            "candidate.vtp",
            "proxy.vtp",
            [],
            hybrid_candidate_path="candidate.vtp",
            hybrid_candidate_produced=True,
            patch_regions=[
                {
                    "id": "region_0040",
                    "selection_status": "use_source_zipper",
                    "artifacts": {
                        "seam_belt_vtp": "region_0040_seam_belt.vtp",
                        "coincident_weld_vtp": "region_0040_coincident_weld.vtp",
                    },
                    "final_provenance": {"consumed_by_final": True},
                }
            ],
            require_hybrid_candidate=True,
        )

        self.assertTrue(gates["patch_artifacts_present"]["passed"])

    def test_policy_reject_blocks_acceptance_without_fake_pending_or_proxy_failures(self) -> None:
        gates = build_gates(
            {"topology": {"components": {"count": 1}}, "quality": {}, "volume": {}},
            {},
            {"items": [{"id": "opening_0001"}], "truncated": False},
            [{"item_id": "opening_0001", "status": "decided", "decision": "reject"}],
            [],
            "candidate.vtp",
            "proxy.vtp",
            [],
            hybrid_candidate_path="candidate.vtp",
            hybrid_candidate_produced=True,
            patch_regions=[
                {
                    "id": "region_0001",
                    "selection_status": "reject_patch",
                    "blocking": True,
                    "rejection_reason": "policy_region_rejected",
                    "artifacts": {},
                    "final_provenance": {"consumed_by_final": False},
                }
            ],
            require_hybrid_candidate=True,
        )
        decision = build_decision(gates, "candidate.vtp")

        self.assertFalse(gates["opening_policy_resolved"]["passed"])
        self.assertFalse(gates["patch_regions_resolved"]["passed"])
        self.assertTrue(gates["patch_artifacts_present"]["passed"])
        self.assertTrue(gates["face_provenance_present"]["passed"])
        self.assertIn("policy_region_rejected", decision["reason_codes"])
        self.assertNotIn("policy_review_pending", decision["reason_codes"])
        self.assertNotIn("proxy_patch_extraction_failed", decision["reason_codes"])

    def test_component_prune_requires_component_atlas_evidence(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [2.1, 0.0, 0.0], [2.0, 0.1, 0.0]],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        sources = np.asarray([10, 11], dtype=np.int64)
        scores = np.asarray([2, 0], dtype=np.int16)
        inventory = {
            "boundary_regions": {
                "items": [
                    {"component_id": 0, "classification": "large_opening_or_missing_surface"},
                    {"component_id": 1, "classification": "internal_or_fragment_component_perimeter"},
                ]
            }
        }

        next_points, next_faces, next_sources, next_scores, repair_pass = prune_classified_internal_components(
            points,
            faces,
            sources,
            scores,
            inventory,
        )

        np.testing.assert_array_equal(next_points, points)
        np.testing.assert_array_equal(next_faces, faces)
        np.testing.assert_array_equal(next_sources, sources)
        np.testing.assert_array_equal(next_scores, scores)
        self.assertEqual(repair_pass["status"], "skipped")
        self.assertEqual(repair_pass["scope"]["removed_source_triangle_count"], 0)
        self.assertEqual(
            repair_pass["thresholds"]["protected_component_reasons"]["1"],
            "missing_component_level_atlas_evidence",
        )

    def test_component_prune_accepts_explicit_closed_component_atlas_evidence(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0], [2.1, 0.0, 0.0], [2.0, 0.1, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        sources = np.asarray([10, 11], dtype=np.int64)
        scores = np.asarray([2, 0], dtype=np.int16)
        inventory = {
            "boundary_regions": {"items": [{"component_id": 0, "classification": "large_opening_or_missing_surface"}]},
            "boundary_classification": {
                "component_thresholds": {"exterior_support_threshold": 0.25},
                "component_evidence": [
                    {
                        "component_id": 1,
                        "automatic_remove_candidate": True,
                        "removal_classification": "internal_or_fragment_component_perimeter",
                        "removal_reason": "contained_low_visibility_small_component",
                        "visible_hard_keep": False,
                        "direct_and_sealed_internal_consensus": True,
                        "physical_scale_small": True,
                    }
                ],
            },
        }

        _, next_faces, next_sources, _, repair_pass = prune_classified_internal_components(
            points,
            faces,
            sources,
            scores,
            inventory,
        )

        self.assertEqual(next_faces.shape[0], 1)
        self.assertEqual(next_sources.tolist(), [10])
        self.assertEqual(repair_pass["status"], "applied")
        self.assertEqual(
            repair_pass["thresholds"]["removed_component_reasons"]["1"],
            "contained_low_visibility_small_component",
        )

    def test_component_prune_rejects_stale_remove_evidence_for_visible_component(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0], [2.1, 0.0, 0.0], [2.0, 0.1, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        sources = np.asarray([10, 11], dtype=np.int64)
        scores = np.asarray([2, 1], dtype=np.int16)
        inventory = {
            "boundary_regions": {"items": []},
            "boundary_classification": {
                "component_evidence": [
                    {
                        "component_id": 1,
                        "automatic_remove_candidate": True,
                        "removal_classification": "internal_or_fragment_component_perimeter",
                        "visible_hard_keep": False,
                        "direct_and_sealed_internal_consensus": True,
                        "physical_scale_small": True,
                    }
                ]
            },
        }

        next_points, next_faces, next_sources, next_scores, repair_pass = (
            prune_classified_internal_components(
                points,
                faces,
                sources,
                scores,
                inventory,
            )
        )

        np.testing.assert_array_equal(next_points, points)
        np.testing.assert_array_equal(next_faces, faces)
        np.testing.assert_array_equal(next_sources, sources)
        np.testing.assert_array_equal(next_scores, scores)
        self.assertEqual(repair_pass["status"], "skipped")
        self.assertEqual(
            repair_pass["thresholds"]["protected_component_reasons"]["1"],
            "multi_view_first_hit_hard_keep",
        )

    def test_component_prune_keeps_whole_component_when_any_face_is_visible(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [2.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [3, 4, 5], [3, 5, 6]], dtype=np.int64)
        sources = np.asarray([10, 11, 12], dtype=np.int64)
        scores = np.asarray([2, 0, 1], dtype=np.int16)
        inventory = {
            "boundary_regions": {"items": []},
            "boundary_classification": {
                "component_evidence": [
                    {
                        "component_id": 1,
                        "automatic_remove_candidate": True,
                        "removal_classification": "internal_or_fragment_component_perimeter",
                        "visible_hard_keep": False,
                        "direct_and_sealed_internal_consensus": True,
                        "physical_scale_small": True,
                    }
                ]
            },
        }

        _, next_faces, next_sources, next_scores, repair_pass = (
            prune_classified_internal_components(
                points,
                faces,
                sources,
                scores,
                inventory,
            )
        )

        np.testing.assert_array_equal(next_faces, faces)
        np.testing.assert_array_equal(next_sources, sources)
        np.testing.assert_array_equal(next_scores, scores)
        self.assertEqual(repair_pass["status"], "skipped")
        self.assertEqual(
            repair_pass["thresholds"]["protected_component_reasons"]["1"],
            "multi_view_first_hit_hard_keep",
        )

    def test_deterministic_repair_records_skipped_safe_ops(self) -> None:
        points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        source_indices = np.asarray([7], dtype=np.int64)

        _, _, candidate_sources, passes = run_deterministic_repair(points, faces, source_indices)

        self.assertEqual(candidate_sources.tolist(), [7])
        self.assertEqual([row["status"] for row in passes], ["skipped", "skipped"])
        self.assertEqual([row["scope"]["removed_source_triangle_count"] for row in passes], [0, 0])

    def test_deterministic_repair_removes_only_degenerate_and_exact_duplicate_faces(self) -> None:
        points, faces, source_indices = dirty_triangle_mesh()

        candidate_points, candidate_faces, candidate_sources, passes = run_deterministic_repair(
            points,
            faces,
            source_indices,
        )

        self.assertEqual(candidate_faces.shape[0], 1)
        self.assertEqual(candidate_sources.tolist(), [10])
        self.assertEqual([row["status"] for row in passes], ["applied", "applied"])
        self.assertEqual(candidate_points.shape[0], 3)

    def test_closure_proxy_and_pending_policy_cannot_be_accepted(self) -> None:
        points, faces, source_indices = open_cube_mesh()
        candidate_points, candidate_faces, candidate_sources, passes = run_deterministic_repair(
            points,
            faces,
            source_indices,
        )
        candidate_metrics = mesh_report(candidate_points, candidate_faces)
        inventory = build_inventory(
            candidate_points,
            candidate_faces,
            candidate_sources,
            "deterministic_repair_candidate",
            max_items=10,
        )
        packet = build_policy_packet(inventory, "watertight-exterior-shell", max_items=10)
        policy_decisions = resolve_policy_decisions(packet, [])
        comparisons = build_comparisons(candidate_metrics, candidate_metrics, {"summary": {"changed_ratio_max": 0.0}})
        unhandled = build_unhandled_items(inventory, packet, has_policy_file=False)
        gates = build_gates(
            candidate_metrics,
            comparisons,
            packet,
            policy_decisions,
            passes,
            "outputs/run/source_preserving_candidate.vtp",
            "outputs/run/closure_proxy.vtp",
            unhandled,
        )

        decision = build_decision(gates, "outputs/run/source_preserving_candidate.vtp")
        outputs = accepted_outputs(
            {"closure_proxy_vtp": "outputs/run/closure_proxy.vtp"},
            decision,
            "outputs/run/source_preserving_candidate.vtp",
        )

        self.assertEqual(decision["status"], "rejected")
        self.assertIsNone(outputs["accepted_mesh_vtp"])
        self.assertEqual(outputs["closure_proxy_vtp"], "outputs/run/closure_proxy.vtp")
        self.assertIn("topology_gate_failed", decision["reason_codes"])
        self.assertIn("required_metric_missing", decision["reason_codes"])
        self.assertIn("policy_review_pending", decision["reason_codes"])

    def test_large_exterior_boundary_requires_semantic_policy_before_patch(self) -> None:
        points, faces, source_indices = open_cube_mesh()
        candidate_points, candidate_faces, candidate_sources, _ = run_deterministic_repair(points, faces, source_indices)
        inventory = build_inventory(
            candidate_points,
            candidate_faces,
            candidate_sources,
            "deterministic_repair_candidate",
            max_items=10,
        )

        packet = build_policy_packet(inventory, "watertight-exterior-shell", max_items=10)

        self.assertEqual(len(packet["items"]), 1)
        self.assertTrue(inventory["boundary_regions"]["items"][0]["requires_policy"])
        self.assertEqual(
            inventory["boundary_regions"]["items"][0]["classification"],
            "large_opening_or_missing_surface",
        )
        self.assertEqual(
            inventory["boundary_regions"]["items"][0]["detector_reason"],
            "exterior_loop_requires_semantic_opening_policy",
        )
        self.assertFalse(inventory["boundary_regions"]["items"][0]["patch_eligible"])
        self.assertIn("component_id", inventory["boundary_regions"]["items"][0])
        self.assertIn("local_scale", inventory["boundary_regions"]["items"][0])
        self.assertIn("nearby_region_ids", inventory["boundary_regions"]["items"][0])

    def test_inventory_report_limit_never_truncates_geometry_truth(self) -> None:
        points, faces, source_indices = dirty_triangle_mesh()
        candidate_points, candidate_faces, candidate_sources, passes = run_deterministic_repair(
            points,
            faces,
            source_indices,
        )
        candidate_metrics = mesh_report(candidate_points, candidate_faces)
        inventory = build_inventory(
            candidate_points,
            candidate_faces,
            candidate_sources,
            "deterministic_repair_candidate",
            max_items=0,
        )
        packet = build_policy_packet(inventory, "watertight-exterior-shell", max_items=10)
        comparisons = {
            "source_distance": {"status": "computed", "passed": True, "max": 0.0, "threshold": 1.0},
            "bbox_drift": {"status": "computed", "passed": True, "max_ratio": 0.0, "threshold": 1.0},
            "silhouette_drift": {"status": "computed", "passed": True, "changed_ratio_max": 0.0, "threshold": 1.0},
        }

        unhandled = hybrid_unhandled_items(
            inventory,
            packet,
            has_policy_file=False,
            patch_regions=[],
            comparisons=comparisons,
        )
        gates = build_gates(
            candidate_metrics,
            comparisons,
            packet,
            [],
            passes,
            "outputs/run/hybrid_fused_candidate.vtp",
            "outputs/run/closure_proxy.vtp",
            unhandled,
        )
        decision = build_decision(gates, "outputs/run/hybrid_fused_candidate.vtp")

        section = inventory["boundary_regions"]
        self.assertFalse(section["truncated"])
        self.assertTrue(section["geometry_truth_complete"])
        self.assertFalse(section["report_limit_applied_to_geometry"])
        self.assertEqual(section["reported_regions"], section["total_regions"])
        self.assertEqual(len(section["items"]), section["total_regions"])
        self.assertNotIn("inventory_after.boundary_regions", {item["item"] for item in unhandled})
        self.assertNotIn("inventory_truncated", decision["reason_codes"])
        self.assertNotIn("opening_inventory_unresolved", decision["reason_codes"])

    def test_hybrid_contract_requires_current_run_final_and_patch_resolution(self) -> None:
        points, faces, source_indices = dirty_triangle_mesh()
        candidate_points, candidate_faces, candidate_sources, passes = run_deterministic_repair(
            points,
            faces,
            source_indices,
        )
        candidate_metrics = mesh_report(candidate_points, candidate_faces)
        inventory = build_inventory(
            candidate_points,
            candidate_faces,
            candidate_sources,
            "deterministic_repair_candidate",
            max_items=10,
        )
        packet = build_policy_packet(inventory, "watertight-exterior-shell", max_items=10)
        policy_decisions = resolve_policy_decisions(packet, [])
        comparisons = build_comparisons(candidate_metrics, candidate_metrics, {"summary": {"changed_ratio_max": 0.0}})
        unhandled = build_unhandled_items(inventory, packet, has_policy_file=False)
        patch_regions = build_patch_regions(inventory, policy_decisions, patch_dir="outputs/run/patches")

        gates = build_gates(
            candidate_metrics,
            comparisons,
            packet,
            policy_decisions,
            passes,
            "outputs/run/hybrid_fused_candidate.vtp",
            "outputs/run/closure_proxy.vtp",
            unhandled,
            hybrid_candidate_path="outputs/run/hybrid_fused_candidate.vtp",
            hybrid_candidate_produced=False,
            patch_regions=patch_regions,
            require_hybrid_candidate=True,
        )
        decision = build_decision(gates, "outputs/run/hybrid_fused_candidate.vtp")

        self.assertFalse(gates["hybrid_candidate_present"]["passed"])
        self.assertFalse(gates["patch_regions_resolved"]["passed"])
        self.assertIn("hybrid_candidate_missing", decision["reason_codes"])
        self.assertIn("patch_region_rejected", decision["reason_codes"])
        self.assertTrue(
            all(
                row["classification"] in {"patch_required", "reject_region", "pending_policy"}
                for row in patch_regions
            )
        )

    def test_missing_topology_or_drift_metrics_reject_fail_closed(self) -> None:
        gates = build_gates(
            {"topology": {"components": {}}, "quality": {}},
            {"source_distance": {"status": "not_implemented", "passed": False}},
            {"items": []},
            [],
            [],
            "outputs/run/hybrid_fused_candidate.vtp",
            "outputs/run/closure_proxy.vtp",
            [],
            hybrid_candidate_path="outputs/run/hybrid_fused_candidate.vtp",
            hybrid_candidate_produced=True,
            patch_regions=[],
            require_hybrid_candidate=True,
        )

        decision = build_decision(gates, "outputs/run/hybrid_fused_candidate.vtp")

        self.assertEqual(decision["status"], "rejected")
        self.assertFalse(gates["boundary_edges_zero"]["passed"])
        self.assertFalse(gates["non_manifold_edges_zero"]["passed"])
        self.assertFalse(gates["degenerate_faces_zero"]["passed"])
        self.assertFalse(gates["required_metrics_present"]["passed"])
        self.assertIn("topology_gate_failed", decision["reason_codes"])
        self.assertIn("degenerate_gate_failed", decision["reason_codes"])
        self.assertIn("required_metric_missing", decision["reason_codes"])

    def test_policy_decision_rerun_marks_packet_item_decided(self) -> None:
        points, faces, source_indices = dirty_triangle_mesh()
        candidate_points, candidate_faces, candidate_sources, _ = run_deterministic_repair(points, faces, source_indices)
        inventory = build_inventory(
            candidate_points,
            candidate_faces,
            candidate_sources,
            "deterministic_repair_candidate",
            max_items=10,
        )
        mark_first_gap_as_semantic_opening(inventory)
        packet = build_policy_packet(
            inventory,
            "watertight-exterior-shell",
            max_items=10,
            run_context={"input": "mesh_a.vtp", "visibility_grid": 720},
        )
        item = packet["items"][0]

        decisions = resolve_policy_decisions(
            packet,
            [
                {
                    "item_id": item["id"],
                    "status": "decided",
                    "decision": "preserve",
                    "semantic_label": "functional_opening",
                    "shape_prior": "none",
                    "run_fingerprint": packet["run_fingerprint"],
                    "packet_fingerprint": packet["packet_fingerprint"],
                    "item_fingerprint": item["item_fingerprint"],
                }
            ],
        )

        self.assertEqual(decisions[0]["status"], "decided")
        self.assertEqual(decisions[0]["decision"], "preserve")
        self.assertEqual(decisions[0]["semantic_label"], "functional_opening")
        self.assertEqual(decisions[0]["shape_prior"], "none")
        self.assertIn("dimensionless_features", item)
        self.assertIn("shape_prior_options", item)

    def test_stale_policy_decision_is_pending_review(self) -> None:
        points, faces, source_indices = dirty_triangle_mesh()
        candidate_points, candidate_faces, candidate_sources, _ = run_deterministic_repair(points, faces, source_indices)
        inventory = build_inventory(
            candidate_points,
            candidate_faces,
            candidate_sources,
            "deterministic_repair_candidate",
            max_items=10,
        )
        mark_first_gap_as_semantic_opening(inventory)
        packet = build_policy_packet(
            inventory,
            "watertight-exterior-shell",
            max_items=10,
            run_context={"input": "mesh_b.vtp", "visibility_grid": 900},
        )
        item = packet["items"][0]

        decisions = resolve_policy_decisions(
            packet,
            [
                {
                    "item_id": item["id"],
                    "status": "decided",
                    "decision": "preserve",
                    "run_fingerprint": "old-run",
                    "packet_fingerprint": packet["packet_fingerprint"],
                    "item_fingerprint": item["item_fingerprint"],
                }
            ],
        )

        self.assertEqual(decisions[0]["status"], "pending_review")
        self.assertIsNone(decisions[0]["reviewer"])
        self.assertTrue(decisions[0]["stale_policy_decision"])
        self.assertIn("mismatched_run_fingerprint", decisions[0]["stale_reasons"])

    def test_policy_packet_carries_semantic_shape_prior_contract(self) -> None:
        inventory = {
            "boundary_regions": {
                "items": [
                    {
                        "id": "boundary_loop_0042",
                        "classification": "large_opening_or_missing_surface",
                        "requires_policy": True,
                        "operator": "proxy_conformal_patch_after_cap_decision",
                        "source_triangle_ids": [10, 11],
                        "source_triangle_count": 2,
                        "bbox": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.1]},
                        "centroid": [0.5, 0.5, 0.0],
                        "normal": [0.0, 0.0, 1.0],
                        "dimensionless_features": {
                            "diameter_bbox_ratio": 0.08,
                            "compactness": 0.7,
                        },
                        "evidence_views": ["/tmp/boundary_loop_0042.png"],
                    }
                ]
            }
        }
        packet = build_policy_packet(inventory, "watertight-exterior-shell", max_items=10)
        item = packet["items"][0]
        supplied = {
            **item["decision_template"],
            "decision": "cap",
            "semantic_label": "missing_cover",
            "shape_prior": "voxel_sdf",
            "confidence": 0.91,
        }

        decisions = resolve_policy_decisions(packet, [supplied])

        self.assertEqual(item["recommended_operator"], "proxy_conformal_patch_after_cap_decision")
        self.assertEqual(item["dimensionless_features"]["compactness"], 0.7)
        self.assertEqual(item["evidence_views"], ["/tmp/boundary_loop_0042.png"])
        self.assertEqual(decisions[0]["status"], "decided")
        self.assertEqual(decisions[0]["semantic_label"], "missing_cover")
        self.assertEqual(decisions[0]["shape_prior"], "voxel_sdf")

    def test_ignored_outputs_marks_proxy_and_previews_unsafe(self) -> None:
        patch_regions = [
            {
                "artifacts": {
                    "proxy_patch_vtp": "outputs/run/patches/region_0001_proxy_patch.vtp",
                    "seam_belt_vtp": "outputs/run/patches/region_0001_seam_belt.vtp",
                    "stitch_band_vtp": "outputs/run/patches/region_0001_stitch_band.vtp",
                    "selection_visuals": ["outputs/run/visual/region_0001_selection.png"],
                }
            }
        ]
        ignored = build_ignored_outputs(
            closure_proxy_path="outputs/run/closure_proxy.vtp",
            stage1_path="outputs/run/stage1_exterior_candidate.vtp",
            candidate_path="outputs/run/source_preserving_candidate.vtp",
            previews=["outputs/run/visual/source_candidate_plus_z_depth.png"],
            group_filter={"output_vtp": "outputs/run/stage0_group_filtered.vtp"},
            hybrid_candidate_path="outputs/run/hybrid_fused_candidate.vtp",
            patch_regions=patch_regions,
            debug_artifacts=["outputs/run/debug/hybrid_fusion_trace.json"],
        )

        self.assertEqual(ignored[0]["kind"], "closure_proxy")
        self.assertIn("rejected_hybrid_fused_candidate", {row["kind"] for row in ignored})
        self.assertIn("patch_only_artifact", {row["kind"] for row in ignored})
        self.assertIn("debug_artifact", {row["kind"] for row in ignored})
        self.assertTrue(all(not row["safe_for_acceptance"] for row in ignored))

    def test_ignored_outputs_marks_stale_mesh_files_unsafe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            closure_proxy = output_dir / "closure_proxy.vtp"
            stage1 = output_dir / "stage1_exterior_candidate.vtp"
            candidate = output_dir / "source_preserving_candidate.vtp"
            stale = output_dir / "stage2_watertight_surface.vtp"
            for path in (closure_proxy, stage1, candidate, stale):
                path.touch()

            ignored = build_ignored_outputs(
                closure_proxy_path=str(closure_proxy),
                stage1_path=str(stage1),
                candidate_path=str(candidate),
                previews=[],
                group_filter=None,
                output_dir=output_dir,
            )

        stale_rows = [row for row in ignored if row["path"].endswith("stage2_watertight_surface.vtp")]
        self.assertEqual(len(stale_rows), 1)
        self.assertEqual(stale_rows[0]["kind"], "stale_output_file")
        self.assertFalse(stale_rows[0]["safe_for_acceptance"])

    def test_gap_capabilities_do_not_claim_individual_classification_without_mapping(self) -> None:
        points, faces, source_indices = dirty_triangle_mesh()
        candidate_points, candidate_faces, _, _ = run_deterministic_repair(points, faces, source_indices)
        stage1_metrics = mesh_report(points, faces)
        candidate_metrics = mesh_report(candidate_points, candidate_faces)

        capabilities = requested_capabilities(
            stage1_metrics,
            candidate_metrics,
            candidate_metrics,
            candidate_metrics,
        )

        for name in ("part_self_gap_closure", "between_part_gap_closure", "micro_hole_filling"):
            with self.subTest(name=name):
                self.assertFalse(capabilities[name]["classified_individually"])
                self.assertEqual(capabilities[name]["classification_status"], "not_individually_classified")
                self.assertFalse(capabilities[name]["repaired"])
                self.assertEqual(capabilities[name]["region_ids"], [])

    def test_rejected_outputs_explicitly_label_unrepaired_source_fallback(self) -> None:
        outputs = accepted_outputs(
            {"source_preserving_candidate_vtp": "source_candidate.vtp"},
            {"status": "rejected"},
            "hybrid_fused_candidate.vtp",
            "hybrid_fused_candidate.vtp",
            source_fallback_path="source_candidate.vtp",
        )

        self.assertIsNone(outputs["accepted_mesh_vtp"])
        self.assertFalse(outputs["accepted_mesh_available"])
        self.assertEqual(outputs["source_fallback_vtp"], "source_candidate.vtp")
        self.assertEqual(outputs["mesh_result"]["status"], "repair_rejected_with_source_fallback")
        self.assertEqual(outputs["mesh_result"]["role"], "source_preserving_unrepaired_fallback")
        self.assertFalse(outputs["mesh_result"]["engineering_ready"])

if __name__ == "__main__":
    unittest.main()
