from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from source_primary_patch_contract import (  # noqa: E402
    BoundaryMapping,
    finalize_patch_candidate,
)
from source_primary_planar_patch import build_planar_patch_candidate  # noqa: E402
from source_primary_quality import audit_source_primary_patch  # noqa: E402
from source_primary_transaction import run_patch_transactions  # noqa: E402
from source_primary_transaction_jobs import PatchTransactionJob  # noqa: E402


def square_annulus(x_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outer = np.asarray(
        [
            [-2.0, -2.0, 0.0],
            [2.0, -2.0, 0.0],
            [2.0, 2.0, 0.0],
            [-2.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    inner = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    points = np.vstack([outer, inner])
    points[:, 0] += x_offset
    faces = []
    for index in range(4):
        following = (index + 1) % 4
        faces.extend(
            [
                [index, following, 4 + following],
                [index, 4 + following, 4 + index],
            ]
        )
    # This is the direction induced by the +Z source-face winding.
    loop = np.asarray([4, 7, 6, 5], dtype=np.int64)
    return points, np.asarray(faces, dtype=np.int64), loop


def two_annuli() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]
]:
    left_points, left_faces, left_loop = square_annulus()
    right_points, right_faces, right_loop = square_annulus(10.0)
    points = np.vstack([left_points, right_points])
    faces = np.vstack([left_faces, right_faces + left_points.shape[0]])
    loops = (left_loop, right_loop + left_points.shape[0])
    source_triangle_index = np.arange(1000, 1000 + faces.shape[0], dtype=np.int64)
    return points, faces, source_triangle_index, loops


def planar_fan_candidate(
    points: np.ndarray,
    faces: np.ndarray,
    source_triangle_index: np.ndarray,
    loop: np.ndarray,
    *,
    region_id: int,
    oriented_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    candidate_vertex_ids: tuple[int, ...] | None = None,
):
    candidate = build_planar_patch_candidate(
        points,
        faces,
        source_triangle_index,
        loop,
        region_id=region_id,
        oriented_normal=np.asarray(oriented_normal, dtype=np.float64),
    )
    if candidate_vertex_ids is None or not candidate.boundary_mapping:
        return candidate
    source_mapping = candidate.boundary_mapping[0]
    mapping = BoundaryMapping(
        region_id=source_mapping.region_id,
        source_vertex_ids=source_mapping.source_vertex_ids,
        candidate_vertex_ids=candidate_vertex_ids,
        source_edge_face_ids=source_mapping.source_edge_face_ids,
        source_triangle_indices=source_mapping.source_triangle_indices,
    )
    return finalize_patch_candidate(
        points,
        faces,
        source_triangle_index,
        method=candidate.method,
        delta=candidate.delta,
        boundary_mapping=(mapping,),
        normal=candidate.normal,
        curvature=candidate.curvature,
        proxy_provenance=candidate.proxy_provenance,
        diagnostics=candidate.diagnostics,
    )


def cap_faces_without_new_points() -> np.ndarray:
    return np.asarray([[4, 5, 6], [4, 6, 7]], dtype=np.int64)


class SourcePrimaryTransactionTest(unittest.TestCase):
    def test_commit_keeps_boundary_indices_and_every_source_prefix_bitwise_fixed(
        self,
    ) -> None:
        points, faces, loop = square_annulus()
        source_triangle_index = np.arange(200, 200 + faces.shape[0], dtype=np.int64)
        repair_mask = np.zeros(points.shape[0], dtype=np.uint8)
        repair_mask[loop] = 1
        source_point_data = {
            "source_vertex_index": np.arange(points.shape[0], dtype=np.int64),
            "repair_mask": repair_mask,
        }
        source_cell_data = {
            "material_id": np.arange(faces.shape[0], dtype=np.int32) + 50,
            "external_direction": np.repeat(
                np.asarray([[0.0, 0.0, 1.0]]), faces.shape[0], axis=0
            ),
        }
        source_point_bytes = points.tobytes()
        source_face_bytes = faces.tobytes()
        source_index_bytes = source_triangle_index.tobytes()
        candidate = planar_fan_candidate(
            points,
            faces,
            source_triangle_index,
            loop,
            region_id=11,
        )

        result = run_patch_transactions(
            points,
            faces,
            source_triangle_index,
            (candidate,),
            source_point_data=source_point_data,
            source_cell_data=source_cell_data,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["transactions"][0]["transaction_status"], "committed")
        self.assertEqual(
            result["points"][: points.shape[0]].tobytes(), source_point_bytes
        )
        self.assertEqual(result["faces"][: faces.shape[0]].tobytes(), source_face_bytes)
        self.assertEqual(
            result["cell_data"]["source_triangle_index"][: faces.shape[0]].tobytes(),
            source_index_bytes,
        )
        np.testing.assert_array_equal(
            result["cell_data"]["source_triangle_index"][faces.shape[0] :],
            np.full(candidate.delta.appended_faces.shape[0], -1, dtype=np.int64),
        )

        displacement = np.linalg.norm(
            result["points"][: points.shape[0]] - points, axis=1
        )
        outside_mask = repair_mask == 0
        self.assertEqual(float(displacement[outside_mask].max()), 0.0)
        self.assertEqual(
            result["points"][: points.shape[0]][outside_mask].tobytes(),
            points[outside_mask].tobytes(),
        )
        np.testing.assert_array_equal(
            result["point_data"]["source_vertex_index"][: points.shape[0]],
            np.arange(points.shape[0], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            result["point_data"]["repair_mask"][: points.shape[0]], repair_mask
        )

        generated_faces = result["faces"][faces.shape[0] :]
        generated_source_ids = np.unique(
            generated_faces[generated_faces < points.shape[0]]
        )
        np.testing.assert_array_equal(np.sort(generated_source_ids), np.sort(loop))
        self.assertEqual(
            tuple(candidate.boundary_mapping[0].source_vertex_ids),
            tuple(candidate.boundary_mapping[0].candidate_vertex_ids),
        )

    def test_candidate_and_committed_transaction_have_complete_row_aligned_provenance(
        self,
    ) -> None:
        points, faces, loop = square_annulus()
        source_triangle_index = np.arange(300, 300 + faces.shape[0], dtype=np.int64)
        candidate = planar_fan_candidate(
            points,
            faces,
            source_triangle_index,
            loop,
            region_id=12,
        )

        candidate_payload = candidate.to_dict()
        required_candidate_fields = {
            "status",
            "method",
            "delta",
            "boundary_mapping",
            "normal",
            "curvature",
            "source_provenance",
            "proxy_provenance",
            "diagnostics",
            "failure_reason_codes",
        }
        self.assertTrue(required_candidate_fields.issubset(candidate_payload))
        required_source_fields = {
            "point_count",
            "face_count",
            "source_triangle_index_count",
            "points_sha256",
            "faces_sha256",
            "source_triangle_index_sha256",
            "source_points_unchanged",
            "source_faces_unchanged",
            "source_triangle_index_unchanged",
            "mutation_policy",
        }
        self.assertTrue(required_source_fields.issubset(candidate.source_provenance))
        self.assertEqual(
            {"used", "role", "geometry_consumed"},
            set(candidate.proxy_provenance),
        )
        for values in candidate.delta.point_provenance.values():
            self.assertEqual(
                np.asarray(values).shape[0], candidate.delta.appended_points.shape[0]
            )
        for values in candidate.delta.face_provenance.values():
            self.assertEqual(
                np.asarray(values).shape[0], candidate.delta.appended_faces.shape[0]
            )

        result = run_patch_transactions(
            points, faces, source_triangle_index, (candidate,)
        )
        transaction = result["transactions"][0]
        self.assertEqual(
            {"source", "proxy", "appended_points", "appended_faces"},
            set(transaction["provenance"]),
        )
        for role, expected_rows in (
            ("appended_points", candidate.delta.appended_points.shape[0]),
            ("appended_faces", candidate.delta.appended_faces.shape[0]),
        ):
            for field in transaction["provenance"][role].values():
                self.assertEqual(field["shape"][0], expected_rows)
                self.assertEqual(len(field["sha256"]), 64)
        for values in result["point_data"].values():
            self.assertEqual(values.shape[0], result["points"].shape[0])
        for values in result["cell_data"].values():
            self.assertEqual(values.shape[0], result["faces"].shape[0])

    def test_non_identity_boundary_mapping_is_rejected_without_consuming_delta(
        self,
    ) -> None:
        points, faces, loop = square_annulus()
        source_triangle_index = np.arange(faces.shape[0], dtype=np.int64)
        shifted_ids = tuple(int(value) for value in np.roll(loop, 1))
        candidate = planar_fan_candidate(
            points,
            faces,
            source_triangle_index,
            loop,
            region_id=13,
            candidate_vertex_ids=shifted_ids,
        )

        self.assertNotEqual(
            tuple(candidate.boundary_mapping[0].source_vertex_ids), shifted_ids
        )
        self.assertEqual(candidate.status, "rejected")
        self.assertEqual(candidate.delta.appended_points.shape, (0, 3))
        self.assertEqual(candidate.delta.appended_faces.shape, (0, 3))
        self.assertTrue(
            any(
                "boundary mapping is not identity" in message
                for message in candidate.diagnostics["contract_validation_errors"]
            )
        )

        result = run_patch_transactions(
            points, faces, source_triangle_index, (candidate,)
        )
        self.assertEqual(result["transactions"][0]["transaction_status"], "rolled_back")
        self.assertEqual(result["points"].tobytes(), points.tobytes())
        self.assertEqual(result["faces"].tobytes(), faces.tobytes())

    def test_failed_second_hole_rolls_back_only_it_and_retains_first_commit(
        self,
    ) -> None:
        points, faces, source_triangle_index, loops = two_annuli()
        successful = planar_fan_candidate(
            points,
            faces,
            source_triangle_index,
            loops[0],
            region_id=10,
        )
        flipped = planar_fan_candidate(
            points,
            faces,
            source_triangle_index,
            loops[1],
            region_id=20,
            oriented_normal=(0.0, 0.0, -1.0),
        )

        # Reverse input order to verify that stable region ordering, not caller
        # ordering, defines which transaction is already committed on failure.
        result = run_patch_transactions(
            points,
            faces,
            source_triangle_index,
            (
                PatchTransactionJob(
                    region_ids=(20,),
                    patch_method=flipped.method,
                    build_candidate=lambda: flipped,
                ),
                PatchTransactionJob(
                    region_ids=(10,),
                    patch_method=successful.method,
                    build_candidate=lambda: successful,
                ),
            ),
            source_cell_data={
                "external_direction": np.repeat(
                    np.asarray([[0.0, 0.0, 1.0]]), faces.shape[0], axis=0
                )
            },
        )

        self.assertEqual(result["status"], "completed_with_local_rollbacks")
        self.assertEqual(
            [
                (row["region_id"], row["transaction_status"])
                for row in result["transactions"]
            ],
            [(10, "committed"), (20, "rolled_back")],
        )
        expected_points = np.vstack([points, successful.delta.appended_points])
        expected_faces = np.vstack([faces, successful.delta.appended_faces])
        np.testing.assert_array_equal(result["points"], expected_points)
        np.testing.assert_array_equal(result["faces"], expected_faces)
        self.assertEqual(result["transactions"][1]["state_revision_before"], 1)
        self.assertEqual(result["transactions"][1]["state_revision_after"], 1)
        self.assertEqual(
            result["transactions"][1]["retained_committed_region_ids_after"], [10]
        )
        self.assertIn(
            "boundary_external_normal_conflicts_source_winding",
            result["transactions"][1]["rollback"]["reason_codes"],
        )
        np.testing.assert_array_equal(
            result["cell_data"]["fusion_region_id"][faces.shape[0] :],
            np.full(successful.delta.appended_faces.shape[0], 10, dtype=np.int64),
        )
        self.assertFalse(np.any(result["cell_data"]["fusion_region_id"] == 20))


class PatchQualityRejectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.points, self.faces, self.loop = square_annulus()

    def audit(self, patch_faces: np.ndarray, expected_normal: np.ndarray):
        expected = np.repeat(
            np.asarray(expected_normal, dtype=np.float64)[None, :],
            patch_faces.shape[0],
            axis=0,
        )
        return audit_source_primary_patch(
            self.points,
            self.faces,
            np.empty((0, 3), dtype=np.float64),
            patch_faces,
            (self.loop,),
            expected_face_normals=expected,
        )

    def test_degenerate_candidate_is_rejected_by_minimum_area_gate(self) -> None:
        degenerate = np.asarray([[4, 4, 5]], dtype=np.int64)
        triangle = self.points[degenerate[0]]
        independent_double_area = np.linalg.norm(
            np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        )
        self.assertEqual(float(independent_double_area), 0.0)

        audit = self.audit(degenerate, np.asarray([0.0, 0.0, 1.0]))

        self.assertFalse(audit["passed"])
        self.assertIn("patch_triangle_area_below_minimum", audit["reason_codes"])
        self.assertFalse(audit["gates"]["triangle_minimum_area"]["passed"])

    def test_flipped_candidate_is_rejected_by_oriented_normal_gate(self) -> None:
        patch_faces = cap_faces_without_new_points()
        triangles = self.points[patch_faces]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        expected = np.asarray([0.0, 0.0, -1.0])
        np.testing.assert_array_equal(normals @ expected, np.full(2, -1.0))

        audit = self.audit(patch_faces, expected)

        self.assertFalse(audit["passed"])
        self.assertEqual(audit["reason_codes"], ["patch_winding_flipped"])
        self.assertFalse(audit["gates"]["patch_orientation"]["passed"])

    def test_non_adjacent_penetration_is_rejected_by_intersection_gate(self) -> None:
        crossing_triangle = np.asarray(
            [
                [0.25, -0.5, -1.0],
                [0.25, 0.5, 1.0],
                [0.75, -0.5, 1.0],
            ],
            dtype=np.float64,
        )
        self.points = np.vstack([self.points, crossing_triangle])
        self.faces = np.vstack([self.faces, [[8, 9, 10]]])
        first_crossing = 0.5 * (crossing_triangle[0] + crossing_triangle[1])
        second_crossing = 0.5 * (crossing_triangle[0] + crossing_triangle[2])
        np.testing.assert_array_equal(first_crossing, [0.25, 0.0, 0.0])
        np.testing.assert_array_equal(second_crossing, [0.5, -0.5, 0.0])
        self.assertTrue(np.all(np.abs(first_crossing[:2]) < 1.0))
        self.assertTrue(np.all(np.abs(second_crossing[:2]) < 1.0))

        audit = self.audit(cap_faces_without_new_points(), np.asarray([0.0, 0.0, 1.0]))

        self.assertFalse(audit["passed"])
        self.assertEqual(audit["reason_codes"], ["patch_self_intersection_detected"])
        intersection = audit["gates"]["patch_non_adjacent_intersection"]
        self.assertFalse(intersection["passed"])
        self.assertGreater(intersection["actual"]["intersection_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
