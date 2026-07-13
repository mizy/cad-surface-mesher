from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from source_primary_curved_patch import build_curved_patch_candidate  # noqa: E402
from source_primary_boundary_curvature import (  # noqa: E402
    select_boundary_owner_one_ring,
)
from source_primary_patch_geometry import analyze_source_boundary  # noqa: E402
from source_primary_transaction import run_patch_transactions  # noqa: E402
from source_primary_transaction_jobs import PatchTransactionJob  # noqa: E402


def cylinder_patch_with_rectangular_hole() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = 3.0
    theta = np.linspace(-0.45, 0.45, 7)
    height = np.linspace(-1.0, 1.0, 7)
    points = np.asarray(
        [
            [radius * np.cos(angle), radius * np.sin(angle), z]
            for angle in theta
            for z in height
        ],
        dtype=np.float64,
    )

    def point_id(theta_id: int, height_id: int) -> int:
        return theta_id * height.size + height_id

    faces: list[list[int]] = []
    for theta_id in range(theta.size - 1):
        for height_id in range(height.size - 1):
            if theta_id in {2, 3} and height_id in {2, 3}:
                continue
            a = point_id(theta_id, height_id)
            b = point_id(theta_id + 1, height_id)
            c = point_id(theta_id + 1, height_id + 1)
            d = point_id(theta_id, height_id + 1)
            faces.extend(([a, b, c], [a, c, d]))
    loop = np.asarray(
        [
            point_id(2, 2),
            point_id(3, 2),
            point_id(4, 2),
            point_id(4, 3),
            point_id(4, 4),
            point_id(3, 4),
            point_id(2, 4),
            point_id(2, 3),
        ],
        dtype=np.int64,
    )
    return points, np.asarray(faces, dtype=np.int64), loop


def attach_vertex_only_tetra(
    points: np.ndarray,
    faces: np.ndarray,
    shared_vertex: int,
) -> tuple[np.ndarray, np.ndarray]:
    origin = points[shared_vertex]
    tetra_points = origin + 0.2 * np.asarray(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    )
    first = points.shape[0]
    a, b, c = first, first + 1, first + 2
    tetra_faces = np.asarray(
        [
            [shared_vertex, a, b],
            [shared_vertex, c, a],
            [shared_vertex, b, c],
            [a, c, b],
        ],
        dtype=np.int64,
    )
    return np.vstack([points, tetra_points]), np.vstack([faces, tetra_faces])


def local_patch_faces(faces: np.ndarray, source_point_count: int) -> np.ndarray:
    result = np.asarray(faces, dtype=np.int64).copy()
    appended = result >= source_point_count
    result[appended] = -(result[appended] - source_point_count + 1)
    return result


class SourcePrimaryOwnerComponentTest(unittest.TestCase):
    def test_every_flipped_owner_one_ring_face_rolls_back(self) -> None:
        points, faces, loop = cylinder_patch_with_rectangular_hole()
        source_ids = np.arange(faces.shape[0], dtype=np.int64)
        base = analyze_source_boundary(
            points, faces, source_ids, loop[::-1], region_id=23
        )
        self.assertTrue(base["success"], base)
        owner_faces, error = select_boundary_owner_one_ring(
            faces,
            base["loop"],
            base["edge_face_ids"],
            points=points,
            minimum_face_normal_consistency=-0.10,
        )
        self.assertIsNone(error)
        self.assertIsNotNone(owner_faces)
        assert owner_faces is not None
        edge_owners = set(base["edge_face_ids"].tolist())
        triangles = points[faces]
        raw_normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        external_directions = raw_normals / np.linalg.norm(raw_normals, axis=1)[:, None]

        roles = set()
        for face_id in owner_faces:
            role = "boundary_owner" if int(face_id) in edge_owners else "one_ring"
            roles.add(role)
            with self.subTest(role=role, face_id=int(face_id)):
                flipped_faces = faces.copy()
                flipped_faces[int(face_id)] = flipped_faces[int(face_id), [0, 2, 1]]
                candidate = build_curved_patch_candidate(
                    points,
                    flipped_faces,
                    source_ids,
                    loop[::-1],
                    region_id=23,
                )
                self.assertEqual(candidate.status, "rejected", candidate.to_dict())
                expected_reason = (
                    "boundary_loop_winding_inconsistent"
                    if role == "boundary_owner"
                    else "boundary_owner_one_ring_winding_inconsistent"
                )
                self.assertIn(expected_reason, candidate.failure_reason_codes)
                result = run_patch_transactions(
                    points,
                    flipped_faces,
                    source_ids,
                    (
                        PatchTransactionJob(
                            region_ids=(23,),
                            patch_method=candidate.method,
                            build_candidate=lambda candidate=candidate: candidate,
                        ),
                    ),
                    source_cell_data={"external_direction": external_directions},
                    required_region_ids=(23,),
                )
                self.assertEqual(
                    result["transactions"][0]["transaction_status"], "rolled_back"
                )
                self.assertIn(
                    expected_reason,
                    result["transactions"][0]["rollback"]["reason_codes"],
                )
                self.assertEqual(result["points"].tobytes(), points.tobytes())
                self.assertEqual(result["faces"].tobytes(), flipped_faces.tobytes())
        self.assertEqual(roles, {"boundary_owner", "one_ring"})

    def test_vertex_only_component_cannot_change_descriptor_or_patch_bytes(self) -> None:
        points, faces, loop = cylinder_patch_with_rectangular_hole()
        attached_points, attached_faces = attach_vertex_only_tetra(
            points, faces, int(loop[0])
        )
        base_ids = np.arange(faces.shape[0], dtype=np.int64)
        attached_ids = np.arange(attached_faces.shape[0], dtype=np.int64)

        base_analysis = analyze_source_boundary(
            points, faces, base_ids, loop[::-1], region_id=23
        )
        attached_analysis = analyze_source_boundary(
            attached_points,
            attached_faces,
            attached_ids,
            loop[::-1],
            region_id=23,
        )
        self.assertTrue(base_analysis["success"], base_analysis)
        self.assertTrue(attached_analysis["success"], attached_analysis)
        self.assertEqual(base_analysis["normal"], attached_analysis["normal"])
        self.assertEqual(base_analysis["curvature"], attached_analysis["curvature"])
        self.assertEqual(base_analysis["diagnostics"], attached_analysis["diagnostics"])
        for name in ("center", "u_axis", "v_axis", "normal", "boundary_uv", "boundary_depth"):
            np.testing.assert_array_equal(
                getattr(base_analysis["frame"], name),
                getattr(attached_analysis["frame"], name),
            )

        base_candidate = build_curved_patch_candidate(
            points, faces, base_ids, loop[::-1], region_id=23
        )
        attached_candidate = build_curved_patch_candidate(
            attached_points,
            attached_faces,
            attached_ids,
            loop[::-1],
            region_id=23,
        )
        self.assertEqual(base_candidate.status, "candidate", base_candidate.to_dict())
        self.assertEqual(
            attached_candidate.status, "candidate", attached_candidate.to_dict()
        )
        self.assertEqual(base_candidate.normal, attached_candidate.normal)
        self.assertEqual(base_candidate.curvature, attached_candidate.curvature)
        self.assertEqual(
            base_candidate.delta.appended_points.tobytes(),
            attached_candidate.delta.appended_points.tobytes(),
        )
        self.assertEqual(
            local_patch_faces(base_candidate.delta.appended_faces, points.shape[0]).tobytes(),
            local_patch_faces(
                attached_candidate.delta.appended_faces, attached_points.shape[0]
            ).tobytes(),
        )
        for name in base_candidate.delta.point_provenance:
            np.testing.assert_array_equal(
                base_candidate.delta.point_provenance[name],
                attached_candidate.delta.point_provenance[name],
            )
        for name in base_candidate.delta.face_provenance:
            np.testing.assert_array_equal(
                base_candidate.delta.face_provenance[name],
                attached_candidate.delta.face_provenance[name],
            )


if __name__ == "__main__":
    unittest.main()
