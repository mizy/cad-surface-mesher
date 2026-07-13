from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from source_primary_boundary_inventory import (  # noqa: E402
    build_source_primary_boundary_inventory,
)
from source_primary_dispatch import build_source_primary_dispatch  # noqa: E402
from source_primary_loop_bridge import (  # noqa: E402
    build_paired_loop_zipper_candidate,
)
from source_primary_multi_loop_patch import (  # noqa: E402
    build_multi_loop_patch_candidate,
)
from source_primary_patch_contract import PatchCandidate  # noqa: E402
from source_primary_quality import (  # noqa: E402
    audit_source_primary_patch,
    validate_transaction_local_quality,
)
from source_primary_quality_geometry import (  # noqa: E402
    build_directed_edge_map,
    calculate_triangle_geometry,
)
from source_primary_slit_patch import build_slit_patch_candidate  # noqa: E402
from source_primary_transaction import run_patch_transactions  # noqa: E402


class SourcePrimaryRepairerProtocolTest(unittest.TestCase):
    def test_three_nested_loops_are_blocked_before_candidate_dispatch(self) -> None:
        points, faces = frame_with_two_inner_islands()
        source_ids = np.arange(faces.shape[0], dtype=np.int64)
        inventory = build_source_primary_boundary_inventory(
            points,
            faces,
            source_ids,
            face_external_directions=np.asarray([0.0, 0.0, 1.0]),
        )
        region = next(
            row
            for row in inventory["regions"]
            if row["classification"] == "multi_loop_or_inner_island_hole"
        )
        self.assertEqual(region["loop_count"], 3)
        self.assertFalse(region["patch_eligible"])
        self.assertIn(
            "multi_loop_requires_exactly_two_nested_loops",
            region["blocking_reason_codes"],
        )

        dispatch = build_source_primary_dispatch(
            points,
            faces,
            source_ids,
            inventory,
            closure_proxy_points=points,
            closure_proxy_faces=faces,
        )
        row = next(
            item
            for item in dispatch["method_dispatch"]["rows"]
            if item["region_id"] == region["region_id"]
        )
        self.assertEqual(row["assignment_status"], "assigned_rejected_candidate")
        self.assertIn(
            "multi_loop_requires_exactly_two_nested_loops", row["reason_codes"]
        )
        job = next(
            item
            for item in dispatch["jobs"]
            if item.region_ids == (region["region_id"],)
        )
        candidate = job.build_candidate()
        self.assertEqual(candidate.status, "rejected")
        self.assertIn(
            "multi_loop_requires_exactly_two_nested_loops",
            candidate.failure_reason_codes,
        )

    def test_multi_loop_constructor_transaction_and_verifier_protocol(self) -> None:
        points, faces = frame_with_inner_island()
        source_ids = np.arange(faces.shape[0], dtype=np.int64)
        inventory = build_source_primary_boundary_inventory(
            points,
            faces,
            source_ids,
            face_external_directions=np.asarray([0.0, 0.0, 1.0]),
        )
        region = next(
            row
            for row in inventory["regions"]
            if row["classification"] == "multi_loop_or_inner_island_hole"
        )
        candidate = build_multi_loop_patch_candidate(
            points,
            faces,
            source_ids,
            [
                np.asarray(loop, dtype=np.int64)
                for loop in region["ordered_boundary_loops"]
            ],
            region_id=71,
        )

        row, patch_faces, external = self.assert_verified_commit(
            points, faces, source_ids, candidate, 71
        )
        flipped = patch_faces[:, [0, 2, 1]]
        self.assert_independent_rejection(
            row, points, faces, candidate.delta.appended_points, flipped, external
        )

    def test_zipper_constructor_transaction_and_verifier_protocol(self) -> None:
        points, faces, first_loop, second_loop = unequal_circular_hole_pair()
        source_ids = np.arange(faces.shape[0], dtype=np.int64)
        candidate = build_paired_loop_zipper_candidate(
            points,
            faces,
            source_ids,
            first_loop,
            second_loop,
            first_region_id=31,
            second_region_id=31,
        )

        row, patch_faces, external = self.assert_verified_commit(
            points, faces, source_ids, candidate, 31
        )
        folded = patch_faces.copy()
        folded[0] = folded[0, [0, 2, 1]]
        self.assert_independent_rejection(
            row, points, faces, candidate.delta.appended_points, folded, external
        )

    def test_slit_constructor_transaction_and_verifier_protocol(self) -> None:
        points, faces, loop = rectangular_annulus(0.3, 0.05)
        source_ids = np.arange(faces.shape[0], dtype=np.int64)
        candidate = build_slit_patch_candidate(
            points,
            faces,
            source_ids,
            loop[::-1],
            region_id=41,
        )

        row, patch_faces, external = self.assert_verified_commit(
            points, faces, source_ids, candidate, 41
        )
        folded = patch_faces.copy()
        folded[-1] = folded[-1, [0, 2, 1]]
        self.assert_independent_rejection(
            row, points, faces, candidate.delta.appended_points, folded, external
        )

    def test_folded_internal_edges_fail_audit_and_verifier_recompute(self) -> None:
        points, faces, interior, patch_faces, loop = folded_checkerboard_patch()
        expected = np.repeat(
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
            patch_faces.shape[0],
            axis=0,
        )
        trial_points = np.vstack([points, interior])
        normals = calculate_triangle_geometry(trial_points, patch_faces)["normals"]
        edge_map = build_directed_edge_map(patch_faces)
        internal_rows = [rows for rows in edge_map.values() if len(rows) == 2]
        dots = np.asarray(
            [
                np.dot(normals[rows[0][0]], normals[rows[1][0]])
                for rows in internal_rows
            ]
        )
        angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
        self.assertLess(float(dots.min()), -0.8)
        self.assertGreater(float(angles.max()), 150.0)

        quality = audit_source_primary_patch(
            points,
            faces,
            interior,
            patch_faces,
            (loop,),
            expected_face_normals=expected,
        )
        self.assertFalse(quality["passed"], quality)
        self.assertFalse(quality["gates"]["patch_internal_normal_transition"]["passed"])
        self.assertFalse(
            quality["gates"]["patch_internal_curvature_continuity"]["passed"]
        )

        flat_interior = interior.copy()
        flat_interior[:, 2] = 0.0
        reported_quality = audit_source_primary_patch(
            points,
            faces,
            flat_interior,
            patch_faces,
            (loop,),
            expected_face_normals=expected,
        )
        self.assertTrue(reported_quality["passed"], reported_quality)
        external = np.repeat(
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
            faces.shape[0],
            axis=0,
        )
        errors = validate_transaction_local_quality(
            1,
            {
                "boundary_loops": [{"source_vertex_ids": loop.tolist()}],
                "quality": reported_quality,
            },
            {
                "points": points,
                "faces": faces,
                "cell_data": {"external_direction": external},
            },
            points,
            faces,
            interior,
            patch_faces,
        )
        self.assertIn("transaction_independent_local_quality_failed:1", errors)

    def assert_verified_commit(
        self,
        points: np.ndarray,
        faces: np.ndarray,
        source_ids: np.ndarray,
        candidate: PatchCandidate,
        region_id: int,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        external = np.repeat(
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
            faces.shape[0],
            axis=0,
        )
        result = run_patch_transactions(
            points,
            faces,
            source_ids,
            (candidate,),
            source_cell_data={"external_direction": external},
            fusion_region_id_by_region={str(region_id): region_id},
        )
        row = result["transactions"][0]
        self.assertEqual(row["transaction_status"], "committed", row)
        patch_faces = np.asarray(result["faces"])[faces.shape[0] :]
        errors = validate_transaction_local_quality(
            1,
            row,
            {
                "points": points,
                "faces": faces,
                "cell_data": {"external_direction": external},
            },
            points,
            faces,
            np.asarray(candidate.delta.appended_points),
            patch_faces,
        )
        self.assertEqual(errors, [])
        return row, patch_faces, external

    def assert_independent_rejection(
        self,
        row: dict,
        points: np.ndarray,
        faces: np.ndarray,
        appended_points: np.ndarray,
        patch_faces: np.ndarray,
        external: np.ndarray,
    ) -> None:
        errors = validate_transaction_local_quality(
            1,
            row,
            {
                "points": points,
                "faces": faces,
                "cell_data": {"external_direction": external},
            },
            points,
            faces,
            appended_points,
            patch_faces,
        )
        self.assertIn("transaction_independent_local_quality_failed:1", errors)


def rectangular_annulus(
    inner_half_x: float, inner_half_y: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [-3.0, -2.5, 0.0],
            [3.0, -2.5, 0.0],
            [3.0, 2.5, 0.0],
            [-3.0, 2.5, 0.0],
            [-inner_half_x, -inner_half_y, 0.0],
            [inner_half_x, -inner_half_y, 0.0],
            [inner_half_x, inner_half_y, 0.0],
            [-inner_half_x, inner_half_y, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
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
    return points, faces, np.arange(4, 8, dtype=np.int64)


def folded_checkerboard_patch() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    coordinates = np.arange(-4.0, 5.0)
    width = coordinates.size

    def grid_id(x_index: int, y_index: int) -> int:
        return y_index * width + x_index

    raw_points = np.asarray(
        [(x, y, 0.0) for y in coordinates for x in coordinates],
        dtype=np.float64,
    )
    raw_source_faces = []
    for y_index in range(width - 1):
        for x_index in range(width - 1):
            if 1 <= x_index <= 6 and 1 <= y_index <= 6:
                continue
            lower_left = grid_id(x_index, y_index)
            lower_right = grid_id(x_index + 1, y_index)
            upper_left = grid_id(x_index, y_index + 1)
            upper_right = grid_id(x_index + 1, y_index + 1)
            raw_source_faces.extend(
                ([lower_left, lower_right, upper_right], [lower_left, upper_right, upper_left])
            )
    used_source_ids = sorted(set(np.asarray(raw_source_faces).reshape(-1)))
    source_id = {raw_id: index for index, raw_id in enumerate(used_source_ids)}
    points = raw_points[used_source_ids]
    source_faces = np.asarray(
        [[source_id[raw_id] for raw_id in face] for face in raw_source_faces],
        dtype=np.int64,
    )

    interior_rows = []
    interior_id = {}
    for y_index in range(2, 7):
        for x_index in range(2, 7):
            x, y = coordinates[x_index], coordinates[y_index]
            z = 0.0 if max(abs(x), abs(y)) >= 2.0 else (
                1.4 if (x_index + y_index) % 2 == 0 else -1.4
            )
            interior_id[grid_id(x_index, y_index)] = len(points) + len(interior_rows)
            interior_rows.append([x, y, z])
    interior = np.asarray(interior_rows, dtype=np.float64)

    def candidate_id(x_index: int, y_index: int) -> int:
        raw_id = grid_id(x_index, y_index)
        return source_id[raw_id] if raw_id in source_id else interior_id[raw_id]

    patch_rows = []
    for y_index in range(1, 7):
        for x_index in range(1, 7):
            lower_left = candidate_id(x_index, y_index)
            lower_right = candidate_id(x_index + 1, y_index)
            upper_left = candidate_id(x_index, y_index + 1)
            upper_right = candidate_id(x_index + 1, y_index + 1)
            patch_rows.extend(
                ([lower_left, lower_right, upper_right], [lower_left, upper_right, upper_left])
            )
    patch_faces = np.asarray(patch_rows, dtype=np.int64)
    raw_loop = (
        [grid_id(x_index, 1) for x_index in range(7, 0, -1)]
        + [grid_id(1, y_index) for y_index in range(2, 8)]
        + [grid_id(x_index, 7) for x_index in range(2, 8)]
        + [grid_id(7, y_index) for y_index in range(6, 1, -1)]
    )
    loop = np.asarray([source_id[raw_id] for raw_id in raw_loop], dtype=np.int64)
    return points, source_faces, interior, patch_faces, loop


def frame_with_inner_island() -> tuple[np.ndarray, np.ndarray]:
    points, faces, _ = rectangular_annulus(1.0, 1.0)
    island = np.asarray(
        [
            [-0.35, -0.30, 0.0],
            [0.35, -0.30, 0.0],
            [0.35, 0.30, 0.0],
            [-0.35, 0.30, 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.vstack([points, island]),
        np.vstack([faces, [[8, 9, 10], [8, 10, 11]]]).astype(np.int64),
    )


def frame_with_two_inner_islands() -> tuple[np.ndarray, np.ndarray]:
    points, faces, _ = rectangular_annulus(1.0, 1.0)
    islands = np.asarray(
        [
            [-0.80, -0.30, 0.0],
            [-0.40, -0.30, 0.0],
            [-0.40, 0.10, 0.0],
            [-0.80, 0.10, 0.0],
            [0.40, -0.10, 0.0],
            [0.80, -0.10, 0.0],
            [0.80, 0.30, 0.0],
            [0.40, 0.30, 0.0],
        ],
        dtype=np.float64,
    )
    island_faces = np.asarray(
        [[8, 9, 10], [8, 10, 11], [12, 13, 14], [12, 14, 15]],
        dtype=np.int64,
    )
    return np.vstack([points, islands]), np.vstack([faces, island_faces])


def unequal_circular_hole_pair(
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first_count, second_count = 5, 7
    first_angles = np.linspace(0.0, 2.0 * np.pi, first_count, endpoint=False)
    first_outer = np.column_stack(
        [2.0 * np.cos(first_angles), 2.0 * np.sin(first_angles), np.zeros(first_count)]
    )
    first_inner = np.column_stack(
        [np.cos(first_angles), np.sin(first_angles), np.zeros(first_count)]
    )
    first_faces = []
    for index in range(first_count):
        following = (index + 1) % first_count
        first_faces.extend(
            (
                [index, following, first_count + following],
                [index, first_count + following, first_count + index],
            )
        )
    second_angles = np.linspace(0.0, 2.0 * np.pi, second_count, endpoint=False)
    second_ring = np.column_stack(
        [
            0.7 * np.cos(second_angles),
            0.7 * np.sin(second_angles),
            np.zeros(second_count),
        ]
    )
    first_points = np.vstack([first_outer, first_inner])
    second_points = np.vstack([np.zeros((1, 3)), second_ring])
    second_offset = first_points.shape[0]
    second_faces = np.asarray(
        [
            [second_offset, second_offset + 1 + index, second_offset + 1 + (index + 1) % second_count]
            for index in range(second_count)
        ],
        dtype=np.int64,
    )
    return (
        np.vstack([first_points, second_points]),
        np.vstack([np.asarray(first_faces, dtype=np.int64), second_faces]),
        np.arange(first_count, 2 * first_count, dtype=np.int64),
        np.arange(second_offset + 1, second_offset + 1 + second_count, dtype=np.int64),
    )


if __name__ == "__main__":
    unittest.main()
