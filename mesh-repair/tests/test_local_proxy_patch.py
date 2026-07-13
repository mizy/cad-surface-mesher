from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import local_proxy_patch  # noqa: E402
from hybrid_proxy_geometry import FACE_ORIGIN  # noqa: E402
from local_proxy_patch import (  # noqa: E402
    LocalProxyPatchConfig,
    build_source_locked_proxy_patch,
)
from mesh_metrics import edge_topology, inconsistent_winding_edges  # noqa: E402


def source_square_annulus(
    *, curved: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def height(x: float, y: float) -> float:
        return 0.15 * (x * x + y * y - 2.0) if curved else 0.0

    outer = np.asarray(
        [
            [x, y, height(x, y)]
            for x, y in ((-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0))
        ],
        dtype=np.float64,
    )
    inner = np.asarray(
        [
            [x, y, height(x, y)]
            for x, y in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
        ],
        dtype=np.float64,
    )
    points = np.vstack([outer, inner])
    faces = []
    for index in range(4):
        following = (index + 1) % 4
        faces.extend(
            [
                [index, following, 4 + following],
                [index, 4 + following, 4 + index],
            ]
        )
    return (
        points,
        np.asarray(faces, dtype=np.int64),
        np.asarray([4, 7, 6, 5], dtype=np.int64),
    )


def closure_grid(
    top_height: object,
    *,
    bottom_height: float,
    count: int = 13,
    extent: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.linspace(-extent, extent, count)
    top_points = np.asarray(
        [[x, y, float(top_height(x, y))] for y in coordinates for x in coordinates],
        dtype=np.float64,
    )
    bottom_points = top_points.copy()
    bottom_points[:, 2] = bottom_height
    points = np.vstack([top_points, bottom_points])
    bottom_offset = top_points.shape[0]

    def point_id(x_index: int, y_index: int) -> int:
        return y_index * count + x_index

    faces: list[list[int]] = []
    for y_index in range(count - 1):
        for x_index in range(count - 1):
            a = point_id(x_index, y_index)
            b = point_id(x_index + 1, y_index)
            c = point_id(x_index + 1, y_index + 1)
            d = point_id(x_index, y_index + 1)
            faces.extend([[a, b, c], [a, c, d]])
            faces.extend(
                [
                    [bottom_offset + a, bottom_offset + c, bottom_offset + b],
                    [bottom_offset + a, bottom_offset + d, bottom_offset + c],
                ]
            )
    perimeter = (
        [point_id(index, 0) for index in range(count)]
        + [point_id(count - 1, index) for index in range(1, count)]
        + [point_id(index, count - 1) for index in range(count - 2, -1, -1)]
        + [point_id(0, index) for index in range(count - 2, 0, -1)]
    )
    for index, top_left in enumerate(perimeter):
        top_right = perimeter[(index + 1) % len(perimeter)]
        bottom_left = bottom_offset + top_left
        bottom_right = bottom_offset + top_right
        faces.extend(
            [
                [top_right, top_left, bottom_left],
                [top_right, bottom_left, bottom_right],
            ]
        )
    return points, np.asarray(faces, dtype=np.int64)


def curved_closure_proxy() -> tuple[np.ndarray, np.ndarray]:
    return closure_grid(
        lambda x, y: 0.15 * (x * x + y * y - 2.0),
        bottom_height=-2.0,
    )


def merge_meshes(
    meshes: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    point_rows = []
    face_rows = []
    offset = 0
    for points, faces in meshes:
        point_rows.append(points)
        face_rows.append(faces + offset)
        offset += points.shape[0]
    return np.vstack(point_rows), np.vstack(face_rows)


def proxy_annulus(count: int = 24) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    outer = np.column_stack(
        [0.75 * np.cos(angles), 0.75 * np.sin(angles), np.full(count, -0.1)]
    )
    inner = np.column_stack(
        [0.30 * np.cos(angles), 0.30 * np.sin(angles), np.full(count, -0.1)]
    )
    points = np.vstack([outer, inner])
    faces = []
    for index in range(count):
        following = (index + 1) % count
        faces.extend(
            [
                [index, following, count + following],
                [index, count + following, count + index],
            ]
        )
    return points, np.asarray(faces, dtype=np.int64)


def passed_self_intersection() -> dict:
    return {
        "method": "vtk_static_cell_locator_triangle_intersection",
        "scope": "focused_faces",
        "status": "computed",
        "passed": True,
        "intersection_pairs": 0,
        "candidate_pairs_tested": 0,
        "reported_pairs": [],
        "truncated": False,
    }


class SourceLockedLocalProxyPatchTest(unittest.TestCase):
    def test_extracts_curved_disk_from_complete_closure_proxy_and_locks_source(
        self,
    ) -> None:
        source_points, source_faces, source_loop = source_square_annulus()
        proxy_points, proxy_faces = curved_closure_proxy()
        proxy_topology, _ = edge_topology(proxy_faces)
        self.assertEqual(proxy_topology["boundary_edges"], 0)
        self.assertEqual(proxy_topology["non_manifold_edges"], 0)
        self.assertEqual(inconsistent_winding_edges(proxy_faces), 0)
        source_indices = np.arange(source_faces.shape[0], dtype=np.int64) + 100

        with mock.patch.object(
            local_proxy_patch,
            "self_intersection_report",
            return_value=passed_self_intersection(),
        ):
            result = build_source_locked_proxy_patch(
                source_points,
                source_faces,
                source_loop,
                proxy_points,
                proxy_faces,
                source_triangle_indices=source_indices,
                region_id=17,
            )

        self.assertTrue(result["success"], result["diagnostics"])
        self.assertTrue(result["accepted"])
        self.assertGreater(result["proxy_face_ids"].size, 0)
        self.assertGreater(result["proxy_inner_loop"].size, source_loop.size)
        self.assertGreater(float(np.ptp(result["proxy_patch_points"][:, 2])), 0.1)
        np.testing.assert_array_equal(
            result["points"][: source_points.shape[0]],
            source_points,
        )
        lock = result["diagnostics"]["source_boundary_lock"]
        self.assertTrue(lock["source_boundary_exactly_locked"])
        self.assertEqual(lock["source_boundary_max_displacement"], 0.0)
        self.assertEqual(
            result["diagnostics"]["extraction"]["ordered_inner_loop"]["loop_count"],
            1,
        )
        self.assertFalse(
            result["diagnostics"]["selection_contract"]["nearest_point_scatter_used"]
        )
        self.assertEqual(
            result["diagnostics"]["stitch"]["correspondence"]["method"],
            "normalized_arc_length_breakpoint_union",
        )
        self.assertEqual(
            result["diagnostics"]["topology_gate"]["boundary_edges_removed"],
            source_loop.size,
        )

        source_parent = result["source_face_parent"]
        proxy_parent = result["proxy_face_parent"]
        source_mask = source_parent >= 0
        proxy_mask = proxy_parent >= 0
        stitch_mask = ~(source_mask | proxy_mask)
        self.assertEqual(
            result["provenance"]["face_origin"].size, result["faces"].shape[0]
        )
        np.testing.assert_array_equal(
            result["provenance"]["source_triangle_index"][source_mask],
            source_indices[source_parent[source_mask]],
        )
        self.assertTrue(
            np.all(np.isin(proxy_parent[proxy_mask], result["proxy_face_ids"]))
        )
        self.assertTrue(
            np.all(result["face_origin"][proxy_mask] == FACE_ORIGIN["proxy_patch"])
        )
        self.assertTrue(
            np.all(result["face_origin"][stitch_mask] == FACE_ORIGIN["stitch_band"])
        )
        self.assertTrue(
            np.all(result["provenance"]["fusion_region_id"][~source_mask] == 17)
        )

    def test_output_and_diagnostics_are_deterministic(self) -> None:
        source_points, source_faces, source_loop = source_square_annulus()
        proxy_points, proxy_faces = curved_closure_proxy()
        with mock.patch.object(
            local_proxy_patch,
            "self_intersection_report",
            return_value=passed_self_intersection(),
        ):
            first = build_source_locked_proxy_patch(
                source_points,
                source_faces,
                source_loop,
                proxy_points,
                proxy_faces,
                region_id=4,
            )
            second = build_source_locked_proxy_patch(
                source_points,
                source_faces,
                source_loop,
                proxy_points,
                proxy_faces,
                region_id=4,
            )

        self.assertTrue(first["success"])
        self.assertTrue(second["success"])
        for key in (
            "points",
            "faces",
            "face_origin",
            "source_face_parent",
            "proxy_face_parent",
            "stitch_face_ids",
            "proxy_face_ids",
            "proxy_point_ids",
            "proxy_inner_loop",
        ):
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(first["diagnostics"], second["diagnostics"])

    def test_local_self_intersection_failure_rolls_back_all_geometry(self) -> None:
        source_points, source_faces, source_loop = source_square_annulus()
        proxy_points, proxy_faces = curved_closure_proxy()
        failed_check = {
            **passed_self_intersection(),
            "passed": False,
            "intersection_pairs": 1,
            "reported_pairs": [[12, 48]],
        }
        with mock.patch.object(
            local_proxy_patch,
            "self_intersection_report",
            return_value=failed_check,
        ):
            result = build_source_locked_proxy_patch(
                source_points,
                source_faces,
                source_loop,
                proxy_points,
                proxy_faces,
            )

        self.assertFalse(result["success"])
        self.assertEqual(
            result["failure_reason_codes"],
            ["local_proxy_patch_self_intersection_detected"],
        )
        self.assertEqual(result["points"].shape, (0, 3))
        self.assertEqual(result["faces"].shape, (0, 3))
        self.assertEqual(result["proxy_face_parent"].size, 0)
        self.assertEqual(
            result["diagnostics"]["local_self_intersection"]["intersection_pairs"],
            1,
        )

    def test_ambiguous_parallel_closure_sheets_fail_closed(self) -> None:
        source_points, source_faces, source_loop = source_square_annulus(curved=False)
        positive = closure_grid(lambda _x, _y: 0.30, bottom_height=0.10, count=9)
        negative = closure_grid(lambda _x, _y: -0.10, bottom_height=-0.30, count=9)
        proxy_points, proxy_faces = merge_meshes([positive, negative])

        result = build_source_locked_proxy_patch(
            source_points,
            source_faces,
            source_loop,
            proxy_points,
            proxy_faces,
            config=LocalProxyPatchConfig(check_self_intersections=False),
        )

        self.assertFalse(result["success"])
        self.assertFalse(result["accepted"])
        self.assertEqual(
            result["failure_reason_codes"], ["proxy_patch_component_ambiguous"]
        )
        self.assertEqual(result["points"].shape, (0, 3))
        self.assertEqual(result["faces"].shape, (0, 3))
        self.assertEqual(result["proxy_face_ids"].size, 0)
        self.assertIn("ambiguity", result["diagnostics"]["extraction"])

    def test_multi_loop_proxy_crop_is_rejected_transactionally(self) -> None:
        source_points, source_faces, source_loop = source_square_annulus(curved=False)
        proxy_points, proxy_faces = proxy_annulus()
        source_before = source_points.copy()
        proxy_before = proxy_points.copy()

        result = build_source_locked_proxy_patch(
            source_points,
            source_faces,
            source_loop,
            proxy_points,
            proxy_faces,
            config=LocalProxyPatchConfig(check_self_intersections=False),
        )

        self.assertFalse(result["success"])
        self.assertEqual(
            result["failure_reason_codes"],
            ["proxy_patch_boundary_loop_count_mismatch"],
        )
        self.assertEqual(result["proxy_patch_points"].shape, (0, 3))
        self.assertEqual(result["proxy_patch_faces"].shape, (0, 3))
        self.assertEqual(result["provenance"]["face_origin"].size, 0)
        np.testing.assert_array_equal(source_points, source_before)
        np.testing.assert_array_equal(proxy_points, proxy_before)

    def test_invalid_source_loop_returns_stable_empty_schema(self) -> None:
        source_points, source_faces, _ = source_square_annulus(curved=False)
        proxy_points, proxy_faces = curved_closure_proxy()

        result = build_source_locked_proxy_patch(
            source_points,
            source_faces,
            np.asarray([0, 2, 3], dtype=np.int64),
            proxy_points,
            proxy_faces,
        )

        self.assertFalse(result["success"])
        self.assertEqual(
            result["failure_reason_codes"], ["source_loop_edge_is_not_boundary"]
        )
        self.assertEqual(result["source_face_parent"].dtype, np.int64)
        self.assertEqual(result["source_face_parent"].size, 0)
        self.assertEqual(result["diagnostics"]["stage"], "source_loop_validation")


if __name__ == "__main__":
    unittest.main()
