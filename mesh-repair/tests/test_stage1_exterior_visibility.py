from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import two_stage_watertight_remesh as stage1  # noqa: E402


CUBE_FACES = np.asarray(
    [
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ],
    dtype=np.int64,
)


def cube_points(half_extent: float) -> np.ndarray:
    h = half_extent
    return np.asarray(
        [
            [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
            [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
        ],
        dtype=np.float64,
    )


def nested_cubes(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    outer = cube_points(scale)
    inner = cube_points(0.5 * scale)
    return np.vstack((outer, inner)), np.vstack((CUBE_FACES, CUBE_FACES + 8))


class Stage1ExteriorVisibilityTest(unittest.TestCase):
    def run_visibility(self, scale: float) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
        points, faces = nested_cubes(scale)
        tolerance, tolerance_report = stage1.resolve_visibility_depth_tolerance(
            points,
            faces,
            requested_absolute=0.02,
        )
        selected, score, reports = stage1.exterior_visibility_mask(
            points,
            faces,
            grid_size=96,
            depth_tolerance=tolerance,
            dilate_rings=0,
        )
        return selected, score, reports, tolerance_report

    def test_triangle_first_hit_atlas_rejects_nested_internal_shell(self) -> None:
        selected, score, reports, _ = self.run_visibility(1.0)

        self.assertTrue(np.all(selected[: CUBE_FACES.shape[0]]))
        self.assertFalse(np.any(selected[CUBE_FACES.shape[0] :]))
        self.assertTrue(np.all(score[: CUBE_FACES.shape[0]] > 0))
        self.assertTrue(np.all(score[CUBE_FACES.shape[0] :] == 0))
        self.assertTrue(
            all(row["method"] == "orthographic_triangle_raster_first_hit" for row in reports)
        )

    def test_outside_flood_rejects_shell_enclosed_by_outer_component(self) -> None:
        points, faces = nested_cubes()
        exposed, report = stage1.outside_flood_face_mask(points, faces, grid_size=64)

        self.assertTrue(np.all(exposed[: CUBE_FACES.shape[0]]))
        self.assertFalse(np.any(exposed[CUBE_FACES.shape[0] :]))
        self.assertEqual(report["outside_adjacent_faces"], CUBE_FACES.shape[0])

    def test_closure_remesh_uses_sealed_far_field_volume_not_raw_shell_fill(self) -> None:
        points = cube_points(1.0)

        proxy_points, proxy_faces, report = stage1.voxel_watertight_remesh(
            points,
            CUBE_FACES,
            0.2,
            seal_radius_voxels=1,
            max_projection_distance=0.4,
        )

        self.assertGreater(proxy_points.shape[0], 0)
        self.assertGreater(proxy_faces.shape[0], 0)
        self.assertTrue(report["output_trimesh_watertight"])
        self.assertEqual(
            report["method"],
            "sealed_exterior_far_field_flood_then_marching_cubes",
        )
        self.assertEqual(
            report["exterior_volume"]["schema"],
            "sealed_exterior_volume/v1",
        )
        self.assertGreater(report["exterior_volume"]["enclosed_empty_voxels"], 0)

    def test_visibility_and_tolerance_are_scale_invariant(self) -> None:
        selected_large, score_large, _, scale_large = self.run_visibility(1.0)
        selected_small, score_small, _, scale_small = self.run_visibility(0.01)

        np.testing.assert_array_equal(selected_large, selected_small)
        np.testing.assert_array_equal(score_large, score_small)
        self.assertAlmostEqual(scale_small["effective"] / scale_large["effective"], 0.01)

    def test_single_view_first_hit_is_hard_keep(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        hits = [np.asarray([True])] + [np.asarray([False])] * (len(stage1.VIEWS) - 1)
        side_effect = [(hit, {"view": view.name}) for hit, view in zip(hits, stage1.VIEWS, strict=True)]

        with patch.object(stage1, "visible_faces_from_view", side_effect=side_effect):
            selected, score, reports = stage1.exterior_visibility_mask(
                points,
                faces,
                grid_size=64,
                depth_tolerance=0.0,
                dilate_rings=0,
                min_visible_views=3,
            )

        self.assertTrue(selected[0])
        self.assertEqual(score.tolist(), [1])
        self.assertTrue(all(row["hard_keep_contract"] == "any_first_hit_view" for row in reports))

    def test_visible_component_is_kept_whole(self) -> None:
        faces = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int64)
        selected, report = stage1.preserve_whole_visible_components(
            faces,
            np.asarray([True, False, False]),
        )

        np.testing.assert_array_equal(selected, np.asarray([True, True, False]))
        self.assertEqual(report["hidden_faces_restored_by_component_guard"], 1)

    def test_six_view_guard_rejects_new_background(self) -> None:
        points = np.asarray(
            [[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 1.0, 1.0], [0.0, -1.0, 1.0]],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        report = stage1.six_view_depth_regression(
            points,
            faces,
            points,
            np.empty((0, 3), dtype=np.int64),
            grid_size=64,
            depth_tolerance=0.0,
        )
        self.assertFalse(report["passed"])
        self.assertGreater(sum(row["new_background_pixels"] for row in report["views"]), 0)

    def test_six_view_guard_allows_occluded_inner_shell_removal(self) -> None:
        points, faces = nested_cubes()
        report = stage1.six_view_depth_regression(
            points,
            faces,
            points,
            faces[: CUBE_FACES.shape[0]],
            grid_size=64,
            depth_tolerance=1.0e-6,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["regression_pixels"], 0)

    def test_six_view_guard_treats_closer_closure_surface_as_diagnostic(self) -> None:
        before_points = cube_points(1.0)
        before_faces = CUBE_FACES
        closure_plane = np.asarray(
            [
                [-1.25, -0.5, -0.5],
                [-1.25, 0.5, -0.5],
                [-1.25, 0.5, 0.5],
                [-1.25, -0.5, 0.5],
            ],
            dtype=np.float64,
        )
        after_points = np.vstack((before_points, closure_plane))
        after_faces = np.vstack(
            (
                before_faces,
                np.asarray([[8, 9, 10], [8, 10, 11]], dtype=np.int64),
            )
        )

        report = stage1.six_view_depth_regression(
            before_points,
            before_faces,
            after_points,
            after_faces,
            grid_size=64,
            depth_tolerance=1.0e-6,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["regression_pixels"], 0)
        self.assertGreater(report["closer_surface_pixels"], 0)

    def test_six_view_guard_uses_one_pixel_registration_tolerance(self) -> None:
        before_points = cube_points(1.0)
        after_points = before_points.copy()
        after_points[:, 1] += 0.01

        raw = stage1.six_view_depth_regression(
            before_points,
            CUBE_FACES,
            after_points,
            CUBE_FACES,
            grid_size=64,
            depth_tolerance=0.02,
            pixel_tolerance=0,
        )
        registered = stage1.six_view_depth_regression(
            before_points,
            CUBE_FACES,
            after_points,
            CUBE_FACES,
            grid_size=64,
            depth_tolerance=0.02,
            pixel_tolerance=1,
        )

        self.assertGreater(raw["regression_pixels"], 0)
        self.assertTrue(registered["passed"])
        self.assertEqual(registered["regression_pixels"], 0)
        self.assertGreater(registered["raw_regression_pixels"], 0)

    def test_closure_proxy_pitch_defaults_to_bbox_scale(self) -> None:
        args = SimpleNamespace(voxel_pitch=0.0, voxel_pitch_bbox_divisor=192.0)
        stage1.resolve_voxel_pitch(args, {"bounds": {"extents": [0.02, 0.0474, 0.015]}})
        self.assertAlmostEqual(args.voxel_pitch, 0.0474 / 192.0)
        self.assertEqual(args.voxel_pitch_source, "bbox_max_extent_divisor")

    def test_gltf_name_mapping_is_diagnostic_only(self) -> None:
        points, faces = nested_cubes()
        diagnostic_keep = np.ones(faces.shape[0], dtype=bool)
        diagnostic_keep[:5] = False
        args = SimpleNamespace(
            group_source_gltf=Path("scene.gltf"),
            remove_name_regex="interior",
        )
        diagnostic = {
            "method": "gltf_geometry_name_diagnostic_by_flattened_face_ranges",
            "role": "diagnostic_only",
            "geometry_filter_applied": False,
        }
        with tempfile.TemporaryDirectory(), patch.object(
            stage1,
            "gltf_group_keep_mask",
            return_value=(diagnostic_keep, diagnostic),
        ):
            indices, candidate_points, candidate_faces, report = stage1.prepare_group_candidate(
                args,
                points,
                faces,
            )

        np.testing.assert_array_equal(indices, np.arange(faces.shape[0]))
        np.testing.assert_array_equal(candidate_points, points)
        np.testing.assert_array_equal(candidate_faces, faces)
        self.assertTrue(report["geometry_truth_preserved"])
        self.assertIsNone(report["output_vtp"])


if __name__ == "__main__":
    unittest.main()
