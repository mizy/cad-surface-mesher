from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from sealed_exterior_atlas import (  # noqa: E402
    _sealed_outside_fill,
    build_sealed_exterior_atlas,
    build_sealed_exterior_volume_grid,
)


CUBE_FACES = np.asarray(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
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


def cube_points(half_extent: float, center: tuple[float, float, float]) -> np.ndarray:
    h = half_extent
    points = np.asarray(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ],
        dtype=np.float64,
    )
    return points + np.asarray(center, dtype=np.float64)


def cube_soup(
    specifications: list[tuple[float, tuple[float, float, float]]],
) -> tuple[np.ndarray, np.ndarray]:
    point_blocks = []
    face_blocks = []
    point_offset = 0
    for half_extent, center in specifications:
        cube = cube_points(half_extent, center)
        point_blocks.append(cube)
        face_blocks.append(CUBE_FACES + point_offset)
        point_offset += cube.shape[0]
    return np.vstack(point_blocks), np.vstack(face_blocks)


class SealedExteriorAtlasTest(unittest.TestCase):
    def test_sealed_outside_fill_closes_a_subgrid_leak_before_flooding(self) -> None:
        shell = np.zeros((9, 9, 9), dtype=bool)
        shell[2:7, 2:7, 2] = True
        shell[2:7, 2:7, 6] = True
        shell[2:7, 2, 2:7] = True
        shell[2:7, 6, 2:7] = True
        shell[2, 2:7, 2:7] = True
        shell[6, 2:7, 2:7] = True
        shell[4, 4, 6] = False

        # A direct fill sees the one-voxel leak and leaves the whole cavity
        # outside.  This is the failure mode of VoxelGrid.fill on panel soup.
        raw_filled = binary_fill_holes(shell)
        self.assertFalse(raw_filled[4, 4, 4])

        _, sealed, outside, filled, _ = _sealed_outside_fill(
            shell,
            seal_radius_voxels=1,
            padding=3,
        )
        padded_center = (7, 7, 7)
        self.assertTrue(sealed[7, 7, 9])
        self.assertFalse(outside[padded_center])
        self.assertTrue(filled[padded_center])
        self.assertFalse(filled[0, 0, 0])

    def test_sealed_exterior_volume_grid_preserves_world_transform(self) -> None:
        center = np.asarray([5.0, -3.0, 2.0])
        points, faces = cube_soup([(1.0, tuple(center))])

        grid, report = build_sealed_exterior_volume_grid(
            points,
            faces,
            pitch=0.2,
            seal_radius_voxels=1,
            max_projection_distance=0.4,
        )
        surface = grid.marching_cubes
        surface.apply_transform(grid.transform)

        np.testing.assert_allclose(surface.bounds.mean(axis=0), center, atol=0.11)
        self.assertGreater(report["enclosed_empty_voxels"], 0)
        self.assertGreater(report["filled_or_shell_voxels"], 0)
        self.assertGreater(
            report["flood_filled_or_shell_voxels_before_offset_restore"],
            report["filled_or_shell_voxels"],
        )
        self.assertEqual(
            report["surface_offset_restore"],
            "erode_filled_solid_by_seal_radius",
        )
        self.assertEqual(
            report["method"],
            "voxel_shell_dilation_six_connected_far_field_flood",
        )
        core = report["projection_erosion_core"]
        self.assertEqual(core["erosion_radius_voxels"], 2)
        self.assertAlmostEqual(core["erosion_radius_world"], 0.4)
        self.assertTrue(core["nonempty"])
        self.assertTrue(core["mesh_watertight"])
        self.assertGreater(core["mesh_signed_abs_volume"], 0.0)
        self.assertGreater(report["estimated_filled_volume"], core["estimated_volume"])

    def test_closed_outer_cube_occludes_enclosed_closed_component(self) -> None:
        points, faces = cube_soup(
            [
                (1.0, (0.0, 0.0, 0.0)),
                (0.35, (0.0, 0.0, 0.0)),
            ]
        )
        original_points = points.copy()
        original_faces = faces.copy()

        exterior, report = build_sealed_exterior_atlas(points, faces, grid_size=64)

        np.testing.assert_array_equal(exterior[: CUBE_FACES.shape[0]], True)
        np.testing.assert_array_equal(exterior[CUBE_FACES.shape[0] :], False)
        np.testing.assert_array_equal(points, original_points)
        np.testing.assert_array_equal(faces, original_faces)
        self.assertEqual(report["role"], "evidence_only")
        self.assertFalse(report["geometry_modified"])
        self.assertEqual(report["output"]["face_count"], faces.shape[0])
        self.assertEqual(report["thresholds"]["seal_radius_voxels"], 1)
        self.assertEqual(report["thresholds"]["surface_band_voxels"], 1.5)
        self.assertGreater(report["voxel_statistics"]["enclosed_empty_voxels"], 0)

    def test_isolated_component_outside_primary_shell_is_exposed(self) -> None:
        points, faces = cube_soup(
            [
                (1.0, (0.0, 0.0, 0.0)),
                (0.25, (3.0, 0.0, 0.0)),
            ]
        )

        exterior, report = build_sealed_exterior_atlas(points, faces, grid_size=96)

        np.testing.assert_array_equal(exterior, True)
        self.assertEqual(report["face_evidence"]["exterior_face_count"], faces.shape[0])
        self.assertEqual(report["face_evidence"]["interior_or_occluded_face_count"], 0)
        self.assertGreater(report["voxel_statistics"]["far_field_seed_voxels"], 0)

    def test_bbox_normalization_makes_classification_scale_invariant(self) -> None:
        points, faces = cube_soup(
            [
                (1.0, (0.0, 0.0, 0.0)),
                (0.35, (0.0, 0.0, 0.0)),
                (0.2, (2.75, 0.0, 0.0)),
            ]
        )

        unit_mask, unit_report = build_sealed_exterior_atlas(points, faces, grid_size=72)
        scaled_mask, scaled_report = build_sealed_exterior_atlas(
            points * 0.01,
            faces,
            grid_size=72,
        )

        np.testing.assert_array_equal(scaled_mask, unit_mask)
        self.assertAlmostEqual(
            scaled_report["thresholds"]["pitch"] / unit_report["thresholds"]["pitch"],
            0.01,
        )
        self.assertEqual(
            scaled_report["voxel_statistics"]["padded_matrix_shape"],
            unit_report["voxel_statistics"]["padded_matrix_shape"],
        )
        self.assertEqual(
            scaled_report["voxel_statistics"]["sealed_shell_voxels"],
            unit_report["voxel_statistics"]["sealed_shell_voxels"],
        )


if __name__ == "__main__":
    unittest.main()
