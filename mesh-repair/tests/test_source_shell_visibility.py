from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from source_shell_visibility import (  # noqa: E402
    build_source_shell_visibility,
    fibonacci_sphere_directions,
    rasterize_directional_first_hits,
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


def cube_points(half_extent: float) -> np.ndarray:
    h = half_extent
    return np.asarray(
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


def nested_cubes() -> tuple[np.ndarray, np.ndarray]:
    points = np.vstack((cube_points(1.0), cube_points(0.35)))
    faces = np.vstack((CUBE_FACES, CUBE_FACES + 8))
    return points, faces


class SourceShellVisibilityTest(unittest.TestCase):
    def test_default_directions_are_deterministic_uniform_sphere_samples(self) -> None:
        first = fibonacci_sphere_directions()
        second = fibonacci_sphere_directions()

        self.assertEqual(first.shape, (42, 3))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)
        np.testing.assert_allclose(np.mean(first, axis=0), 0.0, atol=0.04)

    def test_outer_shell_occludes_nested_inner_shell(self) -> None:
        points, faces = nested_cubes()

        evidence = build_source_shell_visibility(
            points,
            faces,
            grid_size=72,
        )

        np.testing.assert_array_equal(evidence.face_exterior_mask[:12], True)
        np.testing.assert_array_equal(evidence.face_exterior_mask[12:], False)
        self.assertTrue(np.all(evidence.face_first_hit_view_count[:12] > 0))
        self.assertTrue(np.all(evidence.face_first_hit_view_count[12:] == 0))
        self.assertTrue(np.all(evidence.face_first_hit_pixel_support[:12] > 0))
        self.assertTrue(np.all(evidence.face_first_hit_pixel_support[12:] == 0))
        self.assertEqual(evidence.report["selection_scope"], "individual_source_face")
        self.assertFalse(evidence.report["component_expansion_applied"])

    def test_arbitrary_oblique_view_uses_true_orthographic_depth(self) -> None:
        direction = np.asarray([1.0, -2.0, 3.0], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        horizontal = np.cross(np.asarray([0.0, 1.0, 0.0]), direction)
        horizontal /= np.linalg.norm(horizontal)
        vertical = np.cross(direction, horizontal)
        square = np.asarray(
            [
                -horizontal - vertical,
                horizontal - vertical,
                horizontal + vertical,
                -horizontal + vertical,
            ]
        )
        points = np.vstack((square + direction, square - direction))
        faces = np.asarray(
            [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
            dtype=np.int64,
        )

        evidence = rasterize_directional_first_hits(
            points,
            faces,
            direction,
            grid_size=64,
        )

        np.testing.assert_array_equal(
            evidence.face_first_hit_mask,
            np.asarray([True, True, False, False]),
        )
        np.testing.assert_allclose(evidence.external_direction, direction)
        self.assertEqual(
            evidence.report["method"],
            "arbitrary_direction_orthographic_solid_triangle_first_hit",
        )

    def test_visible_face_does_not_keep_its_hidden_connected_component(self) -> None:
        points = np.asarray(
            [
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 5],
                [0, 5, 1],
            ],
            dtype=np.int64,
        )

        evidence = build_source_shell_visibility(
            points,
            faces,
            directions=np.asarray([[0.0, 0.0, 1.0]]),
            grid_size=64,
        )

        np.testing.assert_array_equal(evidence.face_exterior_mask[:2], True)
        np.testing.assert_array_equal(evidence.face_exterior_mask[2:4], False)
        self.assertFalse(evidence.report["component_expansion_applied"])


if __name__ == "__main__":
    unittest.main()
