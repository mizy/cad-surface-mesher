from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from sdf_closure import (  # noqa: E402
    SdfClosureConfig,
    bounded_sdf_smoothing,
    build_tsdf_closure,
    sdf_memory_preflight,
)


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


def cube_points(half_extent: float = 1.0) -> np.ndarray:
    h = half_extent
    return np.asarray(
        [
            [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
            [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
        ],
        dtype=np.float64,
    )


class SignedDistanceClosureTest(unittest.TestCase):
    def test_cube_writes_signed_field_and_watertight_zero_surface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "implicit_field.npz"
            points, faces, report = build_tsdf_closure(
                cube_points(),
                CUBE_FACES,
                config=SdfClosureConfig(
                    pitch=0.25,
                    smoothing_sigma_voxels=0.0,
                    max_memory_gb=0.1,
                    max_projection_distance=0.5,
                ),
                artifact_path=artifact,
            )
            data = np.load(artifact)

        mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
        self.assertTrue(mesh.is_watertight)
        self.assertLess(float(np.min(data["sdf"])), 0.0)
        self.assertGreater(float(np.max(data["sdf"])), 0.0)
        self.assertEqual(str(data["schema"]), "implicit_field/v1")
        self.assertEqual(report["schema"], "signed_distance_closure/v1")
        self.assertEqual(
            report["sign_source"],
            "sealed_exterior_six_connected_far_field_flood",
        )
        self.assertGreater(report["exact_distance"]["query_voxels"], 0)

    def test_memory_budget_fails_without_changing_resolution(self) -> None:
        report = sdf_memory_preflight(
            cube_points(),
            pitch=0.001,
            seal_radius_voxels=1,
            max_memory_gb=0.001,
        )
        self.assertFalse(report["passed"])
        self.assertIn("exceeds", report["failure_reason"])

    def test_large_smoothing_request_falls_back_to_original_field(self) -> None:
        phi = np.ones((9, 9, 9), dtype=np.float32)
        phi[:4] = -1.0
        band = np.ones_like(phi, dtype=bool)
        smoothed, report = bounded_sdf_smoothing(
            phi,
            band,
            pitch=1.0,
            sigma_voxels=4.0,
            max_zero_shift_voxels=0.01,
        )
        self.assertTrue(report["fallback"])
        np.testing.assert_array_equal(smoothed, phi)


if __name__ == "__main__":
    unittest.main()
