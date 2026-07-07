from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from mesh_metrics import silhouette_drift_from_meshes  # noqa: E402


@dataclass(frozen=True)
class ViewSpec:
    name: str
    project_axes: tuple[int, int]
    depth_vector: tuple[float, float, float]


TOP_VIEW = ViewSpec("top_plus_z", (0, 1), (0.0, 0.0, -1.0))


def square_mesh(x0: float, x1: float, y0: float, y1: float) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [x0, y0, 0.0],
            [x1, y0, 0.0],
            [x1, y1, 0.0],
            [x0, y1, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return points, faces


class SilhouetteDriftTest(unittest.TestCase):
    def test_identical_meshes_have_zero_silhouette_drift(self) -> None:
        points, faces = square_mesh(0.0, 1.0, 0.0, 1.0)

        drift = silhouette_drift_from_meshes(points, faces, points, faces, [TOP_VIEW], max_size=64)

        self.assertEqual(drift["summary"]["changed_ratio_max"], 0.0)
        self.assertEqual(drift["summary"]["overlap_ratio_min"], 1.0)

    def test_shifted_mesh_reports_silhouette_drift(self) -> None:
        reference_points, reference_faces = square_mesh(0.0, 1.0, 0.0, 1.0)
        candidate_points, candidate_faces = square_mesh(0.25, 1.25, 0.0, 1.0)

        drift = silhouette_drift_from_meshes(
            reference_points,
            reference_faces,
            candidate_points,
            candidate_faces,
            [TOP_VIEW],
            max_size=64,
        )

        self.assertGreater(drift["summary"]["changed_ratio_max"], 0.0)
        self.assertLess(drift["summary"]["overlap_ratio_min"], 1.0)


if __name__ == "__main__":
    unittest.main()
