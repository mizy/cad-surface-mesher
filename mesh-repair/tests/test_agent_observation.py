from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from agent_observation import render_registered_observation_bundle  # noqa: E402
from mesh_io import write_vtp  # noqa: E402


CUBE_POINTS = np.asarray(
    [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ],
    dtype=np.float64,
)
CUBE_FACES = np.asarray(
    [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ],
    dtype=np.int64,
)


class AgentObservationTest(unittest.TestCase):
    def test_bundle_uses_registered_solid_triangle_views(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_path = root / "cube.vtp"
            write_vtp(mesh_path, CUBE_POINTS, CUBE_FACES)
            manifest = render_registered_observation_bundle(
                {"source_shell": mesh_path},
                root / "visual",
                grid_size=64,
            )
            paths = [Path(row["path"]) for row in manifest["views"]]

        self.assertEqual(manifest["schema"], "mesh_repair_registered_observations/v1")
        self.assertEqual(len(manifest["views"]), 18)
        self.assertTrue(all(row["registered_projection"] for row in manifest["views"]))
        self.assertTrue(all(path.name.endswith(".png") for path in paths))


if __name__ == "__main__":
    unittest.main()
