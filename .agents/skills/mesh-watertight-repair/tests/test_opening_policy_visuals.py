from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from opening_policy_visuals import attach_policy_evidence_views  # noqa: E402


class OpeningPolicyVisualsTest(unittest.TestCase):
    def test_attaches_global_and_local_evidence_only_to_policy_loop(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        policy_item = {
            "id": "boundary_loop_0001",
            "classification": "large_opening_or_missing_surface",
            "requires_policy": True,
            "simple_closed_loop": True,
            "ordered_vertex_ids": [0, 1, 2, 3],
            "normal": [0.0, 0.0, 1.0],
            "diameter": np.sqrt(2.0),
            "mean_edge_length": 1.0,
            "dimensionless_features": {
                "diameter_bbox_ratio": 1.0,
                "compactness": 0.78,
                "planarity": 0.0,
            },
        }
        automatic_item = {
            **policy_item,
            "id": "boundary_loop_0002",
            "classification": "small_exterior_hole",
            "requires_policy": False,
        }
        inventory = {"boundary_regions": {"items": [policy_item, automatic_item]}}

        with tempfile.TemporaryDirectory() as temporary_directory:
            report = attach_policy_evidence_views(
                points,
                faces,
                inventory,
                Path(temporary_directory),
                image_size=128,
            )
            path = Path(policy_item["evidence_views"][0])
            with Image.open(path) as image:
                image_size = image.size

        self.assertEqual(report["requested_regions"], 1)
        self.assertEqual(report["written_regions"], 1)
        self.assertEqual(image_size, (256, 170))
        self.assertNotIn("evidence_views", automatic_item)


if __name__ == "__main__":
    unittest.main()
