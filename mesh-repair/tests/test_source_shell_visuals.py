from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from source_shell_visuals import render_candidate_contact_sheets  # noqa: E402


class SourceShellVisualsTest(unittest.TestCase):
    def test_writes_contact_sheet_and_manifest_for_components(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        labels = np.asarray([0, 1], dtype=np.int64)
        candidates = [
            {
                "candidate_id": "component_000000",
                "component_id": 0,
                "face_count": 1,
                "first_hit_face_count": 0,
                "first_hit_view_max": 0,
            },
            {
                "candidate_id": "component_000001",
                "component_id": 1,
                "face_count": 1,
                "first_hit_face_count": 0,
                "first_hit_view_max": 0,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = render_candidate_contact_sheets(
                points,
                faces,
                labels,
                candidates,
                Path(directory),
                page_size=2,
                tile_width=360,
                tile_height=240,
            )
            page_path = Path(report["pages"][0]["path"])
            manifest_path = Path(report["manifest_path"])
            with Image.open(page_path) as image:
                size = image.size

        self.assertEqual(report["candidate_count"], 2)
        self.assertEqual(report["pages"][0]["candidate_ids"], [
            "component_000000",
            "component_000001",
        ])
        self.assertEqual(size, (720, 240))
        self.assertTrue(manifest_path.name == "candidate_manifest.json")


if __name__ == "__main__":
    unittest.main()
