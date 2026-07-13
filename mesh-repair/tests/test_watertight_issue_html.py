from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from watertight_issue_html import (  # noqa: E402
    comparison_json_path,
    projected_issue_overlay,
)
from watertight_issue_markers import ISSUE_BOUNDARY  # noqa: E402


class WatertightIssueHtmlTest(unittest.TestCase):
    def test_comparison_json_does_not_overwrite_marker_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            marker = root / "watertight_issue_report.json"
            html = root / "watertight_issue_report.html"

            result = comparison_json_path(html, marker)

        self.assertEqual(result.name, "watertight_issue_report_comparison.json")

    def test_projected_overlay_draws_boundary_marker(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)

        image = projected_issue_overlay(
            points,
            faces,
            {ISSUE_BOUNDARY: [(0, 1)], 2: [], 4: []},
            (0, 1),
            max_size=128,
        )

        pixels = np.asarray(image)
        self.assertEqual(image.mode, "RGB")
        self.assertTrue(np.any(np.all(pixels == (255, 151, 20), axis=2)))


if __name__ == "__main__":
    unittest.main()
