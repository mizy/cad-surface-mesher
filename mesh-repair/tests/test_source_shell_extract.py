from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from mesh_io import read_surface, triangle_faces, write_vtp  # noqa: E402
from source_shell_extract import run_source_shell  # noqa: E402


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


class SourceShellExtractTest(unittest.TestCase):
    def test_nested_internal_component_is_reviewed_and_one_shell_is_written(self) -> None:
        points = np.vstack((cube_points(1.0), cube_points(0.35)))
        faces = np.vstack((CUBE_FACES, CUBE_FACES + 8))

        def fake_reviewer(**kwargs):
            return {
                "status": "accepted",
                "decisions": [
                    {
                        "candidate_id": candidate_id,
                        "decision": "remove_internal",
                        "confidence": 0.99,
                        "semantic_role": "nested_internal_shell",
                        "evidence_view_ids": ["candidate_contact_sheet_001"],
                        "reason_codes": ["fully_occluded_nested_component"],
                        "rationale": "The component is fully enclosed by the larger cube.",
                    }
                    for candidate_id in kwargs["candidate_ids"]
                ],
                "error": None,
                "artifacts": {},
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "nested.vtp"
            output_dir = root / "output"
            write_vtp(input_path, points, faces)
            args = SimpleNamespace(
                input=input_path,
                output_dir=output_dir,
                direction_count=18,
                visibility_grid=64,
                depth_tolerance=0.0,
                continuity_rings=0,
                min_first_hit_views=1,
                min_first_hit_pixels=1,
                ai_max_candidates=8,
                ai_min_component_faces=2,
                ai_remove_confidence=0.85,
                ai_timeout_seconds=10.0,
                codex_binary="codex",
                deadline_seconds=60.0,
            )

            report = run_source_shell(args, ai_reviewer=fake_reviewer)
            shell = read_surface(output_dir / "source_shell.vtp")
            shell_faces = triangle_faces(shell)
            mesh_outputs = sorted(path.name for path in output_dir.rglob("*.vtp"))

        self.assertEqual(report["decision"]["status"], "accepted")
        self.assertTrue(report["decision"]["target_achieved"])
        self.assertEqual(shell_faces.shape[0], 12)
        self.assertEqual(mesh_outputs, ["source_shell.vtp"])
        self.assertFalse(report["geometry_contract"]["voxelization"])
        self.assertFalse(report["geometry_contract"]["remesh"])
        self.assertFalse(report["geometry_contract"]["hole_fill"])
        self.assertEqual(report["selection"]["removed_component_ids"], [1])


if __name__ == "__main__":
    unittest.main()
