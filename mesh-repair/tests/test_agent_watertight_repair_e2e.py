from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from agent_watertight_repair import run_agent_repair  # noqa: E402
from mesh_io import write_vtp  # noqa: E402


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


CUBE_POINTS = np.asarray(
    [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ],
    dtype=np.float64,
)


class AgentWatertightRepairE2ETest(unittest.TestCase):
    def test_agent_off_cube_produces_complete_auditable_artifact_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "cube.vtp"
            output_dir = root / "output"
            output_dir.mkdir()
            write_vtp(input_path, CUBE_POINTS, CUBE_FACES)
            args = SimpleNamespace(
                input=input_path,
                output_dir=output_dir,
                target_policy=None,
                voxel_pitch=0.0,
                sdf_grid_size=32,
                sdf_band_voxels=4.0,
                sdf_smoothing_sigma=0.0,
                max_sdf_memory_gb=0.5,
                agent_mode="off",
                max_agent_rounds=5,
                agent_timeout_seconds=30.0,
                codex_binary="codex",
                visibility_grid=64,
                overwrite=False,
                skip_previews=True,
            )
            status = {"status": "running", "stage": "test", "accepted_mesh_vtp": None}
            report = run_agent_repair(args, status)
            persisted = json.loads((output_dir / "agent_repair_report.json").read_text())
            artifacts = {
                "source_shell.vtp",
                "implicit_field.npz",
                "closure_proxy.vtp",
                "source_projected_candidate.vtp",
                "candidate_iter_0.vtp",
                "region_inventory.json",
                "agent_decisions.jsonl",
                "transactions.jsonl",
                "repair_state.json",
                "agent_repair_report.json",
                "agent_repair_report.html",
            }

            self.assertTrue(artifacts.issubset({path.name for path in output_dir.iterdir()}))
            self.assertEqual(report["schema"], "agent_watertight_repair_report/v1")
            self.assertEqual(persisted["output_contract"]["output_kind"], "watertight_mesh")
            self.assertTrue(persisted["output_contract"]["closure_proxy_is_never_accepted"])
            self.assertEqual(persisted["decision"]["status"], "accepted")
            self.assertTrue((output_dir / "watertight_mesh.vtp").is_file())


if __name__ == "__main__":
    unittest.main()
