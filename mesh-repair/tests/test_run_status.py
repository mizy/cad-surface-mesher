from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_status import initial_run_status, prepare_output_dir, run_parameters  # noqa: E402


def args_for(output_dir: Path, *, overwrite: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        input=Path("/models/input.stl"),
        output_dir=output_dir,
        overwrite=overwrite,
        target_name="watertight-exterior-shell",
        group_source_gltf=None,
        remove_name_regex="internal|hidden",
        visibility_grid=720,
        visibility_min_views=1,
        outside_flood_grid=192,
        sealed_exterior_grid=192,
        sealed_exterior_radius_voxels=1,
        sealed_exterior_band_voxels=1.5,
        depth_tolerance=0.0,
        depth_tolerance_bbox_ratio=2.0e-4,
        depth_tolerance_edge_ratio=0.25,
        dilate_rings=1,
        voxel_pitch=0.0,
        voxel_pitch_bbox_divisor=280.0,
        voxel_pitch_bbox_max_extent=None,
        policy_item_limit=500,
        component_filter_thresholds={},
    )


class RunStatusTest(unittest.TestCase):
    def test_overwrite_cleans_known_artifacts_but_preserves_user_notes(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            output_dir = Path(directory_name)
            known = output_dir / "two_stage_report.json"
            note = output_dir / "notes.txt"
            known.write_text("stale", encoding="utf-8")
            note.write_text("keep", encoding="utf-8")

            prepare_output_dir(args_for(output_dir, overwrite=True))

            self.assertFalse(known.exists())
            self.assertEqual(note.read_text(encoding="utf-8"), "keep")

    def test_run_parameters_contain_only_geometry_and_diagnostic_inputs(self) -> None:
        values = run_parameters(args_for(Path("/tmp/output")))

        self.assertEqual(values["target_name"], "watertight-exterior-shell")
        self.assertEqual(
            set(values),
            {
                "input",
                "target_name",
                "group_source_gltf",
                "remove_name_regex",
                "visibility_grid",
                "visibility_min_views",
                "outside_flood_grid",
                "sealed_exterior_grid",
                "sealed_exterior_radius_voxels",
                "sealed_exterior_band_voxels",
                "depth_tolerance",
                "requested_depth_tolerance",
                "depth_tolerance_bbox_ratio",
                "depth_tolerance_edge_ratio",
                "dilate_rings",
                "voxel_pitch",
                "requested_voxel_pitch",
                "voxel_pitch_source",
                "voxel_pitch_bbox_divisor",
                "voxel_pitch_bbox_max_extent",
                "policy_item_limit",
                "component_filter_thresholds",
            },
        )

    def test_initial_status_never_pretends_an_output_is_accepted(self) -> None:
        status = initial_run_status(args_for(Path("/tmp/output")))

        self.assertEqual(status["status"], "running")
        self.assertIsNone(status["accepted_mesh_vtp"])
        self.assertFalse(status["accepted_mesh_available"])


if __name__ == "__main__":
    unittest.main()
