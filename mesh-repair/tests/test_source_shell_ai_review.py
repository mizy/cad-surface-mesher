from __future__ import annotations

import json
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import source_shell_ai_review as ai_review  # noqa: E402


def valid_decision(candidate_id: str, decision: str = "ambiguous") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "decision": decision,
        "confidence": 0.75,
        "semantic_role": "unknown",
        "evidence_view_ids": ["view_01"],
        "reason_codes": ["insufficient_depth_evidence"],
        "rationale": "The available views do not prove that the part is internal.",
    }


class CompletingProcess:
    def __init__(self, command: list[str], response: object, return_code: int = 0) -> None:
        self.command = command
        self.response = response
        self.returncode: int | None = None
        self.final_return_code = return_code
        self.pid = 41001
        self.prompts: list[str | None] = []

    def communicate(self, input: str | None = None, timeout: float | None = None):
        self.prompts.append(input)
        output_path = Path(self.command[self.command.index("-o") + 1])
        output_path.write_text(json.dumps(self.response), encoding="utf-8")
        self.returncode = self.final_return_code
        return None, "simulated failure" if self.final_return_code else ""


class TimeoutProcess:
    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.returncode: int | None = None
        self.pid = 42002
        self.communicate_calls = 0
        self.killed = False

    def communicate(self, input: str | None = None, timeout: float | None = None):
        self.communicate_calls += 1
        if self.communicate_calls <= 2:
            raise subprocess.TimeoutExpired(self.command, timeout)
        self.returncode = -signal.SIGKILL
        return None, ""

    def kill(self) -> None:
        self.killed = True


class SourceShellAiReviewTest(unittest.TestCase):
    def test_runs_codex_exec_with_read_only_contract_and_stdin_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = [root / "view-a.png", root / "view-b.png"]
            for image in images:
                image.write_bytes(b"not-a-real-image")
            response = {
                "decisions": [
                    valid_decision("component_0002", "remove_internal"),
                    valid_decision("component_0001", "keep_exterior"),
                ]
            }
            observed: dict[str, object] = {}

            def fake_popen(command, **kwargs):
                observed["command"] = command
                observed["kwargs"] = kwargs
                process = CompletingProcess(command, response)
                observed["process"] = process
                return process

            with patch.object(ai_review.subprocess, "Popen", side_effect=fake_popen):
                outcome = ai_review.review_source_shell_candidates(
                    candidate_ids=["component_0001", "component_0002"],
                    image_paths=images,
                    prompt="Classify highlighted source-shell candidates.",
                    output_dir=root / "review",
                    workspace=root,
                    timeout_seconds=12.5,
                )

            self.assertEqual(outcome["status"], "accepted")
            self.assertEqual(
                [item["candidate_id"] for item in outcome["decisions"]],
                ["component_0001", "component_0002"],
            )
            command = observed["command"]
            self.assertEqual(command[:2], ["codex", "exec"])
            self.assertIn(["-C", str(root.resolve())], pairwise(command))
            self.assertIn(["--sandbox", "read-only"], pairwise(command))
            self.assertIn("--ephemeral", command)
            self.assertIn("--output-schema", command)
            self.assertIn("-o", command)
            self.assertIn("--json", command)
            self.assertEqual(command[-1], "-")
            self.assertEqual(command.count("--image"), 2)
            self.assertNotIn("Classify highlighted source-shell candidates.", command)
            kwargs = observed["kwargs"]
            self.assertTrue(kwargs["start_new_session"])
            self.assertFalse(kwargs["shell"])
            process = observed["process"]
            self.assertIn("Classify highlighted source-shell candidates.", process.prompts[0])
            self.assertIn('"component_0001"', process.prompts[0])

            schema = json.loads(Path(outcome["artifacts"]["schema"]).read_text(encoding="utf-8"))
            item_properties = schema["properties"]["decisions"]["items"]["properties"]
            self.assertEqual(
                item_properties["decision"]["enum"],
                list(ai_review.DECISION_KINDS),
            )
            self.assertEqual(
                item_properties["candidate_id"]["enum"],
                ["component_0001", "component_0002"],
            )

    def test_unknown_missing_and_duplicate_ids_are_rejected_without_decisions(self) -> None:
        responses = [
            {"decisions": [valid_decision("unknown")]},
            {"decisions": [valid_decision("component_0001")]},
            {
                "decisions": [
                    valid_decision("component_0001"),
                    valid_decision("component_0001"),
                ]
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "view.png"
            image.write_bytes(b"image")
            for index, response in enumerate(responses):
                with self.subTest(index=index):
                    with patch.object(
                        ai_review.subprocess,
                        "Popen",
                        side_effect=lambda command, **_: CompletingProcess(command, response),
                    ):
                        outcome = ai_review.review_source_shell_candidates(
                            candidate_ids=["component_0001", "component_0002"],
                            image_paths=[image],
                            prompt="Review candidates.",
                            output_dir=root / f"review-{index}",
                            workspace=root,
                        )
                    self.assertEqual(outcome["status"], "rejected")
                    self.assertEqual(outcome["decisions"], [])
                    self.assertEqual(outcome["error"]["code"], "invalid_response")

    def test_nonzero_exit_is_fail_closed_even_if_an_output_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "view.png"
            image.write_bytes(b"image")
            response = {"decisions": [valid_decision("component_0001", "remove_internal")]}
            with patch.object(
                ai_review.subprocess,
                "Popen",
                side_effect=lambda command, **_: CompletingProcess(command, response, return_code=1),
            ):
                outcome = ai_review.review_source_shell_candidates(
                    candidate_ids=["component_0001"],
                    image_paths=[image],
                    prompt="Review candidates.",
                    output_dir=root / "review",
                    workspace=root,
                )
            self.assertEqual(outcome["status"], "rejected")
            self.assertEqual(outcome["decisions"], [])
            self.assertEqual(outcome["error"]["code"], "codex_failed")

    def test_timeout_terminates_the_independent_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "view.png"
            image.write_bytes(b"image")
            observed: dict[str, TimeoutProcess] = {}

            def fake_popen(command, **kwargs):
                process = TimeoutProcess(command)
                observed["process"] = process
                return process

            with (
                patch.object(ai_review.subprocess, "Popen", side_effect=fake_popen),
                patch.object(ai_review.os, "killpg") as kill_process_group,
            ):
                outcome = ai_review.review_source_shell_candidates(
                    candidate_ids=["component_0001"],
                    image_paths=[image],
                    prompt="Review candidates.",
                    output_dir=root / "review",
                    workspace=root,
                    timeout_seconds=0.01,
                )

            self.assertEqual(outcome["status"], "rejected")
            self.assertEqual(outcome["decisions"], [])
            self.assertEqual(outcome["error"]["code"], "timeout")
            self.assertEqual(
                kill_process_group.call_args_list,
                [
                    unittest.mock.call(42002, signal.SIGTERM),
                    unittest.mock.call(42002, signal.SIGKILL),
                ],
            )
            self.assertEqual(observed["process"].communicate_calls, 3)

    def test_duplicate_requested_candidate_ids_fail_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "view.png"
            image.write_bytes(b"image")
            with patch.object(ai_review.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(ValueError, "unique"):
                    ai_review.review_source_shell_candidates(
                        candidate_ids=["duplicate", "duplicate"],
                        image_paths=[image],
                        prompt="Review candidates.",
                        output_dir=root / "review",
                        workspace=root,
                    )
            popen.assert_not_called()


def pairwise(values: list[str]) -> list[list[str]]:
    return [values[index : index + 2] for index in range(len(values) - 1)]


if __name__ == "__main__":
    unittest.main()
