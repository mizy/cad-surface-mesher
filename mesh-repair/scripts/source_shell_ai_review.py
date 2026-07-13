from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
from pathlib import Path
from typing import Any, Sequence


DECISION_KINDS = (
    "remove_internal",
    "keep_exterior",
    "split_required",
    "ambiguous",
)
DECISION_KEYS = {
    "candidate_id",
    "decision",
    "confidence",
    "semantic_role",
    "evidence_view_ids",
    "reason_codes",
    "rationale",
}


class CodexReviewTimeout(RuntimeError):
    pass


def build_decision_schema(candidate_ids: Sequence[str]) -> dict[str, Any]:
    """Build a closed JSON schema for exactly one decision per candidate."""

    ids = validate_candidate_ids(candidate_ids)
    decision_item = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(DECISION_KEYS),
        "properties": {
            "candidate_id": {"type": "string", "enum": ids},
            "decision": {"type": "string", "enum": list(DECISION_KINDS)},
            "confidence": {"type": "number"},
            "semantic_role": {"type": "string"},
            "evidence_view_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "reason_codes": {
                "type": "array",
                "items": {"type": "string"},
            },
            "rationale": {"type": "string"},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["decisions"],
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": len(ids),
                "maxItems": len(ids),
                "items": decision_item,
            }
        },
    }


def validate_candidate_ids(candidate_ids: Sequence[str]) -> list[str]:
    ids = list(candidate_ids)
    if not ids:
        raise ValueError("candidate_ids must not be empty")
    for candidate_id in ids:
        if not isinstance(candidate_id, str):
            raise ValueError("every candidate ID must be a string")
        if not candidate_id or candidate_id != candidate_id.strip():
            raise ValueError("candidate IDs must be non-empty and have no surrounding whitespace")
        if len(candidate_id) > 256 or any(ord(character) < 32 for character in candidate_id):
            raise ValueError(f"invalid candidate ID: {candidate_id!r}")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    return ids


def build_codex_command(
    *,
    codex_binary: str,
    workspace: Path,
    image_paths: Sequence[Path],
    schema_path: Path,
    response_path: Path,
) -> list[str]:
    command = [
        str(codex_binary),
        "exec",
        "--skip-git-repo-check",
        "-C",
        str(workspace),
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--model",
        "gpt-5.6-luna",
    ]
    for image_path in image_paths:
        command.extend(("--image", str(image_path)))
    command.extend(
        (
            "--output-schema",
            str(schema_path),
            "-o",
            str(response_path),
            "--json",
            "-",
        )
    )
    return command


def review_source_shell_candidates(
    *,
    candidate_ids: Sequence[str],
    image_paths: Sequence[str | Path],
    prompt: str,
    output_dir: str | Path,
    workspace: str | Path,
    timeout_seconds: float = 120.0,
    codex_binary: str = "codex",
) -> dict[str, Any]:
    """Run a read-only Codex visual review and fail closed on any anomaly.

    A rejected outcome always contains an empty ``decisions`` list. Callers must
    only consume decisions when ``status == "accepted"``.
    """

    ids = validate_candidate_ids(candidate_ids)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be a finite positive number")

    resolved_workspace = Path(workspace).expanduser().resolve()
    if not resolved_workspace.is_dir():
        raise ValueError(f"workspace is not a directory: {resolved_workspace}")
    resolved_images = [Path(path).expanduser().resolve() for path in image_paths]
    if not resolved_images:
        raise ValueError("at least one evidence image is required")
    missing_images = [str(path) for path in resolved_images if not path.is_file()]
    if missing_images:
        raise ValueError(f"evidence images do not exist: {missing_images}")

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = resolved_output_dir / "codex_decision_schema.json"
    response_path = resolved_output_dir / "codex_decisions.json"
    events_path = resolved_output_dir / "codex_events.jsonl"
    artifacts = {
        "schema": str(schema_path),
        "response": str(response_path),
        "events": str(events_path),
    }

    schema_path.write_text(
        json.dumps(build_decision_schema(ids), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    response_path.unlink(missing_ok=True)
    command = build_codex_command(
        codex_binary=codex_binary,
        workspace=resolved_workspace,
        image_paths=resolved_images,
        schema_path=schema_path,
        response_path=response_path,
    )
    review_prompt = build_review_prompt(prompt, ids)

    try:
        return_code, stderr = run_codex_process(
            command,
            prompt=review_prompt,
            workspace=resolved_workspace,
            events_path=events_path,
            timeout_seconds=float(timeout_seconds),
        )
    except CodexReviewTimeout as error:
        try:
            decisions = load_valid_decisions(response_path, events_path, ids)
        except (OSError, json.JSONDecodeError, ValueError):
            return rejected_outcome("timeout", str(error), artifacts)
        persist_recovered_response(response_path, decisions)
        return accepted_outcome(decisions, artifacts)
    except FileNotFoundError as error:
        return rejected_outcome("launch_failed", str(error), artifacts)
    except Exception as error:
        return rejected_outcome("process_failed", str(error), artifacts)

    if return_code != 0:
        detail = stderr.strip()[-2000:] or f"Codex exited with status {return_code}"
        return rejected_outcome("codex_failed", detail, artifacts)

    try:
        decisions = load_valid_decisions(response_path, events_path, ids)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return rejected_outcome("invalid_response", str(error), artifacts)

    return accepted_outcome(decisions, artifacts)


def build_review_prompt(prompt: str, candidate_ids: Sequence[str]) -> str:
    encoded_ids = json.dumps(list(candidate_ids), ensure_ascii=False)
    return (
        f"{prompt.rstrip()}\n\n"
        "Return exactly one schema-valid decision for every requested candidate. "
        "Do not add, omit, rename, merge, or duplicate candidate IDs. "
        f"The exact candidate IDs are: {encoded_ids}"
    )


def run_codex_process(
    command: Sequence[str],
    *,
    prompt: str,
    workspace: Path,
    events_path: Path,
    timeout_seconds: float,
) -> tuple[int, str]:
    with events_path.open("w", encoding="utf-8") as events_file:
        process = subprocess.Popen(
            list(command),
            cwd=str(workspace),
            stdin=subprocess.PIPE,
            stdout=events_file,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            start_new_session=True,
            shell=False,
        )
        try:
            _, stderr = process.communicate(input=prompt, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            terminate_process_group(process)
            raise CodexReviewTimeout(
                f"Codex visual review exceeded {timeout_seconds:.3f} seconds"
            ) from error
        except BaseException:
            terminate_process_group(process)
            raise
    if process.returncode is None:
        raise RuntimeError("Codex process ended without a return code")
    return int(process.returncode), stderr or ""


def terminate_process_group(process: subprocess.Popen[str], *, grace_seconds: float = 2.0) -> None:
    send_process_group_signal(process, signal.SIGTERM)
    try:
        process.communicate(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    send_process_group_signal(process, signal.SIGKILL)
    try:
        process.communicate(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()


def send_process_group_signal(process: subprocess.Popen[str], signal_number: int) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        return


def load_valid_decisions(
    response_path: Path,
    events_path: Path,
    candidate_ids: Sequence[str],
) -> list[dict[str, Any]]:
    responses: list[Any] = []
    errors: list[str] = []
    if response_path.is_file():
        try:
            responses.append(json.loads(response_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as error:
            errors.append(str(error))
    if events_path.is_file():
        for line in reversed(events_path.read_text(encoding="utf-8").splitlines()):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(str(error))
                continue
            if not isinstance(event, dict):
                continue
            item = event.get("item", {})
            if event.get("type") != "item.completed" or item.get("type") != "agent_message":
                continue
            text = item.get("text")
            if isinstance(text, str):
                try:
                    responses.append(json.loads(text))
                    break
                except json.JSONDecodeError as error:
                    errors.append(str(error))
    if not responses:
        detail = "; ".join(errors) or "Codex produced no structured decision response"
        raise ValueError(detail)
    for response in responses:
        try:
            return validate_decisions(response, candidate_ids)
        except ValueError as error:
            errors.append(str(error))
    raise ValueError("; ".join(errors))


def persist_recovered_response(response_path: Path, decisions: list[dict[str, Any]]) -> None:
    response_path.write_text(
        json.dumps({"decisions": decisions}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def accepted_outcome(
    decisions: list[dict[str, Any]], artifacts: dict[str, str]
) -> dict[str, Any]:
    return {
        "status": "accepted",
        "decisions": decisions,
        "error": None,
        "artifacts": artifacts,
    }


def validate_decisions(response: Any, candidate_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not isinstance(response, dict) or set(response) != {"decisions"}:
        raise ValueError("response must be an object containing only 'decisions'")
    decisions = response["decisions"]
    if not isinstance(decisions, list):
        raise ValueError("decisions must be an array")

    expected_ids = list(candidate_ids)
    expected_set = set(expected_ids)
    by_id: dict[str, dict[str, Any]] = {}
    for index, decision in enumerate(decisions):
        validate_decision(decision, index)
        candidate_id = decision["candidate_id"]
        if candidate_id not in expected_set:
            raise ValueError(f"unknown candidate ID: {candidate_id!r}")
        if candidate_id in by_id:
            raise ValueError(f"duplicate candidate ID: {candidate_id!r}")
        by_id[candidate_id] = decision

    missing = [candidate_id for candidate_id in expected_ids if candidate_id not in by_id]
    if missing:
        raise ValueError(f"missing candidate IDs: {missing}")
    return [by_id[candidate_id] for candidate_id in expected_ids]


def validate_decision(decision: Any, index: int) -> None:
    if not isinstance(decision, dict) or set(decision) != DECISION_KEYS:
        raise ValueError(f"decision {index} has an invalid field set")
    if not isinstance(decision["candidate_id"], str):
        raise ValueError(f"decision {index} candidate_id must be a string")
    if decision["decision"] not in DECISION_KINDS:
        raise ValueError(f"decision {index} has an invalid decision kind")
    confidence = decision["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"decision {index} confidence must be a number")
    if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(f"decision {index} confidence must be in [0, 1]")
    semantic_role = decision["semantic_role"]
    if not isinstance(semantic_role, str) or not semantic_role or len(semantic_role) > 160:
        raise ValueError(f"decision {index} semantic_role is invalid")
    validate_string_list(decision["evidence_view_ids"], f"decision {index} evidence_view_ids")
    validate_string_list(decision["reason_codes"], f"decision {index} reason_codes")
    rationale = decision["rationale"]
    if not isinstance(rationale, str) or len(rationale) > 2000:
        raise ValueError(f"decision {index} rationale is invalid")


def validate_string_list(value: Any, field: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field} must be an array of non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} must not contain duplicates")


def rejected_outcome(code: str, message: str, artifacts: dict[str, str]) -> dict[str, Any]:
    return {
        "status": "rejected",
        "decisions": [],
        "error": {"code": code, "message": message},
        "artifacts": artifacts,
    }


# @entry
def main() -> int:
    parser = argparse.ArgumentParser(description="Run a fail-closed Codex source-shell visual review")
    parser.add_argument("--candidate-id", action="append", required=True)
    parser.add_argument("--image", action="append", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--codex-binary", default="codex")
    arguments = parser.parse_args()
    try:
        outcome = review_source_shell_candidates(
            candidate_ids=arguments.candidate_id,
            image_paths=arguments.image,
            prompt=arguments.prompt_file.read_text(encoding="utf-8"),
            output_dir=arguments.output_dir,
            workspace=arguments.workspace,
            timeout_seconds=arguments.timeout_seconds,
            codex_binary=arguments.codex_binary,
        )
    except (OSError, ValueError) as error:
        outcome = {
            "status": "rejected",
            "decisions": [],
            "error": {"code": "invalid_input", "message": str(error)},
            "artifacts": {},
        }
    print(json.dumps(outcome, indent=2, ensure_ascii=False))
    return 0 if outcome["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
