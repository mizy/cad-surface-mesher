from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from agent_repair_contract import decision_schema, validate_agent_response
from source_shell_ai_review import (
    CodexReviewTimeout,
    build_codex_command,
    run_codex_process,
)


POST_REPAIR_PROMPT = """You are the semantic supervisor of a deterministic mesh-repair run.
Review the registered source, SDF proxy, projected-candidate, and defect images. Geometry metrics
and gates are authoritative: never mark a failed deterministic gate as acceptable. For the global
audit, accept only when no semantic omission, incorrect opening closure, lost exterior component,
or conspicuous SDF artifact is visible. For opening regions choose cap, reconstruct_wall,
porous_surface, ask, or reject according to the declared target. Cite only provided view IDs.
Choose region-level actions only; never invent point coordinates, vertices, faces, or triangles.
Return the schema object immediately without calling tools.
"""


def review_repaired_candidate(
    packet: dict[str, Any],
    *,
    output_dir: Path,
    timeout_seconds: float,
    codex_binary: str = "codex",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / "agent_decision_schema.json"
    response_path = output_dir / "agent_response.json"
    events_path = output_dir / "agent_events.jsonl"
    packet_path = output_dir / "observation_packet.json"
    schema_path.write_text(
        json.dumps(decision_schema(packet), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    packet_path.write_text(
        json.dumps(packet, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    image_paths = [Path(row["path"]).resolve() for row in packet.get("views", [])]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if not image_paths or missing:
        return rejected(
            "missing_evidence",
            f"post-repair review requires existing evidence images; missing={missing}",
            schema_path,
            response_path,
            events_path,
            packet_path,
        )
    with tempfile.TemporaryDirectory(prefix="mesh-repair-agent-review-") as workspace:
        command = build_codex_command(
            codex_binary=codex_binary,
            workspace=Path(workspace),
            image_paths=image_paths,
            schema_path=schema_path,
            response_path=response_path,
        )
        prompt = (
            POST_REPAIR_PROMPT
            + "\nObservation packet:\n"
            + json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
        )
        try:
            return_code, stderr = run_codex_process(
                command,
                prompt=prompt,
                workspace=Path(workspace),
                events_path=events_path,
                timeout_seconds=float(timeout_seconds),
            )
        except CodexReviewTimeout as error:
            return rejected("timeout", str(error), schema_path, response_path, events_path, packet_path)
        except FileNotFoundError as error:
            return rejected("launch_failed", str(error), schema_path, response_path, events_path, packet_path)
        except Exception as error:
            return rejected("process_failed", str(error), schema_path, response_path, events_path, packet_path)
    if return_code != 0:
        return rejected(
            "codex_failed",
            stderr.strip()[-2000:] or f"Codex exited with status {return_code}",
            schema_path,
            response_path,
            events_path,
            packet_path,
        )
    try:
        response = json.loads(response_path.read_text(encoding="utf-8"))
        decisions = validate_agent_response(packet, response)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return rejected("invalid_response", str(error), schema_path, response_path, events_path, packet_path)
    return {
        "status": "accepted",
        "decisions": decisions,
        "error": None,
        "artifacts": artifact_paths(schema_path, response_path, events_path, packet_path),
    }


def rejected(
    code: str,
    detail: str,
    schema_path: Path,
    response_path: Path,
    events_path: Path,
    packet_path: Path,
) -> dict[str, Any]:
    return {
        "status": "rejected",
        "decisions": [],
        "error": {"code": code, "detail": detail},
        "artifacts": artifact_paths(schema_path, response_path, events_path, packet_path),
    }


def artifact_paths(
    schema_path: Path,
    response_path: Path,
    events_path: Path,
    packet_path: Path,
) -> dict[str, str]:
    return {
        "schema": str(schema_path),
        "response": str(response_path),
        "events": str(events_path),
        "observation_packet": str(packet_path),
    }
