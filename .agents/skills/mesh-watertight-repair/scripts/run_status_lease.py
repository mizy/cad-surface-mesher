from __future__ import annotations

import fcntl
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any


LEASE_NAME = ".mesh_repair_run.lease"


def clean_output_artifacts(
    output_dir: Path, *, preserve_run_status: bool = False
) -> None:
    for path in output_artifact_paths(output_dir):
        if preserve_run_status and path == output_dir / "run_status.json":
            continue
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def output_artifact_paths(output_dir: Path) -> list[Path]:
    paths = [
        output_dir / "visibility_labeled_source.vtp",
        output_dir / "stage1_exterior_candidate.vtp",
        output_dir / "source_preserving_candidate.vtp",
        output_dir / "closure_proxy.vtp",
        output_dir / "source_projected_watertight_candidate.vtp",
        output_dir / "source_projection_report.json",
        output_dir / "ai_policy_packet.json",
        output_dir / "two_stage_report.json",
        output_dir / "two_stage_report.html",
        output_dir / "run_status.json",
        output_dir / "visual",
        output_dir / "debug",
        output_dir / "opening_policy_evidence",
    ]
    paths.extend(output_dir.glob(".run_status.json.*.tmp"))
    return paths


def acquire_run_lease(args: Any, command_fingerprint: str) -> str:
    """Take a lightweight process lock for one output directory."""
    generation_id = str(uuid.uuid4())
    lease_path = Path(args.output_dir) / LEASE_NAME
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lease_path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise FileExistsError(
            f"output directory is already in use by another mesh-repair run: {lease_path}"
        ) from exc

    try:
        payload = {
            "generation_id": generation_id,
            "pid": os.getpid(),
            "command_fingerprint": command_fingerprint,
        }
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    except BaseException:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise

    args._run_generation_id = generation_id
    args._run_lease_descriptor = descriptor
    return generation_id


def ensure_generation_id(args: Any) -> str:
    generation_id = getattr(args, "_run_generation_id", None)
    if isinstance(generation_id, str) and generation_id:
        return generation_id
    generation_id = str(uuid.uuid4())
    args._run_generation_id = generation_id
    return generation_id


def release_run_lease(args: Any) -> None:
    descriptor = getattr(args, "_run_lease_descriptor", None)
    if not isinstance(descriptor, int):
        return
    fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)
    args._run_lease_descriptor = None
