from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def to_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return to_json_value(value.tolist())
    if isinstance(value, np.generic):
        return to_json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return "NaN"
        return "+Infinity" if value > 0.0 else "-Infinity"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    return value


def to_audit_value(value: Any) -> Any:
    """Represent potentially large provenance arrays with stable hashes."""

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        if array.dtype.hasobject:
            stable = json.dumps(
                to_json_value(array.tolist()),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            digest.update(stable.encode("utf-8"))
        else:
            digest.update(array.tobytes())
        flat = array.reshape(-1)
        return {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": digest.hexdigest(),
            "sample": to_json_value(flat[:16]),
        }
    if isinstance(value, Mapping):
        return {str(key): to_audit_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_audit_value(item) for item in value]
    return to_json_value(value)


def write_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist a report without exposing a partially written decision file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                to_json_value(payload),
                handle,
                indent=2,
                sort_keys=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
