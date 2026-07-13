#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import signal
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any, Callable

import numpy as np
import pyvista as pv

from html_report import write_html_report
from mesh_io import compact_mesh, triangle_faces, write_vtp
from mesh_metrics import mesh_report
from source_shell_ai_review import review_source_shell_candidates
from source_shell_candidates import (
    describe_components,
    face_component_labels,
    remove_low_risk_faces,
)
from source_shell_selection import (
    attach_component_visibility,
    select_ai_candidates,
    select_source_shell_faces,
)
from source_shell_visibility import build_source_shell_visibility
from source_shell_visuals import render_candidate_contact_sheets


TARGET_NAME = "source-accurate-open-exterior-shell"
AI_PROMPT = """You are reviewing deterministic multi-view evidence for components of a dirty
triangle assembly. Classify geometry, not code. A component is remove_internal only when the
images show that it is an enclosed cabin/internal/support/duplicate-inner part and not a real
outer skin feature. Use keep_exterior for a legitimate exposed exterior part, split_required
when exterior and interior geometry are mixed, and ambiguous whenever the evidence is not
decisive. Do not infer removal from small size alone. Cite the contact-sheet page/panel IDs in
evidence_view_ids. This review authorizes a deletion transaction, so prefer ambiguity over an
unsafe removal.
Keep semantic_role to a short phrase, use at most three reason_codes and one evidence_view_id,
and keep rationale under 20 words so every candidate fits in one compact response.
Use only the attached images and exact candidate IDs. Do not call tools, inspect files, or read
repository instructions; return the completed decision object immediately.
"""


class SourceShellDeadlineExceeded(TimeoutError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one source-accurate open exterior shell without voxelization, remeshing, "
            "projection, or hole filling."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--direction-count", type=int, default=42)
    parser.add_argument("--visibility-grid", type=int, default=128)
    parser.add_argument("--depth-tolerance", type=float, default=0.0)
    parser.add_argument("--continuity-rings", type=int, default=1)
    parser.add_argument("--min-first-hit-views", type=int, default=1)
    parser.add_argument("--min-first-hit-pixels", type=int, default=1)
    parser.add_argument("--ai-max-candidates", type=int, default=48)
    parser.add_argument("--ai-min-component-faces", type=int, default=2)
    parser.add_argument("--ai-remove-confidence", type=float, default=0.85)
    parser.add_argument("--ai-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--deadline-seconds", type=float, default=600.0)
    return parser.parse_args()


# @entry dirty mesh -> one open source shell.
def run_source_shell(
    args: argparse.Namespace,
    *,
    ai_reviewer: Callable[..., dict[str, Any]] = review_source_shell_candidates,
) -> dict[str, Any]:
    started = time.monotonic()
    prepare_output_dir(args.output_dir)
    status = initial_status(args)
    write_json(args.output_dir / "run_status.json", status)
    try:
        update_status(args.output_dir, status, "read_input")
        source_points, source_faces = read_source_triangle_mesh(args.input)
        source_ids = np.arange(source_faces.shape[0], dtype=np.int64)
        original_metrics = mesh_report(source_points, source_faces)

        update_status(args.output_dir, status, "low_risk_cleanup")
        clean_points, clean_faces, clean_sources, cleanup = remove_low_risk_faces(
            source_points, source_faces, source_ids
        )
        clean_metrics = mesh_report(clean_points, clean_faces)
        labels = face_component_labels(clean_faces)
        descriptions = describe_components(clean_points, clean_faces, clean_sources, labels)

        update_status(args.output_dir, status, "multi_view_first_hit")
        visibility = build_source_shell_visibility(
            clean_points,
            clean_faces,
            direction_count=args.direction_count,
            grid_size=args.visibility_grid,
            depth_tolerance=args.depth_tolerance,
        )
        component_rows = attach_component_visibility(
            descriptions,
            labels,
            visibility.face_first_hit_view_count,
            visibility.face_first_hit_pixel_support,
        )
        ai_candidates = select_ai_candidates(
            component_rows,
            max_candidates=args.ai_max_candidates,
            min_face_count=args.ai_min_component_faces,
        )

        update_status(args.output_dir, status, "codex_visual_review")
        ai_review, evidence = run_ai_review(
            args,
            clean_points,
            clean_faces,
            labels,
            ai_candidates,
            ai_reviewer,
            started,
        )
        ai_decisions = ai_review.get("decisions", []) if ai_review.get("status") == "accepted" else []

        update_status(args.output_dir, status, "face_level_selection")
        selected, selection_reason, ai_code, selection_report = select_source_shell_faces(
            clean_faces,
            labels,
            visibility.face_first_hit_view_count,
            visibility.face_first_hit_pixel_support,
            ai_decisions,
            min_first_hit_views=args.min_first_hit_views,
            min_first_hit_pixels=args.min_first_hit_pixels,
            ai_remove_confidence=args.ai_remove_confidence,
            continuity_rings=args.continuity_rings,
        )
        if not np.any(selected):
            raise ValueError("source-shell selection produced no triangles")
        selected_face_ids = np.flatnonzero(selected)
        shell_points, shell_faces = compact_mesh(clean_points, clean_faces, selected_face_ids)
        shell_sources = clean_sources[selected_face_ids]
        shell_path = args.output_dir / "source_shell.vtp"
        write_vtp(
            shell_path,
            shell_points,
            shell_faces,
            {
                "source_triangle_index": shell_sources,
                "source_component_id": labels[selected_face_ids],
                "first_hit_view_count": visibility.face_first_hit_view_count[selected_face_ids],
                "first_hit_pixel_support": visibility.face_first_hit_pixel_support[selected_face_ids],
                "external_direction": visibility.face_external_direction[selected_face_ids],
                "external_direction_resultant": visibility.face_external_direction_resultant[selected_face_ids],
                "shell_selection_reason": selection_reason[selected_face_ids],
                "ai_decision_code": ai_code[selected_face_ids],
                "source_geometry_exact": np.ones(selected_face_ids.size, dtype=np.uint8),
            },
        )
        shell_metrics = mesh_report(shell_points, shell_faces)
        final_preview = render_final_preview(shell_points, shell_faces, args.output_dir)

        report = build_report(
            args,
            original_metrics=original_metrics,
            clean_metrics=clean_metrics,
            shell_metrics=shell_metrics,
            cleanup=cleanup,
            visibility=visibility.report,
            component_rows=component_rows,
            ai_candidates=ai_candidates,
            ai_review=ai_review,
            evidence=evidence,
            selection=selection_report,
            shell_path=shell_path,
            final_preview=final_preview,
            elapsed_seconds=time.monotonic() - started,
        )
        report_path = args.output_dir / "source_shell_report.json"
        html_path = args.output_dir / "source_shell_report.html"
        report["outputs"].update(
            {"report_json": str(report_path), "html_report": str(html_path)}
        )
        write_json(report_path, report)
        write_html_report(report, html_path, "Source-Accurate Open Exterior Shell")
        status.update(
            {
                "status": "completed",
                "stage": "completed",
                "updated_at": timestamp(),
                "elapsed_seconds": time.monotonic() - started,
                "decision": report["decision"],
                "outputs": report["outputs"],
            }
        )
        write_json(args.output_dir / "run_status.json", status)
        return report
    except BaseException as error:
        status.update(failure_status(error, started))
        write_json(args.output_dir / "run_status.json", status)
        raise


def read_source_triangle_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = pv.read(path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface(algorithm="dataset_surface")
    surface = mesh.triangulate()
    if surface.n_points == 0 or surface.n_cells == 0:
        raise ValueError(f"empty source mesh: {path}")
    return np.asarray(surface.points, dtype=np.float64), triangle_faces(surface)


def run_ai_review(
    args: argparse.Namespace,
    points: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    candidates: list[dict[str, Any]],
    ai_reviewer: Callable[..., dict[str, Any]],
    started: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not candidates:
        return {"status": "not_needed", "decisions": [], "error": None, "artifacts": {}}, {
            "candidate_count": 0,
            "pages": [],
        }
    visual_rows = [slim_candidate(row) for row in candidates]
    evidence = render_candidate_contact_sheets(
        points,
        faces,
        labels,
        visual_rows,
        args.output_dir / "ai_evidence",
    )
    remaining = max(1.0, float(args.deadline_seconds) - (time.monotonic() - started))
    timeout = min(float(args.ai_timeout_seconds), remaining)
    pages = evidence["pages"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(pages))) as pool:
        futures = [
            pool.submit(review_evidence_page, page, args, ai_reviewer, timeout)
            for page in pages
        ]
        batches = [future.result() for future in futures]
    failed = [index + 1 for index, batch in enumerate(batches) if batch.get("status") != "accepted"]
    artifacts = {"batches": [batch.get("artifacts", {}) for batch in batches]}
    if failed:
        return {
            "status": "rejected",
            "decisions": [],
            "error": {"code": "batch_review_failed", "failed_pages": failed},
            "artifacts": artifacts,
        }, evidence
    decisions = [decision for batch in batches for decision in batch["decisions"]]
    return {
        "status": "accepted",
        "decisions": decisions,
        "error": None,
        "artifacts": artifacts,
    }, evidence


def review_evidence_page(
    page: dict[str, Any],
    args: argparse.Namespace,
    ai_reviewer: Callable[..., dict[str, Any]],
    timeout: float,
) -> dict[str, Any]:
    page_number = int(page["page"])
    with tempfile.TemporaryDirectory(prefix=f"source-shell-codex-{page_number:03d}-") as workspace:
        return ai_reviewer(
            candidate_ids=[str(candidate_id) for candidate_id in page["candidate_ids"]],
            image_paths=[page["path"]],
            prompt=AI_PROMPT,
            output_dir=args.output_dir / "ai_review" / f"page_{page_number:03d}",
            workspace=Path(workspace),
            timeout_seconds=timeout,
            codex_binary=args.codex_binary,
        )


def render_final_preview(points: np.ndarray, faces: np.ndarray, output_dir: Path) -> str:
    labels = np.zeros(faces.shape[0], dtype=np.int64)
    manifest = render_candidate_contact_sheets(
        points,
        faces,
        labels,
        [
            {
                "candidate_id": "source_shell",
                "component_id": 0,
                "face_count": int(faces.shape[0]),
                "first_hit_face_count": int(faces.shape[0]),
                "first_hit_view_max": 0,
            }
        ],
        output_dir / "visual",
        page_size=1,
    )
    return str(manifest["pages"][0]["path"])


def build_report(
    args: argparse.Namespace,
    **data: Any,
) -> dict[str, Any]:
    ai_review = data["ai_review"]
    ai_required = bool(data["ai_candidates"])
    gates = {
        "one_primary_mesh": gate(True, 1, 1),
        "source_shell_nonempty": gate(data["shell_metrics"]["triangles"] > 0, data["shell_metrics"]["triangles"], "> 0"),
        "source_geometry_only": gate(True, "original vertices and selected source triangles", "no generated surface"),
        "no_voxel_remesh_projection_or_hole_fill": gate(True, True, True),
        "codex_visual_review_completed": gate(
            not ai_required or ai_review.get("status") == "accepted",
            ai_review.get("status"),
            "accepted or not_needed",
        ),
    }
    accepted = all(row["passed"] for row in gates.values())
    shell_path = str(data["shell_path"])
    previews = [page["path"] for page in data["evidence"].get("pages", [])]
    previews.append(data["final_preview"])
    return {
        "schema": "source_shell_report/v1",
        "decision": {
            "status": "accepted" if accepted else "rejected",
            "target_achieved": accepted,
            "final_output_path": shell_path if accepted else None,
            "reason_codes": [name for name, row in gates.items() if not row["passed"]],
        },
        "input": {"path": str(args.input), "kind": "mesh", "geometry_truth": "input mesh"},
        "target": {
            "name": TARGET_NAME,
            "one_primary_mesh_object": True,
            "connected_component_count": "diagnostic_only",
            "boundary_edges": "allowed_and_reported",
        },
        "geometry_contract": {
            "voxelization": False,
            "remesh": False,
            "projection": False,
            "hole_fill": False,
            "generated_surface": False,
            "easy_repairs": ["exact_degenerate_removal", "exact_duplicate_removal"],
        },
        "parameters": run_parameters(args),
        "stages": {
            "original_input": metric_summary(data["original_metrics"]),
            "low_risk_cleaned_source": metric_summary(data["clean_metrics"]),
            "source_shell": metric_summary(data["shell_metrics"]),
        },
        "cleanup": data["cleanup"],
        "visibility": data["visibility"],
        "components": component_summary(data["component_rows"]),
        "ai_review": {
            **ai_review,
            "candidate_count": len(data["ai_candidates"]),
            "candidate_ids": [row["candidate_id"] for row in data["ai_candidates"]],
        },
        "selection": data["selection"],
        "gates": gates,
        "outputs": {"source_shell_vtp": shell_path, "previews": previews},
        "elapsed_seconds": float(data["elapsed_seconds"]),
        "limitations": [
            "This stage intentionally leaves real holes and open boundaries unfilled.",
            "Ambiguous zero-first-hit components are excluded unless Codex explicitly protects them.",
            "Local intersection splitting and non-zero-width seam repair are outside this first successful source-shell cut.",
        ],
    }


def slim_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"face_ids", "source_triangle_ids"}
    }


def component_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "faces": int(sum(int(row["face_count"]) for row in rows)),
        "components_with_first_hit": int(sum(int(row["first_hit_face_count"]) > 0 for row in rows)),
        "fully_unseen_components": int(sum(int(row["first_hit_face_count"]) == 0 for row in rows)),
    }


def metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "points": metrics["points"],
        "triangles": metrics["triangles"],
        "boundary_edges": metrics["topology"]["boundary_edges"],
        "non_manifold_edges": metrics["topology"]["non_manifold_edges"],
        "non_manifold_vertices": metrics["topology"]["non_manifold_vertices"],
        "inconsistent_winding_edges": metrics["topology"]["inconsistent_winding_edges"],
        "degenerate_faces": metrics["quality"]["degenerate_faces"],
        "components": metrics["topology"]["components"]["count"],
    }


def gate(passed: bool, value: Any, threshold: Any) -> dict[str, Any]:
    return {"required": True, "passed": bool(passed), "value": value, "threshold": threshold}


def run_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        name: value
        for name, value in vars(args).items()
        if name not in {"input", "output_dir"}
    }


def prepare_output_dir(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"output directory must not exist or must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def initial_status(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "running",
        "stage": "starting",
        "started_at": timestamp(),
        "updated_at": timestamp(),
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "decision": {"status": "pending", "target_achieved": False},
    }


def update_status(output_dir: Path, status: dict[str, Any], stage: str) -> None:
    status["stage"] = stage
    status["updated_at"] = timestamp()
    write_json(output_dir / "run_status.json", status)


def failure_status(error: BaseException, started: float) -> dict[str, Any]:
    return {
        "status": "failed",
        "stage": "failed",
        "updated_at": timestamp(),
        "elapsed_seconds": time.monotonic() - started,
        "decision": {"status": "rejected", "target_achieved": False},
        "failure": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _deadline_handler(_signal: int, _frame: FrameType | None) -> None:
    raise SourceShellDeadlineExceeded("source-shell pipeline exceeded its hard deadline")


def main() -> int:
    args = parse_args()
    previous_handler = signal.signal(signal.SIGALRM, _deadline_handler)
    signal.setitimer(signal.ITIMER_REAL, float(args.deadline_seconds))
    try:
        report = run_source_shell(args)
        print(json.dumps({"decision": report["decision"], "outputs": report["outputs"]}, indent=2))
        return 0 if report["decision"]["status"] == "accepted" else 2
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
