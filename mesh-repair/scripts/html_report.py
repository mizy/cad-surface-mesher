from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.request
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from mesh_io import read_surface, triangle_faces


VTK_JS_URL = "https://unpkg.com/vtk.js@latest/vtk.js"
MAX_VIEWER_TRIANGLES = 60_000


def write_html_report(report: dict[str, Any], path: Path, title: str) -> None:
    path.write_text(render_html(report, title, path.parent), encoding="utf-8")


def render_html(report: dict[str, Any], title: str, report_dir: Path) -> str:
    vtk_js = vtk_js_source()
    preview_meshes = mesh_previews(report, report_dir)
    body = [
        f"<h1>{text(title)}</h1>",
        summary_section(report),
        diagnostics_section(report),
        viewer_section(preview_meshes),
        trace_section(report),
        change_section(report),
        defect_section(report),
        requested_section(report),
        outputs_section(report),
        visual_section(report, report_dir),
        full_json_section(report),
        vtk_script_section(vtk_js, preview_meshes),
    ]
    return "\n".join([
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{text(title)}</title>",
        "<style>",
        css(),
        "</style>",
        "</head>",
        "<body>",
        *body,
        "</body>",
        "</html>",
    ])


def summary_section(report: dict[str, Any]) -> str:
    rows = []
    if "input" in report:
        rows.append(("Input", report["input"]))
    if "target" in report:
        rows.append(("Target", report["target"]))
    if "output_contract" in report:
        rows.append(("Output Contract", report["output_contract"]))
    if "method" in report:
        rows.append(("Method", report["method"]))
    if "gates" in report:
        rows.extend((f"Gate: {key}", value) for key, value in report["gates"].items())
    return section("Summary", kv_table(rows))


def diagnostics_section(report: dict[str, Any]) -> str:
    rows = []
    for key in [
        "parameters",
        "limitations",
        "group_filter",
        "workflow_decision",
        "view_reports",
        "size_field",
        "refinement_iterations",
        "comparisons",
        "diagnostics",
    ]:
        if key in report:
            rows.append((key, report[key]))
    if not rows:
        return ""
    return section("Diagnostics", kv_table(rows))


def trace_section(report: dict[str, Any]) -> str:
    repair = report.get("repair_report") or report.get("change_report") or {}
    rows = repair.get("geometry_to_mesh_trace")
    if not rows:
        return ""
    columns = [
        "stage",
        "operation",
        "status",
        "triangles",
        "removed_triangles",
        "points",
        "boundary_edges",
        "non_manifold_edges",
        "components",
        "output",
    ]
    return section("Geometry To Mesh Trace", row_table(rows, columns))


def change_section(report: dict[str, Any]) -> str:
    repair = report.get("repair_report") or report.get("change_report") or {}
    change = repair.get("change_summary")
    if not change:
        return ""
    return section("Change Summary", object_table(change))


def defect_section(report: dict[str, Any]) -> str:
    repair = report.get("repair_report") or report.get("change_report") or {}
    defects = repair.get("defect_matrix")
    if not defects:
        return ""
    return section("Defect Matrix", object_table(defects))


def requested_section(report: dict[str, Any]) -> str:
    repair = report.get("repair_report", {})
    requested = repair.get("requested_capabilities")
    if not requested:
        return ""
    return section("Requested Capabilities", object_table(requested))


def outputs_section(report: dict[str, Any]) -> str:
    outputs = report.get("outputs")
    if not outputs:
        return ""
    return section("Outputs", object_table(outputs))


def visual_section(report: dict[str, Any], report_dir: Path) -> str:
    items = visual_items(report)
    if not items:
        return ""
    cards = []
    for image, caption in items:
        cards.append(f'<figure><img src="{image_src(image, report_dir)}"><figcaption>{text(caption)}</figcaption></figure>')
    return section("Visual Evidence", '<div class="image-grid">' + "\n".join(cards) + "</div>")


def visual_items(report: dict[str, Any]) -> list[tuple[str, str]]:
    items = [(image, image) for image in report.get("outputs", {}).get("previews", [])]
    for view in report.get("view_reports", []):
        image = view.get("image")
        if image:
            caption = f"{view.get('view', 'view')} critical depth-gradient pixels: {view.get('critical_pixels')}"
            items.append((image, caption))
    return items


def viewer_section(preview_meshes: list[dict[str, Any]]) -> str:
    if len(preview_meshes) < 2:
        return ""
    cards = []
    for index, mesh in enumerate(preview_meshes):
        cards.append(
            "\n".join([
                "<div>",
                f"<h3>{text(mesh['label'])}</h3>",
                f'<div class="vtk-viewer" id="vtk-viewer-{index}"></div>',
                f"<p class=\"muted\">Preview triangles: {mesh['triangles']}; source: {text(mesh['source'])}</p>",
                "</div>",
            ])
        )
    return section("3D Before / After", '<div class="viewer-grid">' + "\n".join(cards) + "</div>")


def full_json_section(report: dict[str, Any]) -> str:
    content = text(json.dumps(report, indent=2, ensure_ascii=False))
    return section("Full JSON", f"<details><summary>Open full machine-readable report</summary><pre>{content}</pre></details>")


def vtk_script_section(vtk_js: str, preview_meshes: list[dict[str, Any]]) -> str:
    if len(preview_meshes) < 2:
        return ""
    return "\n".join([
        "<script>",
        script_source(vtk_js),
        "</script>",
        f'<script type="application/json" id="mesh-preview-data">{script_json(preview_meshes)}</script>',
        "<script>",
        viewer_js(),
        "</script>",
    ])


def section(title: str, content: str) -> str:
    if not content:
        return ""
    return f"<section><h2>{text(title)}</h2>{content}</section>"


def kv_table(rows: list[tuple[str, Any]]) -> str:
    body = "\n".join(f"<tr><th>{text(key)}</th><td>{value_html(value)}</td></tr>" for key, value in rows)
    return f"<table>{body}</table>"


def row_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    head = "".join(f"<th>{text(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{value_html(row.get(column))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def object_table(value: Any) -> str:
    if isinstance(value, dict):
        return kv_table(list(value.items()))
    if isinstance(value, list):
        rows = [(str(index), item) for index, item in enumerate(value)]
        return kv_table(rows)
    return f"<p>{value_html(value)}</p>"


def value_html(value: Any) -> str:
    if isinstance(value, bool):
        return f'<span class="bool-{str(value).lower()}">{str(value).lower()}</span>'
    if isinstance(value, (dict, list)):
        return f"<pre>{text(json.dumps(value, indent=2, ensure_ascii=False))}</pre>"
    if value is None:
        return '<span class="muted">null</span>'
    return text(str(value))


def relative_image(path: str) -> str:
    parts = Path(path).parts
    if "visual" in parts:
        index = parts.index("visual")
        return str(Path(*parts[index:]))
    return path


def image_src(raw_path: str, report_dir: Path) -> str:
    path = resolve_path(raw_path, report_dir)
    if not path or not path.exists():
        return text(relative_image(raw_path))
    mime_type = mimetypes.guess_type(path)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def mesh_previews(report: dict[str, Any], report_dir: Path) -> list[dict[str, Any]]:
    specs = preview_specs(report)
    previews = []
    for label, raw_path in specs:
        path = resolve_path(raw_path, report_dir)
        if path and path.exists():
            try:
                previews.append(mesh_preview(label, path))
            except Exception as exc:
                previews.append({"label": label, "source": str(path), "error": str(exc), "points": [], "polys": []})
    return [preview for preview in previews if preview.get("points") and preview.get("polys")]


def preview_specs(report: dict[str, Any]) -> list[tuple[str, str | None]]:
    outputs = report.get("outputs", {})
    if "stage2_watertight_surface_vtp" in outputs:
        return [
            ("Before: dirty input mesh", report.get("input", {}).get("path")),
            ("After: watertight mesh", outputs.get("stage2_watertight_surface_vtp")),
        ]
    if "adaptive_refined_source_vtp" in outputs:
        return [
            ("Before: source exterior candidate", report.get("input")),
            ("After: adaptive refined source", outputs.get("adaptive_refined_source_vtp")),
        ]
    return []


def resolve_path(raw_path: str | None, report_dir: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = report_dir / path
    return candidate if candidate.exists() else path


def mesh_preview(label: str, path: Path) -> dict[str, Any]:
    mesh = read_surface(path)
    if mesh.n_cells > MAX_VIEWER_TRIANGLES:
        reduction = 1.0 - (MAX_VIEWER_TRIANGLES / float(mesh.n_cells))
        mesh = mesh.decimate_pro(reduction, preserve_topology=False).triangulate().clean()
    faces = triangle_faces(mesh)
    polys = np.column_stack((np.full((faces.shape[0], 1), 3, dtype=np.uint32), faces)).ravel()
    points = np.asarray(mesh.points, dtype=np.float32)
    return {
        "label": label,
        "source": str(path),
        "points": np.round(points, 6).tolist(),
        "polys": polys.astype(np.uint32).tolist(),
        "triangles": int(faces.shape[0]),
        "points_count": int(points.shape[0]),
    }


def vtk_js_source() -> str:
    env_path = os.environ.get("VTK_JS_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path).read_text(encoding="utf-8")
    cache_path = Path.home() / ".cache" / "cad-surface-mesher" / "vtk.js"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(VTK_JS_URL, timeout=30) as response:
            source = response.read().decode("utf-8")
        cache_path.write_text(source, encoding="utf-8")
        return source
    except Exception as exc:
        return f"console.error({json.dumps('vtk.js unavailable: ' + str(exc))});"


def script_source(source: str) -> str:
    return source.replace("</script", "<\\/script")


def script_json(value: Any) -> str:
    data = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def viewer_js() -> str:
    return """
(function () {
  const raw = document.getElementById('mesh-preview-data');
  if (!raw || typeof vtk === 'undefined') {
    document.querySelectorAll('.vtk-viewer').forEach((container) => {
      container.textContent = 'vtk.js failed to load';
    });
    return;
  }
  const meshes = JSON.parse(raw.textContent);
  const vtkGenericRenderWindow = vtk.Rendering.Misc.vtkGenericRenderWindow;
  const vtkActor = vtk.Rendering.Core.vtkActor;
  const vtkMapper = vtk.Rendering.Core.vtkMapper;
  const vtkPolyData = vtk.Common.DataModel.vtkPolyData;

  meshes.forEach((mesh, index) => {
    const container = document.getElementById(`vtk-viewer-${index}`);
    if (!container) return;
    const generic = vtkGenericRenderWindow.newInstance({ background: [0.98, 0.99, 1.0] });
    generic.setContainer(container);
    generic.resize();
    const polyData = vtkPolyData.newInstance();
    polyData.getPoints().setData(Float32Array.from(mesh.points.flat()), 3);
    polyData.getPolys().setData(Uint32Array.from(mesh.polys));
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(polyData);
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    actor.getProperty().setColor(index === 0 ? 0.55 : 0.2, index === 0 ? 0.62 : 0.7, index === 0 ? 0.7 : 0.35);
    actor.getProperty().setOpacity(1.0);
    generic.getRenderer().addActor(actor);
    generic.getRenderer().resetCamera();
    generic.getRenderWindow().render();
  });
})();
"""


def text(value: str) -> str:
    return escape(value, quote=True)


def css() -> str:
    return """
body {
  color: #172026;
  background: #f6f8fa;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.45;
  margin: 0;
  padding: 28px;
}
h1, h2 { color: #111827; }
section {
  background: #fff;
  border: 1px solid #d8dee4;
  border-radius: 8px;
  margin: 18px 0;
  padding: 18px;
}
table {
  border-collapse: collapse;
  width: 100%;
}
th, td {
  border: 1px solid #d8dee4;
  padding: 8px;
  text-align: left;
  vertical-align: top;
}
th { background: #f1f5f9; width: 220px; }
pre {
  background: #0f172a;
  border-radius: 6px;
  color: #e5e7eb;
  max-height: 460px;
  overflow: auto;
  padding: 12px;
  white-space: pre-wrap;
}
.bool-true { color: #067647; font-weight: 700; }
.bool-false { color: #b42318; font-weight: 700; }
.muted { color: #667085; }
.image-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
}
.viewer-grid {
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
}
.vtk-viewer {
  background: #e5e7eb;
  border: 1px solid #d8dee4;
  height: 420px;
  min-width: 0;
}
figure { margin: 0; }
img {
  background: #111827;
  border: 1px solid #d8dee4;
  max-width: 100%;
}
figcaption {
  color: #475467;
  font-size: 12px;
  margin-top: 4px;
  word-break: break-all;
}
"""
