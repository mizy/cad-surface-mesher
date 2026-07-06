from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


def write_html_report(report: dict[str, Any], path: Path, title: str) -> None:
    path.write_text(render_html(report, title), encoding="utf-8")


def render_html(report: dict[str, Any], title: str) -> str:
    body = [
        f"<h1>{text(title)}</h1>",
        summary_section(report),
        trace_section(report),
        change_section(report),
        defect_section(report),
        requested_section(report),
        outputs_section(report),
        visual_section(report),
        full_json_section(report),
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
    if "method" in report:
        rows.append(("Method", report["method"]))
    if "gates" in report:
        rows.extend((f"Gate: {key}", value) for key, value in report["gates"].items())
    return section("Summary", kv_table(rows))


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


def visual_section(report: dict[str, Any]) -> str:
    outputs = report.get("outputs", {})
    images = outputs.get("previews", [])
    if not images:
        return ""
    cards = []
    for image in images:
        cards.append(f'<figure><img src="{text(relative_image(image))}"><figcaption>{text(image)}</figcaption></figure>')
    return section("Visual Evidence", '<div class="image-grid">' + "\n".join(cards) + "</div>")


def full_json_section(report: dict[str, Any]) -> str:
    content = text(json.dumps(report, indent=2, ensure_ascii=False))
    return section("Full JSON", f"<details><summary>Open full machine-readable report</summary><pre>{content}</pre></details>")


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
