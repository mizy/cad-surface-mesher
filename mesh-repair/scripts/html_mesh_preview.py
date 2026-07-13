from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from mesh_io import read_surface, triangle_faces


VTK_JS_UNAVAILABLE_STUB = (
    'console.error("vtk.js unavailable: set VTK_JS_PATH to a readable local file");'
)
DEFAULT_VIEWER_TRIANGLE_LIMIT: int | None = None
SOURCE_PROJECTED_OUTPUT_KEY = "source_projected_watertight_candidate_vtp"


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
    original_comparison = outputs.get("original_mesh_vtp")
    processed_comparison = outputs.get("processed_mesh_vtp")
    if original_comparison and processed_comparison:
        specs = [
            ("Original mesh with watertight issue arrays", original_comparison),
            ("Processed candidate with watertight issue arrays", processed_comparison),
        ]
        if outputs.get("original_issue_faces_vtp"):
            specs.append(("Original issue-adjacent faces", outputs.get("original_issue_faces_vtp")))
        if outputs.get("processed_issue_faces_vtp"):
            specs.append(("Processed issue-adjacent faces", outputs.get("processed_issue_faces_vtp")))
        return specs
    stage1 = outputs.get("stage1_exterior_candidate_vtp") or input_path(report)
    proxy = outputs.get("closure_proxy_vtp")
    patch_specs = patch_preview_specs(report)
    if accepted := outputs.get("accepted_mesh_vtp"):
        projected = outputs.get(SOURCE_PROJECTED_OUTPUT_KEY)
        hybrid_candidates = {outputs.get("hybrid_fused_candidate_vtp")}
        if projected == accepted:
            label = "Accepted source-projected watertight shell"
        elif accepted in hybrid_candidates:
            label = "Accepted hybrid repair"
        else:
            label = "Accepted repair"
        return [
            ("Source exterior", stage1),
            (label, accepted),
            ("Diagnostic closure proxy", proxy),
            *patch_specs,
        ]
    if outputs.get("rejected_candidate_vtp") and outputs.get("closure_proxy_vtp"):
        return [
            ("Rejected candidate", outputs.get("rejected_candidate_vtp")),
            ("Diagnostic closure proxy", outputs.get("closure_proxy_vtp")),
            *patch_specs,
        ]
    if proxy:
        return [("Diagnostic closure proxy", proxy), *patch_specs]
    if refined := outputs.get("adaptive_refined_source_vtp"):
        return [("Original input", input_path(report)), ("Adaptive refined source", refined)]
    return []


def patch_preview_specs(report: dict[str, Any]) -> list[tuple[str, str | None]]:
    specs: list[tuple[str, str | None]] = []
    for region in report.get("patch_regions", []):
        region_id = region.get("id", "patch")
        artifacts = region.get("artifacts", {})
        if artifacts.get("proxy_patch_vtp"):
            specs.append((f"Patch proxy {region_id}", artifacts.get("proxy_patch_vtp")))
        if artifacts.get("stitch_band_vtp"):
            specs.append((f"Patch stitch band {region_id}", artifacts.get("stitch_band_vtp")))
        if artifacts.get("hole_fill_vtp"):
            specs.append((f"Constrained hole fill {region_id}", artifacts.get("hole_fill_vtp")))
        if artifacts.get("seam_belt_vtp"):
            specs.append((f"Patch seam belt {region_id}", artifacts.get("seam_belt_vtp")))
    return specs


def input_path(report: dict[str, Any]) -> str | None:
    raw_input = report.get("input")
    if isinstance(raw_input, dict):
        return raw_input.get("path")
    if isinstance(raw_input, str):
        return raw_input
    return None


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
    original_triangles = int(mesh.n_cells)
    viewer_triangle_limit = viewer_triangle_limit_from_env()
    if viewer_triangle_limit and mesh.n_cells > viewer_triangle_limit:
        reduction = 1.0 - (viewer_triangle_limit / float(mesh.n_cells))
        mesh = mesh.decimate_pro(reduction, preserve_topology=False).triangulate().clean()
    faces = triangle_faces(mesh)
    polys = np.column_stack((np.full((faces.shape[0], 1), 3, dtype=np.uint32), faces)).ravel()
    points = np.asarray(mesh.points, dtype=np.float32)
    displayed_triangles = int(faces.shape[0])
    return {
        "label": label,
        "source": str(path),
        "points": np.round(points, 6).tolist(),
        "polys": polys.astype(np.uint32).tolist(),
        "triangles": displayed_triangles,
        "original_triangles": original_triangles,
        "full_resolution": displayed_triangles == original_triangles,
        "downsampled": displayed_triangles != original_triangles,
        "viewer_triangle_limit": viewer_triangle_limit,
        "points_count": int(points.shape[0]),
        "proxy_weights": proxy_weight_values(mesh, displayed_triangles),
        "issue_values": issue_values(mesh, displayed_triangles),
    }


def viewer_triangle_limit_from_env() -> int | None:
    raw = os.environ.get("CAD_SURFACE_MESHER_VIEWER_TRIANGLES")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            return DEFAULT_VIEWER_TRIANGLE_LIMIT
        return value if value > 0 else None
    return DEFAULT_VIEWER_TRIANGLE_LIMIT


def proxy_weight_values(mesh: Any, cell_count: int) -> list[float]:
    for name in ("proxy_weight", "sdf_blend_weight"):
        if name not in mesh.cell_data:
            continue
        values = np.asarray(mesh.cell_data[name], dtype=np.float64).reshape(-1)
        if values.size != cell_count:
            continue
        return np.round(np.clip(values, 0.0, 1.0), 6).tolist()
    return []


def issue_values(mesh: Any, cell_count: int) -> list[float]:
    name = "watertight_audit_issue_mask"
    if name not in mesh.cell_data:
        return []
    values = np.asarray(mesh.cell_data[name]).reshape(-1)
    if values.size != cell_count:
        return []
    return (values != 0).astype(np.float32).tolist()


def vtk_js_source() -> str:
    env_path = os.environ.get("VTK_JS_PATH")
    if not env_path:
        return VTK_JS_UNAVAILABLE_STUB
    path = Path(env_path)
    if not path.is_file():
        return VTK_JS_UNAVAILABLE_STUB
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return VTK_JS_UNAVAILABLE_STUB


def vtk_script_section(vtk_js: str, preview_meshes: list[dict[str, Any]]) -> str:
    if not preview_meshes:
        return ""
    return "\n".join([
        "<script>", script_source(vtk_js), "</script>",
        f'<script type="application/json" id="mesh-preview-data">{script_json(preview_meshes)}</script>',
        "<script>", viewer_js(), "</script>",
    ])


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
  const vtkDataArray = vtk.Common && vtk.Common.Core && vtk.Common.Core.vtkDataArray;
  const vtkColorTransferFunction = vtk.Rendering.Core.vtkColorTransferFunction;

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
    if (mesh.issue_values && mesh.issue_values.length === mesh.triangles && vtkDataArray && vtkColorTransferFunction) {
      const scalars = vtkDataArray.newInstance({
        name: 'watertight_issue',
        numberOfComponents: 1,
        values: Float32Array.from(mesh.issue_values),
      });
      polyData.getCellData().setScalars(scalars);
      const lookup = vtkColorTransferFunction.newInstance();
      lookup.addRGBPoint(0.0, 0.55, 0.62, 0.70);
      lookup.addRGBPoint(1.0, 0.96, 0.26, 0.12);
      mapper.setLookupTable(lookup);
      if (mapper.setScalarModeToUseCellData) mapper.setScalarModeToUseCellData();
      if (mapper.setColorByArrayName) mapper.setColorByArrayName('watertight_issue');
      if (mapper.setScalarRange) mapper.setScalarRange(0, 1);
    } else if (mesh.proxy_weights && mesh.proxy_weights.length === mesh.triangles && vtkDataArray && vtkColorTransferFunction) {
      const scalars = vtkDataArray.newInstance({
        name: 'proxy_weight',
        numberOfComponents: 1,
        values: Float32Array.from(mesh.proxy_weights),
      });
      polyData.getCellData().setScalars(scalars);
      const lookup = vtkColorTransferFunction.newInstance();
      lookup.addRGBPoint(0.0, 0.22, 0.58, 0.82);
      lookup.addRGBPoint(0.5, 0.95, 0.70, 0.18);
      lookup.addRGBPoint(1.0, 0.88, 0.24, 0.16);
      mapper.setLookupTable(lookup);
      if (mapper.setScalarModeToUseCellData) mapper.setScalarModeToUseCellData();
      if (mapper.setColorByArrayName) mapper.setColorByArrayName('proxy_weight');
      if (mapper.setScalarRange) mapper.setScalarRange(0, 1);
    } else {
      actor.getProperty().setColor(index === 0 ? 0.55 : 0.2, index === 0 ? 0.62 : 0.7, index === 0 ? 0.7 : 0.35);
    }
    actor.getProperty().setOpacity(1.0);
    generic.getRenderer().addActor(actor);
    generic.getRenderer().resetCamera();
    generic.getRenderWindow().render();
  });
})();
"""
