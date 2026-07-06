from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gmsh_pipeline import DependencyError, TessellationControls, TessellationError, TessellationResult


SUPPORTED_MESH_SUFFIXES = {
    ".stl": "stl",
    ".obj": "obj",
    ".vtp": "vtp",
    ".vtk": "vtk",
    ".glb": "glb",
    ".gltf": "gltf",
}


@dataclass(frozen=True)
class MeshRuntimeModules:
    np: Any
    pv: Any
    vtk: Any
    trimesh: Any | None = None


def tessellate_mesh(controls: TessellationControls) -> TessellationResult:
    input_path = controls.input_path.expanduser().resolve()
    output_dir = controls.output_dir.expanduser().resolve()
    if not input_path.exists():
        raise TessellationError(f"Input mesh file does not exist: {input_path}")
    mesh_format = resolve_mesh_format(input_path)
    runtime = load_mesh_runtime(needs_trimesh=mesh_format in {"glb", "gltf"})
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = read_mesh(input_path, mesh_format, runtime)
    surface = extract_triangle_surface(mesh, runtime)
    if surface["triangles"].shape[0] == 0:
        raise TessellationError("Mesh input produced an empty triangle surface.")

    mesh_path = output_dir / "surface_mesh.vtp"
    from .mesh_export import write_vtp

    write_vtp(
        runtime.pv,
        runtime.np,
        mesh_path,
        surface["points"],
        surface["triangles"],
        {"source_triangle_index": surface["source_triangle_index"]},
    )

    from .report import build_mesh_input_report

    report_path = output_dir / "tessellation_report.json"
    warnings = mesh_input_warnings(controls)
    report = build_mesh_input_report(
        np=runtime.np,
        input_path=input_path,
        mesh_format=mesh_format,
        controls=controls,
        mesh=surface,
        mesh_path=mesh_path,
        report_path=report_path,
        warnings=warnings,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return TessellationResult(mesh_path=mesh_path, report_path=report_path, debug_msh_path=None, report=report)


def resolve_mesh_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_MESH_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_MESH_SUFFIXES))
        raise TessellationError(f"Cannot infer mesh format from {path.name}; supported suffixes: {supported}")
    return SUPPORTED_MESH_SUFFIXES[suffix]


def load_mesh_runtime(*, needs_trimesh: bool) -> MeshRuntimeModules:
    modules: dict[str, Any] = {}
    failures: list[tuple[str, BaseException]] = []
    for name in ("numpy", "pyvista", "vtk"):
        try:
            modules[name] = importlib.import_module(name)
        except (ImportError, OSError) as exc:
            failures.append((name, exc))
    trimesh_module = None
    if needs_trimesh:
        try:
            trimesh_module = importlib.import_module("trimesh")
        except (ImportError, OSError) as exc:
            failures.append(("trimesh", exc))

    if failures:
        details = "\n".join(f"  - {name}: {exc}" for name, exc in failures)
        raise DependencyError(
            "Missing mesh surface runtime dependencies:\n"
            f"{details}\n"
            "Install them with: python -m pip install -r requirements.txt"
        )
    return MeshRuntimeModules(np=modules["numpy"], pv=modules["pyvista"], vtk=modules["vtk"], trimesh=trimesh_module)


def read_mesh(path: Path, mesh_format: str, runtime: MeshRuntimeModules) -> Any:
    if mesh_format in {"glb", "gltf"}:
        return read_mesh_with_trimesh(path, runtime)
    mesh = runtime.pv.read(path)
    if isinstance(mesh, runtime.pv.MultiBlock):
        mesh = mesh.combine()
    return mesh


def read_mesh_with_trimesh(path: Path, runtime: MeshRuntimeModules) -> Any:
    if runtime.trimesh is None:
        raise TessellationError("trimesh runtime is required for GLB/GLTF inputs.")
    loaded = runtime.trimesh.load(path, force="scene", process=False)
    if hasattr(loaded, "geometry"):
        geometries = [geom for geom in loaded.dump(concatenate=False) if getattr(geom, "faces", None) is not None]
        if not geometries:
            raise TessellationError("GLB/GLTF scene contains no triangle geometries.")
        loaded = runtime.trimesh.util.concatenate(geometries)
    if getattr(loaded, "vertices", None) is None or getattr(loaded, "faces", None) is None:
        raise TessellationError("trimesh reader did not return vertices and faces.")

    vertices = runtime.np.asarray(loaded.vertices, dtype=runtime.np.float64)
    faces = runtime.np.asarray(loaded.faces, dtype=runtime.np.int64)
    packed_faces = runtime.np.column_stack((runtime.np.full(len(faces), 3, dtype=runtime.np.int64), faces)).reshape(-1)
    return runtime.pv.PolyData(vertices, packed_faces)


def extract_triangle_surface(mesh: Any, runtime: MeshRuntimeModules) -> dict[str, Any]:
    if not isinstance(mesh, runtime.pv.PolyData):
        mesh = mesh.extract_surface(algorithm="dataset_surface")
    surface = mesh.extract_surface(algorithm="dataset_surface").triangulate().clean()
    faces = runtime.np.asarray(surface.faces)
    if faces.size == 0 or faces.size % 4 != 0:
        raise TessellationError("Triangulated mesh has no packed triangle faces.")
    packed = faces.reshape(-1, 4)
    if not runtime.np.all(packed[:, 0] == 3):
        raise TessellationError("Mesh triangulation did not produce triangle-only faces.")
    triangles = packed[:, 1:].astype(runtime.np.int64, copy=False)
    return {
        "points": runtime.np.asarray(surface.points, dtype=runtime.np.float64),
        "triangles": triangles,
        "source_triangle_index": runtime.np.arange(triangles.shape[0], dtype=runtime.np.int64),
        "all_triangles": True,
    }


def mesh_input_warnings(controls: TessellationControls) -> list[str]:
    ignored = []
    for name in (
        "mesh_size",
        "mesh_size_min",
        "mesh_size_max",
        "angle_deg",
        "chord",
        "occ_target_unit",
        "import_labels",
        "save_debug_msh",
    ):
        value = getattr(controls, name)
        if name == "angle_deg" and value == 20.0:
            continue
        if name == "occ_target_unit" and value == "auto":
            continue
        if name == "import_labels" and value is True:
            continue
        if name == "save_debug_msh" and value is False:
            continue
        if value is not None:
            ignored.append(name)
    if not ignored:
        return []
    return [
        "Mesh inputs are already discretized; CAD tessellation controls were ignored: "
        + ", ".join(sorted(ignored))
    ]
