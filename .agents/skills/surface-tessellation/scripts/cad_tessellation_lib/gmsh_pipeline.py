from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_SUFFIXES = {
    ".step": "step",
    ".stp": "step",
    ".iges": "iges",
    ".igs": "iges",
    ".brep": "brep",
    ".brp": "brep",
}


class DependencyError(RuntimeError):
    pass


class TessellationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeModules:
    gmsh: Any
    np: Any
    pv: Any
    vtk: Any


@dataclass(frozen=True)
class TessellationControls:
    input_path: Path
    output_dir: Path
    cad_format: str = "auto"
    occ_target_unit: str = "auto"
    mesh_size: float | None = None
    mesh_size_min: float | None = None
    mesh_size_max: float | None = None
    angle_deg: float = 20.0
    chord: float | None = None
    import_labels: bool = True
    save_debug_msh: bool = False


@dataclass(frozen=True)
class TessellationResult:
    mesh_path: Path
    report_path: Path
    debug_msh_path: Path | None
    report: dict[str, Any]


def load_runtime() -> RuntimeModules:
    modules: dict[str, Any] = {}
    failures: list[tuple[str, BaseException]] = []
    for name in ("gmsh", "numpy", "pyvista", "vtk"):
        try:
            modules[name] = importlib.import_module(name)
        except (ImportError, OSError) as exc:
            failures.append((name, exc))

    if failures:
        details = "\n".join(f"  - {name}: {exc}" for name, exc in failures)
        raise DependencyError(
            "Missing CAD tessellation runtime dependencies:\n"
            f"{details}\n"
            "Install them with: python -m pip install -r requirements.txt"
        )
    return RuntimeModules(gmsh=modules["gmsh"], np=modules["numpy"], pv=modules["pyvista"], vtk=modules["vtk"])


def tessellate_cad(controls: TessellationControls) -> TessellationResult:
    input_path = controls.input_path.expanduser().resolve()
    output_dir = controls.output_dir.expanduser().resolve()
    if not input_path.exists():
        raise TessellationError(f"Input CAD file does not exist: {input_path}")
    cad_format = resolve_cad_format(input_path, controls.cad_format)
    runtime = load_runtime()
    output_dir.mkdir(parents=True, exist_ok=True)

    gmsh = runtime.gmsh
    warnings: list[str] = []
    applied_options: dict[str, Any] = {}
    initialized = False
    try:
        gmsh.initialize()
        initialized = True
        gmsh.model.add("cad_tessellation")
        configure_gmsh(gmsh, controls, applied_options, warnings)

        imported_entities = gmsh.model.occ.importShapes(str(input_path), highestDimOnly=True, format=cad_format)
        gmsh.model.occ.synchronize()
        entity_names = collect_entity_names(gmsh)
        imported = [{"dim": int(dim), "tag": int(tag)} for dim, tag in imported_entities]

        gmsh.model.mesh.generate(2)
        surface_mesh = extract_surface_mesh(gmsh, runtime.np)
        if surface_mesh["triangles"].shape[0] == 0:
            raise TessellationError("Gmsh generated an empty surface mesh.")

        mesh_path = output_dir / "surface_mesh.vtp"
        from .mesh_export import write_vtp

        write_vtp(
            runtime.pv,
            runtime.np,
            mesh_path,
            surface_mesh["points"],
            surface_mesh["triangles"],
            {
                "gmsh_surface_tag": surface_mesh["surface_tags"],
                "gmsh_parent_volume_tag": surface_mesh["parent_volume_tags"],
                "gmsh_element_tag": surface_mesh["element_tags"],
            },
            {"gmsh_node_tag": surface_mesh["node_tags"]},
        )

        debug_msh_path = output_dir / "surface_mesh.msh" if controls.save_debug_msh else None
        if debug_msh_path is not None:
            gmsh.write(str(debug_msh_path))

        from .report import build_tessellation_report

        report_path = output_dir / "tessellation_report.json"
        report = build_tessellation_report(
            np=runtime.np,
            input_path=input_path,
            cad_format=cad_format,
            controls=controls,
            imported_entities=imported,
            entity_names=entity_names,
            entity_counts=collect_entity_counts(gmsh),
            mesh=surface_mesh,
            mesh_path=mesh_path,
            report_path=report_path,
            debug_msh_path=debug_msh_path,
            applied_options=applied_options,
            warnings=warnings,
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return TessellationResult(mesh_path=mesh_path, report_path=report_path, debug_msh_path=debug_msh_path, report=report)
    finally:
        if initialized:
            gmsh.finalize()


def resolve_cad_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise TessellationError(f"Cannot infer CAD format from {path.name}; supported suffixes: {supported}")
    return SUPPORTED_SUFFIXES[suffix]


def configure_gmsh(gmsh: Any, controls: TessellationControls, applied: dict[str, Any], warnings: list[str]) -> None:
    try_set_number(gmsh, "General.Terminal", 0, applied, warnings)
    try_set_number(gmsh, "Geometry.OCCImportLabels", 1 if controls.import_labels else 0, applied, warnings)
    if controls.occ_target_unit != "auto":
        try_set_string(gmsh, "Geometry.OCCTargetUnit", controls.occ_target_unit.upper(), applied, warnings)
    if controls.mesh_size is not None:
        try_set_number(gmsh, "Mesh.MeshSizeMin", controls.mesh_size, applied, warnings)
        try_set_number(gmsh, "Mesh.MeshSizeMax", controls.mesh_size, applied, warnings)
    if controls.mesh_size_min is not None:
        try_set_number(gmsh, "Mesh.MeshSizeMin", controls.mesh_size_min, applied, warnings)
    if controls.mesh_size_max is not None:
        try_set_number(gmsh, "Mesh.MeshSizeMax", controls.mesh_size_max, applied, warnings)
    if controls.angle_deg is not None:
        curvature_segments = max(1, int(round(360.0 / controls.angle_deg)))
        if not try_set_number(gmsh, "Mesh.MeshSizeFromCurvature", curvature_segments, applied, warnings):
            try_set_number(gmsh, "Mesh.CharacteristicLengthFromCurvature", curvature_segments, applied, warnings)
    if controls.chord is not None:
        chord_applied = False
        for option in ("Geometry.OCCDeflection", "Mesh.StlLinearDeflection"):
            chord_applied = try_set_number(gmsh, option, controls.chord, applied, warnings, warn_on_failure=False) or chord_applied
        if chord_applied:
            try_set_number(gmsh, "Mesh.StlLinearDeflectionRelative", 0, applied, warnings, warn_on_failure=False)
        else:
            warnings.append("Gmsh did not accept any known deflection option; --chord is recorded but not applied.")
    try_set_number(gmsh, "Mesh.RecombineAll", 0, applied, warnings)
    try_set_number(gmsh, "Mesh.ElementOrder", 1, applied, warnings)
    try_set_number(gmsh, "Mesh.SecondOrderLinear", 0, applied, warnings)


def extract_surface_mesh(gmsh: Any, np: Any) -> dict[str, Any]:
    node_tags_raw, coords_raw, _ = gmsh.model.mesh.getNodes()
    node_tags_all = np.asarray(node_tags_raw, dtype=np.int64)
    coords_all = np.asarray(coords_raw, dtype=np.float64).reshape(-1, 3)
    if node_tags_all.size == 0:
        raise TessellationError("Gmsh generated no mesh nodes.")
    node_index = {int(tag): index for index, tag in enumerate(node_tags_all.tolist())}

    triangle_nodes: list[list[int]] = []
    element_tags: list[int] = []
    surface_tags: list[int] = []
    parent_volume_tags: list[int] = []
    surface_summaries: dict[int, dict[str, Any]] = {}
    non_triangle_types: list[dict[str, Any]] = []

    for _, surface_tag in gmsh.model.getEntities(2):
        parent_volume = parent_volume_tag(gmsh, surface_tag)
        tri_count = 0
        element_types, element_tag_blocks, node_tag_blocks = gmsh.model.mesh.getElements(2, surface_tag)
        for element_type, block_element_tags, block_node_tags in zip(element_types, element_tag_blocks, node_tag_blocks):
            name, dim, _order, node_count, *_ = gmsh.model.mesh.getElementProperties(element_type)
            if dim != 2:
                continue
            if node_count != 3 or "Triangle" not in name:
                non_triangle_types.append({"surface_tag": int(surface_tag), "element_type": int(element_type), "name": str(name)})
                continue
            nodes = np.asarray(block_node_tags, dtype=np.int64).reshape(-1, 3)
            tags = np.asarray(block_element_tags, dtype=np.int64)
            for element_tag, tri_node_tags in zip(tags.tolist(), nodes.tolist()):
                triangle_nodes.append([int(value) for value in tri_node_tags])
                element_tags.append(int(element_tag))
                surface_tags.append(int(surface_tag))
                parent_volume_tags.append(int(parent_volume))
            tri_count += int(nodes.shape[0])
        surface_summaries[int(surface_tag)] = {"parent_volume_tag": int(parent_volume), "triangle_count": int(tri_count)}

    if non_triangle_types:
        raise TessellationError(f"Gmsh generated non-triangle 2D elements: {non_triangle_types[:10]}")
    if not triangle_nodes:
        raise TessellationError("Gmsh generated no triangle elements.")

    used_node_tags = np.asarray(sorted({tag for tri in triangle_nodes for tag in tri}), dtype=np.int64)
    points = np.asarray([coords_all[node_index[int(tag)]] for tag in used_node_tags.tolist()], dtype=np.float64)
    local_index = {int(tag): index for index, tag in enumerate(used_node_tags.tolist())}
    triangles = np.asarray([[local_index[int(tag)] for tag in tri] for tri in triangle_nodes], dtype=np.int64)

    return {
        "points": points,
        "triangles": triangles,
        "node_tags": used_node_tags,
        "element_tags": np.asarray(element_tags, dtype=np.int64),
        "surface_tags": np.asarray(surface_tags, dtype=np.int64),
        "parent_volume_tags": np.asarray(parent_volume_tags, dtype=np.int64),
        "surface_summaries": surface_summaries,
        "all_triangles": True,
    }


def collect_entity_names(gmsh: Any) -> dict[str, list[dict[str, Any]]]:
    names: dict[str, list[dict[str, Any]]] = {}
    for dim in range(4):
        rows = []
        for _, tag in gmsh.model.getEntities(dim):
            name = gmsh.model.getEntityName(dim, tag)
            rows.append({"tag": int(tag), "name": str(name) if name else None})
        names[str(dim)] = rows
    return names


def collect_entity_counts(gmsh: Any) -> dict[str, int]:
    return {str(dim): len(gmsh.model.getEntities(dim)) for dim in range(4)}


def parent_volume_tag(gmsh: Any, surface_tag: int) -> int:
    upward, _ = gmsh.model.getAdjacencies(2, surface_tag)
    if len(upward) == 1:
        return int(upward[0])
    return -1


def try_set_number(
    gmsh: Any,
    option: str,
    value: float | int,
    applied: dict[str, Any],
    warnings: list[str],
    *,
    warn_on_failure: bool = True,
) -> bool:
    try:
        gmsh.option.setNumber(option, float(value))
    except Exception as exc:  # Gmsh raises generic exceptions for unknown options.
        if warn_on_failure:
            warnings.append(f"Could not set {option}={value}: {exc}")
        return False
    applied[option] = value
    return True


def try_set_string(gmsh: Any, option: str, value: str, applied: dict[str, Any], warnings: list[str]) -> bool:
    try:
        gmsh.option.setString(option, value)
    except Exception as exc:
        warnings.append(f"Could not set {option}={value}: {exc}")
        return False
    applied[option] = value
    return True
