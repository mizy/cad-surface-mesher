---
name: surface-tessellation
description: Convert CAD (STEP, IGES, BREP) or mesh (STL, OBJ, VTP, VTK, GLB, GLTF) inputs into triangle-only VTP surface meshes with a JSON quality and provenance report. Use when Codex needs to prepare a mesh for downstream watertight repair, Cd prediction, or visual QA from a CAD source or an arbitrary mesh format.
---

# Surface Tessellation

Convert CAD or mesh inputs into triangle-only VTP surface meshes with a JSON quality and provenance report.

## I/O Contract

```yaml
input_kind: cad | mesh
output_kind: surface_mesh_vtp
tessellation_domain: mixed
format_provenance: import-session tags only
```

## Quick Start

All commands below assume the working directory is this skill directory (`.agents/skills/surface-tessellation/`).

### Tessellate a CAD file

```bash
python scripts/cad_tessellate.py tessellate /path/to/model.step \
  --output-dir /tmp/tessellation \
  --mesh-size 0.05 \
  --angle-deg 18 \
  --chord 0.005
```

Supported CAD inputs: `.step`, `.stp`, `.iges`, `.igs`, `.brep`, `.brp`.

### Tessellate a mesh file

```bash
python scripts/cad_tessellate.py tessellate /path/to/model.glb \
  --output-dir /tmp/mesh-surface
```

Supported mesh inputs: `.stl`, `.obj`, `.vtp`, `.vtk`, `.glb`, `.gltf`.

### Smoke test

```bash
python scripts/cad_tessellate.py smoke --output-dir /tmp/tessellation-smoke
```

## Output

- `surface_mesh.vtp` — triangle-only surface mesh
- `tessellation_report.json` — quality gates, provenance, and parameters

For CAD inputs, cell data arrays include `gmsh_surface_tag`, `gmsh_parent_volume_tag`, and `gmsh_element_tag` as import-session provenance (not persistent CAD IDs).  
For mesh inputs, a `source_triangle_index` array records the original triangle index.

## Pipeline

- **CAD path**: OCCT-backed through Gmsh Python API — import with `occ.importShapes`, generate 2D surface mesh, export VTP.
- **Mesh path**: Read through PyVista or trimesh, `extract_surface().triangulate().clean()`.

## Documentation

- [README.md](README.md) — detailed CLI reference and caveats
