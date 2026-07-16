# Surface Tessellation

Convert CAD or mesh inputs into triangle-only surface meshes for the personal CAX workflow.

All commands below assume the working directory is this skill directory (`.agents/skills/surface-tessellation/`).

CAD inputs are local and OCCT-backed through the Gmsh Python API:

- import CAD with `gmsh.model.occ.importShapes(..., highestDimOnly=True)`
- generate a first-order 2D surface mesh with `gmsh.model.mesh.generate(2)`
- export `surface_mesh.vtp` as the primary downstream format
- write `tessellation_report.json` with quality gates and provenance limits

Mesh inputs skip Gmsh and are read through PyVista or trimesh, then converted with `extract_surface(...).triangulate().clean()`.

## CLI

```bash
python scripts/cad_tessellate.py --help
python scripts/cad_tessellate.py tessellate /path/to/model.step \
  --output-dir /tmp/tessellation \
  --mesh-size 0.05 \
  --angle-deg 18 \
  --chord 0.005

python scripts/cad_tessellate.py tessellate /path/to/model.glb \
  --output-dir /tmp/mesh-surface
```

Supported CAD inputs are `.step`, `.stp`, `.iges`, `.igs`, `.brep`, and `.brp`.
Supported mesh inputs are `.stl`, `.obj`, `.vtp`, `.vtk`, `.glb`, and `.gltf`.

Key controls:

- `--mesh-size`, `--mesh-size-min`, `--mesh-size-max`: CAD/Gmsh edge-size controls.
- `--angle-deg`: CAD curvature sizing control, mapped to `Mesh.MeshSizeFromCurvature`.
- `--chord`: CAD best-effort deflection hint, mapped to available Gmsh/OCC/STL deflection options when supported. It is recorded in the report and should not be treated as a strict Hausdorff guarantee.
- `--occ-target-unit {auto,mm,cm,m}`: optional CAD/OCC unit conversion target.
- `--import-labels/--no-import-labels`: best-effort CAD import of OCC labels.
- `--save-debug-msh`: also writes raw `surface_mesh.msh` for CAD inputs.

CAD controls are ignored for mesh inputs because the source is already discretized; the report records this as a warning.

The CLI keeps heavy runtime imports lazy, so `--help` works before `gmsh`, `pyvista`, or `vtk` are installed. Execution commands report all missing dependencies together and point back to the repository requirements file.

## Outputs

`tessellate` writes:

- `surface_mesh.vtp`: triangle-only VTP surface mesh.
- `tessellation_report.json`: input summary, controls, import summary, mesh counts, topology, quality percentiles, gates, warnings, and provenance notes.
- `surface_mesh.msh`: optional Gmsh debug output when `--save-debug-msh` is set.

For CAD inputs, the VTP stores cell arrays:

- `gmsh_surface_tag`: per-triangle Gmsh surface entity.
- `gmsh_parent_volume_tag`: unique upward volume adjacency when available, otherwise `-1`.
- `gmsh_element_tag`: original Gmsh element tag.

Point data includes `gmsh_node_tag`.

For mesh inputs, the VTP stores `source_triangle_index`, which is the post-triangulation output cell index, not persistent source CAD or mesh provenance.

## Provenance Limits

This first exporter preserves Gmsh import-session entity tags and best-effort entity names. It does not promise persistent source CAD face/body IDs, assembly hierarchy, instance transforms, colors, layers, or materials. Those limitations are always written into the JSON report.

## Smoke Test

The smoke command generates a small synthetic two-body STEP fixture, tessellates it, and checks that the output mesh is non-empty and triangle-only:

```bash
python cad-tessellation/scripts/cad_tessellate.py smoke --output-dir /tmp/cad-tessellation-smoke
python -m unittest discover -s cad-tessellation/tests -p 'test_*.py'
```

The generated fixture lives under the selected output directory and is not committed.
