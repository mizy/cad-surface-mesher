# CAD Surface Mesher

Personal toolkit for CAD-to-watertight-surface meshing, repair planning, tessellation, and fast vehicle geometry analysis.

## Capabilities

- `predict-vehicle-cd`: estimate vehicle drag coefficient from a mesh using normalized side-view geometry similarity and a packaged public reference library.
- `cad-surface-mesher`: audit CAD-derived vehicle surface meshes, generate target-driven visual checks, and prepare target-specific surface repair decisions.
- `mesh-repair`: planned automatic watertight surface mesh repair operations, quality diagnostics, and repair artifacts.
- `cad-tessellation`: convert CAD or mesh files to triangle-only VTP surface meshes with JSON quality and provenance reports.

## Current Usage

Run the migrated Cd predictor from its agent skill directory:

```bash
cd .agents/skills/predict-vehicle-cd
python scripts/vehicle_cd_similarity.py predict /path/to/car.vtp \
  --library assets/reference-library/reference_cars.json \
  --output-dir /tmp/cd_estimate
```

Tessellate a CAD or mesh file into a downstream-consumable surface mesh:

```bash
python cad-tessellation/scripts/cad_tessellate.py tessellate /path/to/model.step \
  --output-dir /tmp/cad-tessellation \
  --mesh-size 0.05 \
  --angle-deg 18 \
  --chord 0.005
```

Mesh inputs such as STL, OBJ, VTP, VTK, GLB, and GLTF are also accepted:

```bash
python cad-tessellation/scripts/cad_tessellate.py tessellate /path/to/model.glb \
  --output-dir /tmp/mesh-surface
```

The tessellator writes `surface_mesh.vtp` and `tessellation_report.json`. CAD outputs store Gmsh cell provenance, while mesh outputs store post-triangulation `source_triangle_index`.

Run the generated fixture smoke test:

```bash
python cad-tessellation/scripts/cad_tessellate.py smoke --output-dir /tmp/cad-tessellation-smoke
```

Agent skills live under:

```text
.agents/skills/<skill-name>
```

Each skill owns its `SKILL.md`, scripts, references, and bundled assets there.

## Dependencies

Install the local Python runtime dependencies with:

```bash
python -m pip install -r requirements.txt
```

Optional `trimesh` support improves GLB/GLTF loading.

## Repository Layout

```text
.agents/skills/predict-vehicle-cd/   Cd prediction skill, scripts, and reference library
.agents/skills/cad-surface-mesher/      CAD surface meshing skill, audit script, and visual QA prompts
mesh-repair/                       Planned watertight surface mesh repair feature
cad-tessellation/                  CAD tessellation CLI, generated fixture smoke test, and docs
docs/                              Roadmap and cross-feature notes
```
