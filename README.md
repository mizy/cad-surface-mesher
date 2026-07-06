# Vehicle CAX Toolkit

Personal toolkit for vehicle CAX/CAE preprocessing and fast geometry analysis.

## Capabilities

- `predict-vehicle-cd`: estimate vehicle drag coefficient from a mesh using normalized side-view geometry similarity and a packaged public reference library.
- `cad-aero-skin-mesher`: audit CAD-derived vehicle surface meshes, generate target-driven visual checks, and prepare exterior aero-skin repair decisions.
- `mesh-repair`: planned automatic watertight surface mesh repair operations, quality diagnostics, and repair artifacts.
- `cad-tessellation`: planned CAD import/tessellation embedding for STEP/IGES-style workflows.

## Current Usage

Run the migrated Cd predictor from its agent skill directory:

```bash
cd .agents/skills/predict-vehicle-cd
python scripts/vehicle_cd_similarity.py predict /path/to/car.vtp \
  --library assets/reference-library/reference_cars.json \
  --output-dir /tmp/cd_estimate
```

Agent skills live under:

```text
.agents/skills/<skill-name>
```

Each skill owns its `SKILL.md`, scripts, references, and bundled assets there.

## Dependencies

The current predictor expects:

```bash
python -m pip install -r requirements.txt
```

Optional `trimesh` support improves GLB/GLTF loading.

## Repository Layout

```text
.agents/skills/predict-vehicle-cd/   Cd prediction skill, scripts, and reference library
.agents/skills/cad-aero-skin-mesher/ Aero-skin meshing skill, audit script, and visual QA prompts
mesh-repair/                       Planned watertight surface mesh repair feature
cad-tessellation/                  Planned CAD tessellation embedding feature
docs/                              Roadmap and cross-feature notes
```
