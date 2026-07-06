# Vehicle CAX Toolkit

Personal toolkit for vehicle CAX/CAE preprocessing and fast geometry analysis.

## Capabilities

- `predict-vehicle-cd`: estimate vehicle drag coefficient from a mesh using normalized side-view geometry similarity and a packaged public reference library.
- `mesh-repair`: planned automatic watertight surface mesh repair, quality diagnostics, and repair artifacts.
- `cad-tessellation`: planned CAD import/tessellation embedding for STEP/IGES-style workflows.

## Current Usage

Run the migrated Cd predictor from the feature directory:

```bash
cd predict-vehicle-cd
python scripts/vehicle_cd_similarity.py predict /path/to/car.vtp \
  --library assets/reference-library/reference_cars.json \
  --output-dir /tmp/cd_estimate
```

The same capability is exposed as an agent skill through:

```text
.agents/skills/predict-vehicle-cd
```

That path is a symlink to the top-level `predict-vehicle-cd` feature so the implementation has one source of truth.

## Dependencies

The current predictor expects:

```bash
python -m pip install -r requirements.txt
```

Optional `trimesh` support improves GLB/GLTF loading.

## Repository Layout

```text
predict-vehicle-cd/   Cd prediction feature, skill docs, scripts, and reference library
mesh-repair/          Planned watertight surface mesh repair feature
cad-tessellation/     Planned CAD tessellation embedding feature
docs/                 Roadmap and cross-feature notes
.agents/skills/       Agent skill entrypoints
```

