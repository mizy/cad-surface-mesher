---
name: predict-vehicle-cd
description: Estimate a vehicle drag coefficient (Cd) from an arbitrary 3D car mesh by automatically normalizing vehicle axes and units, rendering a standardized side-view depth map with an alpha silhouette mask, comparing those artifacts against a packaged reference vehicle library, and similarity-weighting reference Cd values. Use when Codex needs a fast geometry-similarity Cd estimate for VTP, VTK, STL, OBJ, GLB, or GLTF vehicle meshes, especially before running CFD or a learned model.
---

# Predict Vehicle Cd

Estimate Cd from a 3D vehicle mesh with the packaged side-view reference library. The predictor accepts raw vehicle meshes and first normalizes them to the project vehicle convention: length on `X`, width on `Y`, height on `Z`, top at `+Z`, front target `-X`, meters-scale dimensions, and ground at `z = 0`. Treat the result as a fast shape-similarity prior for triage, candidate ranking, or sanity checks before higher-fidelity CFD/model inference.

## Quick Start

From the repository root, run the standalone predictor from this skill directory:

```bash
cd .agents/skills/predict-vehicle-cd
python scripts/vehicle_cd_similarity.py predict /path/to/car.vtp \
  --library assets/reference-library/reference_cars.json \
  --category pickup \
  --output-dir /tmp/cd_estimate
```

Supported input mesh formats are `vtp`, `vtk`, `stl`, `obj`, `glb`, and `gltf`.

The script prints JSON with:

- `cd`: similarity-weighted Cd estimate
- `confidence`: qualitative confidence from the top side-view match
- `query_category`: optional category passed through `--category`; when supplied, same-category references receive a small ranking bonus
- `orientation`: which normalized front/rear candidate won (`front_as_normalized` or `front_rotated_180_z`)
- `normalization`: raw bounds, axis order, detected unit/scale, final bounds, and any axis warning
- `top_k`: nearest reference cars, scores, weights, and Cd values
- `artifacts`: generated query side-view `depth.png` with alpha mask and debug `normal.png`
- `normalized_mesh`: generated `query_normalized.vtp` when `--output-dir` is set

## Workflow

1. Use `scripts/vehicle_cd_similarity.py predict` for one-off estimates.
2. Inspect `normalization` and `query_normalized.vtp` when a source model has unknown units or orientation.
3. Inspect `top_k` before trusting the scalar Cd.
4. Inspect `cd_source` because the packaged Cd labels come from mixed public wind-tunnel, paper, OEM, and secondary-publication conditions.
5. For reference cars with known real dimensions, pass `--target-dimensions L,W,H` or set `target_dimensions_m` in the manifest so the rendered artifacts use real length, width, and height.
6. Rebuild the packaged library after changing reference cars:

```bash
python scripts/vehicle_cd_similarity.py build-library \
  assets/reference-library/seed_manifest.json \
  --output-dir assets/reference-library
```

## Method

The predictor is intentionally self-contained:

- Reads `vtp`, `vtk`, `stl`, `obj`, `glb`, and `gltf` meshes through PyVista, with a `trimesh` first pass for `glb/gltf` scene files and an optional fallback when PyVista cannot load a file, then triangulates and extracts the surface.
- Normalizes vehicle geometry before rendering:
  - longest axis -> length `X`, middle axis -> width `Y`, shortest axis -> height `Z`
  - uses horizontal PCA to resolve length/width when raw horizontal spans are close
  - detects meter, millimeter, centimeter, and unit-scaled meshes from the length span
  - uniformly scales to meters, keeping realistic 2.5-7.0 m lengths when already plausible
  - can force real exterior dimensions with `--target-dimensions L,W,H` or per-car `target_dimensions_m`
  - flips `Z` only when the top/bottom point distribution strongly indicates the mesh is upside down
  - centers `X/Y` and aligns the lowest point to `z = 0`
  - compares both the normalized pose and a 180 degree rotation around `Z` for front/rear ambiguity
- Renders only the current side view by CPU rasterizing triangles into:
  - normalized side depth PNG whose alpha channel is the binary silhouette mask
  - outward normal RGB map for diagnostics
- Scores each reference with a weighted blend:
  - mask IoU
  - depth similarity on the shared mask
  - normal cosine similarity on the shared mask, currently weighted at `0.0` because leave-one-out validation showed it was not useful for this library
- When `--category` is provided, adds a small same-category ranking bonus before top-k weighting.
- Computes Cd with inverse-distance softmax weighting over top-k references.

## Reference Library

The packaged library lives in `assets/reference-library/reference_cars.json` with side-view artifacts beside it. The Cd values include source provenance in each `cd_source`; align conditions before using the scalar estimate for engineering decisions.

Read `references/method.md` when changing the scoring formula or adding more views. Read `references/reference-library.md` when editing the reference-car schema.
