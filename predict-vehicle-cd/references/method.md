# Cd Similarity Method

This skill estimates Cd from side-view geometry similarity. It is a lightweight prior, not a CFD replacement.

## Pipeline

1. Load a `vtp`, `vtk`, `stl`, `obj`, `glb`, or `gltf` mesh. Use `trimesh` first for `glb/gltf` scene files; otherwise use PyVista, with fallback between the readers when possible.
2. Extract and triangulate the surface.
3. Normalize the vehicle mesh:
   - reorder axes by span so length -> `X`, width -> `Y`, height -> `Z`
   - use horizontal PCA when raw length/width spans are close
   - detect likely units from the length span: meters, millimeters, centimeters, or unitless normalized geometry
   - uniformly scale to meters, using `--target-length` for unitless or unknown-size meshes
   - force real exterior dimensions with `--target-dimensions L,W,H` or manifest `target_dimensions_m` when trustworthy dimensions are available
   - apply a conservative `Z` flip when top/bottom point distribution suggests the model is upside down
   - center `X/Y` and align the lowest point to `z = 0`
4. Render two front/rear candidates: normalized pose and a 180 degree rotation around `Z`.
5. Fit each side projection into a fixed canvas.
6. Rasterize the side view from `-Y` toward `+Y`.
7. Compare the query against each reference using the depth PNG's alpha mask and depth channel. Normal maps are still rendered as diagnostics.
8. Pick the better front/rear candidate and compute weighted Cd from top-k references.

## Similarity Formula

Default score:

```text
score =
  0.60 * mask_iou
+ 0.40 * depth_similarity
+ 0.00 * normal_similarity
```

The script records component scores in the output JSON. Current leave-one-out validation over the reference library showed normal similarity did not improve Cd estimates, so it is generated for inspection but does not affect scoring. Tune weights only with validation evidence, preferably leave-one-out tests over a reference library with authoritative Cd labels.

## Cd Weighting

Top-k weights follow the same idea as the project reference-pool Cd path:

```text
distance_i = 1 - similarity_i
weight_i = softmax(temperature / (distance_i + epsilon))
cd = sum(weight_i * cd_i)
```

Use smaller temperature for sharper nearest-neighbor behavior and larger temperature for smoother averaging.

## Current Limitations

- Only side view is used.
- Axis detection is extent-based and may fail for unusual non-vehicle meshes or partial vehicle meshes.
- PCA length/width alignment is a fallback for rotated or wide-bounds meshes; inspect `normalization.axis_method` for these cases.
- Front/rear selection is similarity-driven against the reference library because arbitrary meshes usually lack trusted semantic part labels.
- Reference Cd values come from mixed public source conditions.
- Wheel, mirror, underbody, and front/rear details are weakly represented from one side view.

Useful next upgrades:

- Add front, rear, and top depth views.
- Add authoritative CFD-labeled references.
- Calibrate score weights with leave-one-out validation.
