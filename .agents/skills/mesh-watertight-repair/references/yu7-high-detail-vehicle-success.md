# High-detail Vehicle Closure-first Success Case

Use this case as a transferable playbook for a high-detail, multi-part vehicle mesh with global
topology defects. Treat its numeric settings as a measured starting point, not as universal
acceptance thresholds or a model-specific allowlist.

## Decision

Use this ownership model:

> Let the source mesh own shape, the sealed-exterior TSDF proxy own topology, and libigl/CGAL own
> final self-intersection judgment.

Prefer this closure-first route when boundaries and intersections are too globally coupled for
independent source-face deletion and hole filling. Keep the source immutable and let the certified
closure scaffold replace topology instead of creating custom per-defect patch geometry.

## Successful Workflow

1. Triangulate the input, preserve source provenance, and extract trusted exterior evidence.
2. Rasterize a conservative surface shell, seal only voxel-scale leaks, flood padded far-field
   space, and use the complement to define inside/outside independently of source winding.
3. Build the truncated signed distance field near the sealed boundary and extract its zero surface.
   Treat this surface as a closed topology scaffold, not as accepted target geometry.
4. Retain the exterior shell and remove only clearly isolated dust components. Do not use component
   count alone as an acceptance rule.
5. Smooth the scaffold with topology-preserving VTK Windowed-Sinc filtering before projection.
6. Project only trustworthy scaffold vertices to trustworthy source triangles with compiled
   PyVista/VTK batch closest-cell queries. Preserve the certified scaffold position as the explicit
   fallback for every vertex.
7. Certify the complete projected mesh with libigl/CGAL exact predicates. For each reported face
   pair, restore the implicated projected vertices, optionally including a bounded vertex ring, to
   their scaffold positions and certify the complete mesh again.
8. Inspect registered smooth-shaded side, front, top, and rear views. If a bounded region still
   shows voxel rings or staircase grain, blend a stronger local Windowed-Sinc result into that
   region, roll back any exact-intersection participants, and rerun every gate.
9. Export both the authoritative VTP and requested interchange format, read both back, and rerun
   topology, orientation, volume, degeneracy, exact self-intersection, and fidelity checks.

## Resolution and Fidelity Lessons

- Reject the longest-axis divisor `280` result for this case because smooth-shaded views exposed
  obvious voxel grain.
- Use longest-axis divisor `560` for the successful final scaffold. It produced about 28 million
  voxels and 1.287 million output triangles, so budget memory before starting the final round.
- Increase resolution only when measured silhouette, source distance, or visual evidence requires
  it. A two-times finer linear grid costs roughly eight-times the dense voxel memory and work.
- Do not infer visual smoothness from watertightness. Zero boundary and non-manifold counts can
  coexist with objectionable staircase or ring artifacts.
- Do not infer fidelity from a one-way distance alone. Measure final-to-source and visible
  source-to-final distances, plus six fixed-camera silhouette/depth comparisons.

## Automation Rules

Automate these choices:

| Evidence | Action |
| --- | --- |
| Global part soup, widespread overlap, or ambiguous coupled boundaries | Build a sealed-exterior TSDF scaffold. |
| Projection creates exact self-intersections | Restore only vertices incident to reported faces to the certified scaffold, then recertify globally. |
| Smooth-shaded views show a bounded scaffold artifact | Apply masked local Windowed-Sinc smoothing, then recertify and remeasure fidelity. |
| Any required gate fails after serialization | Reject the exported artifact even if the in-memory candidate passed. |

Require human or Skill-agent semantic review only for ambiguous exterior openings, uncertain
component ownership, or visual artifacts that deterministic evidence cannot classify. Keep
geometry generation, rollback, measurements, and acceptance deterministic.

## Library and Script Ownership

Prefer mature compiled library operations over new handwritten geometry kernels:

| Responsibility | Preferred implementation |
| --- | --- |
| Sealed exterior, TSDF construction, and zero-isosurface extraction | `two_stage_watertight_remesh.py`, `sealed_exterior_atlas.py`, and `sdf_closure.py` using NumPy, SciPy, and scikit-image |
| Topology-preserving smoothing and closest-source projection | VTK through PyVista; use the batch path in `source_projected_closure.py` |
| Exact all-face self-intersection detection | libigl's CGAL binding through `source_projected_validation.py` |
| Topology, volume, quality, and six-view evidence | `mesh_metrics.py` and `solid_triangle_raster.py` |

Do not reimplement triangle-triangle predicates, point-to-triangle locators, marching cubes, or
Windowed-Sinc smoothing in Python. Add custom code only for workflow-specific evidence, masks,
provenance, decision routing, and rollback bookkeeping.

## Failed Routes and Their Meaning

- Do not use a Python/tolerance-driven all-face intersection scan for acceptance. In this case it
  ran for about 22 minutes and reported 22 proxy pairs that libigl/CGAL rejected as false positives;
  exact certification completed in about 2.3 seconds with zero pairs.
- Do not project every scaffold vertex unconditionally. An aggressive projected candidate created
  4,528 exact intersecting face pairs.
- Do not delete all faces involved in self-intersections and then fill the resulting holes. The
  removed faces do not necessarily bound simple disks, adjacent defects merge, and source shape
  evidence is lost.
- Do not accept generic MeshFix-style repair solely because it closes the mesh. The tested route
  changed too much vehicle shape and topology.
- Do not assume a CGAL boolean union is automatically manifold. The tested result retained
  non-manifold, degenerate, or self-intersecting elements.
- Do not use source-only Screened Poisson as the default vehicle exterior reconstruction. It
  smoothed the surface but bulged the underbody; the tested Poisson fusion route introduced visible
  ripples.

## Successful Validation Snapshot

Record these values as provenance for this run, not as hard-coded gates for future models:

| Measurement | Result |
| --- | ---: |
| Points | 642,808 |
| Triangles | 1,286,964 |
| Connected components | 1 |
| Boundary edges | 0 |
| Non-manifold edges / vertices | 0 / 0 |
| Inconsistent-winding edges | 0 |
| Degenerate faces | 0 |
| Exact self-intersection pairs, VTP / STL readback | 0 / 0 |
| Reliable signed enclosed volume | `7.45715907326226e-06` source-unit cubed |
| Maximum bbox drift / longest bbox extent | 0.1104% |
| Minimum six-view silhouette overlap | 99.3974% |
| Final-to-source surface distance p95 / longest extent | 0.1640% |
| Visible source-to-final distance p95 / longest extent | 0.4217% |

The final visual gate passed because registered side, front, top, and rear smooth-shaded renders
showed no obvious voxel grain after a stronger bounded upper-surface blend. The final VTP and STL
readbacks preserved the same 642,808 points and 1,286,964 triangles after STL vertex cleaning.
The recorded environment used PyVista 0.48.4, VTK 9.6.2, libigl 2.6.2, NumPy 2.5.1, SciPy 1.18.0,
scikit-image 0.26.0, and trimesh 4.12.2; record rather than silently assume equivalent versions in
future runs.

## Reuse Checklist

- Preserve the original mesh as immutable geometry truth.
- Save the TSDF scaffold separately from the source-projected candidate.
- Keep a per-vertex projection-applied/fallback provenance mask.
- Certify the scaffold before projecting anything.
- Recertify the whole candidate after rollback or smoothing.
- Compare fixed registered views before and after every visually motivated change.
- Save exact library versions and serialized readback results in the final report.
- Accept only the artifact named by `outputs.accepted_mesh_vtp` after all gates pass.
