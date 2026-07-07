# Mesh Repair

Planned responsibility: repair target-specific surface meshes into watertight, inspectable artifacts suitable for downstream CAX/CAE preprocessing.

This module currently implements mesh-domain prototypes. The broader skill contract is:

- CAD input -> watertight CAD output when CAD-domain healing/sewing/filling is requested
- CAD input -> watertight mesh output when meshing is requested
- mesh input -> watertight mesh output for the current local prototype
- mesh input -> watertight CAD output only through explicit reverse surface fitting, with approximation loss reported

Initial scope:

- use a two-stage watertight flow: extract exterior wall surfaces first, then remesh/seal only that exterior candidate set
- prefer group/name visibility reduction when GLTF/CAD assembly metadata is available
- diagnose boundaries, non-manifold topology, duplicated vertices, degenerate faces, inverted normals, and disconnected shells
- propose a repair plan before changing geometry
- write repaired mesh artifacts and a JSON report
- keep units, coordinate convention, and provenance explicit

Do not apply watertight remeshing to a full dirty assembly before removing interior, hidden, duplicate, or irrelevant components. For external-flow or CAX skins, the intermediate exterior wall candidate may be non-watertight; water-tightness is introduced only after the target wall set is known.

## Two-Stage Prototype

Run the local prototype on a mesh input:

```bash
python mesh-repair/scripts/two_stage_watertight_remesh.py \
  /path/to/assembly.vtp \
  --group-source-gltf /path/to/scene.gltf \
  --target-name external-flow-skin \
  --remove-name-regex "internal|interior|hidden|cavity" \
  --output-dir outputs/assembly-two-stage \
  --visibility-grid 900 \
  --depth-tolerance 0.0 \
  --dilate-rings 0 \
  --voxel-pitch 0.02
```

Outputs:

- `stage0_group_filtered.vtp` when `--group-source-gltf` is provided
- `stage1_exterior_candidate.vtp`
- `stage2_watertight_surface.vtp`
- `two_stage_report.json`
- depth preview PNGs under `visual/`

The prototype uses GLTF geometry names to remove explicit non-target groups, then keeps first-hit exterior candidates from six orthographic directions, then runs a voxel fill plus marching-cubes remesh. A `watertight_topology_pass` means boundary, non-manifold, and degenerate face checks pass. It is not the same as `engineering_pass`; final acceptance still requires target-specific opening review, self-intersection checks, and target drift limits.

Voxel output must also pass shared-projection silhouette drift against Stage 1. If the voxel remesh caps wheel wells, underbody detail, functional openings, or other high-area features, `stage2_silhouette_drift_within_default_0_05` fails and the output is only a closure proxy, not accepted final geometry.

For vehicles, `--remove-name-regex` can include `carInternal`, `seat`, `dashboard`, `centerConsole`, and `steeringWheel`. For other assemblies, provide domain-specific remove/keep naming rules before relying on geometry visibility.

## Adaptive Refinement

Adaptive refinement must use the original model or `stage1_exterior_candidate.vtp` as the geometric source. Do not subdivide a coarse watertight shell and treat that as recovered detail; the shell has already lost source information.

The intended flow is:

1. classify the target exterior on the original assembly or Stage 1 candidate
2. build a target size field from source curvature, feature edges, openings, materials, names, and visibility evidence
3. keep coarse cells in low-detail source regions
4. refine critical regions directly against the source exterior geometry
5. use the coarse watertight shell only as topology/closure guidance
6. report source-to-output drift per critical region

Depth maps are the preferred first signal for mesh-only inputs. Render depth plus face IDs from several directions, detect strong depth gradients and silhouette/face-id discontinuities, then map those pixels back to source faces. Merge the resulting scores into a size field and create a conforming adaptive mesh: fine regions and coarse regions must share stitched transition edges rather than overlapping as separate meshes.

Run the current source-refinement prototype on a Stage 1 exterior candidate:

```bash
python mesh-repair/scripts/adaptive_depth_refine.py \
  outputs/vehicle-two-stage/stage1_exterior_candidate.vtp \
  --output-dir outputs/vehicle-adaptive-depth-refine \
  --grid-size 900 \
  --gradient-percentile 98 \
  --disable-silhouette \
  --base-size 0.03 \
  --transition-size 0.015 \
  --fine-size 0.008
```

This writes `refinement_field.vtp`, `adaptive_refined_source.vtp`, per-view critical-pixel PNGs, and `adaptive_refinement_report.json`. The script refines source triangles conformingly; it does not coarsen already-dense source areas and does not seal the mesh by itself.

## Workflow Report

When a run includes both watertight remeshing and adaptive refinement diagnostics, publish one primary workflow report:

```bash
python mesh-repair/scripts/workflow_report.py \
  --two-stage-report outputs/vehicle-two-stage/two_stage_report.json \
  --adaptive-report outputs/vehicle-adaptive-depth-refine/adaptive_refinement_report.json \
  --output-dir outputs/vehicle-workflow-report
```

The workflow report is a single-file HTML report with embedded vtk.js, embedded before/after mesh data, embedded diagnostic images, and a JSON companion. It records whether adaptive refinement was used by the final watertight output or only run as a diagnostic branch.

The embedded 3D viewer defaults to full-resolution before/after meshes so dirty input and watertight output are inspectable as real artifacts. Set `CAD_SURFACE_MESHER_VIEWER_TRIANGLES` only when intentionally producing a smaller viewer payload; the HTML caption will then mark the mesh as not full resolution.

## Report Contract

Repair outputs must include an HTML report and a machine-readable JSON report with:

- `geometry_to_mesh_trace`: original, group filter, exterior candidate, refinement, repair, sealing, and final mesh stages
- `change_summary`: removed, refined, offset, sealed, capped, filled, or regenerated regions
- `defect_matrix`: before/after free edges, non-manifold/shared edges, degenerate faces, components, volume reliability, and leak checks
- `requested_capabilities`: part self gaps, between-part gaps, free edges, overlaps, normal inconsistency, micro holes, common edges, target-specific offsets such as front-bumper CAS offset
- `comparisons.stage2_silhouette_vs_stage1`: shared-projection visual drift so voxelized outputs that visibly diverge from the source candidate are rejected
- explicit `not_implemented` or `not_individually_classified` entries for anything the script did not truly check or repair

The current prototypes report implicit voxel closure separately from per-gap classification. They must not claim overlap repair, normal repair, CAS offset, or source-loop hole inventory until those detectors and repair steps exist.

Current report files:

- `workflow_report.html` plus `workflow_report.json` as the primary user-facing report
- `two_stage_report.html/json` and `adaptive_refinement_report.html/json` as stage debug artifacts
