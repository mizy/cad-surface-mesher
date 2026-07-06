---
name: cad-surface-mesher
description: Build and validate target-specific CAD surface meshes from CAD or mesh inputs. Use when Codex needs to convert CAD or mesh inputs to triangle-only surface meshes, remove non-target internal or hidden parts with group-level show/hide tests, prune AABB-contained internals, extract only exterior wall surfaces, seal target-specific gaps, handle opening policy, generate target-driven visual inspection screenshots, audit watertightness/non-manifold topology/triangle quality, or produce a surface mesh quality report for CFD/CAE preprocessing.
---

# CAD Surface Mesher

Create a target-specific CAD surface mesh, not a faithful full assembly mesh. The common default target is an exterior flow or analysis skin for CAX/CAE preprocessing with explicit visibility, visual, and topology evidence. Vehicles are only one target family.

## Core Rule

Use AI to classify and review visual evidence. Use deterministic geometry checks to decide mesh quality. Do not claim success from screenshots alone.

## Target Policy

Start every task by writing or confirming the target policy:

```yaml
target: external-flow-skin
remove:
  non_target_internal_parts: true
  hidden_internal_faces: true
  fixtures_supports_fasteners: ask
seal:
  small_assembly_gaps: true
  intentional_panel_gaps: target_specific
openings:
  functional_openings: ask
  through_holes: ask
refinement:
  base_size_mm: 20
  feature_size_mm: 5
  curvature_sensitive: true
limits:
  max_gap_fill_width_mm: 30
  max_surface_offset_mm: 5
  max_bbox_drift_ratio: 0.002
```

If the user asks for simple external Cd triage, use the vehicle target policy. For non-vehicle assemblies, first define target-specific remove, keep, opening, and refinement rules.

## Workflow

1. Load CAD assembly metadata before meshing when available: group tree, names, layers, colors/materials, body IDs, face IDs, transforms, and bounding boxes.
2. Reduce the assembly at group level first with the visibility workflow below.
3. Keep only exterior wall candidates, then tessellate or remesh those surfaces with face/body provenance preserved.
4. Generate deterministic audit metrics and target-driven screenshots.
5. Run separate AI visual checks for each goal:
   - exterior skin completeness
   - interior part residue
   - panel gap sealing
   - functional opening policy
   - exterior feature preservation
   - patch artifacts
   - silhouette drift
6. Convert visual findings into a repair plan.
7. Apply one deterministic repair step at a time.
8. Re-audit and regenerate screenshots after each repair.
9. Stop when validators pass or when policy/tolerance limits require user confirmation.

## Two-Stage Watertight Strategy

Never watertight-remesh the full dirty assembly directly.

1. Exterior wall extraction:
   - remove interior, hidden, duplicate, and AABB-contained internal groups first
   - keep only the target exterior wall candidate set
   - accept that this intermediate mesh may still have open boundaries, panel gaps, and functional openings
2. Watertight remesh:
   - run remesh or hole sealing only on the exterior wall candidate set
   - cap or preserve functional openings according to target policy
   - validate silhouette/bbox drift against the pre-remesh exterior wall, not the unfiltered assembly

The first stage decides what surface should exist. The second stage makes that surface watertight.

Use `mesh-repair/scripts/two_stage_watertight_remesh.py` for the current local prototype when only mesh inputs are available:

```bash
python mesh-repair/scripts/two_stage_watertight_remesh.py /path/to/assembly.vtp \
  --group-source-gltf /path/to/scene.gltf \
  --target-name external-flow-skin \
  --remove-name-regex "internal|interior|hidden|cavity" \
  --output-dir outputs/assembly-two-stage \
  --depth-tolerance 0.0 \
  --voxel-pitch 0.02
```

When `--group-source-gltf` is supplied, the script first removes names matching `--remove-name-regex` by flattened GLTF face ranges, then runs mesh-only exterior visibility. This assumes the GLTF flatten order matches the input triangle order; validate with screenshots and metrics. For vehicles the regex can include `carInternal`, `INT_*`, seats, dashboard, center console, and steering wheel. For other assemblies, use domain-specific names.

Treat `watertight_topology_pass` as a narrow topology result only. Do not call the mesh engineering-ready until component count, volume reliability, self-intersection checks, visual opening policy, and bbox/silhouette drift all pass.

## Group Visibility Workflow

Prefer group-level decisions before body-level or face-level work. A human CAD cleanup usually starts by hiding named assembly groups and checking whether the exterior changed; follow that pattern.

1. Build a normalized group table with `path`, `name`, `parent`, `children`, `aabb`, `area`, `volume`, `material`, `body_count`, and source IDs.
2. Assign a weak semantic label from names and paths:
   - likely remove: target-specific interior/hidden/service/fixture/fastener/support groups
   - likely keep: target-specific exterior shell, walls, boundary surfaces, visible accessories, and required functional geometry
3. Find large exterior-shell candidates from name, area, and bbox span. Use them as containment references, not the full assembly bbox alone.
4. Mark a group as `internal_candidate` when its AABB is strictly contained inside an exterior-shell candidate with clearance, it has no exterior name token, and its projected silhouette contribution is small.
5. Render A/B screenshots at group granularity:
   - baseline full assembly
   - hide each candidate group or candidate parent group
   - only exterior-shell candidates
   - removed-candidate overlay
6. Compare front, rear, left, right, top, bottom, and three-quarter views. Remove a group when hiding it has near-zero silhouette/visible-area delta; use AI to explain non-zero changed pixels, not to override a large exterior delta.
7. If hiding a parent group changes exterior silhouette, recurse into children instead of deleting the parent.

AABB containment is a speed filter, not proof. Do not delete small but target-visible functional parts from AABB containment alone.

## Exterior Wall Extraction

After group reduction, keep exterior wall surfaces before final meshing.

- Prefer original CAD faces marked as exterior by group visibility and exterior-name evidence.
- When only mesh geometry is available, ray-cast from outside views and keep the first surface hit along each ray; use this as an outermost-wall score per face/body.
- Remove duplicate inner sheets only when they sit behind an exterior sheet with similar normals/curvature and no exterior visibility.
- Preserve face/body IDs through tessellation so later patches can report their source.
- Mesh the reduced exterior wall set, then repair gaps and openings according to target policy.

## Two-Face Gaps

For each boundary loop, classify the local situation before repair:

- `seal_panel_gap`: target-specific panel seam, such as hood, door, trunk, or window seams in vehicle external-aero mode.
- `seal_cad_gap`: small assembly/tessellation gap under the configured width limit.
- `remove_inner_sheet`: duplicate or hidden internal sheet behind the exterior skin.
- `keep_separate`: true separated exterior components such as wheel/body gaps or separate exposed assemblies.
- `ask`: ambiguous functional opening or over-limit patch.

Use boundary-loop distance, normal/curvature compatibility, and visual closeups together. Do not stitch duplicate inner and outer sheets together.

## Functional Openings

Treat target-specific functional openings as policy openings, not ordinary holes. Vehicle examples include grilles, cooling intakes, wheel wells, underbody openings, and exhausts.

- `preserve`: use when modeling cooling flow or true internal ducting.
- `cap`: use for simplified external-aero skin when internal flow is out of scope.
- `porous_surface`: use when a surrogate porous grille boundary is desired.
- `ask`: default when the target policy is unclear.

Every generated cap or porous patch must be marked in the report with `patch_type`, source visual check, area, and confidence.

## Deterministic Gates

Report these metrics every time:

- boundary edges
- non-manifold edges
- degenerate faces
- connected components
- surface area
- signed/enclosed volume when topology permits
- triangle aspect-ratio percentiles
- bbox and dimension drift when comparing before/after

A final engineering pass requires all required metrics to be present. Missing metrics are not a pass.

## Bundled Tool

From the repository root, use `cad-tessellation/scripts/cad_tessellate.py` first when a CAD or mesh input must be converted to a triangle-only surface mesh:

```bash
python cad-tessellation/scripts/cad_tessellate.py tessellate /path/to/model.step --output-dir /tmp/cad-tessellation
```

STEP, IGES, and BREP inputs use Gmsh/OCC. STL, OBJ, VTP, VTK, GLB, and GLTF inputs are read as existing meshes and converted with surface extraction plus triangulation.

The tessellator writes `surface_mesh.vtp` plus `tessellation_report.json`. CAD outputs include `gmsh_surface_tag`, `gmsh_parent_volume_tag`, and `gmsh_element_tag` cell arrays. Mesh outputs include `source_triangle_index`. Treat all of these as import/output-session provenance, not persistent source CAD IDs.

Use `scripts/audit_surface_mesh.py` for the current deterministic audit and screenshot generation:

```bash
python scripts/audit_surface_mesh.py /path/to/car.vtp --output-dir /tmp/cad-surface-audit
```

The script writes:

- `surface_mesh.vtp`
- `surface_mesh_quality.json`
- multi-view depth/normal/boundary-overlay PNGs
- `visual_checks.json` with separate AI inspection prompts
