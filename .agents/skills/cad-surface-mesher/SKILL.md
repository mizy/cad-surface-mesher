---
name: cad-surface-mesher
description: Build and validate target-specific CAD surface meshes from CAD or vehicle mesh inputs. Use when Codex needs to tessellate CAD, remove interior vehicle parts with group-level show/hide tests, prune AABB-contained internals, extract only exterior wall surfaces, seal panel gaps, handle grille/opening policy, generate target-driven visual inspection screenshots, audit watertightness/non-manifold topology/triangle quality, or produce a surface mesh quality report for CFD/CAE preprocessing.
---

# CAD Surface Mesher

Create a target-specific CAD surface mesh, not a faithful full assembly mesh. The default target is an exterior aerodynamic skin for CFD/CAE preprocessing with explicit visibility, visual, and topology evidence.

## Core Rule

Use AI to classify and review visual evidence. Use deterministic geometry checks to decide mesh quality. Do not claim success from screenshots alone.

## Target Policy

Start every task by writing or confirming the target policy:

```yaml
target: external-aero-cfd-skin
remove:
  cabin_interior: true
  engine_bay_internals: true
  hidden_internal_faces: true
seal:
  panel_gaps: true
  window_body_cracks: true
  hood_trunk_door_seams: true
openings:
  grille: ask
  cooling_intake: ask
  wheel_well: keep
  underbody: ask
limits:
  max_gap_fill_width_mm: 30
  max_surface_offset_mm: 5
  max_bbox_drift_ratio: 0.002
```

If the user asks for simple external Cd triage, cap small panel gaps and ask before changing functional openings such as grilles, cooling intakes, wheel wells, underbody openings, and exhausts.

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

## Group Visibility Workflow

Prefer group-level decisions before body-level or face-level work. A human CAD cleanup usually starts by hiding named assembly groups and checking whether the exterior changed; follow that pattern.

1. Build a normalized group table with `path`, `name`, `parent`, `children`, `aabb`, `area`, `volume`, `material`, `body_count`, and source IDs.
2. Assign a weak semantic label from names and paths:
   - likely remove: `interior`, `seat`, `dashboard`, `steering`, `trim`, `speaker`, `engine`, `battery`, `hvac`, `wire`, `bolt`, `fastener`, `bracket`, `suspension_internal`
   - likely keep: `body`, `exterior`, `skin`, `panel`, `door`, `hood`, `trunk`, `bumper`, `glass`, `window`, `mirror`, `lamp`, `wheel`, `tire`, `grille`, `splitter`, `spoiler`, `wiper`, `sensor`
3. Find large exterior-shell candidates from name, area, and bbox span. Use them as containment references, not the full vehicle bbox alone.
4. Mark a group as `internal_candidate` when its AABB is strictly contained inside an exterior-shell candidate with clearance, it has no exterior name token, and its projected silhouette contribution is small.
5. Render A/B screenshots at group granularity:
   - baseline full assembly
   - hide each candidate group or candidate parent group
   - only exterior-shell candidates
   - removed-candidate overlay
6. Compare front, rear, left, right, top, bottom, and three-quarter views. Remove a group when hiding it has near-zero silhouette/visible-area delta; use AI to explain non-zero changed pixels, not to override a large exterior delta.
7. If hiding a parent group changes exterior silhouette, recurse into children instead of deleting the parent.

AABB containment is a speed filter, not proof. Do not delete mirrors, lights, tires, wheels, grilles, glass, aero add-ons, sensors, or underbody-visible parts from AABB containment alone.

## Exterior Wall Extraction

After group reduction, keep exterior wall surfaces before final meshing.

- Prefer original CAD faces marked as exterior by group visibility and exterior-name evidence.
- When only mesh geometry is available, ray-cast from outside views and keep the first surface hit along each ray; use this as an outermost-wall score per face/body.
- Remove duplicate inner sheets only when they sit behind an exterior sheet with similar normals/curvature and no exterior visibility.
- Preserve face/body IDs through tessellation so later patches can report their source.
- Mesh the reduced exterior wall set, then repair gaps and openings according to target policy.

## Two-Face Gaps

For each boundary loop, classify the local situation before repair:

- `seal_panel_gap`: hood, door, trunk, or window seam in external-aero mode.
- `seal_cad_gap`: small assembly/tessellation gap under the configured width limit.
- `remove_inner_sheet`: duplicate or hidden internal sheet behind the exterior skin.
- `keep_separate`: true separated exterior components such as wheel/body gaps.
- `ask`: ambiguous functional opening or over-limit patch.

Use boundary-loop distance, normal/curvature compatibility, and visual closeups together. Do not stitch duplicate inner and outer sheets together.

## Grilles And Functional Openings

Treat grilles as policy openings, not ordinary holes.

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

From the repository root, use `cad-tessellation/scripts/cad_tessellate.py` first when the input is STEP, IGES, or BREP CAD and a triangle-only surface mesh is needed:

```bash
python cad-tessellation/scripts/cad_tessellate.py tessellate /path/to/model.step --output-dir /tmp/cad-tessellation
```

The tessellator writes `surface_mesh.vtp` with `gmsh_surface_tag`, `gmsh_parent_volume_tag`, and `gmsh_element_tag` cell arrays plus `tessellation_report.json`. Treat those tags as Gmsh import-session provenance, not persistent source CAD IDs.

Use `scripts/audit_surface_mesh.py` for the current deterministic audit and screenshot generation:

```bash
python scripts/audit_surface_mesh.py /path/to/car.vtp --output-dir /tmp/cad-surface-audit
```

The script writes:

- `surface_mesh.vtp`
- `surface_mesh_quality.json`
- multi-view depth/normal/boundary-overlay PNGs
- `visual_checks.json` with separate AI inspection prompts
