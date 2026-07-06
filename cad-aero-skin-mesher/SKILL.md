---
name: cad-aero-skin-mesher
description: Build and validate exterior aerodynamic surface meshes from CAD or vehicle mesh inputs. Use when Codex needs to tessellate CAD, remove interior vehicle parts, seal panel gaps, handle grille/opening policy, generate target-driven visual inspection screenshots, audit watertightness/non-manifold topology/triangle quality, or produce an aero-skin mesh quality report for CFD/CAE preprocessing.
---

# CAD Aero Skin Mesher

Create an exterior aerodynamic skin, not a faithful full assembly mesh. The target is a CFD/CAE-ready surface mesh with explicit visual and topology evidence.

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

1. Tessellate CAD or load the input mesh with face/body IDs when available.
2. Generate deterministic audit metrics and target-driven screenshots.
3. Run separate AI visual checks for each goal:
   - exterior skin completeness
   - interior part residue
   - panel gap sealing
   - functional opening policy
   - exterior feature preservation
   - patch artifacts
   - silhouette drift
4. Convert visual findings into a repair plan.
5. Apply one deterministic repair step at a time.
6. Re-audit and regenerate screenshots after each repair.
7. Stop when validators pass or when policy/tolerance limits require user confirmation.

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

Use `scripts/audit_surface_mesh.py` for the current deterministic audit and screenshot generation:

```bash
python scripts/audit_surface_mesh.py /path/to/car.vtp --output-dir /tmp/aero-skin-audit
```

The script writes:

- `surface_mesh.vtp`
- `surface_mesh_quality.json`
- multi-view depth/normal/boundary-overlay PNGs
- `visual_checks.json` with separate AI inspection prompts

