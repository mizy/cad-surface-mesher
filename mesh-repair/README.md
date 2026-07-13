# Mesh Repair

`mesh-repair` builds and audits a geometry-driven watertight exterior shell from a mesh input.

## Agent-supervised TSDF workflow

The primary combined workflow uses deterministic exterior extraction and validation around a
flood-signed TSDF closure.  The Agent classifies semantic geometry before and after closure; it
never edits vertices or overrides a failed geometry gate.

```bash
python mesh-repair/scripts/agent_watertight_repair.py INPUT_MESH \
  --output-dir OUTPUT_DIR \
  --agent-mode auto \
  --sdf-grid-size 256 \
  --sdf-band-voxels 6 \
  --max-sdf-memory-gb 4
```

Use `--agent-mode off` for deterministic offline runs.  That mode conservatively keeps uncertain
pre-closure components and only accepts a result when the deterministic pipeline passes and no
semantic opening remains unresolved by the target policy.

The TSDF sign comes from a padded six-connected far-field flood, not source winding.  Exact
point-to-triangle distance replaces the EDT magnitude in a narrow band before bounded smoothing
and zero-isosurface extraction.  `implicit_field.npz` records the field, solid mask, transform,
pitch, and SDF parameters.

The authoritative combined artifacts are:

```text
OUTPUT_DIR/
  source_shell.vtp
  implicit_field.npz
  closure_proxy.vtp
  source_projected_candidate.vtp
  candidate_iter_0.vtp
  observation_packet.json
  region_inventory.json
  agent_decisions.jsonl
  transactions.jsonl
  repair_state.json
  watertight_mesh.vtp             # only when every required gate passes
  agent_repair_report.json
  agent_repair_report.html
```

`closure_proxy.vtp` is always diagnostic.  Only `watertight_mesh.vtp` and
`outputs.accepted_mesh_vtp` represent an accepted engineering result.

The current v1 executes one complete deterministic geometry transaction round followed by the
post-SDF Agent audit.  If that audit requests another geometry-changing action, the run records it
as blocking and returns a rejected candidate; it never pretends the requested follow-up was
applied.  `repair_state.json` already enforces stale-hash, no-progress, oscillation, and round-budget
rules for later multi-round execution.

The target is deliberately component-agnostic. The output may contain one or several connected components; component count is reported but never used as an acceptance gate.

## Target contract

```yaml
input_kind: mesh
output_kind: watertight_mesh
repair_domain: mesh
target: watertight-exterior-shell
geometry_truth: input mesh
component_count: diagnostic_only
```

Acceptance answers only whether the result is a trustworthy watertight exterior shell:

- zero boundary edges;
- zero non-manifold edges and vertices;
- zero inconsistent-winding edges and degenerate faces;
- complete all-face self-intersection audit with zero intersections;
- every naturally occurring component has a finite non-zero enclosed volume and outward orientation;
- bbox, six-axis silhouette/depth, and source-distance fidelity remain within configured limits;
- committed patches pass seam incidence, local quality, provenance, and transaction rollback checks;
- the written VTP survives readback with the same geometry and topology.

## Run

```bash
python mesh-repair/scripts/two_stage_watertight_remesh.py INPUT_MESH \
  --output-dir OUTPUT_DIR \
  --visibility-grid 720 \
  --outside-flood-grid 192 \
  --sealed-exterior-grid 192 \
  --voxel-pitch-bbox-divisor 280
```

`--group-source-gltf` is optional diagnostic metadata. It never supplies geometry and never changes the STL/VTP triangle truth.

## Geometry pipeline

1. Triangulate the input and preserve `source_triangle_index`.
2. Build six-axis conservative first-hit/depth evidence.
3. Add far-field flood and sealed-exterior evidence.
4. Remove only whole contained internal or isolated flying components when the evidence agrees; never cut a low-visibility strip out of an otherwise retained component.
5. Inventory every boundary region without report truncation.
6. Route simple small holes to constrained triangulation.
7. Route near-coincident loops to conformal weld or zipper after feasibility checks.
8. Use the closure proxy only as topology or local missing-surface guidance.
9. Apply each patch as an independent transaction; a failed patch rolls back only itself.
10. Project safe scaffold locations back to trusted source geometry and run topology/fidelity gates.

The source-projected candidate is eligible for acceptance when every required geometry gate passes. The raw closure proxy and local hybrid candidate remain diagnostic alternatives unless their role is explicitly promoted by the report contract.

## Boundary semantics

A free edge is not automatically a hole. Boundary inventory distinguishes:

- `small_exterior_hole`;
- `near_coincident_part_seam`;
- `large_opening_or_missing_surface`;
- `part_perimeter_or_opening_unknown`;
- `internal_or_fragment_component_perimeter`;
- `isolated_floating_fragment_perimeter`;
- `non_simple_boundary_graph`.

Large or ambiguous regions require an opening decision. AI may classify the semantic choice, but accepted geometry is always produced and validated by deterministic mesh operations.

## Repair operators

- constrained planar or curved hole fill using the original ordered source loop;
- fixed-source loop weld for near-zero-width component interfaces;
- annular zipper for two feasible ordered loops;
- multi-loop, slit, and local proxy-guided patching;
- local quality, orientation, seam-incidence, and self-intersection checks before commit.

Component merges caused by a valid weld or zipper are allowed. They are judged by the declared topology delta and geometry gates, not rejected merely because component count changed.

## Output roles

```text
OUTPUT_DIR/
  visibility_labeled_source.vtp
  stage1_exterior_candidate.vtp
  source_preserving_candidate.vtp
  closure_proxy.vtp
  source_projected_watertight_candidate.vtp
  hybrid_fused_candidate.vtp
  two_stage_report.json
  two_stage_report.html
  run_status.json
  patches/
  debug/
  visual/
```

Only `outputs.accepted_mesh_vtp` is an accepted result. A descriptive filename is not acceptance evidence by itself.

## Full-resolution issue report

```bash
python mesh-repair/scripts/watertight_issue_markers.py \
  INPUT_MESH OUTPUT_DIR/source_projected_watertight_candidate.vtp \
  --output-dir OUTPUT_DIR/watertight-issues

python mesh-repair/scripts/watertight_issue_html.py \
  OUTPUT_DIR/watertight-issues/watertight_issue_report.json \
  --output OUTPUT_DIR/watertight-issues/watertight_issue_report.html \
  --two-stage-report OUTPUT_DIR/two_stage_report.json
```

`*_surface_with_issue_arrays.vtp` contains the complete normalized triangle surface with issue arrays. `*_watertight_issue_faces.vtp` is intentionally an issue-adjacent subset, not a decimated model. `--preview-size` changes only static PNG size. HTML embeds every triangle by default; a positive `CAD_SURFACE_MESHER_VIEWER_TRIANGLES` value is an explicit opt-in preview limit.

See [report_contract.md](report_contract.md) for the machine-readable report rules.
