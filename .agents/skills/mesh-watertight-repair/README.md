# Mesh Watertight Repair

Build a geometry-driven watertight exterior shell from a mesh input.

All commands below assume the working directory is this skill directory (`.agents/skills/mesh-watertight-repair/`).

## Skill entrypoint and worker boundary

[`SKILL.md`](SKILL.md) is the only AI orchestration entrypoint. The active Skill agent reads
deterministic reports and visual evidence, makes semantic decisions, and invokes the scripts in this
directory as geometry workers. No worker launches an AI model, accepts an `--agent-mode`, or overrides
a failed geometry gate.

The internal geometry worker for one repair round is:

```bash
python scripts/two_stage_watertight_remesh.py INPUT_MESH \
  --output-dir OUTPUT_DIR \
  --visibility-grid 720 \
  --outside-flood-grid 192 \
  --sealed-exterior-grid 192 \
  --voxel-pitch-bbox-divisor 560 \
  --sdf-band-voxels 6 \
  --max-sdf-memory-gb 4
```

This command is intentionally deterministic. It performs one geometry round and writes
`two_stage_report.json`, `ai_policy_packet.json`, visual evidence, transaction diagnostics, and the
candidate artifacts needed by the Skill agent. Direct model execution from a mesh-repair CLI is not
part of the contract.

`--group-source-gltf` is optional diagnostic metadata. It never supplies geometry and never changes
the STL/VTP triangle truth.

The TSDF sign comes from a padded six-connected far-field flood, not source winding.  Exact
point-to-triangle distance replaces the EDT magnitude in a narrow band before bounded smoothing
and zero-isosurface extraction.  `implicit_field.npz` records the field, solid mask, transform,
pitch, and SDF parameters.

Closest-point projection is a compiled PyVista/VTK batch query. Formal self-intersection acceptance
uses libigl's CGAL exact-predicate detector on every face; the worker never nudges a proxy to satisfy
an approximate scanner. Projected intersections are repaired only by restoring the implicated
projected vertices to the already certified proxy and certifying the complete candidate again.

The primary round artifacts are:

```text
OUTPUT_DIR/
  visibility_labeled_source.vtp
  stage1_exterior_candidate.vtp
  source_preserving_candidate.vtp
  implicit_field.npz
  closure_proxy.vtp
  source_projected_watertight_candidate.vtp
  ai_policy_packet.json
  two_stage_report.json
  two_stage_report.html
  run_status.json
```

`closure_proxy.vtp` is always diagnostic. Only the path named by
`outputs.accepted_mesh_vtp` represents an accepted engineering result.

When semantic review is needed, the Skill agent may use `agent_observation.py` and
`agent_repair_contract.py` to build registered evidence, hash-bound decisions, transaction records,
and repair state. These are protocol helpers, not a second workflow entrypoint. Any requested
geometry change must be executed by a declared deterministic worker operation and re-audited.

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
- the written VTP survives readback with the same geometry and topology.

## Geometry pipeline

1. Triangulate the input and preserve `source_triangle_index`.
2. Build six-axis conservative first-hit/depth evidence.
3. Add far-field flood and sealed-exterior evidence.
4. Remove only whole contained internal or isolated flying components when the evidence agrees; never cut a low-visibility strip out of an otherwise retained component.
5. Remove only degenerate and exact-duplicate source triangles before reconstruction.
6. Build and certify the sealed-exterior TSDF closure scaffold.
7. Project safe scaffold locations back to trusted source geometry with PyVista/VTK batch locators.
8. Certify proxy and projected candidates with libigl/CGAL exact predicates; roll back only
    implicated projected vertices.
9. When visual QA finds residual scaffold grain, use topology-preserving VTK Windowed-Sinc
    smoothing and repeat complete CGAL, topology, volume, silhouette, and distance validation.
10. Serialize, read back, and re-audit every requested output format.

The source-projected candidate is eligible for acceptance when every required geometry gate passes.
The raw closure proxy remains diagnostic and can never be promoted directly.

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

## Output roles

```text
OUTPUT_DIR/
  visibility_labeled_source.vtp
  stage1_exterior_candidate.vtp
  source_preserving_candidate.vtp
  closure_proxy.vtp
  source_projected_watertight_candidate.vtp
  two_stage_report.json
  two_stage_report.html
  run_status.json
  debug/
  visual/
```

Only `outputs.accepted_mesh_vtp` is an accepted result. A descriptive filename is not acceptance evidence by itself.

## Full-resolution issue report

```bash
python scripts/watertight_issue_markers.py \
  INPUT_MESH OUTPUT_DIR/source_projected_watertight_candidate.vtp \
  --output-dir OUTPUT_DIR/watertight-issues

python scripts/watertight_issue_html.py \
  OUTPUT_DIR/watertight-issues/watertight_issue_report.json \
  --output OUTPUT_DIR/watertight-issues/watertight_issue_report.html \
  --two-stage-report OUTPUT_DIR/two_stage_report.json
```

`*_surface_with_issue_arrays.vtp` contains the complete normalized triangle surface with issue arrays. `*_watertight_issue_faces.vtp` is intentionally an issue-adjacent subset, not a decimated model. `--preview-size` changes only static PNG size. HTML embeds every triangle by default; a positive `CAD_SURFACE_MESHER_VIEWER_TRIANGLES` value is an explicit opt-in preview limit.

## References

- See [report_contract.md](report_contract.md) for the machine-readable report rules.
- See [references/yu7-high-detail-vehicle-success.md](references/yu7-high-detail-vehicle-success.md)
  for the closure-first high-detail vehicle playbook, measured tradeoffs, and rejected alternatives.
