---
name: mesh-watertight-repair
description: Build, audit, and validate a geometry-driven watertight exterior shell from an arbitrary mesh input. Use when Codex needs to take a non-watertight mesh (STL, OBJ, VTP, VTK, GLB, GLTF) and produce a watertight exterior shell with zero boundary edges, zero non-manifold elements, zero self-intersections, reliable enclosed volume, and outward orientation — or to generate full-resolution watertightness issue reports with HTML viewer evidence.
---

# Mesh Watertight Repair

Build a watertight exterior shell from a mesh input using Skill-orchestrated, deterministic geometry operations.

## Entrypoint and Worker Boundary

This `SKILL.md` is the only AI orchestration entrypoint. The active Skill agent inspects evidence,
makes semantic decisions, invokes deterministic geometry workers, and owns any multi-round repair
loop. Files under `scripts/` never represent or launch an Agent, and no mesh-repair CLI may claim an
`--agent-mode`.

The geometry workers may generate observation packets, registered views, transaction records, and
validation reports for the Skill agent. Those artifacts support the Skill loop; they do not create a
second Agent entrypoint and cannot override a failed geometry gate.

## I/O Contract

```yaml
input_kind: mesh
output_kind: watertight_mesh
repair_domain: mesh
target: watertight-exterior-shell
geometry_truth: input mesh
component_count: diagnostic_only
```

## Quick Start

All commands below assume the working directory is this skill directory (`.agents/skills/mesh-watertight-repair/`).

### Internal deterministic geometry worker

```bash
python scripts/two_stage_watertight_remesh.py INPUT_MESH \
  --output-dir OUTPUT_DIR \
  --visibility-grid 720 \
  --outside-flood-grid 192 \
  --sealed-exterior-grid 192 \
  --voxel-pitch-bbox-divisor 560
```

The command performs one deterministic geometry round. It is called by this Skill; it is not a
separate user-facing or Agent entrypoint.

### Skill-owned repair loop

1. Run the deterministic worker in a fresh round directory.
2. Read `two_stage_report.json`, `ai_policy_packet.json`, registered evidence, and failed gates.
3. Make semantic decisions only for stable component or region IDs and bind them to the packet
   fingerprints. Never issue raw point, vertex, face, triangle, or coordinate edits.
4. Rerun the worker in a new round only to change declared exterior policy, TSDF resolution,
   smoothing, or projection thresholds from measured evidence.
5. Stop only when `outputs.accepted_mesh_vtp` is populated or when policy, no-progress,
   oscillation, or round-budget rules block safe progress.

`agent_observation.py` and `agent_repair_contract.py` are reusable protocol helpers for this loop,
not executable Agent workflows.

## Route Selection

Use the closure-first route for arbitrary dirty meshes. Never use global "delete every intersecting
face, then fill" as the default route; it destroys source evidence, couples unrelated defects, and
makes the replacement surface ambiguous. Enforce this ownership boundary:

1. Let the sealed-exterior TSDF proxy own topology and inside/outside classification.
2. Let trusted source triangles own target shape through gated closest-point projection.
3. Let all-face libigl/CGAL exact predicates own self-intersection acceptance.
4. Roll back implicated projected vertices to the certified proxy; never repair an exact-predicate
   failure by heuristic nudging or unconditional face deletion.
5. Apply local topology-preserving smoothing only to visually defective scaffold regions, then
   repeat exact certification and every fidelity gate.

Read [references/yu7-high-detail-vehicle-success.md](references/yu7-high-detail-vehicle-success.md)
when repairing a high-detail vehicle or another large globally defective mesh, tuning TSDF
resolution, diagnosing visible voxel grain, or choosing a scalable self-intersection strategy.

## Performance Guardrails

- Use compiled PyVista/VTK batch closest-cell queries; never loop over hundreds of thousands of
  vertices with one locator call per vertex in Python.
- Use libigl/CGAL exact-predicate detection for formal all-face certification. Keep approximate or
  tolerance-based scanners diagnostic-only.
- Escalate TSDF resolution from a cheap diagnostic proxy to the lowest resolution that passes
  silhouette, source-distance, and smooth-shaded visual QA; do not promote a visibly faceted proxy.
- Run cheap decode, topology, degeneracy, and volume gates before expensive full-candidate checks,
  but repeat exact self-intersection certification after every geometry mutation eligible for
  acceptance.

### Full-resolution watertightness issue report

```bash
python scripts/watertight_issue_markers.py \
  INPUT_MESH OUTPUT_DIR/source_projected_watertight_candidate.vtp \
  --output-dir OUTPUT_DIR/watertight-issues

python scripts/watertight_issue_html.py \
  OUTPUT_DIR/watertight-issues/watertight_issue_report.json \
  --output OUTPUT_DIR/watertight-issues/watertight_issue_report.html \
  --two-stage-report OUTPUT_DIR/two_stage_report.json
```

## Pipeline

1. Triangulate input and preserve `source_triangle_index`.
2. Build six-axis conservative first-hit/depth evidence.
3. Add far-field flood and sealed-exterior evidence.
4. Remove wholly contained internal or isolated flying components when evidence agrees.
5. Remove only degenerate and exact-duplicate source triangles before reconstruction.
6. Build the sealed-exterior TSDF and extract one certified closure scaffold.
7. Project safe scaffold locations to trusted source geometry with PyVista/VTK batch locators.
8. Certify the proxy and every projected candidate with libigl/CGAL exact predicates; roll back
    only projected vertices touched by reported intersecting face pairs.
9. Use topology-preserving VTK Windowed-Sinc smoothing only when visual QA finds scaffold grain,
    then repeat CGAL certification and topology/fidelity gates.
10. Serialize, read back, and re-audit every requested output format.

## Required Geometry Gates

Every required gate must pass for acceptance:

1. Candidate artifact exists and decodes.
2. Boundary edges == 0.
3. Non-manifold edges == 0 and non-manifold vertices == 0.
4. Inconsistent-winding edges == 0.
5. Degenerate faces == 0.
6. All-face libigl/CGAL exact-predicate self-intersection certification: zero pairs.
7. Every component has finite non-zero enclosed volume.
8. Every component is outward oriented.
9. Bbox drift within tolerance.
10. All six axis-aligned silhouette/depth comparisons within tolerance.
11. Bidirectional source-distance evidence complete and within limits.
12. Serialized candidate survives readback with identical topology and geometry.

## Output Roles

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

Only `outputs.accepted_mesh_vtp` is an accepted result.

## Boundary Semantics

- `small_exterior_hole`
- `near_coincident_part_seam`
- `large_opening_or_missing_surface`
- `part_perimeter_or_opening_unknown`
- `internal_or_fragment_component_perimeter`
- `isolated_floating_fragment_perimeter`
- `non_simple_boundary_graph`

## Core Operators

- NumPy/SciPy/scikit-image sealed-exterior TSDF and marching cubes
- PyVista/VTK batch closest-cell projection and Windowed-Sinc smoothing
- libigl/CGAL exact all-face self-intersection certification
- deterministic projection rollback to certified proxy positions
- topology, volume, silhouette, source-distance, and serialized-readback gates

## Documentation

- [README.md](README.md) — detailed pipeline, semantics, and usage
- [report_contract.md](report_contract.md) — machine-readable report contract
