# Watertight Exterior Shell Report Contract

The JSON report is authoritative. HTML is a human-readable rendering of the same facts.

## Scope

```text
input_kind = mesh
output_kind = watertight_mesh
repair_domain = mesh
target = watertight-exterior-shell
component_count = diagnostic_only
```

The contract is geometry-driven and input-agnostic. Runtime and optional metadata are diagnostic only and have no acceptance authority.

## Required top-level sections

- `decision`
- `input`
- `input_truth`
- `output_contract`
- `target`
- `parameters`
- `stages`
- `inventory_before`
- `inventory_after`
- `deterministic_passes`
- `patch_regions`
- `comparisons`
- `gates`
- `outputs`
- `ignored_outputs`
- `unhandled_items`
- `repair_report`

## Required geometry gates

Every required gate must contain `passed`, the measured value, its threshold, and a stable failure reason or reason code.

Required acceptance checks:

1. Candidate artifact exists and can be decoded.
2. Boundary edges equal zero.
3. Non-manifold edges and non-manifold vertices equal zero.
4. Inconsistent-winding edges equal zero.
5. Degenerate faces equal zero.
6. The all-face self-intersection scan completed and found zero pairs.
7. Every connected component has finite non-zero enclosed volume.

The volume change between the closure proxy and the source-projected candidate
is reported for diagnosis only. The proxy owns connectivity, not target shape,
so no fixed proxy-relative volume percentage is an acceptance gate.

The closure proxy volume is constructed by locally sealing the voxelized source
shell, flooding outside space from a padded far field, filling the complement,
and restoring the temporary dilation offset before marching cubes. Direct hole
filling of a still-leaky voxel shell is not an exterior-volume construction.

Acceptance also requires the projected candidate's reliable signed volume to
remain above the sealed volume's erosion core. The erosion radius is the
smallest integer voxel radius covering the configured maximum projection
distance. This absolute, resolution-derived lower bound prevents a
topologically closed thin coating from passing; it is not a fixed percentage
comparison with proxy volume.
8. Every connected component is outward oriented.
9. Bbox drift is within the configured dimensionless tolerance.
10. All six axis-aligned silhouette/depth comparisons are within tolerance.
11. Bidirectional source-distance evidence is complete and within bbox/local-edge normalized limits.
12. Every committed patch passes seam incidence, orientation, local triangle quality, local self-intersection, provenance, and transaction rollback checks.
13. The serialized candidate preserves topology and geometry on readback.

Connected-component count is always reported but is never itself a pass/fail condition. A legitimate conformal weld may reduce it; a legitimate disconnected exterior shell may contain more than one component.

## Geometry truth and provenance

The input mesh is the only source geometry truth. Optional GLTF/CAD metadata may label or diagnose source regions but cannot move, add, remove, or accept triangles by itself.

Source faces retain `source_triangle_index`. Generated faces record at least:

- `face_origin`;
- `fusion_region_id`;
- source/proxy consumption state;
- operator or patch method;
- parent source region where applicable.

The report may include file and geometry SHA-256 values to bind evidence to artifacts. Hashes are lineage metadata, not model-specific allowlists and not geometric acceptance gates.

## Patch transactions

Each repair region is independent. A patch commits only when its declared local topology delta and all local gates pass. Rejected patches leave the current mesh unchanged. A bad patch cannot roll back unrelated successful patches.

Cross-component operations are allowed when the operator explicitly declares a seam weld or zipper. They must prove the expected component delta, edge incidence two, opposite directed seam occurrences, local quality, no self-intersection, and bounded source drift.

## Agent-supervised TSDF extension

The combined Agent workflow adds the following authoritative records without weakening any
geometry gate:

- `implicit_field.npz` uses `implicit_field/v1` and records the flood-signed TSDF, solid mask,
  transform, pitch, band width, and smoothing request;
- `observation_packet.json` binds every Agent decision to the current candidate geometry hash,
  stable region IDs, registered view IDs, deterministic gates, and a defect signature;
- `agent_decisions.jsonl` contains region-level semantic or operator choices only; raw point,
  vertex, face, triangle, or coordinate edits are invalid;
- `transactions.jsonl` records deterministic dry-run/commit/rollback ownership and geometry hashes;
- `repair_state.json` records round budget, no-progress and oscillation detection, and the final stop reason.

An automatic Agent timeout, invalid schema, stale geometry hash, unknown region or view ID, missing
region decision, or unauthorized action is fail-closed.  Agent approval is an additional semantic
gate; it cannot turn a rejected deterministic candidate into an accepted mesh.

The raw TSDF zero surface remains a closure proxy.  The combined workflow may populate
`accepted_mesh_vtp` only with the final source-projected, transaction-audited candidate after both
the deterministic gates and Agent semantic gates pass.

## Output roles

- `closure_proxy_vtp`: diagnostic topology/shape guide.
- `source_preserving_candidate_vtp`: cleaned source fallback; may remain open.
- `source_projected_watertight_candidate_vtp`: main whole-shell candidate.
- `hybrid_fused_candidate_vtp`: local-patch diagnostic candidate.
- `accepted_mesh_vtp`: populated only when every required gate passes.
- `rejected_candidate_vtp`: candidate retained for diagnosis after rejection.

`decision.final_output_path` and `outputs.accepted_mesh_vtp` must be identical for an accepted run. Both are null for a rejected run.

## Full-resolution issue evidence

The original and processed annotated VTPs retain every normalized input triangle. The water-tightness comparison HTML also embeds full-resolution geometry. The report records:

```text
triangles == original_triangles
full_resolution == true
downsampled == false
```

Issue-face VTPs are subsets by definition and must be labeled as such.

## Non-geometric diagnostics

Elapsed time, optional timeouts, atomic-write state, and output-directory locks may be reported for operations. They never change whether a decoded candidate is geometrically watertight. Unknown units or coordinate semantics are reported as metadata limitations rather than replaced with guessed values.
