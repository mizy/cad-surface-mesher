# AI-Directed 3D Mesh Engineering Plan

Status: planning only. This document does not authorize implementation or mesh mutation.

## Motivation

Watertight repair is not one fixed geometry pipeline. The correct action depends on both the
model and the engineering target. A vehicle body and four disconnected wheels can be valid as
five closed wall components for external-flow CFD, while a printable single solid may require
boolean union or explicit bridges. Geometry alone cannot resolve that policy choice.

The long-term goal is therefore an agent-directed workflow:

- the AI agent observes the 3D scene, forms semantic and spatial hypotheses, and chooses an
  engineering strategy;
- the skill supplies reusable observation, mutation, transaction, and validation operations;
- deterministic geometry code remains the authority for topology, distances, intersections,
  provenance, and acceptance gates;
- the agent asks for target clarification before an irreversible choice when multiple engineering
  interpretations remain valid.

The current fixed sequence of visibility filtering followed by repair should become one possible
playbook assembled by the agent, not the hard-coded definition of watertight meshing.

## Ownership and truth

- The source CAD or mesh is immutable geometry truth.
- A scene manifest owns component IDs, face provenance, transforms, units, axes, and exact measured
  relationships.
- AI labels such as `wheel`, `body`, `internal_support`, and `functional_opening` are derived
  hypotheses with confidence and evidence references.
- A target policy owns the intended engineering result: external-flow walls, one printable solid,
  moving assembly, structural surface, or another explicit target.
- Every mutation creates a new mesh version with a transaction record. It never silently rewrites
  the source.
- Deterministic validators own pass/fail results. An AI visual judgment cannot override a failed
  topology or geometry gate.

## Agent operating loop

1. Confirm or infer the target policy, asking when the target changes the valid result.
2. Inspect the scene manifest and an initial observation bundle.
3. Form component-role and relationship hypotheses.
4. Request targeted observations to resolve uncertainty.
5. Select an engineering playbook and propose atomic operations.
6. Dry-run the operations and inspect predicted changes, affected IDs, and geometric drift.
7. Commit one reversible transaction.
8. Re-observe and run deterministic validators.
9. Stop on acceptance or ask when the next action exceeds policy or tolerance.

## What it means for AI to understand a 3D model

The first version should not assume that a vision model natively understands raw STL triangles,
normal-map colors, or a continuous WebGL scene. It should give the agent active perception: a
structured scene plus tools for requesting useful, grounded observations.

### Observation channels

| Channel | Primary value | Limitation | Planned role |
| --- | --- | --- | --- |
| Neutral shaded RGB render | Object semantics, silhouette, familiar manufactured forms | Hides depth ordering and occluded structure | Primary semantic image |
| Highlighted component render | Grounds a semantic label to stable component IDs | Requires reliable ID rendering | Primary component-review image |
| Depth render | Occlusion, relative ordering, cavities, front/back separation | Raw grayscale is ambiguous without camera range and legend | Paired geometric evidence |
| Normal render | Surface orientation and discontinuities | Encoded RGB normals are weak semantic evidence and encoding-dependent | Secondary diagnostic channel |
| Component-ID buffer | Exact component grounding and changed-pixel attribution | Arbitrary colors carry no meaning by themselves | Machine grounding plus labeled overlay |
| Face-ID buffer | Maps visual findings back to source triangles | Too detailed for whole-scene semantic review | Local inspection and provenance |
| Boundary/contact/intersection overlay | Makes engineering defects and relationships explicit | Depends on deterministic preprocessing | High-value task-specific evidence |
| Transparent or clipped render | Reveals containment, wheel wells, ducts, and hidden relationships | A single clipping choice can still mislead | Agent-requested ambiguity resolver |
| Turntable video | Rapid whole-object overview and continuity across angles | Redundant frames, uncertain sampled frames, weak exact grounding | Optional orientation aid, not authority |
| Agent-controlled viewport | Lets the agent choose the next most informative observation | Requires a deterministic action and camera-state contract | Preferred active-perception interface |

Depth and normal maps should always be paired with a shaded render from the same camera. Depth
must include near/far or metric range metadata and a visible legend. Normal maps must state world-
space or camera-space encoding. Derived overlays such as depth discontinuities and normal breaks
may be more useful to the AI than raw encoded maps, so both must be evaluated rather than assumed.
Dirty meshes may have unreliable winding, so every normal image must also state whether orientation
is trusted; an unoriented two-sided geometric-normal diagnostic may be needed. Exact gap, contact,
containment, and intersection facts must come from geometry queries rather than visual inference
from depth or normal images.

### Initial view policy

The initial bundle should be small and deterministic:

- six orthographic views aligned to known model axes or an oriented bounding box: front, rear,
  left, right, top, and bottom;
- four perspective three-quarter views plus at least one lower oblique view;
- a neutral shaded render and a component-ID-grounded counterpart for every view;
- camera position, target, up vector, projection, field of view or orthographic scale, clipping
  range, visible component IDs, and model-axis labels stored with every image.

For a generic unoriented model, views must initially be labeled by geometric axes rather than
pretending that an axis is the semantic front. The agent may infer semantic orientation and record
that as a hypothesis.

Fixed views are only the opening observation. They cannot reliably expose wheel wells, underbody
recesses, nested parts, thin gaps, or intersections. When uncertainty remains, the agent should
request targeted views instead of receiving dozens of arbitrary screenshots.

For a vehicle wheel/body question, side, front or rear, bottom, and a focused oblique wheel-arch
view carry more unique information than a dense set of evenly distributed distant cameras. A
42-direction set can remain useful for deterministic visibility statistics, but should not be sent
to the AI as one undifferentiated image dump.

### Agent-controlled 3D view

The essential capability is a deterministic view controller, whether it is backed by a browser
Viewer, an offscreen renderer, or another engine. A human-facing UI is useful, but the agent-facing
action contract is the first requirement.

Planned observation actions include:

- fit the whole scene or selected components;
- set an exact camera or orbit by azimuth and elevation;
- zoom, pan, and capture a grounded screenshot;
- isolate, hide, highlight, or make components transparent;
- show an exploded arrangement without changing source transforms;
- add and move clipping planes;
- show only a component pair and their closest/contact/intersection region;
- pick a pixel and return component ID, face ID, world position, depth, and normal;
- render shaded RGB, depth, normal, component ID, face ID, and defect overlays from the identical
  camera state;
- save every observation under a stable `view_id` with full camera and visibility metadata.

The first agent-facing contract should be equivalent to:

```text
observe_3d(
  camera,
  target = all | component_ids | region_id,
  hidden_component_ids,
  transparent_component_ids,
  clip_planes,
  render_modes = [shaded, depth, normal, component_id, face_id, boundary, wireframe],
  resolution
) -> registered images + camera metadata + pixel-to-source mapping
```

The AI does not need to consume a continuously streaming 3D viewport. It can operate the viewport,
receive discrete screenshots and structured query results, then choose its next action. This is
more precise, reproducible, and auditable than relying on a video alone.

The first renderer must show solid shaded triangles with real z-buffer occlusion. Projected
triangle centroids or sparse point samples are unsuitable for measuring AI 3D understanding
because they are far from the image distribution used to train general vision models.

## Skill architecture

The skill should contain engineering playbooks and stopping rules, while scripts expose reusable
operations. The agent chooses and sequences them.

### Observation operations

- inspect scene, components, units, axes, topology, and provenance;
- build containment, proximity, contact, intersection, symmetry, and visibility relationships;
- render exact camera states and auxiliary buffers;
- perform hide/show A/B comparisons;
- isolate a component or relationship for local review;
- query exterior accessibility and deterministic geometry facts.

### Mutation operations

- keep, remove, split, or regroup components and face patches;
- split and corefine intersecting surfaces;
- boolean union, difference, or trim when preconditions pass;
- weld coincident seams and stitch compatible boundary loops;
- cap or fill classified openings under explicit tolerances;
- make selected components independently watertight;
- apply local surface wrapping or remeshing under a source-drift contract;
- combine multiple closed shells into one output dataset while preserving zones and provenance.

### Transaction and validation operations

- dry-run and report affected source IDs, predicted topology changes, and drift;
- checkpoint, commit, compare, and roll back;
- validate boundary and non-manifold edges, self intersections, orientation, component closure,
  volume reliability, source distance, silhouette drift, and target-specific opening policy;
- produce a decision trace explaining the evidence, selected playbook, operations, and gates.

Atomic operations should be large enough to own a real engineering responsibility. Per-triangle
editing is too low-level for agent planning, while `make_watertight` is too broad to be auditable.

## Existing prototypes to consolidate

The repository already contains useful prototypes. They should inform the observation contract and
be consolidated only after the first-step evaluation; several are currently untracked or modified
and must not be treated as stable public APIs yet.

- `.agents/skills/cad-surface-mesher/scripts/audit_surface_mesh.py` already emits fixed-view depth,
  normal, and boundary images. Its vehicle-axis assumptions, missing bottom view, and center-based
  normal orientation need to be removed from a generic observer.
- `mesh-repair/scripts/solid_triangle_raster.py` provides conservative solid-triangle depth,
  silhouette, and first-hit face-ID rasterization and is the strongest current renderer base.
- `mesh-repair/scripts/source_shell_visibility.py` proves that deterministic arbitrary-direction
  cameras are feasible, while `source_shell_visuals.py` provides component location and isolation
  contact sheets.
- `source_shell_candidates.py`, `boundary_classification.py`, `watertight_issue_markers.py`, and
  `mesh_metrics.py` provide much of the component and topology manifest data.
- `opening_policy_visuals.py` already demonstrates global context plus a focused local region.
- `source_shell_ai_review.py` provides a read-only, schema-validated, fail-closed Codex review
  adapter, but currently consumes a preselected static image set.
- `html_mesh_preview.py` provides a human-operated vtk.js Viewer but does not expose deterministic
  camera state, component visibility, clipping, picking, screenshots, or an agent command protocol.

The largest missing geometry primitive is a component relationship graph covering nearest points,
minimum gap, contact, intersection, containment, relative pose and size, symmetry, approximate
rotation axes, and component-level occlusion contribution.

Active perception also needs session orchestration. A one-shot ephemeral review that forbids all
tools is appropriate for a bounded static classification, while an observing agent must retain its
hypotheses across `request view -> render -> return observation` rounds. This can use a resumable
model session or an outer orchestrator that owns the observation history and action budget.

## First step: 3D perception evaluation and observation contract

### Goal

Determine, with controlled evidence, which observation bundle lets the current AI agent understand
component roles and spatial relationships well enough to choose a watertight engineering strategy.
This step performs no mesh repair and no geometry mutation.

### Questions to answer

1. How well does the model understand neutral shaded multi-view renders by themselves?
2. Does paired depth improve containment, gap, and front/back judgments?
3. Does a raw normal map help, or are explicit normal-discontinuity overlays more useful?
4. Which initial views contribute unique information, especially bottom and three-quarter views?
5. Can an agent-controlled viewport resolve cases that remain ambiguous under fixed views?
6. Does turntable video add useful evidence beyond selected frames and dynamic screenshots?
7. How much latency, token use, and model cost does each observation mode require?

### Evaluation scenes

Create at least six controlled synthetic scenes and two complex public or local-only scenes,
covering different relationship types:

- one simple closed exterior object;
- one body with four disconnected closed wheel-like components;
- wheel-like components intersecting or touching the body;
- a fully contained internal component;
- an exterior-accessible wheel well, duct, or recessed cavity;
- an assembly mixing a visible accessory with a visually similar hidden support;
- a locally available complex vehicle such as YU7 for non-committed integration evaluation.

Synthetic and public fixtures may be committed. Private or local vehicle assets remain outside the
repository; only permitted derived evaluation metrics may be retained.

Each scene needs machine-readable ground truth for component IDs, semantic roles when known,
containment, contact, intersection, separation distance, exterior accessibility, and the correct
strategy under at least two target policies.

### Observation modes to compare

- **A — RGB baseline:** fixed neutral shaded views only.
- **B — grounded baseline:** RGB plus registered component-ID images and a deterministic component
  relationship manifest.
- **C — auxiliary geometry channels:** mode B plus paired depth and raw normal or
  normal-discontinuity images from identical cameras.
- **D — active viewport:** mode B plus at most two rounds and four images per round of
  agent-requested camera, visibility, clipping, picking, and measurement actions. Mode C channels
  remain available on request.
- **E — video baseline:** a deterministic turntable video or its exact ordered frames, evaluated as
  an optional overview channel.

The current Codex CLI observation path accepts images rather than a directly attached video, so the
first experiment should evaluate mode E through exact extracted frames with camera poses. Direct
video ingestion can be evaluated later through a separate adapter if it becomes available.

Use the same primary model, prompt contract, target policy, and decision schema across modes. Start
with `gpt-5.6-luna` for routine evaluation; a stronger model may be run only as a reference ceiling,
not as a replacement for a usable observation contract.

### Tasks scored for every mode

- count and identify meaningful components;
- distinguish separate, touching, intersecting, and contained relationships;
- identify surfaces or cavities accessible from the exterior;
- distinguish a legitimate exterior component from an internal support;
- choose `keep_separate`, `independently_watertight`, `boolean_union`, `bridge_required`,
  `remove_internal`, or `ask` under the supplied target policy;
- state uncertainty and request the next useful observation rather than inventing geometry.

### Measurements

- component-role accuracy;
- relationship and exterior-accessibility accuracy;
- engineering-strategy accuracy by target policy;
- high-confidence destructive false-positive count;
- uncertainty calibration and useful `ask` decisions;
- number and type of dynamic observations requested;
- wall time, model calls, input images or video frames, tokens, and estimated cost;
- reproducibility from stable camera and scene state.

### Deliverables

1. A versioned 3D observation contract covering scene state, camera state, render modes, stable IDs,
   image metadata, and dynamic actions.
2. A small ground-truthed perception benchmark with synthetic/public scenes plus a local-only YU7
   integration case.
3. An ablation report comparing modes A through E with exact prompts, observations, decisions, and
   errors.
4. A recommendation for the minimum default observation bundle and the conditions that trigger
   active viewport actions.
5. A list of observation operations that can be promoted into reusable Skill scripts.

### Exit criteria

- Every evaluation fact has deterministic ground truth and stable component IDs.
- Every model conclusion cites one or more `view_id` or structured relationship query IDs.
- The experiment establishes whether depth, normals, video, and active control add measurable value
  instead of assuming that they do.
- Relationship accuracy is at least 95% on controlled synthetic scenes and engineering-strategy
  accuracy is at least 90% across the supplied target-policy cases.
- The selected default bundle has no high-confidence destructive mistake on the controlled scenes.
- At least one intentionally ambiguous scene demonstrates a correct request for another view or a
  policy clarification.
- Active mode uses no more than eight supplemental images per case.
- No repair operator is implemented or invoked during this step.

## Current recommendation to validate

Use shaded RGB plus stable component grounding as the semantic baseline. Pair depth with RGB for
occlusion and separation, and treat normal maps as optional diagnostics until the ablation proves
their value. Begin with six orthographic and four three-quarter views, then let the agent request
precise additional screenshots through a controlled viewport. Keep turntable video as an optional
overview experiment; do not use it as the sole source for geometry-changing decisions.
