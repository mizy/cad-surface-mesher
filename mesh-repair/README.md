# Mesh Repair

Planned responsibility: repair target-specific surface meshes into watertight, inspectable artifacts suitable for downstream CAX/CAE preprocessing.

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
