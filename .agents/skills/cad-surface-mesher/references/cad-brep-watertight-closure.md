# CAD/B-Rep Watertight Closure

Use this route for `cad -> watertight_cad`. Keep the analytic CAD faces as geometry truth and close topology before tessellation.

## Route decision

Prefer CAD-domain closure when all of the following hold:

- STEP/BREP import produces mostly valid faces or sewable shells.
- Physical openings are local and can be classified geometrically.
- The requested output is closed STEP/BREP or a high-fidelity mesh derived from it.
- Global wrapping would erase features or move the exterior surface beyond policy limits.

Use mesh-domain wrapping only when CAD healing cannot produce trustworthy shells, the source is already a mesh, or the user explicitly accepts reconstruction loss.

## Audit physical openings

1. Import with OpenCASCADE/FreeCAD and remove non-target farfield or fixture surfaces before sewing.
2. Sew with a conservative tolerance. Sweep several tolerances when necessary and compare shell count, free-edge count, area, and bounding box; identical results across the sweep are evidence that closure did not depend on aggressive welding.
3. Count edge owners on the candidate shape.
4. Classify a free edge as physical only when it has one owning face and `edge.Degenerated == false`.
5. Keep OCCT degenerated parameter edges. Do not fill or delete them as holes merely because they have one owning face.
6. Group physical free edges with `Part.sortEdges`, construct wires, and require each repair candidate wire to be closed and valid.
7. Record loop perimeter, bounding box, planarity/flatness, neighboring faces, and intended design meaning before choosing a patch.

Do not equate `free_edge_count` with physical openings. Analytic surfaces legitimately contain zero-length or near-zero-length degenerated seams.

## Apply local CAD repairs

Use the smallest operator that matches the opening:

- Closed valid shell: construct a Solid directly; do not rebuild its surfaces.
- Planar truncation or contact opening: construct `Part.Face(wire)`, combine it with the owning shell, run `sewShape()`, require exactly one closed shell, then call `Part.makeSolid`.
- Small compatible CAD gap: sew or bridge at a measured tolerance bounded by policy.
- Missing curved surface: build a local surface using surrounding continuity evidence and report approximation error.
- Ambiguous functional opening: stop for policy instead of treating it as corruption.

After solid construction, reverse orientation only when volume is negative. Require every Solid to be valid, closed, and positively oriented.

Avoid whole-model voxelization, implicit reconstruction, or inflated sewing tolerance when a few local loops explain the remaining physical openings.

## Export STEP defensively

A valid native BREP can become invalid after default STEP export. OpenCASCADE shape processing may split a small trimmed B-spline face, producing a self-intersecting wire or `Unorientable shape` on readback.

Set units and write precision explicitly, then validate the serialized file:

```python
Part.exportUnits("MM")
Part.setStaticValue("write.precision.mode", 2)
Part.setStaticValue("write.precision.val", 0.05)
Part.setStaticValue("write.surfacecurve.mode", 1)
shape.exportStep(output_step)
```

Treat `0.05 mm` as a proven DrivAer value, not a universal constant. Probe the smallest value that preserves topology and acceptable geometry for the current model. Record the emitted STEP `UNCERTAINTY_MEASURE_WITH_UNIT` and the configured precision in the report.

Do not assume that an empty `write.step.sequence` disables processing; FreeCAD/OCCT may restore defaults. A custom resource or sequence setting is effective only when entity counts and readback topology demonstrate the change.

## Require serialized round-trip validation

Validate three states independently:

1. In-memory repaired shape.
2. Native BREP re-import.
3. STEP re-import.

Require all applicable gates:

- `shape.isValid() == true`
- `shape.isClosed() == true`
- expected Solid and shell counts
- every Solid valid, closed, and positive-volume
- physical free edges = 0
- non-manifold edges = 0
- `shape.check(False)` has no errors
- face, shell, and Solid counts preserved across serialization
- units and bounding box preserved
- area and volume drift within declared tolerances
- STEP contains closed solid representation and no unintended `OPEN_SHELL`

Count degenerated free edges separately and report them as parameter seams. Never fail physical watertightness solely because their count is non-zero.

Use the bundled validator under `FreeCADCmd`:

```bash
FreeCADCmd -c "exec(open('scripts/freecad_validate_watertight_cad.py').read())" \
  --pass output.step \
  --pass --reference --pass output.brep \
  --pass --expected-solids --pass 5 \
  --pass --max-bbox-drift-mm --pass 0.001 \
  --pass --max-relative-measure-drift --pass 5e-7 \
  --pass --report --pass output.validation.json
```

FreeCAD 1.0 forwards one argument per `--pass`, so repeat it for every option and value. If another FreeCAD version differs, inspect `FreeCADCmd --help` and preserve the same script arguments. Do not skip validation because of launcher syntax.

## DrivAer case evidence

The DrivAer fastback case established these reusable lessons:

- Native sewing reduced 45 vehicle shells to 5 without changing the vehicle bounding box.
- Of 189 apparent free edges, 130 were degenerated parameter edges rather than physical cracks.
- The remaining physical edges formed four closed, nearly planar wheel-contact loops.
- Adding four planar caps converted the result to 5 valid closed Solids while preserving all original exterior faces.
- Default STEP export made the main body invalid by splitting a small B-spline face.
- A user-defined `0.05 mm` STEP precision preserved 3,533 faces, 5 closed shells, and 5 `MANIFOLD_SOLID_BREP` entities on readback.
- The accepted STEP had zero physical free and non-manifold edges; relative area and volume drift were `3.960e-7` and `1.055e-7`.

Generalize the classification and validation rules. Do not hard-code the DrivAer shell count, cap count, or precision for unrelated models.
