# Mesh Repair

Planned responsibility: repair vehicle surface meshes into watertight, inspectable artifacts suitable for downstream CAX/CAE preprocessing.

Initial scope:

- use a two-stage watertight flow: extract exterior wall surfaces first, then remesh/seal only that exterior candidate set
- diagnose boundaries, non-manifold topology, duplicated vertices, degenerate faces, inverted normals, and disconnected shells
- propose a repair plan before changing geometry
- write repaired mesh artifacts and a JSON report
- keep units, coordinate convention, and provenance explicit

Do not apply watertight remeshing to a full dirty vehicle assembly before removing interior, hidden, duplicate, or irrelevant components. For external CFD/CAE skins, the intermediate exterior wall candidate may be non-watertight; water-tightness is introduced only after the target wall set is known.
