# Mesh Repair

Planned responsibility: repair vehicle surface meshes into watertight, inspectable artifacts suitable for downstream CAX/CAE preprocessing.

Initial scope:

- diagnose boundaries, non-manifold topology, duplicated vertices, degenerate faces, inverted normals, and disconnected shells
- propose a repair plan before changing geometry
- write repaired mesh artifacts and a JSON report
- keep units, coordinate convention, and provenance explicit

