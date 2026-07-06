# Roadmap

## Predict Vehicle Cd

- Keep the current shape-similarity predictor standalone.
- Add regression fixtures for known public/reference meshes.
- Add a small report command that packages normalized mesh, side depth, top matches, and confidence into one folder.

## Mesh Repair

- Detect open boundaries, non-manifold edges, duplicated vertices, zero-area faces, self intersections, inverted normals, and disconnected shells.
- Produce a repair plan before mutating geometry.
- Repair common surface issues with measurable before/after diagnostics.
- Preserve original units, orientation, named surfaces, and user-visible provenance when possible.
- Export repaired watertight meshes plus a JSON repair report.

## CAD Tessellation

- Import CAD files through a local OpenCascade-compatible stack.
- Tessellate with explicit chord tolerance, angular tolerance, min/max edge length, and unit handling.
- Preserve body/face names and source identifiers when available.
- Emit mesh artifacts that downstream repair and Cd prediction can consume.

