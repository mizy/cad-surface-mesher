# CAD Tessellation

Planned responsibility: embed CAD import and tessellation into the personal vehicle CAX workflow.

Initial scope:

- read CAD sources through a local OpenCascade-compatible toolchain
- control chord tolerance, angular tolerance, edge length, and unit conversion explicitly
- preserve body and face identities when the source file provides them
- emit surface meshes that can feed mesh repair and Cd prediction

