from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv


def read_mesh(mesh_path: Path) -> pv.PolyData:
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    if mesh_path.suffix.lower() in {".glb", ".gltf"}:
        try:
            mesh = read_mesh_with_trimesh(mesh_path)
        except Exception as trimesh_error:
            try:
                mesh = pv.read(mesh_path)
            except Exception as pv_error:
                raise ValueError(
                    f"Could not read GLB/GLTF mesh with optional trimesh fallback ({trimesh_error}) "
                    f"or PyVista ({pv_error})"
                ) from pv_error
    else:
        try:
            mesh = pv.read(mesh_path)
        except Exception as pv_error:
            try:
                mesh = read_mesh_with_trimesh(mesh_path)
            except Exception as trimesh_error:
                raise ValueError(
                    f"Could not read mesh with PyVista ({pv_error}) or optional trimesh fallback ({trimesh_error})"
                ) from pv_error

    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    mesh = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError(f"Mesh has no usable surface triangles: {mesh_path}")
    return mesh


def read_mesh_with_trimesh(mesh_path: Path) -> pv.PolyData:
    try:
        import trimesh  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Install trimesh for this fallback reader") from exc

    loaded = trimesh.load(mesh_path, force="scene", process=False)
    if hasattr(loaded, "geometry"):
        geometries = [geom for geom in loaded.dump(concatenate=False) if getattr(geom, "faces", None) is not None]
        if not geometries:
            raise ValueError("trimesh scene contains no triangle geometries")
        loaded = trimesh.util.concatenate(geometries)
    if getattr(loaded, "vertices", None) is None or getattr(loaded, "faces", None) is None:
        raise ValueError("trimesh reader did not return vertices and faces")

    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    packed_faces = np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces)).reshape(-1)
    return pv.PolyData(vertices, packed_faces)
