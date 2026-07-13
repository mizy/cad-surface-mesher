from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mesh_io import read_surface, triangle_faces
from solid_triangle_raster import rasterize_solid_views


def render_registered_observation_bundle(
    meshes: dict[str, Path],
    output_dir: Path,
    *,
    grid_size: int,
) -> dict[str, Any]:
    """Render solid z-buffer depth, face-ID, and discontinuity evidence."""
    if grid_size < 32:
        raise ValueError("observation grid_size must be at least 32")
    output_dir.mkdir(parents=True, exist_ok=True)
    views: list[dict[str, Any]] = []
    mesh_rows = []
    for role, path in meshes.items():
        surface = read_surface(path)
        points = np.asarray(surface.points, dtype=np.float64)
        faces = triangle_faces(surface)
        evidence = rasterize_solid_views(
            points,
            faces,
            grid_size=grid_size,
            depth_tolerance=0.0,
            collect_face_evidence=True,
        )
        role_rows = []
        for raster in evidence.views:
            depth_path = output_dir / f"{role}_{raster.view.name}_depth.png"
            face_id_path = output_dir / f"{role}_{raster.view.name}_face_id.png"
            edge_path = output_dir / f"{role}_{raster.view.name}_discontinuity.png"
            Image.fromarray(depth_image(raster.depth, raster.silhouette)).save(depth_path)
            Image.fromarray(face_id_image(raster.first_hit_face)).save(face_id_path)
            Image.fromarray(discontinuity_image(raster.first_hit_face)).save(edge_path)
            for mode, image_path in (
                ("depth", depth_path),
                ("face_id", face_id_path),
                ("face_id_discontinuity", edge_path),
            ):
                row = {
                    "view_id": f"{role}_{raster.view.name}_{mode}",
                    "mesh_role": role,
                    "camera": raster.view.name,
                    "render_mode": mode,
                    "path": str(image_path.resolve()),
                    "registered_projection": True,
                }
                views.append(row)
                role_rows.append(row)
        mesh_rows.append(
            {
                "role": role,
                "path": str(path),
                "points": int(points.shape[0]),
                "triangles": int(faces.shape[0]),
                "raster_report": evidence.report,
                "views": role_rows,
            }
        )
    manifest = {
        "schema": "mesh_repair_registered_observations/v1",
        "method": "conservative_solid_triangle_zbuffer_registered_by_mesh_role",
        "meshes": mesh_rows,
        "views": views,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def depth_image(depth: np.ndarray, silhouette: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float64)
    mask = np.asarray(silhouette, dtype=bool)
    image = np.full((*values.shape, 3), 255, dtype=np.uint8)
    if not np.any(mask):
        return image
    finite = values[mask]
    minimum = float(np.min(finite))
    span = max(float(np.max(finite) - minimum), 1e-12)
    shade = 235 - np.rint((finite - minimum) / span * 190.0).astype(np.uint8)
    image[mask] = np.column_stack((shade, shade, shade))
    return image


def face_id_image(first_hit_face: np.ndarray) -> np.ndarray:
    ids = np.asarray(first_hit_face, dtype=np.int64)
    image = np.full((*ids.shape, 3), 255, dtype=np.uint8)
    mask = ids >= 0
    values = ids[mask].astype(np.uint64)
    image[mask] = np.column_stack(
        (
            ((values * 73 + 41) % 251 + 2).astype(np.uint8),
            ((values * 151 + 67) % 251 + 2).astype(np.uint8),
            ((values * 199 + 101) % 251 + 2).astype(np.uint8),
        )
    )
    return image


def discontinuity_image(first_hit_face: np.ndarray) -> np.ndarray:
    ids = np.asarray(first_hit_face, dtype=np.int64)
    occupied = ids >= 0
    discontinuity = np.zeros(ids.shape, dtype=bool)
    for axis in (0, 1):
        left = [slice(None), slice(None)]
        right = [slice(None), slice(None)]
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        lhs = ids[tuple(left)]
        rhs = ids[tuple(right)]
        changed = lhs != rhs
        discontinuity[tuple(left)] |= changed
        discontinuity[tuple(right)] |= changed
    image = np.full((*ids.shape, 3), 255, dtype=np.uint8)
    image[occupied] = (210, 218, 224)
    image[discontinuity] = (220, 38, 38)
    return image
