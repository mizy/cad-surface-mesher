from __future__ import annotations

from pathlib import Path
from typing import Any


def write_vtp(
    pv: Any,
    np: Any,
    path: Path,
    points: Any,
    triangles: Any,
    cell_data: dict[str, Any],
    point_data: dict[str, Any] | None = None,
) -> None:
    faces = np.column_stack((np.full((triangles.shape[0], 1), 3, dtype=np.int64), triangles)).ravel()
    mesh = pv.PolyData(points, faces)
    for name, values in cell_data.items():
        mesh.cell_data[name] = values
    for name, values in (point_data or {}).items():
        mesh.point_data[name] = values
    mesh.save(path)
