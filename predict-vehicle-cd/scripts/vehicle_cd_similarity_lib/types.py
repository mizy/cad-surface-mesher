from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyvista as pv


@dataclass(frozen=True)
class SideArtifacts:
    mask: np.ndarray
    depth: np.ndarray
    normal: np.ndarray


@dataclass(frozen=True)
class NormalizedVehicle:
    mesh: pv.PolyData
    info: dict[str, Any]
