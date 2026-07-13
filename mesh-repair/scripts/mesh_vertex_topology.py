from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def count_non_manifold_vertices(
    faces: np.ndarray,
    edge_to_faces: Mapping[tuple[int, int], Sequence[int]],
) -> int:
    """Count vertices whose incident-face link is not one cycle or path."""

    return int(non_manifold_vertex_ids(faces, edge_to_faces).size)


def non_manifold_vertex_ids(
    faces: np.ndarray,
    edge_to_faces: Mapping[tuple[int, int], Sequence[int]],
) -> np.ndarray:
    """Return vertices whose incident-face link is not one cycle or path."""

    links: dict[int, dict[int, set[int]]] = {}
    repeated_link_edges: set[int] = set()
    for face in faces:
        a, b, c = (int(value) for value in face)
        for center, left, right in ((a, b, c), (b, c, a), (c, a, b)):
            graph = links.setdefault(center, {})
            left_neighbors = graph.setdefault(left, set())
            if right in left_neighbors:
                repeated_link_edges.add(center)
            left_neighbors.add(right)
            graph.setdefault(right, set()).add(left)

    boundary_counts: dict[int, int] = {}
    invalid_edge_vertices: set[int] = set()
    for (left, right), face_ids in edge_to_faces.items():
        if len(face_ids) == 1:
            boundary_counts[left] = boundary_counts.get(left, 0) + 1
            boundary_counts[right] = boundary_counts.get(right, 0) + 1
        elif len(face_ids) != 2:
            invalid_edge_vertices.update((left, right))

    invalid: list[int] = []
    for vertex, graph in links.items():
        start = next(iter(graph))
        reached = {start}
        pending = [start]
        while pending:
            for neighbor in graph[pending.pop()]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        degrees = [len(neighbors) for neighbors in graph.values()]
        boundary_count = boundary_counts.get(vertex, 0)
        cycle = boundary_count == 0 and all(degree == 2 for degree in degrees)
        path = (
            boundary_count == 2
            and degrees.count(1) == 2
            and all(degree in (1, 2) for degree in degrees)
        )
        if (
            vertex in repeated_link_edges
            or vertex in invalid_edge_vertices
            or len(reached) != len(graph)
            or not (cycle or path)
        ):
            invalid.append(vertex)
    return np.asarray(sorted(invalid), dtype=np.int64)
