from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from hybrid_proxy_geometry import (  # noqa: E402
    FACE_ORIGIN,
    conformal_loop_stitch,
    conformal_same_mesh_loop_stitch,
    extract_ordered_boundary_loops,
    triangulate_simple_boundary_loop,
)
from mesh_metrics import edge_topology, mesh_report  # noqa: E402


def source_square_annulus() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outer = np.asarray(
        [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    inner = np.asarray(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    points = np.vstack([outer, inner])
    faces = []
    for index in range(4):
        next_index = (index + 1) % 4
        faces.append([index, next_index, 4 + next_index])
        faces.append([index, 4 + next_index, 4 + index])
    # The inner boundary follows the clockwise, face-induced direction.
    return points, np.asarray(faces, dtype=np.int64), np.asarray([4, 7, 6, 5], dtype=np.int64)


def proxy_disk(
    count: int = 8,
    *,
    radius: float = 0.8,
    reverse_faces: bool = False,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    boundary = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros(count)]
    )
    points = np.vstack([boundary, np.zeros((1, 3))]) + np.asarray(offset, dtype=np.float64)
    faces = np.asarray(
        [[count, index, (index + 1) % count] for index in range(count)],
        dtype=np.int64,
    )
    if reverse_faces:
        faces = faces[:, [0, 2, 1]]
    return points, faces


def open_torus_segment(
    longitudinal_count: int = 20,
    ring_count: int = 8,
    gap_angle: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    major_angles = np.linspace(
        gap_angle * 0.5,
        2.0 * np.pi - gap_angle * 0.5,
        longitudinal_count,
    )
    ring_angles = np.linspace(0.0, 2.0 * np.pi, ring_count, endpoint=False)
    points = []
    for major_angle in major_angles:
        for ring_angle in ring_angles:
            radius = 4.0 + np.cos(ring_angle)
            points.append(
                [
                    radius * np.cos(major_angle),
                    radius * np.sin(major_angle),
                    np.sin(ring_angle),
                ]
            )
    faces = []
    for longitudinal_index in range(longitudinal_count - 1):
        for ring_index in range(ring_count):
            next_ring = (ring_index + 1) % ring_count
            lower_left = longitudinal_index * ring_count + ring_index
            upper_left = (longitudinal_index + 1) * ring_count + ring_index
            upper_right = (longitudinal_index + 1) * ring_count + next_ring
            lower_right = longitudinal_index * ring_count + next_ring
            faces.extend(
                [
                    [lower_left, upper_left, upper_right],
                    [lower_left, upper_right, lower_right],
                ]
            )
    return np.asarray(points, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def split_boundary_edge(
    points: np.ndarray,
    faces: np.ndarray,
    left: int,
    right: int,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    _, edge_faces = edge_topology(faces)
    face_id = edge_faces[tuple(sorted((left, right)))][0]
    face = faces[face_id].astype(int).tolist()
    midpoint_id = points.shape[0]
    split_point = points[left] * (1.0 - fraction) + points[right] * fraction
    for index in range(3):
        directed_left = face[index]
        directed_right = face[(index + 1) % 3]
        opposite = face[(index + 2) % 3]
        if {directed_left, directed_right} == {left, right}:
            replacement = np.asarray(
                [
                    [directed_left, midpoint_id, opposite],
                    [midpoint_id, directed_right, opposite],
                ],
                dtype=np.int64,
            )
            break
    else:  # pragma: no cover - guarded by edge topology lookup
        raise AssertionError("boundary edge not found in its incident triangle")
    return (
        np.vstack([points, split_point]),
        np.vstack([np.delete(faces, face_id, axis=0), replacement]),
    )


class OrderedBoundaryLoopTest(unittest.TestCase):
    def test_extracts_deterministic_face_oriented_loops(self) -> None:
        points, faces, inner_loop = source_square_annulus()

        first = extract_ordered_boundary_loops(points, faces)
        second = extract_ordered_boundary_loops(points, faces)

        self.assertTrue(first["success"])
        self.assertEqual(first["diagnostics"]["loop_count"], 2)
        self.assertEqual(first["diagnostics"]["boundary_edge_count"], 8)
        np.testing.assert_array_equal(first["loops"][0], np.asarray([0, 1, 2, 3]))
        np.testing.assert_array_equal(first["loops"][1], inner_loop)
        for left, right in zip(first["loops"], second["loops"]):
            np.testing.assert_array_equal(left, right)

    def test_rejects_branched_boundary_graph(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [0, 3, 4]], dtype=np.int64)

        result = extract_ordered_boundary_loops(points, faces)

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_reason_codes"], ["boundary_graph_not_simple"])
        self.assertEqual(result["diagnostics"]["stage"], "boundary_loop_extraction")
        self.assertEqual(result["diagnostics"]["invalid_vertex_degrees"][0], 4)

    def test_rejects_geometrically_self_intersecting_cycle(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

        result = extract_ordered_boundary_loops(points, faces)

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_reason_codes"], ["boundary_loop_self_intersection"])
        self.assertEqual(result["diagnostics"]["segment_pairs"], [[0, 2]])


class ConformalLoopStitchTest(unittest.TestCase):
    def test_splits_unequal_rings_and_builds_real_annular_bridge(self) -> None:
        source_points, source_faces, source_loop = source_square_annulus()
        proxy_points, proxy_faces = proxy_disk(8)
        proxy_loop = extract_ordered_boundary_loops(proxy_points, proxy_faces)["loops"][0]

        result = conformal_loop_stitch(
            source_points,
            source_faces,
            proxy_points,
            proxy_faces,
            source_loop=source_loop,
            proxy_loop=proxy_loop,
        )

        self.assertTrue(result["success"], result["diagnostics"])
        correspondence = result["diagnostics"]["correspondence"]
        validation = result["diagnostics"]["validation"]
        self.assertEqual(correspondence["method"], "normalized_arc_length_breakpoint_union")
        self.assertEqual(correspondence["resampling"]["target_ring_vertices"], 8)
        self.assertEqual(result["diagnostics"]["source_split"]["inserted_boundary_vertices"], 4)
        self.assertEqual(result["diagnostics"]["proxy_split"]["inserted_boundary_vertices"], 0)
        self.assertEqual(result["stitch_face_ids"].size, 16)
        self.assertTrue(validation["edge_incidence_two"])
        self.assertTrue(validation["opposite_directed_edge_pairs"])
        self.assertTrue(validation["face_origin_pairing_valid"])
        self.assertEqual(validation["degenerate_generated_faces"], [])
        self.assertEqual(
            np.count_nonzero(result["face_origin"] == FACE_ORIGIN["stitch_band"]),
            16,
        )

        topology, edge_faces = edge_topology(result["faces"])
        self.assertEqual(topology["non_manifold_edges"], 0)
        for ring in (result["source_ring_vertex_ids"], result["proxy_ring_vertex_ids"]):
            for index, left in enumerate(ring):
                edge = tuple(sorted((int(left), int(ring[(index + 1) % ring.size]))))
                self.assertEqual(len(edge_faces[edge]), 2)

    def test_orientation_pairing_flips_reversed_proxy_faces(self) -> None:
        source_points, source_faces, source_loop = source_square_annulus()
        proxy_points, proxy_faces = proxy_disk(8, reverse_faces=True)
        proxy_loop = extract_ordered_boundary_loops(proxy_points, proxy_faces)["loops"][0]

        result = conformal_loop_stitch(
            source_points,
            source_faces,
            proxy_points,
            proxy_faces,
            source_loop=source_loop,
            proxy_loop=proxy_loop,
        )

        self.assertTrue(result["success"], result["diagnostics"])
        orientation = result["diagnostics"]["correspondence"]["orientation_pairing"]
        self.assertTrue(orientation["proxy_faces_flipped"])
        self.assertFalse(orientation["proxy_traversal_reversed_from_induced"])
        self.assertGreater(orientation["normal_dot_mean"], 0.99)
        self.assertTrue(result["diagnostics"]["validation"]["opposite_directed_edge_pairs"])

    def test_output_is_bitwise_deterministic(self) -> None:
        source_points, source_faces, source_loop = source_square_annulus()
        proxy_points, proxy_faces = proxy_disk(7)
        proxy_loop = extract_ordered_boundary_loops(proxy_points, proxy_faces)["loops"][0]

        first = conformal_loop_stitch(
            source_points,
            source_faces,
            proxy_points,
            proxy_faces,
            source_loop=source_loop,
            proxy_loop=proxy_loop,
        )
        second = conformal_loop_stitch(
            source_points,
            source_faces,
            proxy_points,
            proxy_faces,
            source_loop=source_loop,
            proxy_loop=proxy_loop,
        )

        self.assertTrue(first["success"])
        self.assertTrue(second["success"])
        np.testing.assert_array_equal(first["points"], second["points"])
        np.testing.assert_array_equal(first["faces"], second["faces"])
        np.testing.assert_array_equal(first["face_origin"], second["face_origin"])
        self.assertEqual(first["diagnostics"], second["diagnostics"])

    def test_rejects_degenerate_zero_width_band_transactionally(self) -> None:
        source_points, source_faces, source_loop = source_square_annulus()
        proxy_points = np.asarray(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        proxy_faces = np.asarray([[4, index, (index + 1) % 4] for index in range(4)], dtype=np.int64)

        result = conformal_loop_stitch(
            source_points,
            source_faces,
            proxy_points,
            proxy_faces,
            source_loop=source_loop,
            proxy_loop=np.arange(4, dtype=np.int64),
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_reason_codes"], ["stitch_degenerate_faces_detected"])
        self.assertEqual(result["diagnostics"]["stage"], "stitch_validation")
        self.assertEqual(result["points"].shape, (0, 3))
        self.assertEqual(result["faces"].shape, (0, 3))
        self.assertGreater(
            len(result["diagnostics"]["validation"]["degenerate_generated_faces"]),
            0,
        )

    def test_rejects_proxy_outside_correspondence_gate(self) -> None:
        source_points, source_faces, source_loop = source_square_annulus()
        proxy_points, proxy_faces = proxy_disk(8, offset=(20.0, 0.0, 0.0))
        proxy_loop = extract_ordered_boundary_loops(proxy_points, proxy_faces)["loops"][0]

        result = conformal_loop_stitch(
            source_points,
            source_faces,
            proxy_points,
            proxy_faces,
            source_loop=source_loop,
            proxy_loop=proxy_loop,
            max_correspondence_distance=1.0,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_reason_codes"], ["loop_correspondence_distance_exceeded"])
        self.assertEqual(result["diagnostics"]["stage"], "arc_length_correspondence")
        self.assertGreater(result["diagnostics"]["distance_max"], 1.0)

    def test_requires_explicit_selection_when_mesh_has_multiple_boundaries(self) -> None:
        source_points, source_faces, _ = source_square_annulus()
        proxy_points, proxy_faces = proxy_disk(8)

        result = conformal_loop_stitch(
            source_points,
            source_faces,
            proxy_points,
            proxy_faces,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_reason_codes"], ["source_boundary_loop_count_mismatch"])
        self.assertEqual(result["diagnostics"]["loop_count"], 2)

    def test_same_component_loops_split_both_rings_and_close_conformally(self) -> None:
        points, faces = open_torus_segment()
        loops = extract_ordered_boundary_loops(points, faces)["loops"]
        points, faces = split_boundary_edge(
            points,
            faces,
            int(loops[0][0]),
            int(loops[0][1]),
            0.5,
        )
        loops = extract_ordered_boundary_loops(points, faces)["loops"]
        points, faces = split_boundary_edge(
            points,
            faces,
            int(loops[1][2]),
            int(loops[1][3]),
            0.33,
        )
        extraction = extract_ordered_boundary_loops(points, faces)

        result = conformal_same_mesh_loop_stitch(
            points,
            faces,
            source_loop=extraction["loops"][0],
            target_loop=extraction["loops"][1],
            max_correspondence_distance=3.0,
            normal_score_weight=0.0,
        )

        self.assertTrue(result["success"], result["diagnostics"])
        diagnostics = result["diagnostics"]
        self.assertEqual(
            diagnostics["method"],
            "same_mesh_paired_arc_length_edge_split_annular_bridge",
        )
        self.assertGreater(diagnostics["source_split"]["inserted_boundary_vertices"], 0)
        self.assertGreater(diagnostics["target_split"]["inserted_boundary_vertices"], 0)
        self.assertTrue(diagnostics["validation"]["edge_incidence_two"])
        self.assertTrue(diagnostics["validation"]["opposite_directed_edge_pairs"])
        self.assertTrue(diagnostics["validation"]["face_origin_pairing_valid"])
        self.assertEqual(diagnostics["validation"]["degenerate_generated_faces"], [])
        self.assertGreater(
            diagnostics["validation"]["adjacent_normal_dot_min"],
            diagnostics["validation"]["minimum_adjacent_normal_dot"],
        )
        source_mask = result["source_face_parent"] >= 0
        self.assertTrue(np.all(result["source_face_parent"][source_mask] < faces.shape[0]))
        self.assertTrue(np.all(result["source_face_parent"][~source_mask] == -1))
        self.assertEqual(int(np.count_nonzero(~source_mask)), result["stitch_face_ids"].size)
        topology, _ = edge_topology(result["faces"])
        self.assertEqual(topology["boundary_edges"], 0)
        self.assertEqual(topology["non_manifold_edges"], 0)


class ConstrainedHoleFillTest(unittest.TestCase):
    def test_fill_reuses_source_boundary_and_reduces_free_edges(self) -> None:
        points, faces, inner_loop = source_square_annulus()

        fill = triangulate_simple_boundary_loop(points, inner_loop)

        self.assertTrue(fill["success"], fill["diagnostics"])
        self.assertEqual(fill["faces"].shape[0], inner_loop.size - 2)
        topology, edge_faces = edge_topology(np.vstack([faces, fill["faces"]]))
        self.assertEqual(topology["boundary_edges"], 4)
        self.assertEqual(topology["non_manifold_edges"], 0)
        for index, left in enumerate(inner_loop):
            edge = tuple(sorted((int(left), int(inner_loop[(index + 1) % inner_loop.size]))))
            self.assertEqual(len(edge_faces[edge]), 2)

    def test_concave_loop_is_triangulated_deterministically(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [1.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=np.float64,
        )
        loop = np.arange(points.shape[0], dtype=np.int64)

        first = triangulate_simple_boundary_loop(points, loop)
        second = triangulate_simple_boundary_loop(points, loop)

        self.assertTrue(first["success"])
        np.testing.assert_array_equal(first["faces"], second["faces"])
        self.assertEqual(first["faces"].shape[0], 3)

    def test_fill_orients_an_arbitrarily_ordered_inventory_loop_from_source_faces(self) -> None:
        points, faces, inner_loop = source_square_annulus()

        fill = triangulate_simple_boundary_loop(
            points,
            inner_loop[::-1],
            source_faces=faces,
        )
        report = mesh_report(points, np.vstack([faces, fill["faces"]]))

        self.assertTrue(fill["success"], fill["diagnostics"])
        self.assertIsNotNone(fill["diagnostics"]["source_boundary_validation"])
        self.assertEqual(report["topology"]["inconsistent_winding_edges"], 0)


if __name__ == "__main__":
    unittest.main()
