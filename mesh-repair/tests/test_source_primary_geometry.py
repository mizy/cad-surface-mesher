from __future__ import annotations

import hashlib
import importlib
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from source_primary_boundary_inventory import (  # noqa: E402
    build_source_primary_boundary_inventory,
)
from source_primary_boundary_topology import canonical_cycle  # noqa: E402
from source_primary_curved_patch import build_curved_patch_candidate  # noqa: E402
from source_primary_loop_bridge import build_paired_loop_zipper_candidate  # noqa: E402
from source_primary_patch_contract import (  # noqa: E402
    PatchDelta,
    finalize_patch_candidate,
)
from source_primary_patch_geometry import (  # noqa: E402
    analyze_source_boundary,
    world_from_local,
)
from source_primary_patch_validation import validate_patch_candidate  # noqa: E402
from source_primary_planar_patch import (  # noqa: E402
    build_planar_patch_candidate,
    triangulate_constrained_polygon,
)
from source_primary_quality import audit_source_primary_patch  # noqa: E402
from source_primary_slit_patch import build_slit_patch_candidate  # noqa: E402
from source_primary_transaction import run_patch_transactions  # noqa: E402


def rectangular_annulus(
    inner_half_x: float = 1.0,
    inner_half_y: float = 1.0,
    *,
    outer_half_x: float = 3.0,
    outer_half_y: float = 2.5,
    z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [-outer_half_x, -outer_half_y, z],
            [outer_half_x, -outer_half_y, z],
            [outer_half_x, outer_half_y, z],
            [-outer_half_x, outer_half_y, z],
            [-inner_half_x, -inner_half_y, z],
            [inner_half_x, -inner_half_y, z],
            [inner_half_x, inner_half_y, z],
            [-inner_half_x, inner_half_y, z],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return points, faces, np.arange(4, 8, dtype=np.int64)


def frame_with_inner_island() -> tuple[np.ndarray, np.ndarray]:
    points, faces, _ = rectangular_annulus()
    island = np.asarray(
        [
            [-0.35, -0.30, 0.0],
            [0.35, -0.30, 0.0],
            [0.35, 0.30, 0.0],
            [-0.35, 0.30, 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.vstack([points, island]),
        np.vstack([faces, [[8, 9, 10], [8, 10, 11]]]).astype(np.int64),
    )


def circular_annulus(
    vertex_count: int,
    *,
    z: float,
    point_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, vertex_count, endpoint=False)
    outer = np.column_stack(
        [2.0 * np.cos(angles), 2.0 * np.sin(angles), np.full(vertex_count, z)]
    )
    inner = np.column_stack(
        [np.cos(angles), np.sin(angles), np.full(vertex_count, z)]
    )
    faces: list[list[int]] = []
    for index in range(vertex_count):
        following = (index + 1) % vertex_count
        faces.extend(
            [
                [index, following, vertex_count + following],
                [index, vertex_count + following, vertex_count + index],
            ]
        )
    return (
        np.vstack([outer, inner]),
        np.asarray(faces, dtype=np.int64) + point_offset,
        np.arange(vertex_count, 2 * vertex_count, dtype=np.int64) + point_offset,
    )


def unequal_circular_hole_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first_points, first_faces, first_loop = circular_annulus(5, z=0.0)
    second_count = 7
    second_angles = np.linspace(0.0, 2.0 * np.pi, second_count, endpoint=False)
    second_ring = np.column_stack(
        [
            0.7 * np.cos(second_angles),
            0.7 * np.sin(second_angles),
            np.zeros(second_count),
        ]
    )
    second_points = np.vstack([np.zeros((1, 3), dtype=np.float64), second_ring])
    second_faces = np.asarray(
        [
            [0, 1 + index, 1 + (index + 1) % second_count]
            for index in range(second_count)
        ],
        dtype=np.int64,
    )
    second_offset = first_points.shape[0]
    second_loop = np.arange(
        second_offset + 1,
        second_offset + 1 + second_count,
        dtype=np.int64,
    )
    return (
        np.vstack([first_points, second_points]),
        np.vstack([first_faces, second_faces + second_offset]),
        first_loop,
        second_loop,
    )


def cylinder_patch_with_rectangular_hole(
    *, radius: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    theta = np.linspace(-0.45, 0.45, 7)
    height = np.linspace(-1.0, 1.0, 7)
    points = np.asarray(
        [
            [radius * np.cos(angle), radius * np.sin(angle), z]
            for angle in theta
            for z in height
        ],
        dtype=np.float64,
    )

    def point_id(theta_id: int, height_id: int) -> int:
        return theta_id * height.size + height_id

    faces: list[list[int]] = []
    for theta_id in range(theta.size - 1):
        for height_id in range(height.size - 1):
            if theta_id in {2, 3} and height_id in {2, 3}:
                continue
            a = point_id(theta_id, height_id)
            b = point_id(theta_id + 1, height_id)
            c = point_id(theta_id + 1, height_id + 1)
            d = point_id(theta_id, height_id + 1)
            faces.extend(([a, b, c], [a, c, d]))
    loop = np.asarray(
        [
            point_id(2, 2),
            point_id(3, 2),
            point_id(4, 2),
            point_id(4, 3),
            point_id(4, 4),
            point_id(3, 4),
            point_id(2, 4),
            point_id(2, 3),
        ],
        dtype=np.int64,
    )
    return points, np.asarray(faces, dtype=np.int64), loop, radius


def annulus_with_crossing_triangle() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, faces, loop = rectangular_annulus()
    crossing = np.asarray(
        [
            [0.0, -0.75, -1.0],
            [0.0, 0.75, 1.0],
            [0.65, 0.0, 0.45],
        ],
        dtype=np.float64,
    )
    return np.vstack([points, crossing]), np.vstack([faces, [[8, 9, 10]]]), loop


def directed_edge_occurrences(
    faces: np.ndarray,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    result: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face in np.asarray(faces, dtype=np.int64):
        for index in range(3):
            left, right = int(face[index]), int(face[(index + 1) % 3])
            result.setdefault(tuple(sorted((left, right))), []).append((left, right))
    return result


def face_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(points, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    raw = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(raw, axis=1)
    return np.divide(
        raw,
        lengths[:, None],
        out=np.zeros_like(raw),
        where=lengths[:, None] > 0.0,
    )


def strictly_inside_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    inside = False
    x, y = (float(value) for value in point)
    for left, right in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        edge = right - left
        denominator = float(np.dot(edge, edge))
        parameter = float(
            np.clip(np.dot(point - left, edge) / max(denominator, 1e-30), 0.0, 1.0)
        )
        if np.linalg.norm(point - (left + parameter * edge)) <= 1e-12:
            return False
        if (left[1] > y) != (right[1] > y):
            crossing = float(
                left[0]
                + (y - left[1]) * (right[0] - left[0]) / (right[1] - left[1])
            )
            if crossing > x:
                inside = not inside
    return inside


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def hole_region(inventory: dict) -> dict:
    rows = [row for row in inventory["regions"] if row["classification"] == "single_loop_hole"]
    if len(rows) != 1:
        raise AssertionError(f"expected one single-loop hole, got {[row['classification'] for row in inventory['regions']]}")
    return rows[0]


class SourcePrimaryBoundaryInventoryTest(unittest.TestCase):
    def test_closed_boundary_loop_order_is_deterministic(self) -> None:
        points, faces, _ = rectangular_annulus()
        source_ids = np.asarray([41, 17, 73, 29, 101, 7, 89, 53], dtype=np.int64)
        external = np.asarray([0.0, 0.0, 1.0])

        first = build_source_primary_boundary_inventory(
            points,
            faces,
            source_ids,
            face_external_directions=external,
        )
        permutation = np.asarray([5, 2, 7, 0, 6, 1, 4, 3], dtype=np.int64)
        second = build_source_primary_boundary_inventory(
            points,
            faces[permutation],
            source_ids[permutation],
            face_external_directions=external,
        )

        first_loops = {
            row["loop_id"]: row["canonical_source_vertex_ids"] for row in first["loops"]
        }
        second_loops = {
            row["loop_id"]: row["canonical_source_vertex_ids"] for row in second["loops"]
        }
        self.assertEqual(first_loops, second_loops)
        self.assertEqual(first["summary"]["ordered_loop_count"], 2)
        self.assertEqual(first["summary"]["anomalous_boundary_graph_count"], 0)
        for row in first["loops"]:
            ordered = row["source_winding_vertex_ids"]
            listed_edges = {tuple(sorted(edge)) for edge in row["boundary_edges"]}
            traversed = {
                tuple(sorted((ordered[index], ordered[(index + 1) % len(ordered)])))
                for index in range(len(ordered))
            }
            self.assertEqual(traversed, listed_edges)
            self.assertEqual(row["patch_boundary_vertex_ids"], list(reversed(ordered)))
        values = [11, 4, 19, 7]
        expected = canonical_cycle(values)
        self.assertEqual(expected, canonical_cycle(values[2:] + values[:2]))
        self.assertEqual(expected, canonical_cycle(list(reversed(values))))

    def test_nested_loops_form_one_multi_loop_inner_island_region(self) -> None:
        points, faces = frame_with_inner_island()
        inventory = build_source_primary_boundary_inventory(
            points,
            faces,
            np.arange(faces.shape[0], dtype=np.int64),
            face_external_directions=np.asarray([0.0, 0.0, 1.0]),
        )

        regions = [
            row
            for row in inventory["regions"]
            if row["classification"] == "multi_loop_or_inner_island_hole"
        ]
        self.assertEqual(len(regions), 1, inventory["relationship_diagnostics"])
        region = regions[0]
        self.assertEqual(region["loop_count"], 2)
        self.assertEqual(region["recommended_operator"], "multi_loop_constrained_patch")
        self.assertTrue(region["patch_eligible"], region["blocking_reason_codes"])
        self.assertEqual(region["stable_center"]["method"], "outer_minus_inner_polygon_centroids")
        np.testing.assert_allclose(region["stable_center"]["value"], [0.0, 0.0, 0.0], atol=1e-15)
        nesting = [row for row in inventory["relationships"] if row["kind"] == "nested_loops"]
        self.assertEqual(len(nesting), 1)
        self.assertTrue(nesting[0]["accepted"])

    def test_multi_loop_patch_is_append_only_and_does_not_fill_inner_island(self) -> None:
        points, faces = frame_with_inner_island()
        source_ids = np.arange(faces.shape[0], dtype=np.int64)
        inventory = build_source_primary_boundary_inventory(
            points,
            faces,
            source_ids,
            face_external_directions=np.asarray([0.0, 0.0, 1.0]),
        )
        region = next(
            row
            for row in inventory["regions"]
            if row["classification"] == "multi_loop_or_inner_island_hole"
        )
        try:
            module = importlib.import_module("source_primary_multi_loop_patch")
            repairer = getattr(module, "build_multi_loop_patch_candidate")
        except (ModuleNotFoundError, AttributeError) as exc:
            self.fail(
                "recommended_operator=multi_loop_constrained_patch has no canonical "
                f"append-only repairer API: {exc}"
            )

        original_points = points.copy()
        original_faces = faces.copy()
        candidate = repairer(
            points,
            faces,
            source_ids,
            [np.asarray(loop, dtype=np.int64) for loop in region["ordered_boundary_loops"]],
            region_id=71,
        )
        self.assertEqual(candidate.status, "candidate", candidate.to_dict())
        self.assertNotIn("expected_face_normals", candidate.delta.face_provenance)
        np.testing.assert_array_equal(points, original_points)
        np.testing.assert_array_equal(faces, original_faces)
        self.assertEqual(len(candidate.boundary_mapping), 2)
        for mapping in candidate.boundary_mapping:
            self.assertEqual(mapping.source_vertex_ids, mapping.candidate_vertex_ids)

        loops = [np.asarray(loop, dtype=np.int64) for loop in region["ordered_boundary_loops"]]
        outer_loop = max(loops, key=lambda loop: np.ptp(points[loop, 0]) * np.ptp(points[loop, 1]))
        inner_loop = min(loops, key=lambda loop: np.ptp(points[loop, 0]) * np.ptp(points[loop, 1]))
        combined_points = np.vstack([points, candidate.delta.appended_points])
        centroids = combined_points[candidate.delta.appended_faces].mean(axis=1)[:, :2]
        outer_polygon = points[outer_loop, :2]
        island_polygon = points[inner_loop, :2]
        self.assertTrue(all(strictly_inside_polygon(value, outer_polygon) for value in centroids))
        self.assertFalse(any(strictly_inside_polygon(value, island_polygon) for value in centroids))

    def test_adjacent_normals_are_oriented_by_external_direction(self) -> None:
        points, faces, _ = rectangular_annulus()
        source_ids = np.arange(faces.shape[0], dtype=np.int64)
        positive = build_source_primary_boundary_inventory(
            points,
            faces,
            source_ids,
            face_external_directions=np.asarray([0.0, 0.0, 1.0]),
        )
        negative = build_source_primary_boundary_inventory(
            points,
            faces,
            source_ids,
            face_external_directions=np.asarray([0.0, 0.0, -1.0]),
        )

        positive_region = hole_region(positive)
        negative_region = hole_region(negative)
        np.testing.assert_allclose(
            positive_region["normals"]["area_weighted_source"]["normal"],
            [0.0, 0.0, 1.0],
            atol=1e-15,
        )
        self.assertEqual(positive_region["normals"]["oriented"]["status"], "aligned")
        self.assertEqual(
            negative_region["normals"]["oriented"]["status"],
            "flipped_descriptor_only",
        )
        for inventory, expected in ((positive, [0.0, 0.0, 1.0]), (negative, [0.0, 0.0, -1.0])):
            oriented = np.asarray(hole_region(inventory)["normals"]["oriented"]["normal"])
            self.assertGreater(float(np.dot(oriented, expected)), 1.0 - 1e-14)
            self.assertFalse(inventory["mutation_policy"]["source_points_mutated"])
            self.assertFalse(inventory["mutation_policy"]["source_faces_removed"])

    def test_incident_outside_evidence_must_be_complete_and_consistent(self) -> None:
        points, faces, _ = rectangular_annulus()
        source_ids = np.arange(faces.shape[0], dtype=np.int64)
        complete = np.repeat(
            np.asarray([[0.0, 0.0, 1.0]]), faces.shape[0], axis=0
        )
        reference = build_source_primary_boundary_inventory(
            points,
            faces,
            source_ids,
            face_external_directions=complete,
        )
        incident = np.asarray(hole_region(reference)["incident_face_ids"], dtype=np.int64)

        partial = np.zeros_like(complete)
        partial[incident[0]] = [0.0, 0.0, 1.0]
        partial_region = hole_region(
            build_source_primary_boundary_inventory(
                points,
                faces,
                source_ids,
                face_external_directions=partial,
            )
        )
        self.assertFalse(partial_region["patch_eligible"])
        self.assertIn(
            "external_direction_incomplete",
            partial_region["blocking_reason_codes"],
        )
        self.assertLess(
            partial_region["normals"]["external_direction"]["coverage"], 1.0
        )

        contradictory = complete.copy()
        contradictory[incident[-1]] = [0.0, 0.0, -1.0]
        contradictory_region = hole_region(
            build_source_primary_boundary_inventory(
                points,
                faces,
                source_ids,
                face_external_directions=contradictory,
            )
        )
        self.assertFalse(contradictory_region["patch_eligible"])
        self.assertIn(
            "external_direction_multivalued",
            contradictory_region["blocking_reason_codes"],
        )
        self.assertLess(
            contradictory_region["normals"]["external_direction"][
                "minimum_dot_to_resultant"
            ],
            0.0,
        )


class SourcePrimaryPatchGeometryTest(unittest.TestCase):
    def test_planar_patch_uses_local_plane_and_constrained_triangulation(self) -> None:
        points, faces, loop = rectangular_annulus()
        source_ids = np.arange(faces.shape[0], dtype=np.int64)
        candidate = build_planar_patch_candidate(
            points,
            faces,
            source_ids,
            loop[::-1],
            region_id=17,
        )

        self.assertEqual(candidate.status, "candidate", candidate.failure_reason_codes)
        self.assertEqual(candidate.method, "planar_cap")
        np.testing.assert_array_equal(candidate.boundary_mapping[0].source_vertex_ids, loop[::-1])
        np.testing.assert_allclose(candidate.delta.appended_points[:, 2], 0.0, atol=1e-15)
        self.assertTrue(
            all(
                strictly_inside_polygon(point[:2], points[loop, :2])
                for point in candidate.delta.appended_points
            )
        )

        concave = np.asarray(
            [[0.0, 0.0], [0.0, 2.0], [1.0, 2.0], [1.0, 1.0], [2.0, 1.0], [2.0, 0.0]],
            dtype=np.float64,
        )
        triangulation = triangulate_constrained_polygon(
            concave,
            np.arange(concave.shape[0], dtype=np.int64),
            source_point_count=concave.shape[0],
            steiner_rounds=1,
            max_appended_points=100,
            tolerance=1e-12,
        )
        self.assertTrue(triangulation["success"], triangulation["diagnostics"])
        all_uv = np.vstack([concave, triangulation["appended_uv"]])
        for point in triangulation["appended_uv"]:
            self.assertTrue(strictly_inside_polygon(point, concave))
        occurrences = directed_edge_occurrences(triangulation["faces"])
        boundary_edges = {
            tuple(sorted((index, (index + 1) % concave.shape[0])))
            for index in range(concave.shape[0])
        }
        for edge, rows in occurrences.items():
            self.assertEqual(len(rows), 1 if edge in boundary_edges else 2)
            if len(rows) == 2:
                self.assertEqual(rows[0], tuple(reversed(rows[1])))
        for face in triangulation["faces"]:
            triangle = all_uv[face]
            first = triangle[1] - triangle[0]
            second = triangle[2] - triangle[0]
            signed_double_area = float(first[0] * second[1] - first[1] * second[0])
            self.assertGreater(signed_double_area, 0.0)

    def test_cylindrical_hole_continues_source_normal_and_curvature(self) -> None:
        points, faces, loop, radius = cylinder_patch_with_rectangular_hole()
        source_ids = np.arange(faces.shape[0], dtype=np.int64)
        analysis = analyze_source_boundary(
            points,
            faces,
            source_ids,
            loop,
            region_id=23,
        )
        self.assertTrue(analysis["success"], analysis.get("diagnostics"))
        self.assertTrue(analysis["curvature"]["reliable"], analysis["curvature"])
        principal = np.sort(np.abs(analysis["curvature"]["principal_curvatures"]))
        self.assertLess(principal[0], 0.08)
        self.assertGreater(principal[1], 0.20)
        self.assertLess(principal[1], 0.45)

        candidate = build_curved_patch_candidate(
            points,
            faces,
            source_ids,
            loop[::-1],
            region_id=23,
        )
        self.assertEqual(candidate.status, "candidate", candidate.to_dict())
        boundary_center = points[loop].mean(axis=0)
        expected_outward = boundary_center.copy()
        expected_outward[2] = 0.0
        expected_outward /= np.linalg.norm(expected_outward)
        self.assertGreater(
            float(np.dot(candidate.normal["oriented_normal"], expected_outward)),
            0.98,
        )
        offsets = np.asarray(candidate.delta.point_provenance["normal_offset"])
        self.assertGreater(float(np.ptp(offsets)), 1e-5)
        candidate_radius = np.linalg.norm(candidate.delta.appended_points[:, :2], axis=1)
        flat_points = world_from_local(
            analysis["frame"],
            np.asarray(candidate.delta.point_provenance["uv"]),
            0.0,
        )
        flat_radius = np.linalg.norm(flat_points[:, :2], axis=1)
        self.assertLess(
            float(np.mean(np.abs(candidate_radius - radius))),
            float(np.mean(np.abs(flat_radius - radius))),
        )
        self.assertLess(float(np.max(np.abs(candidate_radius - radius))), 0.02)

        combined_points = np.vstack([points, candidate.delta.appended_points])
        patch_normals = face_normals(combined_points, candidate.delta.appended_faces)
        source_normals = face_normals(points, faces)
        patch_edges = directed_edge_occurrences(candidate.delta.appended_faces)
        mapping = candidate.boundary_mapping[0]
        dots = []
        for index, left in enumerate(mapping.source_vertex_ids):
            right = mapping.source_vertex_ids[(index + 1) % len(mapping.source_vertex_ids)]
            patch_face_id = next(
                face_id
                for face_id, face in enumerate(candidate.delta.appended_faces)
                if tuple(sorted((left, right)))
                in {
                    tuple(sorted((int(face[edge]), int(face[(edge + 1) % 3]))))
                    for edge in range(3)
                }
            )
            self.assertEqual(len(patch_edges[tuple(sorted((left, right)))]), 1)
            dots.append(
                float(
                    np.dot(
                        patch_normals[patch_face_id],
                        source_normals[mapping.source_edge_face_ids[index]],
                    )
                )
            )
        self.assertGreater(min(dots), 0.90)

        proxy_points = combined_points
        proxy_faces = candidate.delta.appended_faces.copy()
        with_proxy = build_curved_patch_candidate(
            points,
            faces,
            source_ids,
            loop,
            region_id=24,
            oriented_normal=np.asarray(candidate.normal["oriented_normal"]),
            closure_proxy_points=proxy_points,
            closure_proxy_faces=proxy_faces,
            closure_proxy_triangle_index=np.arange(
                proxy_faces.shape[0], dtype=np.int64
            ),
            closure_proxy_component_id=np.zeros(
                proxy_faces.shape[0], dtype=np.int64
            ),
        )
        self.assertEqual(with_proxy.status, "candidate", with_proxy.to_dict())
        self.assertTrue(with_proxy.proxy_provenance["used"])
        self.assertGreater(
            float(np.min(with_proxy.proxy_provenance["normal_dot"])), 0.0
        )
        transaction_run = run_patch_transactions(
            points,
            faces,
            source_ids,
            (with_proxy,),
            source_cell_data={
                "external_direction": np.repeat(
                    np.asarray(with_proxy.normal["oriented_normal"])[None, :],
                    faces.shape[0],
                    axis=0,
                )
            },
            fusion_region_id_by_region={"24": 24},
        )
        self.assertEqual(
            transaction_run["transactions"][0]["transaction_status"], "committed"
        )
        for incompatible_faces in (
            proxy_faces[:, [0, 2, 1]],
            np.vstack(
                [proxy_faces[:1, [0, 2, 1]], proxy_faces[1:]]
            ),
        ):
            rejected = build_curved_patch_candidate(
                points,
                faces,
                source_ids,
                loop,
                region_id=25,
                oriented_normal=np.asarray(candidate.normal["oriented_normal"]),
                closure_proxy_points=proxy_points,
                closure_proxy_faces=incompatible_faces,
                closure_proxy_triangle_index=np.arange(
                    proxy_faces.shape[0], dtype=np.int64
                ),
                closure_proxy_component_id=np.zeros(
                    proxy_faces.shape[0], dtype=np.int64
                ),
            )
            self.assertEqual(rejected.status, "rejected", rejected.to_dict())
            self.assertIn("proxy_winding_incompatible", rejected.failure_reason_codes)

    def test_unequal_reversed_loops_zip_deterministically(self) -> None:
        points, faces, first_loop, second_loop = unequal_circular_hole_pair()
        source_ids = np.arange(faces.shape[0], dtype=np.int64)
        first = build_paired_loop_zipper_candidate(
            points,
            faces,
            source_ids,
            first_loop,
            second_loop,
            first_region_id=31,
            second_region_id=32,
        )
        reversed_inputs = build_paired_loop_zipper_candidate(
            points,
            faces,
            source_ids,
            first_loop[::-1],
            second_loop[::-1],
            first_region_id=31,
            second_region_id=32,
        )

        self.assertEqual(first.status, "candidate", first.to_dict())
        self.assertEqual(reversed_inputs.status, "candidate", reversed_inputs.to_dict())
        self.assertEqual(first.delta.appended_points.shape, (0, 3))
        self.assertNotIn("expected_face_normals", first.delta.face_provenance)
        self.assertEqual(
            first.delta.appended_faces.shape[0],
            first_loop.size + second_loop.size,
        )
        np.testing.assert_array_equal(first.delta.appended_faces, reversed_inputs.delta.appended_faces)
        occurrences = directed_edge_occurrences(first.delta.appended_faces)
        source_occurrences = directed_edge_occurrences(faces)
        declared_boundary_edges: set[tuple[int, int]] = set()
        for mapping in first.boundary_mapping:
            self.assertEqual(mapping.source_vertex_ids, mapping.candidate_vertex_ids)
            for index, left in enumerate(mapping.source_vertex_ids):
                right = mapping.source_vertex_ids[(index + 1) % len(mapping.source_vertex_ids)]
                edge = tuple(sorted((left, right)))
                declared_boundary_edges.add(edge)
                self.assertEqual(len(occurrences[edge]), 1)
                self.assertEqual(occurrences[edge][0], tuple(reversed(source_occurrences[edge][0])))
        for edge, rows in occurrences.items():
            if edge not in declared_boundary_edges:
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0], tuple(reversed(rows[1])))

    def test_finite_slit_bridges_and_zero_width_requires_rejected_weld(self) -> None:
        # Immutable source boundary edges place a hard lower bound on patch
        # triangle aspect ratio.  This finite slit is narrow but still feasible
        # under the production <=25 quality gate without splitting an edge.
        points, faces, loop = rectangular_annulus(inner_half_x=0.3, inner_half_y=0.05)
        source_ids = np.arange(faces.shape[0], dtype=np.int64)
        bridge = build_slit_patch_candidate(
            points,
            faces,
            source_ids,
            loop[::-1],
            region_id=41,
        )
        self.assertEqual(bridge.status, "candidate", bridge.to_dict())
        self.assertEqual(bridge.method, "slit_bridge")
        self.assertNotIn("expected_face_normals", bridge.delta.face_provenance)
        self.assertGreater(bridge.delta.appended_points.shape[0], 0)
        used_source_ids = np.unique(
            bridge.delta.appended_faces[
                bridge.delta.appended_faces < points.shape[0]
            ]
        )
        np.testing.assert_array_equal(np.sort(used_source_ids), np.sort(loop))
        self.assertEqual(
            bridge.boundary_mapping[0].source_vertex_ids,
            bridge.boundary_mapping[0].candidate_vertex_ids,
        )

        zero_points, zero_faces, zero_loop = rectangular_annulus(
            inner_half_x=2.0,
            inner_half_y=0.0,
        )
        zero_width = build_slit_patch_candidate(
            zero_points,
            zero_faces,
            np.arange(zero_faces.shape[0], dtype=np.int64),
            zero_loop,
            region_id=42,
        )
        self.assertEqual(zero_width.status, "rejected")
        self.assertEqual(zero_width.method, "slit_weld")
        self.assertIn(
            "slit_weld_requires_source_connectivity_edit",
            zero_width.failure_reason_codes,
        )
        self.assertEqual(zero_width.delta.appended_faces.shape, (0, 3))

    def test_boundary_indices_and_patch_provenance_are_complete(self) -> None:
        points, faces, loop = rectangular_annulus()
        source_ids = np.arange(900, 900 + faces.shape[0], dtype=np.int64)
        original_points = points.copy()
        original_faces = faces.copy()
        original_source_ids = source_ids.copy()
        candidate = build_planar_patch_candidate(
            points,
            faces,
            source_ids,
            loop,
            region_id=51,
        )

        self.assertEqual(validate_patch_candidate(points, faces, source_ids, candidate), [])
        np.testing.assert_array_equal(points, original_points)
        np.testing.assert_array_equal(faces, original_faces)
        np.testing.assert_array_equal(source_ids, original_source_ids)
        mapping = candidate.boundary_mapping[0]
        self.assertEqual(mapping.source_vertex_ids, mapping.candidate_vertex_ids)
        self.assertEqual(set(mapping.source_vertex_ids), set(loop))
        self.assertTrue(np.all(candidate.delta.appended_faces >= 0))
        source_references = candidate.delta.appended_faces[
            candidate.delta.appended_faces < points.shape[0]
        ]
        self.assertEqual(set(source_references), set(loop))

        self.assertEqual(
            set(candidate.delta.point_provenance),
            {"patch_method", "region_id", "uv", "normal_offset", "placement"},
        )
        self.assertEqual(
            set(candidate.delta.face_provenance),
            {
                "patch_method",
                "region_id",
                "source_triangle_index",
                "source_geometry_consumed",
                "proxy_geometry_consumed",
            },
        )
        for values in candidate.delta.point_provenance.values():
            self.assertEqual(np.asarray(values).shape[0], candidate.delta.appended_points.shape[0])
        for values in candidate.delta.face_provenance.values():
            self.assertEqual(np.asarray(values).shape[0], candidate.delta.appended_faces.shape[0])
        np.testing.assert_array_equal(
            candidate.delta.face_provenance["source_triangle_index"],
            -np.ones(candidate.delta.appended_faces.shape[0], dtype=np.int64),
        )
        self.assertFalse(np.any(candidate.delta.face_provenance["source_geometry_consumed"]))
        self.assertFalse(np.any(candidate.delta.face_provenance["proxy_geometry_consumed"]))
        self.assertEqual(candidate.source_provenance["points_sha256"], array_sha256(points))
        self.assertEqual(candidate.source_provenance["faces_sha256"], array_sha256(faces))
        self.assertEqual(
            candidate.source_provenance["source_triangle_index_sha256"],
            array_sha256(source_ids),
        )
        self.assertFalse(candidate.proxy_provenance["used"])
        self.assertFalse(candidate.proxy_provenance["geometry_consumed"])


class SourcePrimaryCandidateRejectionTest(unittest.TestCase):
    def test_degenerate_and_flipped_candidates_are_rejected(self) -> None:
        points, faces, loop = rectangular_annulus()
        source_ids = np.arange(faces.shape[0], dtype=np.int64)
        valid = build_planar_patch_candidate(
            points,
            faces,
            source_ids,
            loop,
            region_id=61,
        )
        self.assertEqual(valid.status, "candidate", valid.to_dict())

        degenerate_faces = valid.delta.appended_faces.copy()
        degenerate_faces[0, 1] = degenerate_faces[0, 0]
        degenerate_delta = PatchDelta(
            valid.delta.appended_points,
            degenerate_faces,
            valid.delta.point_provenance,
            valid.delta.face_provenance,
        )
        degenerate = finalize_patch_candidate(
            points,
            faces,
            source_ids,
            method=valid.method,
            delta=degenerate_delta,
            boundary_mapping=valid.boundary_mapping,
            normal=valid.normal,
            curvature=valid.curvature,
            proxy_provenance=valid.proxy_provenance,
        )
        self.assertEqual(degenerate.status, "rejected")
        self.assertIn("patch_candidate_contract_invalid", degenerate.failure_reason_codes)
        self.assertTrue(
            any(
                "repeated point IDs" in message or "degenerate triangles" in message
                for message in degenerate.diagnostics["contract_validation_errors"]
            )
        )

        flipped_faces = valid.delta.appended_faces[:, [0, 2, 1]].copy()
        expected = np.repeat(
            np.asarray(valid.normal["oriented_normal"])[None, :],
            flipped_faces.shape[0],
            axis=0,
        )
        flipped = audit_source_primary_patch(
            points,
            faces,
            valid.delta.appended_points,
            flipped_faces,
            (np.asarray(valid.boundary_mapping[0].source_vertex_ids),),
            expected_face_normals=expected,
        )
        self.assertFalse(flipped["passed"])
        self.assertIn("patch_winding_flipped", flipped["reason_codes"])
        self.assertLess(
            flipped["gates"]["patch_orientation"]["actual"]["minimum_signed_dot"],
            0.0,
        )

    def test_non_adjacent_self_intersection_candidate_is_rejected(self) -> None:
        points, faces, loop = annulus_with_crossing_triangle()
        source_ids = np.arange(faces.shape[0], dtype=np.int64)
        candidate = build_planar_patch_candidate(
            points,
            faces,
            source_ids,
            loop,
            region_id=62,
        )
        self.assertEqual(candidate.status, "candidate", candidate.to_dict())
        expected = np.repeat(
            np.asarray(candidate.normal["oriented_normal"])[None, :],
            candidate.delta.appended_faces.shape[0],
            axis=0,
        )
        audit = audit_source_primary_patch(
            points,
            faces,
            candidate.delta.appended_points,
            candidate.delta.appended_faces,
            (np.asarray(candidate.boundary_mapping[0].source_vertex_ids),),
            expected_face_normals=expected,
        )

        self.assertFalse(audit["passed"])
        self.assertIn("patch_self_intersection_detected", audit["reason_codes"])
        intersection = audit["gates"]["patch_non_adjacent_intersection"]
        self.assertEqual(intersection["actual"]["status"], "computed")
        self.assertGreater(intersection["actual"]["intersection_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
