from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from hybrid_proxy_fusion import (  # noqa: E402
    FusionThresholds,
    build_fusion_comparisons,
    effective_region_operator,
    run_hybrid_proxy_fusion,
    topology_counts,
    transaction_topology_counts,
)
from hybrid_proxy_geometry import (  # noqa: E402
    FACE_ORIGIN,
    edge_reason,
    extract_ordered_boundary_loops,
    face_centroids_normals,
    local_scale,
)
from hybrid_proxy_regions import build_proxy_face_index, proxy_trust_report  # noqa: E402
from mesh_io import read_surface  # noqa: E402
from mesh_metrics import mesh_report  # noqa: E402
from mesh_metrics import edge_topology  # noqa: E402


def source_plate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    source_indices = np.asarray([10, 11], dtype=np.int64)
    return points, faces, source_indices


def seam_touching_proxy_patch() -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.08, 0.0, 0.0],
            [1.0, 0.08, 0.0],
            [4.0, 4.0, 0.0],
            [4.1, 4.0, 0.0],
            [4.0, 4.1, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    return points, faces


def far_proxy_patch() -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [3.0, 3.0, 0.0],
            [3.1, 3.0, 0.0],
            [3.0, 3.1, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    return points, faces


def orthogonal_proxy_patch() -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.08, 0.0],
            [1.0, 0.0, 0.08],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    return points, faces


def open_cube_with_top_hole(
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    ) + np.asarray(offset, dtype=np.float64)
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return points, faces, np.asarray([4, 7, 6, 5], dtype=np.int64)


def open_torus_source(
    longitudinal_count: int = 20,
    ring_count: int = 8,
    gap_angle: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    triangle_faces = np.asarray(faces, dtype=np.int64)
    return (
        np.asarray(points, dtype=np.float64),
        triangle_faces,
        np.arange(triangle_faces.shape[0], dtype=np.int64) + 1000,
    )


def coincident_open_tube_halves(
    ring_count: int = 12,
    seam_gap: float = 0.005,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, ring_count, endpoint=False)
    ring_xy = np.column_stack([np.cos(angles), np.sin(angles)])

    def half(z_min: float, z_max: float) -> tuple[np.ndarray, np.ndarray]:
        half_points = np.vstack(
            [
                np.column_stack([ring_xy, np.full(ring_count, z_min)]),
                np.column_stack([ring_xy, np.full(ring_count, z_max)]),
            ]
        )
        half_faces = []
        for index in range(ring_count):
            following = (index + 1) % ring_count
            half_faces.extend(
                [
                    [index, following, ring_count + following],
                    [index, ring_count + following, ring_count + index],
                ]
            )
        return half_points, np.asarray(half_faces, dtype=np.int64)

    lower_points, lower_faces = half(-1.0, 0.0)
    upper_points, upper_faces = half(seam_gap, 1.0 + seam_gap)
    point_offset = lower_points.shape[0]
    points = np.vstack([lower_points, upper_points])
    faces = np.vstack([lower_faces, upper_faces + point_offset])
    lower_loop = np.arange(ring_count, 2 * ring_count, dtype=np.int64)
    upper_loop = np.arange(point_offset, point_offset + ring_count, dtype=np.int64)
    source_indices = np.arange(faces.shape[0], dtype=np.int64) + 2000
    return points, faces, source_indices, lower_loop, upper_loop


def paired_loop_inventory_item(
    item_id: str,
    paired_id: str,
    pair_id: int,
    loop: np.ndarray,
    points: np.ndarray,
    faces: np.ndarray,
) -> dict:
    _, edge_faces = edge_topology(faces)
    incident_faces = sorted(
        {
            edge_faces[tuple(sorted((int(left), int(loop[(index + 1) % loop.size]))))][0]
            for index, left in enumerate(loop)
        }
    )
    loop_points = points[loop]
    lengths = np.linalg.norm(loop_points - np.roll(loop_points, -1, axis=0), axis=1)
    return {
        "id": item_id,
        "type": "boundary_loop",
        "edge_count": int(loop.size),
        "length": float(lengths.sum()),
        "bbox": {
            "min": loop_points.min(axis=0).tolist(),
            "max": loop_points.max(axis=0).tolist(),
        },
        "face_ids": incident_faces,
        "source_triangle_ids": incident_faces,
        "edge_vertex_ids": loop.astype(int).tolist(),
        "ordered_vertex_ids": loop.astype(int).tolist(),
        "paired_region_id": paired_id,
        "pair_id": pair_id,
        "requires_policy": False,
        "classification": "near_coincident_part_seam",
        "patch_eligible": True,
        "operator": "loop_pair_zipper",
    }


def inset_square_proxy(
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [[-0.6, -0.6, 1.0], [0.6, -0.6, 1.0], [0.6, 0.6, 1.0], [-0.6, 0.6, 1.0]],
        dtype=np.float64,
    ) + np.asarray(offset, dtype=np.float64)
    return points, np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


def closed_grid_proxy_for_top_hole(count: int = 13) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.linspace(-1.2, 1.2, count)
    top = np.asarray([[x, y, 1.0] for y in coordinates for x in coordinates], dtype=np.float64)
    bottom = top.copy()
    bottom[:, 2] = -1.0
    points = np.vstack([top, bottom])
    bottom_offset = top.shape[0]

    def point_id(x_index: int, y_index: int) -> int:
        return y_index * count + x_index

    faces = []
    for y_index in range(count - 1):
        for x_index in range(count - 1):
            a = point_id(x_index, y_index)
            b = point_id(x_index + 1, y_index)
            c = point_id(x_index + 1, y_index + 1)
            d = point_id(x_index, y_index + 1)
            faces.extend([[a, b, c], [a, c, d]])
            faces.extend(
                [
                    [bottom_offset + a, bottom_offset + c, bottom_offset + b],
                    [bottom_offset + a, bottom_offset + d, bottom_offset + c],
                ]
            )
    perimeter = (
        [point_id(index, 0) for index in range(count)]
        + [point_id(count - 1, index) for index in range(1, count)]
        + [point_id(index, count - 1) for index in range(count - 2, -1, -1)]
        + [point_id(0, index) for index in range(count - 2, 0, -1)]
    )
    for index, top_left in enumerate(perimeter):
        top_right = perimeter[(index + 1) % len(perimeter)]
        bottom_left = bottom_offset + top_left
        bottom_right = bottom_offset + top_right
        faces.extend([[top_right, top_left, bottom_left], [top_right, bottom_left, bottom_right]])
    return points, np.asarray(faces, dtype=np.int64)


def conformal_hole_inventory(
    source_loop: np.ndarray,
    *,
    face_offset: int = 0,
    vertex_offset: int = 0,
    region_index: int = 1,
    x_offset: float = 0.0,
    operator: str = "proxy_conformal_patch_after_cap_decision",
) -> dict:
    face_ids = list(range(face_offset + 2, face_offset + 10))
    return {
        "id": f"boundary_loop_{region_index:04d}",
        "type": "boundary_loop",
        "edge_count": 4,
        "length": 8.0,
        "bbox": {"min": [x_offset - 1.1, -1.1, 0.9], "max": [x_offset + 1.1, 1.1, 1.1]},
        "face_ids": face_ids,
        "source_triangle_ids": face_ids,
        "edge_vertex_ids": (source_loop + vertex_offset).astype(int).tolist(),
        "ordered_vertex_ids": (source_loop + vertex_offset).astype(int).tolist(),
        "requires_policy": False,
        "classification": "small_exterior_hole",
        "patch_eligible": True,
        "operator": operator,
        "blocking": True,
    }


def patch_inventory() -> dict:
    return {
        "boundary_regions": {
            "items": [
                {
                    "id": "boundary_loop_0001",
                    "type": "boundary_loop",
                    "edge_count": 4,
                    "length": 0.4,
                    "bbox": {"min": [0.0, 0.0, 0.0], "max": [4.1, 4.1, 0.0]},
                    "source_triangle_ids": [10, 11],
                    "requires_policy": False,
                    "classification": "patch_required",
                    "detector_reason": "topology_boundary_defect",
                    "blocking": True,
                }
            ]
        },
        "gap_regions": {"items": []},
        "non_manifold_regions": {"items": []},
    }


def patch_inventory_with_z_belt() -> dict:
    inventory = patch_inventory()
    inventory["boundary_regions"]["items"][0]["bbox"] = {
        "min": [0.0, 0.0, 0.0],
        "max": [4.1, 4.1, 0.1],
    }
    return inventory


def preserve_opening_inventory() -> tuple[dict, dict, list[dict]]:
    inventory = {
        "boundary_regions": {"items": []},
        "non_manifold_regions": {"items": []},
        "gap_regions": {
            "items": [
                {
                    "id": "gap_or_opening_candidate_0003",
                    "type": "gap_or_opening_candidate",
                    "edge_count": 4,
                    "length": 4.0,
                    "bbox": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]},
                    "source_triangle_ids": [10, 11],
                    "requires_policy": True,
                    "classification": "pending_policy",
                    "detector_reason": "semantic_opening_candidate",
                    "blocking": False,
                }
            ]
        },
    }
    packet = {
        "items": [
            {
                "id": "opening_0003",
                "source_region": "gap_or_opening_candidate_0003",
            }
        ]
    }
    decisions = [{"item_id": "opening_0003", "status": "decided", "decision": "preserve"}]
    return inventory, packet, decisions


class HybridProxyFusionTest(unittest.TestCase):
    def test_policy_shape_prior_selects_deterministic_geometry_operator(self) -> None:
        base = {
            "operator": "proxy_conformal_patch_after_cap_decision",
            "classification": "patch_required",
            "policy_decision": {"status": "decided", "decision": "cap"},
        }

        self.assertEqual(
            effective_region_operator(
                {**base, "policy_decision": {**base["policy_decision"], "shape_prior": "planar"}}
            ),
            "constrained_loop_triangulation",
        )
        self.assertEqual(
            effective_region_operator(
                {**base, "policy_decision": {**base["policy_decision"], "shape_prior": "voxel_sdf"}}
            ),
            "source_locked_proxy_patch",
        )
        self.assertEqual(
            effective_region_operator(
                {
                    **base,
                    "classification": "pending_policy",
                    "policy_decision": {"status": "pending_review", "decision": None},
                }
            ),
            "proxy_conformal_patch_after_cap_decision",
        )
        self.assertEqual(
            effective_region_operator(
                {
                    **base,
                    "classification": "reject_region",
                    "policy_decision": {"status": "decided", "decision": "reject"},
                }
            ),
            "policy_reject",
        )

    def test_lightweight_transaction_counts_match_full_mesh_report(self) -> None:
        points, faces, _ = open_cube_with_top_hole()

        self.assertEqual(
            transaction_topology_counts(points, faces),
            topology_counts(mesh_report(points, faces)),
        )

    def test_boundary_regions_are_independent_transactions_unless_explicitly_paired(self) -> None:
        base = {
            "type": "boundary_loop",
            "component_id": 7,
            "expanded_bbox": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
        }
        small_hole = {**base, "id": "boundary_loop_0001", "paired_region_id": None}
        policy_opening = {**base, "id": "boundary_loop_0002", "paired_region_id": None}

        self.assertIsNone(edge_reason(small_hole, policy_opening))
        paired = {**small_hole, "paired_region_id": policy_opening["id"]}
        self.assertEqual(
            edge_reason(paired, policy_opening),
            "explicit_compatible_boundary_loop_pair",
        )

    def test_patch_local_scale_is_not_inflated_by_global_proxy_pitch(self) -> None:
        thresholds = FusionThresholds(voxel_pitch=0.04)
        item = {
            "local_scale": 1.2e-4,
            "length": 9.0e-4,
            "edge_count": 23,
        }

        self.assertEqual(local_scale(item, thresholds), 1.2e-4)

    def test_proxy_face_index_matches_direct_bbox_filter(self) -> None:
        proxy_points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [1.0, 1.0, 0.0],
                [1.1, 1.0, 0.0],
                [1.0, 1.1, 0.0],
                [2.0, 0.0, 0.0],
                [2.1, 0.0, 0.0],
                [2.0, 0.1, 0.0],
            ],
            dtype=np.float64,
        )
        proxy_faces = np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)
        bbox = {"min": [0.9, 0.9, -0.1], "max": [1.2, 1.2, 0.1]}
        index = build_proxy_face_index(proxy_points, proxy_faces)
        centroids, _ = face_centroids_normals(proxy_points, proxy_faces)
        expected = np.flatnonzero(
            np.all((centroids >= np.asarray(bbox["min"])) & (centroids <= np.asarray(bbox["max"])), axis=1)
        )

        np.testing.assert_array_equal(np.sort(index.query_bbox(bbox)), expected)

    def test_conformal_proxy_patch_closes_real_source_loop(self) -> None:
        source_points, source_faces, source_loop = open_cube_with_top_hole()
        source_indices = np.arange(source_faces.shape[0], dtype=np.int64)
        proxy_points, proxy_faces = inset_square_proxy()
        inventory = {
            "boundary_regions": {"items": [conformal_hole_inventory(source_loop)]},
            "gap_regions": {"items": []},
            "non_manifold_regions": {"items": []},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_hybrid_proxy_fusion(
                source_points,
                source_faces,
                source_indices,
                proxy_points,
                proxy_faces,
                inventory,
                {"items": []},
                [],
                Path(tmp_dir),
                voxel_pitch=0.05,
            )
            region = result["patch_regions"][0]
            artifacts = region["artifacts"]
            final_mesh = read_surface(Path(result["hybrid_fused_candidate_vtp"]))
            proxy_weight = np.asarray(final_mesh.cell_data["proxy_weight"], dtype=np.float64)
            sdf_blend_weight = np.asarray(final_mesh.cell_data["sdf_blend_weight"], dtype=np.float64)

            self.assertTrue(Path(artifacts["proxy_patch_vtp"]).exists())
            self.assertTrue(Path(artifacts["seam_belt_vtp"]).exists())
            self.assertTrue(Path(artifacts["stitch_band_vtp"]).exists())
            self.assertEqual(Path(artifacts["proxy_patch_vtp"]).name, "region_0001_proxy_patch.vtp")
            self.assertTrue(Path(result["debug_artifacts"]["region_patch_graph_json"]).exists())
            self.assertTrue(Path(result["debug_artifacts"]["hybrid_fusion_trace_json"]).exists())
            self.assertEqual(proxy_weight.shape[0], final_mesh.n_cells)
            self.assertTrue(np.all((proxy_weight >= 0.0) & (proxy_weight <= 1.0)))
            self.assertTrue(np.array_equal(proxy_weight, sdf_blend_weight))

        self.assertEqual(region["selection_status"], "use_proxy_patch")
        self.assertEqual(region["classification"], "patch_required")
        self.assertEqual(region["proxy_trust"]["status"], "trusted")
        self.assertEqual(region["proxy_trust"]["values"]["proxy_face_count"], 2)
        self.assertEqual(result["final_metrics"]["topology"]["boundary_edges"], 0)
        self.assertEqual(result["final_metrics"]["topology"]["non_manifold_edges"], 0)
        self.assertEqual(result["source_proxy_blend"]["origin_face_counts"]["source"], 10)
        self.assertEqual(result["source_proxy_blend"]["origin_face_counts"]["proxy_patch"], 2)
        self.assertEqual(result["source_proxy_blend"]["origin_face_counts"]["stitch_band"], 8)
        self.assertTrue(result["comparisons"]["self_intersection"]["passed"])
        self.assertEqual(region["proxy_weight"]["max"], 1.0)
        self.assertGreater(result["source_proxy_blend"]["area_weighted"]["proxy_ratio"], 0.0)
        self.assertGreater(result["source_proxy_blend"]["face_count_weighted"]["proxy_ratio"], 0.0)
        self.assertEqual(result["face_provenance"]["source_faces_missing_source_triangle_index"], 0)
        self.assertIn(1, result["face_provenance"]["face_origin_values"])
        self.assertIn(2, result["face_provenance"]["face_origin_values"])

    def test_policy_cap_extracts_source_locked_patch_from_complete_proxy(self) -> None:
        source_points, source_faces, source_loop = open_cube_with_top_hole()
        source_indices = np.arange(source_faces.shape[0], dtype=np.int64)
        proxy_points, proxy_faces = closed_grid_proxy_for_top_hole()
        item = conformal_hole_inventory(source_loop)
        item.update(
            requires_policy=True,
            classification="large_opening_or_missing_surface",
            operator="proxy_conformal_patch_after_cap_decision",
        )
        inventory = {
            "boundary_regions": {"items": [item]},
            "gap_regions": {"items": []},
            "non_manifold_regions": {"items": []},
        }
        packet = {"items": [{"id": "opening_0001", "source_region": "boundary_loop_0001"}]}
        decisions = [
            {
                "item_id": "opening_0001",
                "status": "decided",
                "decision": "cap",
                "semantic_label": "missing_cover",
                "shape_prior": "voxel_sdf",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_hybrid_proxy_fusion(
                source_points,
                source_faces,
                source_indices,
                proxy_points,
                proxy_faces,
                inventory,
                packet,
                decisions,
                Path(tmp_dir),
                voxel_pitch=0.05,
            )
            region = result["patch_regions"][0]
            proxy_artifact_exists = Path(region["artifacts"]["proxy_patch_vtp"]).exists()

        self.assertEqual(region["selection_status"], "use_proxy_patch")
        self.assertEqual(region["operator"], "source_locked_proxy_patch")
        self.assertEqual(region["proxy_trust"]["status"], "trusted")
        self.assertFalse(region["proxy_trust"]["values"]["nearest_point_scatter_used"])
        self.assertTrue(proxy_artifact_exists)
        self.assertEqual(result["final_metrics"]["topology"]["boundary_edges"], 0)
        self.assertEqual(result["final_metrics"]["topology"]["non_manifold_edges"], 0)

    def test_small_hole_uses_constrained_fill_without_proxy_geometry(self) -> None:
        source_points, source_faces, source_loop = open_cube_with_top_hole()
        source_indices = np.arange(source_faces.shape[0], dtype=np.int64)
        proxy_points, proxy_faces = far_proxy_patch()
        item = conformal_hole_inventory(
            source_loop,
            operator="constrained_loop_triangulation",
        )
        inventory = {
            "boundary_regions": {"items": [item]},
            "gap_regions": {"items": []},
            "non_manifold_regions": {"items": []},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_hybrid_proxy_fusion(
                source_points,
                source_faces,
                source_indices,
                proxy_points,
                proxy_faces,
                inventory,
                {"items": []},
                [],
                Path(tmp_dir),
                voxel_pitch=0.05,
            )
            region = result["patch_regions"][0]
            self.assertTrue(Path(region["artifacts"]["hole_fill_vtp"]).exists())

        self.assertEqual(region["selection_status"], "use_hole_fill")
        self.assertEqual(result["final_metrics"]["topology"]["boundary_edges"], 0)
        self.assertEqual(result["source_proxy_blend"]["origin_face_counts"]["proxy_patch"], 0)
        self.assertEqual(result["source_proxy_blend"]["origin_face_counts"]["hole_fill"], 2)
        self.assertTrue(result["comparisons"]["self_intersection"]["passed"])

    def test_small_hole_transaction_rolls_back_local_self_intersection(self) -> None:
        source_points, source_faces, source_loop = open_cube_with_top_hole()
        crossing_points = np.asarray(
            [[0.0, -0.5, 0.5], [0.0, 0.5, 1.5], [0.0, 0.5, 0.5]],
            dtype=np.float64,
        )
        source_points = np.vstack([source_points, crossing_points])
        source_faces = np.vstack([source_faces, [[8, 9, 10]]])
        source_indices = np.arange(source_faces.shape[0], dtype=np.int64)
        item = conformal_hole_inventory(
            source_loop,
            operator="constrained_loop_triangulation",
        )
        inventory = {
            "boundary_regions": {"items": [item]},
            "gap_regions": {"items": []},
            "non_manifold_regions": {"items": []},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_hybrid_proxy_fusion(
                source_points,
                source_faces,
                source_indices,
                *far_proxy_patch(),
                inventory,
                {"items": []},
                [],
                Path(tmp_dir),
                voxel_pitch=0.05,
            )

        region = result["patch_regions"][0]
        self.assertEqual(region["selection_status"], "reject_patch")
        self.assertEqual(region["rejection_reason"], "patch_local_self_intersection_detected")
        self.assertGreater(
            region["seam_results"]["local_self_intersection"]["intersection_pairs"],
            0,
        )
        self.assertNotIn("hole_fill_vtp", region["artifacts"])

    def test_seeded_proxy_speckle_is_not_mistaken_for_a_conformal_patch(self) -> None:
        source_points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
            dtype=np.float64,
        )
        source_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        source_indices = np.asarray([20], dtype=np.int64)
        proxy_points = np.asarray(
            [
                [0.10, 0.02, 0.0],
                [0.20, 0.02, 0.0],
                [0.10, 0.12, 0.0],
                [0.02, 10.00, 0.0],
                [0.12, 10.00, 0.0],
                [0.02, 10.10, 0.0],
            ],
            dtype=np.float64,
        )
        proxy_faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        inventory = {
            "boundary_regions": {
                "items": [
                    {
                        "id": "boundary_loop_0001",
                        "type": "boundary_loop",
                        "edge_count": 1,
                        "length": 0.2,
                        "bbox": {"min": [0.0, 0.0, -0.1], "max": [1.0, 10.2, 0.1]},
                        "face_ids": [0],
                        "source_triangle_ids": [20],
                        "edge_vertex_ids": [0, 1],
                        "requires_policy": False,
                        "classification": "patch_required",
                        "blocking": True,
                    }
                ]
            },
            "gap_regions": {"items": []},
            "non_manifold_regions": {"items": []},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_hybrid_proxy_fusion(
                source_points,
                source_faces,
                source_indices,
                proxy_points,
                proxy_faces,
                inventory,
                {"items": []},
                [],
                Path(tmp_dir),
                voxel_pitch=0.05,
            )

        region = result["patch_regions"][0]
        self.assertEqual(region["selection_status"], "reject_patch")
        self.assertEqual(region["proxy_trust"]["values"]["proxy_face_count"], 1)
        self.assertNotIn("stitch_band_vtp", region["artifacts"])

    def test_patch_that_worsens_final_topology_is_kept_out_of_final_mesh(self) -> None:
        source_points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        source_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        source_indices = np.asarray([30], dtype=np.int64)
        proxy_points = np.asarray(
            [
                [-0.1, -0.1, 0.0],
                [0.8, -0.1, 0.0],
                [0.8, 0.8, 0.0],
                [-0.1, 0.8, 0.0],
            ],
            dtype=np.float64,
        )
        proxy_faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        inventory = {
            "boundary_regions": {
                "items": [
                    {
                        "id": "boundary_loop_0001",
                        "type": "boundary_loop",
                        "edge_count": 3,
                        "length": 0.3,
                        "bbox": {"min": [-0.2, -0.2, -0.1], "max": [1.0, 1.0, 0.1]},
                        "face_ids": [0],
                        "source_triangle_ids": [30],
                        "edge_vertex_ids": [0, 1, 2],
                        "requires_policy": False,
                        "classification": "patch_required",
                        "blocking": True,
                    }
                ]
            },
            "gap_regions": {"items": []},
            "non_manifold_regions": {"items": []},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_hybrid_proxy_fusion(
                source_points,
                source_faces,
                source_indices,
                proxy_points,
                proxy_faces,
                inventory,
                {"items": []},
                [],
                Path(tmp_dir),
                voxel_pitch=0.05,
            )

        region = result["patch_regions"][0]
        self.assertEqual(region["selection_status"], "reject_patch")
        self.assertIn(
            region["rejection_reason"],
            {"loop_correspondence_distance_exceeded", "patch_topology_worsened"},
        )
        self.assertEqual(result["source_proxy_blend"]["origin_face_counts"]["proxy_patch"], 0)

    def test_patch_transactions_commit_good_region_and_only_roll_back_bad_region(self) -> None:
        left_points, left_faces, left_loop = open_cube_with_top_hole((-3.0, 0.0, 0.0))
        right_points, right_faces, right_loop = open_cube_with_top_hole((3.0, 0.0, 0.0))
        source_points = np.vstack([left_points, right_points])
        source_faces = np.vstack([left_faces, right_faces + left_points.shape[0]])
        source_indices = np.arange(source_faces.shape[0], dtype=np.int64)
        proxy_points, proxy_faces = inset_square_proxy((-3.0, 0.0, 0.0))
        inventory = {
            "boundary_regions": {
                "items": [
                    conformal_hole_inventory(left_loop, region_index=1, x_offset=-3.0),
                    conformal_hole_inventory(
                        right_loop,
                        face_offset=left_faces.shape[0],
                        vertex_offset=left_points.shape[0],
                        region_index=2,
                        x_offset=3.0,
                    ),
                ]
            },
            "gap_regions": {"items": []},
            "non_manifold_regions": {"items": []},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_hybrid_proxy_fusion(
                source_points,
                source_faces,
                source_indices,
                proxy_points,
                proxy_faces,
                inventory,
                {"items": []},
                [],
                Path(tmp_dir),
                voxel_pitch=0.05,
            )

        self.assertEqual(
            [region["selection_status"] for region in result["patch_regions"]],
            ["use_proxy_patch", "reject_patch"],
        )
        self.assertEqual(result["final_metrics"]["topology"]["boundary_edges"], 4)
        transaction_summary = next(
            step for step in result["fusion_trace"]["steps"]
            if step["step"] == "per_patch_transaction_summary"
        )
        self.assertEqual(transaction_summary["accepted"], 1)
        self.assertEqual(transaction_summary["rejected"], 1)
        self.assertEqual(result["source_proxy_blend"]["origin_face_counts"]["proxy_patch"], 2)

    def test_near_coincident_source_loops_use_real_zipper_band(self) -> None:
        lower = np.asarray(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        source_points = np.vstack([lower, lower + np.asarray([0.0, 0.0, 0.1])])
        source_faces = np.asarray([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7]], dtype=np.int64)
        source_indices = np.arange(4, dtype=np.int64)
        items = [
            {
                "id": "boundary_loop_0001",
                "type": "boundary_loop",
                "edge_count": 4,
                "length": 8.0,
                "bbox": {"min": [-1.0, -1.0, -0.01], "max": [1.0, 1.0, 0.11]},
                "face_ids": [0, 1],
                "source_triangle_ids": [0, 1],
                "edge_vertex_ids": [0, 1, 2, 3],
                "ordered_vertex_ids": [0, 3, 2, 1],
                "paired_region_id": "boundary_loop_0002",
                "pair_id": 1,
                "requires_policy": False,
                "classification": "near_coincident_part_seam",
                "patch_eligible": True,
                "operator": "loop_pair_zipper",
            },
            {
                "id": "boundary_loop_0002",
                "type": "boundary_loop",
                "edge_count": 4,
                "length": 8.0,
                "bbox": {"min": [-1.0, -1.0, -0.01], "max": [1.0, 1.0, 0.11]},
                "face_ids": [2, 3],
                "source_triangle_ids": [2, 3],
                "edge_vertex_ids": [4, 5, 6, 7],
                "ordered_vertex_ids": [4, 5, 6, 7],
                "paired_region_id": "boundary_loop_0001",
                "pair_id": 1,
                "requires_policy": False,
                "classification": "near_coincident_part_seam",
                "patch_eligible": True,
                "operator": "loop_pair_zipper",
            },
        ]
        inventory = {
            "boundary_regions": {"items": items},
            "gap_regions": {"items": []},
            "non_manifold_regions": {"items": []},
        }
        proxy_points, proxy_faces = far_proxy_patch()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_hybrid_proxy_fusion(
                source_points,
                source_faces,
                source_indices,
                proxy_points,
                proxy_faces,
                inventory,
                {"items": []},
                [],
                Path(tmp_dir),
                voxel_pitch=0.01,
            )
            region = result["patch_regions"][0]
            self.assertTrue(Path(region["artifacts"]["stitch_band_vtp"]).exists())

        self.assertEqual(region["selection_status"], "use_source_zipper")
        self.assertEqual(result["final_metrics"]["topology"]["boundary_edges"], 0)
        self.assertEqual(result["final_metrics"]["topology"]["components"]["count"], 1)
        self.assertEqual(result["source_proxy_blend"]["origin_face_counts"]["proxy_patch"], 0)
        self.assertEqual(result["source_proxy_blend"]["origin_face_counts"]["stitch_band"], 8)

    def test_near_zero_width_component_seam_uses_index_weld_without_band_faces(self) -> None:
        source_points, source_faces, source_indices, lower_loop, upper_loop = (
            coincident_open_tube_halves()
        )
        first_id = "boundary_loop_0001"
        second_id = "boundary_loop_0002"
        inventory = {
            "boundary_regions": {
                "items": [
                    paired_loop_inventory_item(
                        first_id,
                        second_id,
                        1,
                        lower_loop,
                        source_points,
                        source_faces,
                    ),
                    paired_loop_inventory_item(
                        second_id,
                        first_id,
                        1,
                        upper_loop,
                        source_points,
                        source_faces,
                    ),
                ]
            },
            "gap_regions": {"items": []},
            "non_manifold_regions": {"items": []},
        }
        proxy_points, proxy_faces = far_proxy_patch()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_hybrid_proxy_fusion(
                source_points,
                source_faces,
                source_indices,
                proxy_points,
                proxy_faces,
                inventory,
                {"items": []},
                [],
                Path(tmp_dir),
                voxel_pitch=0.01,
            )
            region = result["patch_regions"][0]
            weld_artifact_exists = Path(
                region["artifacts"]["coincident_weld_vtp"]
            ).exists()

        self.assertEqual(region["selection_status"], "use_source_zipper")
        self.assertEqual(
            region["seam_results"]["method"],
            "arc_length_union_edge_split_fixed_source_ring_index_weld",
        )
        self.assertTrue(weld_artifact_exists)
        self.assertEqual(result["final_metrics"]["topology"]["boundary_edges"], 24)
        self.assertEqual(result["final_metrics"]["topology"]["components"]["count"], 1)
        self.assertEqual(result["source_proxy_blend"]["origin_face_counts"]["stitch_band"], 0)
        self.assertEqual(region["final_provenance"]["face_origin_values"], ["source"])

    def test_same_component_source_loops_commit_real_zipper_and_keep_parent_provenance(self) -> None:
        source_points, source_faces, source_indices = open_torus_source()
        extraction = extract_ordered_boundary_loops(source_points, source_faces)
        self.assertTrue(extraction["success"])
        self.assertEqual(len(extraction["loops"]), 2)
        first_id = "boundary_loop_0001"
        second_id = "boundary_loop_0002"
        inventory = {
            "boundary_regions": {
                "items": [
                    paired_loop_inventory_item(
                        first_id,
                        second_id,
                        1,
                        extraction["loops"][0],
                        source_points,
                        source_faces,
                    ),
                    paired_loop_inventory_item(
                        second_id,
                        first_id,
                        1,
                        extraction["loops"][1],
                        source_points,
                        source_faces,
                    ),
                ]
            },
            "gap_regions": {"items": []},
            "non_manifold_regions": {"items": []},
        }
        proxy_points, proxy_faces = far_proxy_patch()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_hybrid_proxy_fusion(
                source_points,
                source_faces,
                source_indices,
                proxy_points,
                proxy_faces,
                inventory,
                {"items": []},
                [],
                Path(tmp_dir),
                voxel_pitch=0.6,
            )
            candidate = read_surface(Path(result["hybrid_fused_candidate_vtp"]))
            origins = np.asarray(candidate.cell_data["face_origin"])
            parent_indices = np.asarray(candidate.cell_data["source_triangle_index"])

        region = result["patch_regions"][0]
        self.assertEqual(region["selection_status"], "use_source_zipper")
        self.assertEqual(
            region["seam_results"]["diagnostics"]["method"],
            "same_mesh_paired_arc_length_edge_split_annular_bridge",
        )
        self.assertEqual(result["final_metrics"]["topology"]["boundary_edges"], 0)
        self.assertEqual(result["final_metrics"]["topology"]["non_manifold_edges"], 0)
        self.assertEqual(result["final_metrics"]["topology"]["components"]["count"], 1)
        self.assertEqual(result["source_proxy_blend"]["origin_face_counts"]["proxy_patch"], 0)
        self.assertEqual(result["source_proxy_blend"]["origin_face_counts"]["stitch_band"], 16)
        source_mask = origins == FACE_ORIGIN["source"]
        stitch_mask = origins == FACE_ORIGIN["stitch_band"]
        self.assertTrue(np.all(parent_indices[source_mask] >= 1000))
        self.assertTrue(np.all(parent_indices[stitch_mask] == -1))

    def test_policy_preserved_opening_keeps_source_without_proxy_patch(self) -> None:
        source_points, source_faces, source_indices = source_plate()
        proxy_points, proxy_faces = seam_touching_proxy_patch()
        inventory, packet, decisions = preserve_opening_inventory()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_hybrid_proxy_fusion(
                source_points,
                source_faces,
                source_indices,
                proxy_points,
                proxy_faces,
                inventory,
                packet,
                decisions,
                Path(tmp_dir),
                voxel_pitch=0.05,
            )

        region = result["patch_regions"][0]
        self.assertEqual(region["classification"], "preserve_opening")
        self.assertEqual(region["selection_status"], "keep_source")
        self.assertTrue(region["accepted"])
        self.assertEqual(region["artifacts"].get("proxy_patch_vtp"), None)

    def test_policy_rejected_opening_is_explicit_and_does_not_extract_proxy(self) -> None:
        source_points, source_faces, source_indices = source_plate()
        proxy_points, proxy_faces = seam_touching_proxy_patch()
        inventory, packet, decisions = preserve_opening_inventory()
        decisions[0]["decision"] = "reject"

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_hybrid_proxy_fusion(
                source_points,
                source_faces,
                source_indices,
                proxy_points,
                proxy_faces,
                inventory,
                packet,
                decisions,
                Path(tmp_dir),
                voxel_pitch=0.05,
            )

        region = result["patch_regions"][0]
        self.assertEqual(region["classification"], "reject_region")
        self.assertEqual(region["operator"], "policy_reject")
        self.assertEqual(region["selection_status"], "reject_patch")
        self.assertEqual(region["rejection_reason"], "policy_region_rejected")
        self.assertEqual(region["proxy_trust"]["status"], "not_applicable")
        self.assertNotIn("proxy_patch_vtp", region["artifacts"])
        self.assertNotIn("stitch_band_vtp", region["artifacts"])
        self.assertEqual(result["final_metrics"]["triangles"], source_faces.shape[0])

    def test_unstitchable_proxy_patch_is_rejected_when_seam_contact_fails(self) -> None:
        source_points, source_faces, source_indices = source_plate()
        proxy_points, proxy_faces = far_proxy_patch()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_hybrid_proxy_fusion(
                source_points,
                source_faces,
                source_indices,
                proxy_points,
                proxy_faces,
                patch_inventory(),
                {"items": []},
                [],
                Path(tmp_dir),
                voxel_pitch=0.05,
            )

        region = result["patch_regions"][0]
        self.assertEqual(region["selection_status"], "reject_patch")
        self.assertFalse(region["accepted"])
        self.assertEqual(region["rejection_reason"], "proxy_patch_extraction_failed")

    def test_distance_and_patch_local_drift_gates_are_computed_from_final_mesh(self) -> None:
        source_points, source_faces, _ = source_plate()
        final_points = source_points + np.asarray([0.5, 0.0, 0.0])
        thresholds = FusionThresholds(voxel_pitch=0.01, source_distance_ratio=0.001)

        comparisons = build_fusion_comparisons(
            source_points,
            source_faces,
            final_points,
            source_faces,
            np.asarray([0, 0], dtype=np.int16),
            thresholds,
        )

        self.assertEqual(comparisons["source_distance"]["status"], "computed")
        self.assertFalse(comparisons["source_distance"]["passed"])
        self.assertEqual(comparisons["patch_local_drift"]["status"], "not_applicable")

    def test_source_distance_acceptance_threshold_does_not_expand_with_proxy_pitch(self) -> None:
        source_points, source_faces, _ = source_plate()
        fine = build_fusion_comparisons(
            source_points,
            source_faces,
            source_points,
            source_faces,
            np.asarray([0, 0], dtype=np.int16),
            FusionThresholds(voxel_pitch=0.001),
        )
        coarse = build_fusion_comparisons(
            source_points,
            source_faces,
            source_points,
            source_faces,
            np.asarray([0, 0], dtype=np.int16),
            FusionThresholds(voxel_pitch=10.0),
        )

        self.assertEqual(
            fine["source_distance"]["threshold"],
            coarse["source_distance"]["threshold"],
        )

    def test_normal_and_topology_gates_reject_bad_proxy_patch(self) -> None:
        source_points, source_faces, _ = source_plate()
        proxy_points, proxy_faces = orthogonal_proxy_patch()
        thresholds = FusionThresholds(voxel_pitch=0.05)
        proxy_face_ids = np.asarray([0], dtype=np.int64)
        source_face_ids = np.asarray([0, 1], dtype=np.int64)

        trust = proxy_trust_report(
            source_points,
            source_faces,
            source_face_ids,
            proxy_points,
            proxy_faces,
            proxy_face_ids,
            source_points,
            thresholds,
        )

        self.assertEqual(trust["status"], "untrusted")
        self.assertIn("normal_compatibility_failed", trust["reason_codes"])
        self.assertIn("seam_contact_passed", trust["evidence_codes"])
        self.assertNotIn("seam_contact_passed", trust["failure_reason_codes"])

    def test_low_normal_proxy_candidate_still_requires_conformal_seam(self) -> None:
        source_points, source_faces, source_indices = source_plate()
        proxy_points, proxy_faces = orthogonal_proxy_patch()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_hybrid_proxy_fusion(
                source_points,
                source_faces,
                source_indices,
                proxy_points,
                proxy_faces,
                patch_inventory_with_z_belt(),
                {"items": []},
                [],
                Path(tmp_dir),
                voxel_pitch=0.05,
            )

        region = result["patch_regions"][0]
        self.assertEqual(region["selection_status"], "reject_patch")
        self.assertIsNotNone(region["rejection_reason"])
        self.assertIn("seam_contact_passed", region["proxy_trust"]["evidence_codes"])
        self.assertIn("normal_compatibility_low_nonblocking", region["proxy_trust"]["evidence_codes"])
        self.assertNotIn("seam_contact_passed", region["proxy_trust"]["failure_reason_codes"])
        self.assertNotIn("normal_compatibility_failed", region["proxy_trust"]["failure_reason_codes"])
        self.assertNotIn("stitch_band_vtp", region["artifacts"])


if __name__ == "__main__":
    unittest.main()
