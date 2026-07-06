from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import load_artifacts, resize_artifacts
from .constants import DEFAULT_SAME_CATEGORY_BONUS, DEFAULT_WEIGHTS
from .types import SideArtifacts


def compare_artifacts(
    query: SideArtifacts,
    reference: SideArtifacts,
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    weights = weights or DEFAULT_WEIGHTS
    if query.mask.shape != reference.mask.shape:
        reference = resize_artifacts(reference, width=query.mask.shape[1], height=query.mask.shape[0])

    union = query.mask | reference.mask
    overlap = query.mask & reference.mask
    mask_iou = float(np.count_nonzero(overlap) / max(np.count_nonzero(union), 1))

    if np.any(overlap):
        depth_mae = float(np.mean(np.abs(query.depth[overlap] - reference.depth[overlap])))
        depth_score = float(np.clip(1.0 - depth_mae / 0.30, 0.0, 1.0))
        qn = query.normal[overlap]
        rn = reference.normal[overlap]
        qn_norm = qn / np.linalg.norm(qn, axis=1, keepdims=True).clip(min=1e-8)
        rn_norm = rn / np.linalg.norm(rn, axis=1, keepdims=True).clip(min=1e-8)
        cosine = np.einsum("ij,ij->i", qn_norm, rn_norm)
        normal_score = float(np.clip(np.mean((cosine + 1.0) * 0.5), 0.0, 1.0))
    else:
        depth_score = 0.0
        normal_score = 0.0

    total = weights["mask"] * mask_iou + weights["depth"] * depth_score + weights["normal"] * normal_score
    return {
        "score": float(total),
        "mask_iou": mask_iou,
        "depth": depth_score,
        "normal": normal_score,
    }


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    denom = sum(exps)
    return [v / denom for v in exps]


def confidence_from_score(score: float) -> str:
    if score >= 0.78:
        return "high"
    if score >= 0.58:
        return "medium"
    return "low"


def normalized_category(value: str | None) -> str | None:
    if value is None:
        return None
    clean = value.strip().lower()
    return clean or None


def score_library(
    query: SideArtifacts,
    library: dict[str, Any],
    library_dir: Path,
    *,
    query_category: str | None = None,
) -> list[dict[str, Any]]:
    results = []
    query_category = normalized_category(query_category)
    same_category_bonus = float(library.get("same_category_bonus", DEFAULT_SAME_CATEGORY_BONUS))
    for car in library.get("cars", []):
        reference = load_artifacts(library_dir, car["artifacts"])
        components = compare_artifacts(query, reference, weights=library.get("score_weights") or DEFAULT_WEIGHTS)
        car_category = car.get("category", "unknown")
        category_match = query_category is not None and normalized_category(car_category) == query_category
        category_bonus = same_category_bonus if category_match else 0.0
        score = float(min(1.0, components["score"] + category_bonus))
        results.append(
            {
                "id": car["id"],
                "display_name": car.get("display_name", car["id"]),
                "category": car_category,
                "cd": float(car["cd"]),
                "cd_confidence": car.get("cd_confidence", "unknown"),
                "cd_source": car.get("cd_source", ""),
                "score": score,
                "category_match": category_match if query_category is not None else None,
                "components": {
                    "base_score": components["score"],
                    "category_bonus": category_bonus,
                    "mask_iou": components["mask_iou"],
                    "depth": components["depth"],
                    "normal": components["normal"],
                },
            }
        )
    return sorted(results, key=lambda item: item["score"], reverse=True)
