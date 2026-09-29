"""Canonical geometry, selection, and validation helpers for FactorAtlas.

All FactorAtlas audits import these functions. Keeping centroid
normalization and validation tie-breaking here prevents cross-factor, cross-backbone, and control scripts
from silently implementing a different intervention than the split-specific main
experiment.
"""

from __future__ import annotations


import numpy as np
from sklearn.metrics import average_precision_score


ALPHAS = np.linspace(0.0, 8.0, 161)


def norm(value: np.ndarray) -> np.ndarray:
    return value / (np.linalg.norm(value, axis=-1, keepdims=True) + 1e-12)


def factor_metrics(image, labels, mask, queries, names):
    """Return image-wise K-way top-1 and query-wise one-vs-rest macro-mAP."""
    scores = image[mask] @ queries.T
    prediction = np.asarray(names)[scores.argmax(1)]
    aps = [
        average_precision_score(labels[mask] == name, scores[:, index])
        for index, name in enumerate(names)
    ]
    return {
        "top1": float(np.mean(prediction == labels[mask])),
        "mean_ap": float(np.mean(aps)),
        "per_class_ap": dict(zip(names, map(float, aps))),
        "image_count": int(mask.sum()),
    }


def normalized_centroids(image, labels, names, mask):
    """Mean individually normalized images, then L2-normalize each value mean."""
    return norm(np.stack([
        image[mask & (labels == name)].mean(0) for name in names
    ]))


def ovr_directions(centroids):
    """Normalized one-vs-rest directions from normalized value centroids."""
    others = (centroids.sum(0, keepdims=True) - centroids) / (len(centroids) - 1)
    return norm(centroids - others)



def canonical_directions(image, labels, names, calibration):
    centers = normalized_centroids(image, labels, names, calibration)
    return centers, ovr_directions(centers)


def select_alpha(image, labels, validation, native, direction, names,
                 grid=ALPHAS):
    """Select by validation macro-mAP, then top-1, then smaller alpha."""
    candidates = []
    for alpha in grid:
        query = norm(native + alpha * direction)
        result = factor_metrics(image, labels, validation, query, names)
        candidates.append((result["mean_ap"], result["top1"], -float(alpha),
                           float(alpha), query, result))
    _, _, _, alpha, query, result = max(candidates)
    return alpha, query, result


def protocol_manifest():
    return {
        "centroid": "normalize(mean(individually L2-normalized image embeddings))",
        "direction": "normalized one-vs-rest residual of normalized value centroids",
        "alpha_selection": "validation macro-mAP; ties by label top-1 then smaller alpha",
        "alpha_grid": ALPHAS.tolist(),
    }
