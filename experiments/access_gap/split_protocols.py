"""Canonical FactorAtlas validation and held-out split definitions.
This module validates the expected FactorAtlas vocabulary. All split-specific
alignment experiments import these masks so that calibration,
validation, and test examples cannot silently drift between methods.
"""
from pathlib import Path

import numpy as np


FACTORS = ("pattern", "hue", "shape")
EXPECTED_SHAPES = (
    "cube", "sphere", "cylinder", "cone", "torus", "pyramid",
    "triangular_prism", "octahedron",
)
EXPECTED_COUNTS = {
    "context_ood": (7680, 3840, 11520),
    "hue_shape_interaction": (17280, 2880, 2880),
}


def ordered_unique(values):
    return tuple(dict.fromkeys(values))


def validate_factoratlas(data: Path, records, image, text):
    assert len(records) == 23040, len(records)
    assert image.ndim == 2 and image.shape[0] == 23040, image.shape
    dimension = image.shape[1]
    assert dimension > 0, image.shape
    assert set(text) == set(FACTORS), set(text)
    assert text["pattern"].shape == (10, dimension)
    assert text["hue"].shape == (12, dimension)
    assert text["shape"].shape == (8, dimension)
    shapes = ordered_unique(row["shape"] for row in records)
    assert shapes == EXPECTED_SHAPES, shapes
    banned = {"monkey", "icosphere", "beveled_cube"}
    assert not banned.intersection(shapes), shapes


def split_masks(records):
    context = np.asarray([row["context"] for row in records])
    hue = np.asarray([row["hue"] for row in records])
    shape = np.asarray([row["shape"] for row in records])
    hues = ordered_unique(hue)
    shapes = ordered_unique(shape)

    context_split = {
        "calibration": context <= 3,
        "validation": (context >= 4) & (context <= 5),
        "test": context >= 6,
        "details": {
            "calibration_contexts": [0, 1, 2, 3],
            "validation_contexts": [4, 5],
            "test_contexts": [6, 7, 8, 9, 10, 11],
        },
    }

    test_cells = {(h, shapes[i % len(shapes)]) for i, h in enumerate(hues)}
    validation_cells = {
        (h, shapes[(i + 1) % len(shapes)]) for i, h in enumerate(hues)
    }
    test = np.asarray([(h, s) in test_cells for h, s in zip(hue, shape)])
    validation = np.asarray([
        (h, s) in validation_cells for h, s in zip(hue, shape)
    ])
    interaction_split = {
        "calibration": ~(test | validation),
        "validation": validation,
        "test": test,
        "details": {
            "unit": "hue x shape cell; all contexts and both seeds occur in each split",
            "validation_cells": sorted(map(list, validation_cells)),
            "test_cells": sorted(map(list, test_cells)),
        },
    }

    splits = {
        "context_ood": context_split,
        "hue_shape_interaction": interaction_split,
    }
    for name, split in splits.items():
        masks = [split[key] for key in ("calibration", "validation", "test")]
        assert not np.any(masks[0] & masks[1])
        assert not np.any(masks[0] & masks[2])
        assert not np.any(masks[1] & masks[2])
        counts = tuple(int(mask.sum()) for mask in masks)
        assert counts == EXPECTED_COUNTS[name], (name, counts)
    return splits
