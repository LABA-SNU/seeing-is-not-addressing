"""D/A/R and specificity controls for every backbone, split, and factor."""

import argparse
import json
from pathlib import Path

import numpy as np

from factor_access_protocol import (
    ALPHAS,
    canonical_directions,
    factor_metrics,
    norm,
    ovr_directions,
    protocol_manifest,
    select_alpha,
)
from split_protocols import FACTORS, ordered_unique, split_masks, validate_factoratlas


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/factoratlas"


def calibration_halves(records, calibration):
    context = np.asarray([row["context"] for row in records])
    seed = np.asarray([row["renderer_seed"] for row in records])
    groups = []
    for context_value in ordered_unique(context[calibration]):
        for seed_value in ordered_unique(
            seed[calibration & (context == context_value)]
        ):
            groups.append((context_value, seed_value))
    first_groups = set(groups[::2])
    first = calibration & np.asarray([
        (context_value, seed_value) in first_groups
        for context_value, seed_value in zip(context, seed)
    ])
    second = calibration & ~first
    if not first.any() or not second.any():
        raise ValueError("empty stability half")
    return first, second, groups


def run_factor(image, native_text, records, split, factor, rng):
    names = ordered_unique(row[factor] for row in records)
    labels = np.asarray([row[factor] for row in records])
    calibration = split["calibration"]
    validation = split["validation"]
    test = split["test"]
    centers, directions = canonical_directions(
        image, labels, names, calibration
    )
    alpha, repaired, validation_result = select_alpha(
        image,
        labels,
        validation,
        native_text,
        directions,
        names,
        ALPHAS,
    )
    native = factor_metrics(image, labels, test, native_text, names)
    recovery = factor_metrics(image, labels, test, repaired, names)
    image_only = factor_metrics(image, labels, test, centers, names)
    first, second, stability_groups = calibration_halves(records, calibration)
    _, first_direction = canonical_directions(image, labels, names, first)
    _, second_direction = canonical_directions(image, labels, names, second)
    stability = np.sum(first_direction * second_direction, axis=1)
    wrong = directions[np.roll(np.arange(len(names)), -1)]
    wrong_result = factor_metrics(
        image, labels, test, norm(native_text + alpha * wrong), names
    )
    random_map = []
    random_top1 = []
    for _ in range(64):
        random_direction = norm(rng.standard_normal(directions.shape))
        metrics = factor_metrics(
            image,
            labels,
            test,
            norm(native_text + alpha * random_direction),
            names,
        )
        random_map.append(metrics["mean_ap"])
        random_top1.append(metrics["top1"])
    text_direction = ovr_directions(norm(native_text))
    gamma, text_query, text_validation = select_alpha(
        image,
        labels,
        validation,
        native_text,
        text_direction,
        names,
        ALPHAS,
    )
    text_result = factor_metrics(image, labels, test, text_query, names)
    return {
        "classes": list(names),
        "counts": {
            part: int(split[part].sum())
            for part in ("calibration", "validation", "test")
        },
        "D_image_centroid_test": image_only,
        "A_native_test": native,
        "R_visual_test": recovery,
        "visual_alpha": alpha,
        "visual_validation": validation_result,
        "visual_gain": {
            "top1": recovery["top1"] - native["top1"],
            "mean_ap": recovery["mean_ap"] - native["mean_ap"],
        },
        "stability": {
            "group_order": [list(map(int, group)) for group in stability_groups],
            "split_rule": "alternating calibration context x seed groups",
            "mean_cosine": float(stability.mean()),
            "min_cosine": float(stability.min()),
            "per_value": dict(zip(names, map(float, stability))),
        },
        "wrong_cyclic_test": wrong_result,
        "random_64_test": {
            "mean_ap_mean": float(np.mean(random_map)),
            "mean_ap_std": float(np.std(random_map, ddof=1)),
            "mean_ap_max": float(np.max(random_map)),
            "top1_mean": float(np.mean(random_top1)),
            "top1_std": float(np.std(random_top1, ddof=1)),
        },
        "text_OVR": {
            "gamma": gamma,
            "validation": text_validation,
            "test": text_result,
            "mean_ap_gain": text_result["mean_ap"] - native["mean_ap"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="feature-cache prefix")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--data", type=Path, default=DATA,
                        help="FactorAtlas root containing metadata, images, and feature cache")
    parser.add_argument("--out", type=Path,
                        help="directory for this run's JSON result (default: <data>/results)")
    args = parser.parse_args()
    data = args.data
    out = args.out or data / "results"
    records = json.loads((data / "metadata.json").read_text())["records"]
    feature_file = f"{args.tag}_features.npz"
    cache = np.load(data / feature_file, allow_pickle=True)
    image = norm(cache["image"]).astype(np.float32)
    text = {
        factor: norm(value).astype(np.float32)
        for factor, value in cache["text"].item().items()
    }
    validate_factoratlas(data, records, image, text)
    report = {
        "protocol": {
            "tag": args.tag,
            "feature_file": feature_file,
            "canonical": protocol_manifest(),
            "scope": "2 held-out conditions x 3 factors",
            "specificity": "cyclic wrong, 64 matched-shape isotropic random banks, native-text OVR",
            "seed": args.seed,
        },
        "splits": {},
    }
    for split_index, (split_name, split) in enumerate(split_masks(records).items()):
        row = {}
        for factor_index, factor in enumerate(FACTORS):
            rng = np.random.default_rng(
                args.seed + split_index * 100 + factor_index
            )
            row[factor] = run_factor(
                image, text[factor], records, split, factor, rng
            )
            print(json.dumps({
                "tag": args.tag,
                "split": split_name,
                "factor": factor,
                "D": row[factor]["D_image_centroid_test"]["mean_ap"],
                "A": row[factor]["A_native_test"]["mean_ap"],
                "R": row[factor]["R_visual_test"]["mean_ap"],
                "wrong": row[factor]["wrong_cyclic_test"]["mean_ap"],
                "random": row[factor]["random_64_test"]["mean_ap_mean"],
            }), flush=True)
        report["splits"][split_name] = row
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.tag}.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"wrote": str(path)}))


if __name__ == "__main__":
    main()
