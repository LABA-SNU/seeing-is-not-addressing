"""Cache frozen image and text embeddings for a FactorAtlas backbone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/factoratlas"
PROMPT_TEMPLATES = {
    "pattern": ("{value}", "a {value} surface", "an object with a {value} pattern"),
    "hue": ("{value}", "a {value} object", "a {value} surface"),
    "shape": ("{value}", "a {value}", "an object shaped like a {value}"),
}


def normalize(values: np.ndarray) -> np.ndarray:
    return values / (np.linalg.norm(values, axis=-1, keepdims=True) + 1e-12)


def features(output: object) -> torch.Tensor:
    return output.pooler_output if hasattr(output, "pooler_output") else output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--tag", required=True, help="output feature-cache name")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu for a smoke test")

    records = json.loads((args.data / "metadata.json").read_text())["records"]
    if not records:
        raise ValueError("metadata has no records")
    model_kwargs = {"local_files_only": args.local_files_only}
    processor = AutoProcessor.from_pretrained(args.model, **model_kwargs)
    model = AutoModel.from_pretrained(args.model, **model_kwargs).to(args.device).eval()
    max_length = min(64, getattr(AutoConfig.from_pretrained(args.model, **model_kwargs).text_config, "max_position_embeddings", 64))
    image_batches = []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            images = [Image.open(args.data / row["file"]).convert("RGB") for row in batch]
            inputs = processor(images=images, return_tensors="pt").to(args.device)
            image_batches.append(normalize(features(model.get_image_features(**inputs)).cpu().numpy()))
            print(f"{args.tag} images {start + len(batch)}/{len(records)}", flush=True)

        text = {}
        for factor, templates in PROMPT_TEMPLATES.items():
            values = tuple(dict.fromkeys(row[factor] for row in records))
            prompts = [template.format(value=value) for value in values for template in templates]
            inputs = processor(text=prompts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").to(args.device)
            vectors = features(model.get_text_features(**inputs)).cpu().numpy()
            text[factor] = normalize(normalize(vectors).reshape(len(values), len(templates), -1).mean(axis=1))

    destination = args.data / f"{args.tag}_features.npz"
    np.savez(destination, image=np.concatenate(image_batches), text=np.array(text, dtype=object))
    print(destination)


if __name__ == "__main__":
    main()
