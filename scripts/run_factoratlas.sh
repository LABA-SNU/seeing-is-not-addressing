#!/usr/bin/env bash
# Download FactorAtlas, cache frozen VLM features, and run the complete audit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="google/siglip2-base-patch16-224"
TAG="siglip2_base"
DEVICE="cuda"
DATA="$ROOT/data/factoratlas"

usage() {
  cat <<EOF
Usage: bash scripts/run_factoratlas.sh [options]

  --model MODEL   Hugging Face vision-language model ID (default: $MODEL)
  --tag TAG       local feature-cache and result name (default: $TAG)
  --device DEVICE torch device, e.g. cuda or cpu (default: $DEVICE)
  --data PATH     local FactorAtlas directory (default: $DATA)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

hf download 6uvsoomJ/FactorAtlas --repo-type dataset --local-dir "$DATA"

mkdir -p "$DATA/images"
shards=("$DATA"/data/factoratlas-*.tar)
if [[ ! -e "${shards[0]}" ]]; then
  echo "No FactorAtlas image shards found under $DATA/data." >&2
  exit 1
fi
for shard in "${shards[@]}"; do
  tar -xf "$shard" -C "$DATA/images"
done

python "$ROOT/experiments/access_gap/embed_hard_nuisance_backbone.py" \
  --model "$MODEL" --tag "$TAG" --device "$DEVICE" --data "$DATA"
python "$ROOT/experiments/access_gap/factor_access_audit.py" \
  --tag "$TAG" --data "$DATA" --out "$DATA/results"

echo "FactorAtlas report: $DATA/results/$TAG.json"
