#!/usr/bin/env bash
# Reproduce the public command-model workflow from the citable HF dataset.
# This intentionally excludes the live holdout split from training.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"
DATASET="airimonda/ai231-me2-voice-commands"
RAW_DIR="data/raw/ai231_me2_voice_commands"
EXP_ID="${EXP_ID:-AI231_REPRO_$(date +%Y%m%d_%H%M%S)}"

if [[ "${1:-}" == "--download" ]]; then
  command -v hf >/dev/null || {
    echo "Hugging Face CLI is required for --download. Install huggingface_hub first." >&2
    exit 1
  }
  mkdir -p "$RAW_DIR"
  # Required train/test and synthetic-negative shards only. The live holdout is
  # deliberately neither trained on nor copied into this repository.
  hf download "$DATASET" --repo-type dataset --local-dir "$RAW_DIR" \
    variations.csv \
    data/train-00000-of-00002.parquet data/train-00001-of-00002.parquet \
    data/test-00000-of-00001.parquet \
    synthetic_negatives/train-00000-of-00001.parquet \
    synthetic_negatives/test-00000-of-00001.parquet
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv "$VENV"
  "$VENV/bin/pip" install -r requirements.txt
fi

"$VENV/bin/python" scripts/prepare_ai231_commands.py --force
"$VENV/bin/python" -m src.train --config configs/ai231_hpo_nosampler.yaml --exp-id "$EXP_ID"
"$VENV/bin/python" -m src.evaluate \
  --checkpoint "models/checkpoints/$EXP_ID/best.pt" \
  --config configs/ai231_hpo_nosampler.yaml --split test --exp-id "$EXP_ID"
"$VENV/bin/python" -m src.export_onnx \
  --checkpoint "models/checkpoints/$EXP_ID/best.pt" \
  --output "models/onnx/$EXP_ID.onnx"

echo "Reproduction complete: $EXP_ID"
