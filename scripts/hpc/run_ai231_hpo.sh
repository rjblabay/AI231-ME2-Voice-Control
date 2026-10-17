#!/usr/bin/env bash
# Validation-only HPO.  The untouched offline test set is evaluated once,
# after choosing the best trial by validation macro-F1.
set -euo pipefail
ROOT="$HOME/ME2-Voice-Control"
PYTHON="$HOME/ME2-Voice-Control-Old/.hpc-venv/bin/python"
GPU="${CUDA_VISIBLE_DEVICES:-4}"
cd "$ROOT"
mkdir -p logs/hpc
for CONFIG in configs/ai231_hpo_capacity.yaml configs/ai231_hpo_focal.yaml configs/ai231_hpo_nosampler.yaml; do
  NAME="$(basename "$CONFIG" .yaml | tr '[:lower:]' '[:upper:]')_$(date +%Y%m%d_%H%M%S)"
  LOG="logs/hpc/${NAME}.log"
  {
    echo "[hpo] config=$CONFIG experiment=$NAME gpu=$GPU started=$(date -Is)"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m src.train --config "$CONFIG" --exp-id "$NAME"
    echo "[hpo] validation-only completed=$(date -Is)"
  } 2>&1 | tee "$LOG"
done
