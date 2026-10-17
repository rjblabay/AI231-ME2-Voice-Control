#!/usr/bin/env bash
# Submit/run manually from the UP HPC login node after data preparation.
set -euo pipefail
ROOT="$HOME/ME2-Voice-Control"
PYTHON="$HOME/ME2-Voice-Control-Old/.hpc-venv/bin/python"
EXP_ID="${1:-AI231_CRNN_BIGRU_$(date +%Y%m%d_%H%M%S)}"
cd "$ROOT"
mkdir -p logs/hpc
LOG="logs/hpc/${EXP_ID}.log"
{
  echo "[run] experiment=$EXP_ID host=$(hostname) started=$(date -Is)"
  nvidia-smi || true
  "$PYTHON" -m src.train --config configs/ai231_crnn_bigru.yaml --exp-id "$EXP_ID"
  "$PYTHON" -m src.evaluate --checkpoint "models/checkpoints/${EXP_ID}/best.pt" --config configs/ai231_crnn_bigru.yaml --split test --exp-id "$EXP_ID"
  mkdir -p models/onnx
  "$PYTHON" -m src.export_onnx --checkpoint "models/checkpoints/${EXP_ID}/best.pt" --output "models/onnx/${EXP_ID}.onnx"
  echo "[run] completed=$(date -Is)"
} 2>&1 | tee "$LOG"
