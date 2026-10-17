#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${HOME}/ME2-Voice-Control-Old/.hpc-venv"
EXP_ID="${1:-ASTRA_CRNN_BIGRU_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$ROOT"
PYTHON="$VENV/bin/python"
test -x "$PYTHON"
mkdir -p "logs/hpc" "models/onnx"
LOG="logs/hpc/${EXP_ID}.log"
exec > >(tee -a "$LOG") 2>&1
echo "[hpc] experiment=$EXP_ID gpu=$CUDA_VISIBLE_DEVICES started=$(date -Is)"
"$PYTHON" -m src.train --config configs/astra_crnn_bigru.yaml --exp-id "$EXP_ID"
CKPT="models/checkpoints/${EXP_ID}/best.pt"
"$PYTHON" -m src.evaluate --checkpoint "$CKPT" --config configs/astra_crnn_bigru.yaml --split test --exp-id "$EXP_ID"
"$PYTHON" -m src.export_onnx --checkpoint "$CKPT" --output "models/onnx/${EXP_ID}.onnx"
sha256sum "$CKPT" "models/onnx/${EXP_ID}.onnx" > "results/astra_wake/${EXP_ID}/artifact_sha256.txt"
echo "[hpc] complete=$(date -Is)"
