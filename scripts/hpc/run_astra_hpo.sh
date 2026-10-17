#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${HOME}/ME2-Voice-Control-Old/.hpc-venv/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$ROOT"
mkdir -p logs/hpc
for cfg in astra_hpo_syn80 astra_hpo_syn160 astra_hpo_syn240; do
  exp="${cfg^^}_$(date +%Y%m%d_%H%M%S)"
  echo "[hpo] $exp" | tee "logs/hpc/${exp}.log"
  "$PYTHON" -m src.train --config "configs/${cfg}.yaml" --exp-id "$exp" 2>&1 | tee -a "logs/hpc/${exp}.log"
done
