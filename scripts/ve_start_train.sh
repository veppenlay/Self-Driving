#!/usr/bin/env bash
# Start Seq-CfC Temporal3 2D training on ve (run after switching to GPU instance).
set -euo pipefail

REPO="/root/autodl-tmp/v-Net"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/vnet
cd "$REPO"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable. Switch AutoDL to a GPU instance before training.")
print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
PY

LABEL_CSV="locked_model/labels/remote/labels_2d.csv"
OUT_DIR="locked_model/runs/ve_seq_cfc_temporal3_2d"
if [[ ! -f "$LABEL_CSV" ]]; then
  echo "missing $LABEL_CSV — run scripts/ve_prepare_labels.sh first" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
exec python locked_model/train.py \
  --label-csv "$LABEL_CSV" \
  --output-dir "$OUT_DIR" \
  --model-variant seq_cfc_temporal3 \
  --num-frames 3 \
  --epochs 80 \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda
