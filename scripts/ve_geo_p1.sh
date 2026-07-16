#!/usr/bin/env bash
# P1 (preferred): train affordance model (e_y/theta) with strong augmentation, then run the
# basement clean + held-out perturbation stress evaluation. Train/val = 26合并; test = 25地下室.
set -euo pipefail

REPO="/root/autodl-tmp/v-Net_cursor"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/vnet_cursor
cd "$REPO"

GEO_LABELS="mp_cursor/geo_control/labels/labels_geo.csv"
EXP="mp_cursor/exp_P1_affordance"
CTRL="mp_cursor/geo_control/frozen/controller_pid.json"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable. Switch to a GPU instance before training.")
print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
PY

mkdir -p "$EXP/checkpoints" "$EXP/eval"

python -m mp_cursor.geo_control.train_affordance \
  --label-csv "$GEO_LABELS" \
  --output-dir "$EXP/checkpoints" \
  --epochs 150 --batch-size 32 --num-workers 8 --device cuda \
  --aug-strength 1.0

python -m mp_cursor.geo_control.infer_affordance \
  --ckpt "$EXP/checkpoints/best_affordance_p1.pth" \
  --label-csv "$GEO_LABELS" --split test \
  --controller "$CTRL" \
  --output-dir "$EXP/eval" --device cuda --stress

echo "P1 done. Clean metrics + stress degradation in $EXP/eval"
