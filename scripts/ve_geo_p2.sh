#!/usr/bin/env bash
# P2 (diagnostic / fallback only, NOT a robustness candidate): quantify the fixed-threshold
# CV detector's fragility on the basement, clean + held-out perturbation. HSV thresholds are
# NOT tuned; width prior is the frozen source-domain constant.
set -euo pipefail

REPO="/root/autodl-tmp/v-Net_cursor"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/vnet_cursor
cd "$REPO"

GEO_LABELS="mp_cursor/geo_control/labels/labels_geo.csv"
EXP="mp_cursor/exp_P2_cv"
CTRL="mp_cursor/geo_control/frozen/controller_pid.json"

mkdir -p "$EXP/eval"

python -m mp_cursor.geo_control.diagnose_cv \
  --label-csv "$GEO_LABELS" --split test \
  --width-prior 411.0 --hold-frames 5 \
  --controller "$CTRL" \
  --output-dir "$EXP/eval" --stress

echo "P2 diagnostic done. CV fragility in $EXP/eval/cv_fragility.json"
