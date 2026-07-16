#!/usr/bin/env bash
# Prepare geometry-control exploration: derive theta labels + calibrate/freeze controllers.
# Run on the ve GPU/dev instance. Uses the SOURCE domain (26合并) only for calibration.
set -euo pipefail

REPO="/root/autodl-tmp/v-Net_cursor"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/vnet_cursor
cd "$REPO"

BASE_LABELS="locked_model/labels/remote/labels_2d.csv"
GEO_LABELS="mp_cursor/geo_control/labels/labels_geo.csv"
FROZEN_DIR="mp_cursor/geo_control/frozen"

if [[ ! -f "$BASE_LABELS" ]]; then
  echo "missing $BASE_LABELS — run scripts/ve_prepare_labels.sh first" >&2
  exit 1
fi

# 1) Automatic heading (theta) labels, zero manual work (reuses locked CV detector).
python -m mp_cursor.geo_control.generate_geo_labels \
  --base-label-csv "$BASE_LABELS" \
  --output-csv "$GEO_LABELS" \
  --far-y-ratio 0.42

# 2) Calibrate + FREEZE controllers on the source domain (split=train) only.
python -m mp_cursor.geo_control.calibrate_controller \
  --label-csv "$GEO_LABELS" \
  --output-dir "$FROZEN_DIR" \
  --nominal-speed 0.5 --fps 30

echo "prepared geo labels -> $GEO_LABELS"
echo "frozen controllers  -> $FROZEN_DIR (recommended in calibration_summary.json)"
