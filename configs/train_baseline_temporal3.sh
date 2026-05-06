#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-dataset/temporal3_data}"

python train.py \
  --data_folder "$DATA_DIR" \
  --model_variant temporal3 \
  --num_frames 3 \
  --frame_stride 1 \
  --epochs "${EPOCHS:-80}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --output_dir runs/baseline_3frame_temporal_v2 \
  --save_name baseline_3frame_temporal_v2.pth \
  --best_save_name best_baseline_3frame_temporal_v2.pth
