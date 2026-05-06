#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-dataset/temporal3_data}"
TEACHER_CKPT="${TEACHER_CKPT:-checkpoints/baseline_3frame_temporal_v2/best_baseline_3frame_temporal_v2.pth}"

python train.py \
  --data_folder "$DATA_DIR" \
  --model_variant temporal3 \
  --num_frames 3 \
  --frame_stride 1 \
  --epochs "${EPOCHS:-80}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --use_distillation \
  --teacher_ckpt "$TEACHER_CKPT" \
  --distill_weight 0.10 \
  --output_dir runs/distill_temporal3_w010 \
  --save_name distill_temporal3_w010.pth \
  --best_save_name best_distill_temporal3_w010.pth
