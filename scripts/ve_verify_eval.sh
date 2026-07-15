#!/usr/bin/env bash
# CPU-safe verify on no-card (2GB cgroup): single-thread, small batch.
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/vnet
cd /root/autodl-tmp/v-Net

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "==== LOCKED CKPT ===="
python locked_model/evaluate.py \
  --ckpt locked_model/checkpoints/best_seq_cfc_temporal3_2d.pth \
  --label-csv locked_model/labels/remote/labels_2d.csv \
  --split test \
  --output-dir locked_model/evaluation/ve_locked_recheck \
  --batch-size 4 \
  --num-workers 0 \
  --device cpu

echo "==== NEW TRAIN CKPT ===="
python locked_model/evaluate.py \
  --ckpt locked_model/runs/ve_seq_cfc_temporal3_2d/best_seq_cfc_temporal3_2d.pth \
  --label-csv locked_model/labels/remote/labels_2d.csv \
  --split test \
  --output-dir locked_model/evaluation/ve_newtrain_recheck \
  --batch-size 4 \
  --num-workers 0 \
  --device cpu

echo "==== RESULTS ===="
echo "local_baseline_steeringMAE=0.2045544220148927"
echo "---locked---"
cat locked_model/evaluation/ve_locked_recheck/summary_test.json
echo "---newtrain---"
cat locked_model/evaluation/ve_newtrain_recheck/summary_test.json
