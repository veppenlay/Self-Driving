#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/v-Net_cursor
PY=/root/autodl-tmp/envs/vnet_cursor/bin/python
LABELS=locked_model/labels/remote/labels_2d.csv
HMM=epaicar_deploy/steering_only_hmm_online_model.json

evaluate_one() {
  local experiment_id="$1"
  local run_dir="$2"
  local eval_dir="${run_dir}/basement_eval"
  mkdir -p "$eval_dir/raw" "$eval_dir/hmm"
  "$PY" locked_model/evaluate.py \
    --ckpt "${run_dir}/best_seq_cfc_temporal3_2d.pth" \
    --label-csv "$LABELS" --split test --batch-size 32 \
    --output-dir "$eval_dir/raw"
  "$PY" locked_model/apply_current_hmm_postprocess.py \
    --model-json "$HMM" \
    --input-pred-csv "$eval_dir/raw/predictions_test.csv" \
    --output-dir "$eval_dir/hmm"
  "$PY" mp_codex/scripts/summarize_predictions.py \
    --raw-csv "$eval_dir/raw/predictions_test.csv" \
    --hmm-csv "$eval_dir/hmm/current_postprocess_predictions.csv" \
    --output "$eval_dir/comparison.json" \
    --experiment-id "$experiment_id"
}

evaluate_one W0-warm-control mp_codex/runs/w0_warm_control_seed20260715
evaluate_one W1-warm-photometric mp_codex/runs/w1b_warm_photo_fast_seed20260715
