#!/usr/bin/env bash
set -euo pipefail
REPO="/root/autodl-tmp/v-Net_cursor"
ENV="/root/autodl-tmp/envs/vnet_cursor"
LABEL_CSV="${REPO}/locked_model/labels/remote/labels_2d.csv"
LOCKED_CKPT="${REPO}/locked_model/checkpoints/best_seq_cfc_temporal3_2d.pth"
MP="${REPO}/mp_cursor"
LOG="${MP}/explore_shots.log"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV}"
cd "${REPO}"
export PYTHONPATH="${REPO}:${REPO}/locked_model:${PYTHONPATH:-}"

{
  echo "======== EXPLORE RESUME $(date -Iseconds) ========"

  if [[ -f "${MP}/exp_D1_framediff/checkpoints/best.pth" && ! -f "${MP}/exp_D1_framediff/verdict.md" ]]; then
    echo "======== D1 eval-only ========"
    python mp_cursor/challengers/eval_only.py \
      --experiment D1_framediff \
      --label-csv "${LABEL_CSV}" \
      --output-dir "${MP}/exp_D1_framediff" \
      --device cuda
    cat "${MP}/exp_D1_framediff/verdict.md"
  elif [[ ! -f "${MP}/exp_D1_framediff/verdict.md" ]]; then
    python mp_cursor/challengers/d1_framediff.py \
      --label-csv "${LABEL_CSV}" --output-dir "${MP}/exp_D1_framediff" \
      --epochs 150 --batch-size 32 --num-workers 8 --device cuda
  fi

  if [[ ! -f "${MP}/exp_B0_nll/verdict.md" ]]; then
    echo "======== exp_B0_nll ========"
    python mp_cursor/challengers/b0_nll.py \
      --label-csv "${LABEL_CSV}" --output-dir "${MP}/exp_B0_nll" \
      --epochs 150 --batch-size 32 --num-workers 8 --device cuda \
      --init-checkpoint "${LOCKED_CKPT}"
    cat "${MP}/exp_B0_nll/verdict.md"
  fi

  if [[ ! -f "${MP}/exp_A2_temporal_ln/verdict.md" ]]; then
    echo "======== exp_A2_temporal_ln ========"
    python mp_cursor/challengers/a2_temporal_ln.py \
      --label-csv "${LABEL_CSV}" --output-dir "${MP}/exp_A2_temporal_ln" \
      --epochs 150 --batch-size 32 --num-workers 8 --device cuda \
      --init-checkpoint "${LOCKED_CKPT}"
    cat "${MP}/exp_A2_temporal_ln/verdict.md"
  fi

  echo "======== SUMMARY ========"
  for d in exp_D1_framediff exp_B0_nll exp_A2_temporal_ln; do
    echo "--- ${d} ---"
    cat "${MP}/${d}/verdict.md" || true
  done
  echo "======== EXPLORE DONE $(date -Iseconds) ========"
} >> "${LOG}" 2>&1
