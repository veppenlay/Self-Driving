#!/usr/bin/env bash
# Exploration shots on ve: D1 → B0 → A2 (single GPU, sequential).
# Data boundary: train/val=26合并, test=25地下室 via labels/remote.
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

mkdir -p "${MP}"
{
  echo "======== EXPLORE START $(date -Iseconds) ========"
  echo "labels=${LABEL_CSV}"
  python - <<'PY'
import torch
assert torch.cuda.is_available(), "need GPU"
print("device", torch.cuda.get_device_name(0))
PY

  run_one() {
    local name="$1"
    local script="$2"
    local out="${MP}/${name}"
    local extra=()
    if [[ $# -ge 3 ]]; then
      extra+=(--init-checkpoint "$3")
    fi
    echo ""
    echo "======== ${name} $(date -Iseconds) ========"
    mkdir -p "${out}"
    python "mp_cursor/challengers/${script}" \
      --label-csv "${LABEL_CSV}" \
      --output-dir "${out}" \
      --epochs 150 \
      --batch-size 32 \
      --num-workers 8 \
      --device cuda \
      "${extra[@]}"
    echo "======== ${name} DONE $(date -Iseconds) ========"
    cat "${out}/verdict.md"
  }

  # D1: cold ImageNet-adapted stem (4ch); no locked init (channel mismatch)
  run_one "exp_D1_framediff" "d1_framediff.py"

  # B0: optional warm start from locked (reg_head size mismatch → partial load)
  run_one "exp_B0_nll" "b0_nll.py" "${LOCKED_CKPT}"

  # A2: warm start from locked preferred
  run_one "exp_A2_temporal_ln" "a2_temporal_ln.py" "${LOCKED_CKPT}"

  echo ""
  echo "======== SUMMARY ========"
  for d in exp_D1_framediff exp_B0_nll exp_A2_temporal_ln; do
    echo "--- ${d} ---"
    cat "${MP}/${d}/verdict.md" || true
  done
  echo "======== EXPLORE DONE $(date -Iseconds) ========"
} > "${LOG}" 2>&1
