#!/usr/bin/env bash
# Timed training wrapper for ve GPU instance.
set -euo pipefail

REPO="/root/autodl-tmp/v-Net_cursor"
OUT_DIR="${REPO}/locked_model/runs/ve_seq_cfc_temporal3_2d"
LOG="${OUT_DIR}/train.log"
CLOCK="${OUT_DIR}/wall_clock.txt"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/vnet_cursor
cd "${REPO}"
mkdir -p "${OUT_DIR}"

START="$(date +%s)"
START_ISO="$(date -Iseconds)"
{
  echo "START_ISO=${START_ISO}"
  bash "${REPO}/scripts/ve_start_train.sh"
  EC=$?
  END="$(date +%s)"
  END_ISO="$(date -Iseconds)"
  ELAPSED="$((END - START))"
  echo "END_ISO=${END_ISO}"
  echo "EXIT_CODE=${EC}"
  echo "ELAPSED_SEC=${ELAPSED}"
  cat > "${CLOCK}" <<EOF
START_ISO=${START_ISO}
END_ISO=${END_ISO}
ELAPSED_SEC=${ELAPSED}
EXIT_CODE=${EC}
EOF
  exit "${EC}"
} > "${LOG}" 2>&1
