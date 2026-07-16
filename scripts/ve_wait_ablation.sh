#!/usr/bin/env bash
set -euo pipefail
LOG=/root/autodl-tmp/v-Net_cursor/locked_model/runs/ablation_roi_chain_e150/ablation.log
SUM=/root/autodl-tmp/v-Net_cursor/locked_model/runs/ablation_roi_chain_e150/comparison_summary.json
while [[ ! -f "$SUM" ]]; do
  echo "waiting $(date -Iseconds)"
  tail -n 3 "$LOG" 2>/dev/null || true
  sleep 45
done
echo DONE
cat "$SUM"
tail -n 40 "$LOG"
