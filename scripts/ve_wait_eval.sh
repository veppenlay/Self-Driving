#!/usr/bin/env bash
set -euo pipefail
while pgrep -f 'locked_model/evaluate.py' >/dev/null; do
  echo "still_running $(date -Iseconds)"
  pgrep -af 'locked_model/evaluate.py' || true
  sleep 30
done
echo DONE
echo '===LOCKED==='
cat /root/autodl-tmp/v-Net_cursor/locked_model/evaluation/ve_locked_recheck/summary_test.json 2>&1 || true
echo '===NEW==='
cat /root/autodl-tmp/v-Net_cursor/locked_model/evaluation/ve_newtrain_recheck/summary_test.json 2>&1 || true
ls -la /root/autodl-tmp/v-Net_cursor/locked_model/evaluation/ve_locked_recheck/ /root/autodl-tmp/v-Net_cursor/locked_model/evaluation/ve_newtrain_recheck/ 2>&1 || true
