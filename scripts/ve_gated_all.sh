#!/usr/bin/env bash
# Gated one-shot exploration+confirmation on ve (single GPU sequential).
#
# Gates (from 网络侧鲁棒性实验指导):
#   1) Always run missing B0, A2 (对照完整)
#   2) S-pack only if NONE of D1/B0/A2 晋级
#   3) F-lite skipped here (needs pseudo-label pipeline) — logged if gate reached
#   4) For every 晋级 model: mini OOD confirmation (dark/blur/drop)
set -euo pipefail

REPO="/root/autodl-tmp/v-Net_cursor"
ENV="/root/autodl-tmp/envs/vnet_cursor"
LABEL_CSV="${REPO}/locked_model/labels/remote/labels_2d.csv"
LOCKED_CKPT="${REPO}/locked_model/checkpoints/best_seq_cfc_temporal3_2d.pth"
MP="${REPO}/mp_cursor"
LOG="${MP}/gated_all.log"
BASELINE_HMM="0.167248"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV}"
cd "${REPO}"
export PYTHONPATH="${REPO}:${REPO}/locked_model:${PYTHONPATH:-}"

promoted() {
  local v="$1/verdict.md"
  [[ -f "$v" ]] && grep -q "结论：晋级" "$v"
}

run_train() {
  local name="$1"
  local script="$2"
  shift 2
  local out="${MP}/${name}"
  if [[ -f "${out}/verdict.md" ]]; then
    echo "[SKIP train] ${name} already has verdict"
    return 0
  fi
  echo "======== TRAIN ${name} $(date -Iseconds) ========"
  mkdir -p "${out}"
  python "mp_cursor/challengers/${script}" \
    --label-csv "${LABEL_CSV}" \
    --output-dir "${out}" \
    --epochs 150 --batch-size 32 --num-workers 8 --device cuda \
    "$@"
  cat "${out}/verdict.md"
}

run_confirm() {
  local name="$1"
  local exp_key="$2"
  local out="${MP}/${name}"
  if [[ ! -f "${out}/verdict.md" ]]; then
    return 0
  fi
  if [[ -f "${out}/confirmation/ood_mini/summary.json" ]]; then
    echo "[SKIP confirm] ${name}"
    return 0
  fi
  if ! grep -q "结论：晋级" "${out}/verdict.md"; then
    echo "[SKIP confirm] ${name} not 晋级"
    return 0
  fi
  echo "======== CONFIRM mini-OOD ${name} $(date -Iseconds) ========"
  python mp_cursor/challengers/confirm_mini_ood.py \
    --experiment "${exp_key}" \
    --label-csv "${LABEL_CSV}" \
    --output-dir "${out}" \
    --device cuda
}

{
  echo "======== GATED ALL START $(date -Iseconds) ========"
  python - <<'PY'
import torch
assert torch.cuda.is_available()
print("device", torch.cuda.get_device_name(0))
PY

  # --- Exploration ---
  # D1 may already exist (晋级)
  if [[ ! -f "${MP}/exp_D1_framediff/verdict.md" ]]; then
    run_train exp_D1_framediff d1_framediff.py
  else
    echo "[KEEP] exp_D1_framediff"
    cat "${MP}/exp_D1_framediff/verdict.md"
  fi

  run_train exp_B0_nll b0_nll.py --init-checkpoint "${LOCKED_CKPT}"
  run_train exp_A2_temporal_ln a2_temporal_ln.py --init-checkpoint "${LOCKED_CKPT}"

  ANY_WIN=0
  for d in exp_D1_framediff exp_B0_nll exp_A2_temporal_ln; do
    if promoted "${MP}/${d}"; then ANY_WIN=1; fi
  done

  if [[ "${ANY_WIN}" -eq 0 ]]; then
    echo "[GATE] no single-shot win → run S-pack (D1+B0)"
    # warm from D1 if available
    if [[ -f "${MP}/exp_D1_framediff/checkpoints/best.pth" ]]; then
      run_train exp_S_pack_D1_B0 s_pack_d1_b0.py --init-checkpoint "${MP}/exp_D1_framediff/checkpoints/best.pth"
    else
      run_train exp_S_pack_D1_B0 s_pack_d1_b0.py
    fi
    if promoted "${MP}/exp_S_pack_D1_B0"; then ANY_WIN=1; fi
  else
    echo "[GATE] at least one of D1/B0/A2 晋级 → skip S-pack"
  fi

  if [[ "${ANY_WIN}" -eq 0 ]]; then
    echo "[GATE] still no win → F-lite requires pseudo-label pipeline; SKIP with stub"
    mkdir -p "${MP}/exp_F_lite_stub"
    cat > "${MP}/exp_F_lite_stub/verdict.md" <<EOF
数据：train/val=26合并；test=25地下室
实验：F_lite
结论：跳过（未实现伪标签/辅助头流水线；需单独立项）
EOF
  else
    echo "[GATE] win exists → skip F-lite"
  fi

  # --- Confirmation for every 晋级 ---
  run_confirm exp_D1_framediff D1_framediff
  run_confirm exp_B0_nll B0_nll
  run_confirm exp_A2_temporal_ln A2_temporal_ln
  run_confirm exp_S_pack_D1_B0 S_pack_D1_B0

  echo ""
  echo "======== FINAL SUMMARY ========"
  python - <<'PY'
from pathlib import Path
import re, json
mp = Path("/root/autodl-tmp/v-Net_cursor/mp_cursor")
rows = []
for d in sorted(mp.glob("exp_*")):
    v = d / "verdict.md"
    if not v.is_file():
        continue
    text = v.read_text(encoding="utf-8")
    hmm = re.search(r"HMM MAE = ([0-9.]+)", text)
    raw = re.search(r"RAW MAE = ([0-9.]+)", text)
    verdict = "晋级" if "结论：晋级" in text else ("跳过" if "跳过" in text else "淘汰")
    rows.append((d.name, raw.group(1) if raw else "-", hmm.group(1) if hmm else "-", verdict))
print(f"{'exp':28} {'RAW':10} {'HMM':10} {'verdict'}")
for r in rows:
    print(f"{r[0]:28} {r[1]:10} {r[2]:10} {r[3]}")
summary = {"baselineHmm": 0.167248, "results": [{"exp": a, "raw": b, "hmm": c, "verdict": d} for a,b,c,d in rows]}
(mp / "gated_all_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  echo "======== GATED ALL DONE $(date -Iseconds) ========"
} > "${LOG}" 2>&1
