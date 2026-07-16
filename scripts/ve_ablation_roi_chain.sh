#!/usr/bin/env bash
# Ablation on ve (GPU):
#   A) locked ROI best -> train non-ROI -> test (25basement via labels/remote)
#   B) non-ROI from scratch -> best; then ROI init from that best -> test
set -euo pipefail

REPO="/root/autodl-tmp/v-Net_cursor"
LABEL_CSV="${REPO}/locked_model/labels/remote/labels_2d.csv"
LOCKED_ROI="${REPO}/locked_model/checkpoints/best_seq_cfc_temporal3_2d.pth"
OUT_ROOT="${REPO}/locked_model/runs/ablation_roi_chain_e150"
LOG="${OUT_ROOT}/ablation.log"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/vnet_cursor
cd "${REPO}"
mkdir -p "${OUT_ROOT}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
PY

EPOCHS=150
BATCH=32
WORKERS=8
COMMON=(
  --label-csv "${LABEL_CSV}"
  --model-variant seq_cfc_temporal3
  --num-frames 3
  --epochs "${EPOCHS}"
  --batch-size "${BATCH}"
  --num-workers "${WORKERS}"
  --device cuda
  --early-stop
  --early-stop-patience 0.005
  --early-stop-patience-epochs 10
)

run_eval() {
  local ckpt="$1"
  local out_dir="$2"
  python locked_model/evaluate.py \
    --ckpt "${ckpt}" \
    --label-csv "${LABEL_CSV}" \
    --split test \
    --output-dir "${out_dir}" \
    --batch-size 16 \
    --num-workers 0 \
    --device cuda
}

{
  echo "======== ABLATION START $(date -Iseconds) ========"
  echo "labels=${LABEL_CSV}"
  echo "locked_roi=${LOCKED_ROI}"

  echo ""
  echo "======== EXP A: ROI-best -> non-ROI train ========"
  A_DIR="${OUT_ROOT}/A_roi_init_nonroi_train"
  A_START="$(date +%s)"
  python locked_model/train.py \
    "${COMMON[@]}" \
    --output-dir "${A_DIR}" \
    --init-checkpoint "${LOCKED_ROI}" \
    --no-use-roi \
    --save-name nonroi_from_roi.pth \
    --best-save-name best_nonroi_from_roi.pth
  A_END="$(date +%s)"
  echo "EXP_A_TRAIN_ELAPSED_SEC=$((A_END - A_START))"
  run_eval "${A_DIR}/best_nonroi_from_roi.pth" "${A_DIR}/eval_test"
  echo "EXP_A_TEST_SUMMARY=$(cat "${A_DIR}/eval_test/summary_test.json")"

  echo ""
  echo "======== EXP B1: non-ROI from scratch ========"
  B1_DIR="${OUT_ROOT}/B1_nonroi_scratch"
  B1_START="$(date +%s)"
  python locked_model/train.py \
    "${COMMON[@]}" \
    --output-dir "${B1_DIR}" \
    --no-use-roi \
    --save-name nonroi_scratch.pth \
    --best-save-name best_nonroi_scratch.pth
  B1_END="$(date +%s)"
  echo "EXP_B1_TRAIN_ELAPSED_SEC=$((B1_END - B1_START))"
  run_eval "${B1_DIR}/best_nonroi_scratch.pth" "${B1_DIR}/eval_test"
  echo "EXP_B1_TEST_SUMMARY=$(cat "${B1_DIR}/eval_test/summary_test.json")"

  echo ""
  echo "======== EXP B2: non-ROI-best -> ROI train ========"
  B2_DIR="${OUT_ROOT}/B2_roi_from_nonroi"
  B2_START="$(date +%s)"
  python locked_model/train.py \
    "${COMMON[@]}" \
    --output-dir "${B2_DIR}" \
    --init-checkpoint "${B1_DIR}/best_nonroi_scratch.pth" \
    --use-roi \
    --save-name roi_from_nonroi.pth \
    --best-save-name best_roi_from_nonroi.pth
  B2_END="$(date +%s)"
  echo "EXP_B2_TRAIN_ELAPSED_SEC=$((B2_END - B2_START))"
  run_eval "${B2_DIR}/best_roi_from_nonroi.pth" "${B2_DIR}/eval_test"
  echo "EXP_B2_TEST_SUMMARY=$(cat "${B2_DIR}/eval_test/summary_test.json")"

  echo ""
  echo "======== BASELINE LOCKED ROI (reference) ========"
  run_eval "${LOCKED_ROI}" "${OUT_ROOT}/ref_locked_roi/eval_test"

  python - <<'PY'
import json
from pathlib import Path

root = Path("/root/autodl-tmp/v-Net_cursor/locked_model/runs/ablation_roi_chain_e150")
paths = {
    "locked_roi_ref": root / "ref_locked_roi/eval_test/summary_test.json",
    "A_roi_init_nonroi": root / "A_roi_init_nonroi_train/eval_test/summary_test.json",
    "B1_nonroi_scratch": root / "B1_nonroi_scratch/eval_test/summary_test.json",
    "B2_roi_from_nonroi": root / "B2_roi_from_nonroi/eval_test/summary_test.json",
}
rows = {}
for name, path in paths.items():
    data = json.loads(path.read_text(encoding="utf-8"))
    rows[name] = {
        "steeringMAE": data["steeringMAE"],
        "eyWeightedMAE": data.get("eyWeightedMAE"),
        "count": data.get("count"),
        "useRoi": (data.get("preprocess") or {}).get("useRoi"),
        "checkpoint": data.get("checkpoint"),
    }
out = {
    "protocol": {
        "epochs": 150,
        "earlyStopPatienceMinDelta": 0.005,
        "earlyStopPatienceEpochs": 10,
        "batchSize": 32,
        "labelCsv": "/root/autodl-tmp/v-Net_cursor/locked_model/labels/remote/labels_2d.csv",
        "testSplit": "25basement via labels/remote test",
        "experiments": {
            "A": "init locked ROI best -> train --no-use-roi -> test",
            "B1": "ImageNet backbone + train --no-use-roi from scratch -> test",
            "B2": "init B1 best -> train --use-roi -> test",
            "locked_roi_ref": "evaluate locked ROI checkpoint (no retrain)",
        },
    },
    "testResults": rows,
}
(root / "comparison_summary.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(out["testResults"], ensure_ascii=False, indent=2))
PY

  echo "======== ABLATION DONE $(date -Iseconds) ========"
} > "${LOG}" 2>&1
