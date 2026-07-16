#!/usr/bin/env bash
# CPU-safe readiness check before flipping to GPU billing.
set -euo pipefail

REPO="/root/autodl-tmp/v-Net_cursor"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/vnet_cursor
cd "$REPO"

python - <<'PY'
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "locked_model")
from datasets import AutoDrive2DDataset
from steering_preprocess import PreprocessConfig

label_csv = Path("locked_model/labels/remote/labels_2d.csv")
assert label_csv.is_file(), label_csv
ds = AutoDrive2DDataset(
    label_csv,
    split="train",
    preprocess=PreprocessConfig(),
    num_frames=3,
    frame_stride=1,
)
assert len(ds) > 0, "empty train split"
image, label, meta = ds[0]
print("sample_ok", tuple(image.shape), tuple(label.tolist()), meta["path"])
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
batch = next(iter(loader))
print("batch_ok", tuple(batch[0].shape), tuple(batch[1].shape))
print("cuda_now", torch.cuda.is_available())
print("READY_FOR_GPU_TRAIN")
PY
