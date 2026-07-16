#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Remote smoke check for warm-start augmentation and checkpoint compatibility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "mp_codex" / "src", REPO_ROOT / "locked_model"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models import AutoDriveNetSeqCfC  # type: ignore  # noqa: E402
from steering_preprocess import PreprocessConfig  # type: ignore  # noqa: E402
from warm_dataset import WarmAutoDrive2DDataset  # noqa: E402
from warm_photometric import WarmPhotometricAugmentor  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label-csv", required=True)
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()
    preprocess = PreprocessConfig(color_space="hsv", input_size=(144, 192), use_roi=True, illumination_profile="none")
    common = dict(label_csv=args.label_csv, split="train", preprocess=preprocess, num_frames=3, frame_stride=1)
    plain = WarmAutoDrive2DDataset(**common)
    augmented = WarmAutoDrive2DDataset(**common, augmentor=WarmPhotometricAugmentor(seed=20260715))
    x0, y0, _ = plain[0]
    x1, y1, _ = augmented[0]
    assert x0.shape == x1.shape == (9, 144, 192)
    assert torch.equal(y0, y1)
    assert torch.isfinite(x1).all()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = AutoDriveNetSeqCfC(num_frames=3)
    model.load_state_dict(checkpoint["model"], strict=True)
    with torch.no_grad():
        output = model(x0.unsqueeze(0))
    print(f"shape={tuple(x1.shape)} mean_abs_change={float((x1-x0).abs().mean()):.6f} output={output.tolist()}")


if __name__ == "__main__":
    main()
