#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Locked 2D dataset with optional training-only stack-consistent augmentation."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_DIR = REPO_ROOT / "locked_model"
if str(LOCKED_DIR) not in sys.path:
    sys.path.insert(0, str(LOCKED_DIR))

from datasets import AutoDrive2DDataset  # type: ignore  # noqa: E402
from steering_preprocess import apply_bottom_roi, imread_bgr, preprocess_bgr_to_tensor  # type: ignore  # noqa: E402

from warm_photometric import WarmPhotometricAugmentor


class WarmAutoDrive2DDataset(AutoDrive2DDataset):
    def __init__(self, *args, augmentor: WarmPhotometricAugmentor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentor = augmentor

    def __getitem__(self, index: int):
        row = self.rows[index]
        frames = []
        for frame_path in self._resolve_frame_paths(row["image"]):
            bgr = imread_bgr(frame_path)
            if bgr is None:
                raise FileNotFoundError(f"failed to read image: {frame_path}")
            frames.append(bgr)
        tensor_preprocess = self.preprocess
        if self.augmentor is not None:
            # Photometric transforms are geometry preserving, so execute them at the
            # exact locked-model input resolution.  This is equivalent to augmenting
            # the full ROI before its final resize and removes the CPU bottleneck.
            height, width = self.preprocess.input_size
            frames = [
                cv2.resize(
                    apply_bottom_roi(frame, config=self.preprocess),
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )
                for frame in frames
            ]
            frames = self.augmentor.augment_stack(frames)
            tensor_preprocess = replace(self.preprocess, use_roi=False)
        tensors = [preprocess_bgr_to_tensor(frame, config=tensor_preprocess).squeeze(0) for frame in frames]
        label = torch.tensor([row["steering"], row["e_y"]], dtype=torch.float32)
        meta = {
            "eyQuality": torch.tensor(row["ey_quality"], dtype=torch.float32),
            "path": str(row["image"]),
            "status": row["status"],
            "sequence": row["sequence"],
            "frame": row["frame"],
        }
        return torch.cat(tensors, dim=0), label, meta


__all__ = ["WarmAutoDrive2DDataset"]
