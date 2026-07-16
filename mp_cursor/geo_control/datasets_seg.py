#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 segmentation dataset: image -> automatic CV yellow pseudo-mask (geometry-preserving aug).

The pseudo-mask target is computed from the CLEAN, ROI-cropped, resized frame using the locked
`_yellow_mask` (fully automatic). Photometric augmentation is applied to the INPUT only, so the
network learns to predict the true lane geometry under appearance shifts (learned invariance),
while the target stays geometrically valid.

Needs torch + cv2. Single frame per sample (144x192 network input, matching the locked ROI).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_DIR = REPO_ROOT / "locked_model"
if str(LOCKED_DIR) not in sys.path:
    sys.path.insert(0, str(LOCKED_DIR))

from steering_preprocess import PreprocessConfig, apply_bottom_roi, imread_bgr, preprocess_bgr_to_tensor  # type: ignore  # noqa: E402

from .augment import PhotometricAugmentor

INPUT_H, INPUT_W = 144, 192
_INPUT_CFG = PreprocessConfig(color_space="hsv", input_size=(INPUT_H, INPUT_W), use_roi=False, illumination_profile="none")
_ROI_CFG = PreprocessConfig(color_space="hsv", input_size=(INPUT_H, INPUT_W), use_roi=True, illumination_profile="none")


def _roi_resized_bgr(bgr: np.ndarray):
    import cv2

    roi = apply_bottom_roi(bgr, config=_ROI_CFG)
    return cv2.resize(roi, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)


class LaneSegDataset(Dataset):
    def __init__(
        self,
        label_csv: str | Path,
        *,
        split: str,
        augmentor: PhotometricAugmentor | None = None,
        perturb_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.label_csv = Path(label_csv).resolve()
        self.split = split
        self.augmentor = augmentor
        self.perturb_fn = perturb_fn
        self.rows: list[dict[str, Any]] = []
        self._yellow = None
        with self.label_csv.open("r", encoding="utf-8-sig", newline="") as fh:
            for raw in csv.DictReader(fh):
                if split != "all" and str(raw.get("split", "")).lower() != split.lower():
                    continue
                self.rows.append({
                    "image": Path(raw["image"]),
                    "sequence": raw.get("sequence", ""),
                    "frame": int(float(raw.get("frame") or 0)),
                })
        if not self.rows:
            raise FileNotFoundError(f"no rows for split={split!r} in {self.label_csv}")

    def _yellow_mask(self, bgr: np.ndarray) -> np.ndarray:
        if self._yellow is None:
            import generate_2d_labels as g2d  # type: ignore

            self._yellow = g2d._yellow_mask
        return self._yellow(bgr)

    def __getitem__(self, index: int):
        row = self.rows[index]
        bgr = imread_bgr(row["image"])
        if bgr is None:
            raise FileNotFoundError(f"failed to read image: {row['image']}")
        resized = _roi_resized_bgr(bgr)
        mask = (self._yellow_mask(resized) > 0).astype(np.float32)  # target from CLEAN frame

        frame_in = resized
        if self.augmentor is not None:
            frame_in = self.augmentor.augment_stack([resized])[0]
        if self.perturb_fn is not None:
            frame_in = self.perturb_fn(frame_in)

        image_tensor = preprocess_bgr_to_tensor(frame_in, config=_INPUT_CFG).squeeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        meta = {"path": str(row["image"]), "sequence": row["sequence"], "frame": row["frame"]}
        return image_tensor, mask_tensor, meta

    def __len__(self) -> int:
        return len(self.rows)


__all__ = ["LaneSegDataset", "INPUT_H", "INPUT_W"]
