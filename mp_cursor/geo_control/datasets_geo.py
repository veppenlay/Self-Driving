#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset yielding geometric affordance targets [steering, e_y, theta] with augmentation.

Reads `labels_geo.csv` (from `generate_geo_labels.py`). Temporal frame stacking mirrors
`locked_model/datasets.py`. Photometric augmentation (train only) is applied to the BGR
frames BEFORE preprocessing, using the geometry-preserving `PhotometricAugmentor`.

Also supports an optional held-out perturbation axis for the stress evaluation (applied to
ALL frames deterministically, no training randomness).

Needs torch + cv2 (runs on the GPU/dev environment).
"""

from __future__ import annotations

import csv
import re
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

from steering_preprocess import PreprocessConfig, imread_bgr, preprocess_bgr_to_tensor  # type: ignore  # noqa: E402

from .augment import PhotometricAugmentor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _frame_index(path: Path) -> int:
    match = re.match(r"^(\d+)_", path.name)
    if not match:
        raise ValueError(f"image filename does not start with a frame index: {path}")
    return int(match.group(1))


def _frame_candidate(parent: Path, frame_index: int) -> Path | None:
    for ext in IMAGE_EXTS:
        matches = sorted(parent.glob(f"{frame_index}_*{ext}"))
        if matches:
            return matches[0]
    return None


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    val = row.get(key, None)
    if val in (None, "", "None"):
        return float(default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


class AffordanceDataset(Dataset):
    def __init__(
        self,
        label_csv: str | Path,
        *,
        split: str,
        preprocess: PreprocessConfig,
        num_frames: int = 3,
        frame_stride: int = 1,
        augmentor: PhotometricAugmentor | None = None,
        perturb_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        frame_drop_rate: float = 0.0,
        include_zero_quality: bool = True,
    ):
        self.label_csv = Path(label_csv).resolve()
        self.split = split
        self.preprocess = preprocess
        self.num_frames = max(1, int(num_frames))
        self.frame_stride = max(1, int(frame_stride))
        self.augmentor = augmentor
        self.perturb_fn = perturb_fn
        self.frame_drop_rate = float(frame_drop_rate)
        self.rows: list[dict[str, Any]] = []
        self._drop_rng = np.random.default_rng(0)

        with self.label_csv.open("r", encoding="utf-8-sig", newline="") as fh:
            for raw in csv.DictReader(fh):
                if split != "all" and str(raw.get("split", "")).lower() != split.lower():
                    continue
                ey_quality = _f(raw, "eyQuality", 0.0)
                if not include_zero_quality and ey_quality <= 0:
                    continue
                image = Path(raw["image"])
                self.rows.append({
                    "image": image,
                    "steering": _f(raw, "steering", 0.0),
                    "e_y": _f(raw, "eY", 0.0),
                    "theta": _f(raw, "theta", 0.0),
                    "ey_quality": ey_quality,
                    "theta_quality": _f(raw, "thetaQuality", 0.0),
                    "status": raw.get("status", ""),
                    "sequence": raw.get("sequence", ""),
                    "frame": int(_f(raw, "frame", _frame_index(image))),
                })
        if not self.rows:
            raise FileNotFoundError(f"no rows for split={split!r} in {self.label_csv}")

    def _resolve_frame_paths(self, image_path: Path) -> list[Path]:
        current_index = _frame_index(image_path)
        parent = image_path.parent
        paths: list[Path] = []
        last_valid = image_path
        for offset in range(self.num_frames - 1, -1, -1):
            target_index = current_index - offset * self.frame_stride
            candidate = _frame_candidate(parent, target_index) if target_index >= 0 else None
            frame_path = candidate if candidate is not None else last_valid
            paths.append(frame_path)
            last_valid = frame_path
        return paths

    def __getitem__(self, index: int):
        row = self.rows[index]
        bgr_frames: list[np.ndarray] = []
        for frame_path in self._resolve_frame_paths(row["image"]):
            bgr = imread_bgr(frame_path)
            if bgr is None:
                raise FileNotFoundError(f"failed to read image: {frame_path}")
            bgr_frames.append(bgr)

        if self.augmentor is not None:
            bgr_frames = self.augmentor.augment_stack(bgr_frames)
        if self.perturb_fn is not None:
            bgr_frames = [self.perturb_fn(f) for f in bgr_frames]
        if self.frame_drop_rate > 0.0:
            held: list[np.ndarray] = []
            prev = None
            for f in bgr_frames:
                if prev is not None and self._drop_rng.random() < self.frame_drop_rate:
                    held.append(prev)
                else:
                    held.append(f)
                    prev = f
            bgr_frames = held

        frame_tensors = [preprocess_bgr_to_tensor(f, config=self.preprocess).squeeze(0) for f in bgr_frames]
        image_tensor = torch.cat(frame_tensors, dim=0)
        label = torch.tensor([row["steering"], row["e_y"], row["theta"]], dtype=torch.float32)
        meta = {
            "eyQuality": torch.tensor(row["ey_quality"], dtype=torch.float32),
            "thetaQuality": torch.tensor(row["theta_quality"], dtype=torch.float32),
            "path": str(row["image"]),
            "status": row["status"],
            "sequence": row["sequence"],
            "frame": row["frame"],
        }
        return image_tensor, label, meta

    def __len__(self) -> int:
        return len(self.rows)


__all__ = ["AffordanceDataset"]
