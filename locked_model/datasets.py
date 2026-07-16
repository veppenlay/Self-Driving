#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""2D steering/e_y dataset with temporal frame stacking."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from steering_preprocess import PreprocessConfig, imread_bgr, preprocess_bgr_to_tensor


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


class AutoDrive2DDataset(Dataset):
    def __init__(
        self,
        label_csv: str | Path,
        *,
        split: str,
        preprocess: PreprocessConfig,
        num_frames: int = 3,
        frame_stride: int = 1,
        include_zero_quality: bool = True,
    ):
        self.label_csv = Path(label_csv).resolve()
        self.split = split
        self.preprocess = preprocess
        self.num_frames = max(1, int(num_frames))
        self.frame_stride = max(1, int(frame_stride))
        self.rows: list[dict[str, Any]] = []

        with self.label_csv.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                if split != "all" and str(raw.get("split", "")).lower() != split.lower():
                    continue
                ey_quality = float(raw.get("eyQuality") or 0.0)
                if not include_zero_quality and ey_quality <= 0:
                    continue
                image = Path(raw["image"])
                e_y_text = raw.get("eY", "")
                e_y = float(e_y_text) if e_y_text not in ("", "None", None) else 0.0
                self.rows.append(
                    {
                        "image": image,
                        "steering": float(raw["steering"]),
                        "e_y": e_y,
                        "ey_quality": ey_quality,
                        "status": raw.get("status", ""),
                        "sequence": raw.get("sequence", ""),
                        "frame": int(raw.get("frame") or _frame_index(image)),
                    }
                )

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
        frame_tensors = []
        for frame_path in self._resolve_frame_paths(row["image"]):
            bgr = imread_bgr(frame_path)
            if bgr is None:
                raise FileNotFoundError(f"failed to read image: {frame_path}")
            frame_tensors.append(preprocess_bgr_to_tensor(bgr, config=self.preprocess).squeeze(0))
        image_tensor = torch.cat(frame_tensors, dim=0)
        label = torch.tensor([row["steering"], row["e_y"]], dtype=torch.float32)
        meta = {
            "eyQuality": torch.tensor(row["ey_quality"], dtype=torch.float32),
            "path": str(row["image"]),
            "status": row["status"],
            "sequence": row["sequence"],
            "frame": row["frame"],
        }
        return image_tensor, label, meta

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def steering_values(self) -> list[float]:
        return [float(row["steering"]) for row in self.rows]
