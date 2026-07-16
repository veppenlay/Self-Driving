#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 lane-segmentation model: tiny decoder on the locked MobileNetV3 backbone.

Predicts a single-channel lane (yellow-line) probability mask. Trained with automatic CV
pseudo-masks (`_yellow_mask`) so labeling complexity is unchanged. The mask is later turned
into e_y/theta by `seg_geometry.py`, then fed to the frozen controller.

Aggregating the whole ROI (vs a single band) is the structural robustness argument over P1.
Single-frame perception keeps it simple and ONNX-friendly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_DIR = REPO_ROOT / "locked_model"
if str(LOCKED_DIR) not in sys.path:
    sys.path.insert(0, str(LOCKED_DIR))

from steering_models import _MobileNetFeatureBase  # type: ignore  # noqa: E402


class LaneSegNet(_MobileNetFeatureBase):
    def __init__(self, *, use_pretrained: bool = True):
        super().__init__(use_pretrained=use_pretrained, in_channels=3)
        self.reduce_final = nn.Sequential(
            nn.Conv2d(576, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Hardswish(),
        )
        self.reduce_mid = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.Hardswish(),
            nn.Conv2d(48, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.Hardswish(),
        )
        self.head = nn.Conv2d(24, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]
        mid, final = self.extract_feature_maps(x)
        final = self.reduce_final(final)
        final = F.interpolate(final, size=mid.shape[-2:], mode="bilinear", align_corners=False)
        mid = self.reduce_mid(mid)
        fused = self.fuse(torch.cat([mid, final], dim=1))
        logits = self.head(fused)
        return F.interpolate(logits, size=input_hw, mode="bilinear", align_corners=False)


__all__ = ["LaneSegNet"]
