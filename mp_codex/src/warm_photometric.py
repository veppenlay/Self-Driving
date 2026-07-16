#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Moderate geometry-preserving augmentation for locked-model fine-tuning.

One parameter set is sampled per temporal stack and applied to every frame.  This
preserves steering/e_y labels and temporal appearance continuity.  Magnitudes are
generic and are not derived from the held-out basement images.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class WarmAugmentConfig:
    p_apply: float = 0.80
    p_gamma: float = 0.45
    p_brightness: float = 0.45
    p_contrast: float = 0.35
    p_hsv: float = 0.35
    p_shadow: float = 0.15
    p_blur: float = 0.15
    p_noise: float = 0.15


class WarmPhotometricAugmentor:
    def __init__(self, config: WarmAugmentConfig | None = None, *, seed: int = 20260715):
        self.cfg = config or WarmAugmentConfig()
        self.rng = np.random.default_rng(seed)

    def _sample(self, height: int, width: int) -> dict[str, object] | None:
        if self.rng.random() >= self.cfg.p_apply:
            return None
        enabled = lambda p: bool(self.rng.random() < p)
        use_shadow = enabled(self.cfg.p_shadow)
        projection = None
        if use_shadow:
            angle = float(self.rng.uniform(0.0, np.pi))
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
            projection = np.cos(angle) * xx / max(1, width) + np.sin(angle) * yy / max(1, height)
            projection = (projection - projection.min()) / (float(np.ptp(projection)) + 1e-6)
        return {
            "gamma": float(self.rng.uniform(0.80, 1.25)) if enabled(self.cfg.p_gamma) else None,
            "brightness": float(self.rng.uniform(-0.12, 0.12)) if enabled(self.cfg.p_brightness) else None,
            "contrast": float(self.rng.uniform(0.85, 1.15)) if enabled(self.cfg.p_contrast) else None,
            "hue": float(self.rng.uniform(-4.0, 4.0)) if enabled(self.cfg.p_hsv) else None,
            "saturation": float(self.rng.uniform(0.85, 1.15)),
            "value": float(self.rng.uniform(0.90, 1.10)),
            "shadow": projection,
            "shadow_depth": float(self.rng.uniform(0.10, 0.25)),
            "blur": float(self.rng.uniform(0.4, 1.2)) if enabled(self.cfg.p_blur) else None,
            "noise": float(self.rng.uniform(0.005, 0.02)) if enabled(self.cfg.p_noise) else None,
        }

    def _apply(self, bgr: np.ndarray, params: dict[str, object]) -> np.ndarray:
        out = bgr.astype(np.float32) / 255.0
        gamma = params["gamma"]
        if gamma is not None:
            out = np.power(np.clip(out, 0.0, 1.0), float(gamma))
        brightness = params["brightness"]
        if brightness is not None:
            out += float(brightness)
        contrast = params["contrast"]
        if contrast is not None:
            mean = float(out.mean())
            out = (out - mean) * float(contrast) + mean
        out = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)

        hue = params["hue"]
        if hue is not None:
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + float(hue)) % 180.0
            hsv[..., 1] = np.clip(hsv[..., 1] * float(params["saturation"]), 0.0, 255.0)
            hsv[..., 2] = np.clip(hsv[..., 2] * float(params["value"]), 0.0, 255.0)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        shadow = params["shadow"]
        if shadow is not None:
            gain = (1.0 - float(params["shadow_depth"])) + float(params["shadow_depth"]) * shadow[..., None]
            out = np.clip(out.astype(np.float32) * gain, 0.0, 255.0).astype(np.uint8)
        blur = params["blur"]
        if blur is not None:
            k = max(3, 2 * int(round(3.0 * float(blur))) + 1)
            out = cv2.GaussianBlur(out, (k, k), float(blur))
        noise = params["noise"]
        if noise is not None:
            eps = self.rng.normal(0.0, float(noise) * 255.0, size=out.shape)
            out = np.clip(out.astype(np.float32) + eps, 0.0, 255.0).astype(np.uint8)
        return out

    def augment_stack(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        if not frames:
            return frames
        params = self._sample(*frames[0].shape[:2])
        if params is None:
            return frames
        return [self._apply(frame, params) for frame in frames]


__all__ = ["WarmAugmentConfig", "WarmPhotometricAugmentor"]
