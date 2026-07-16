#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Synthetic, geometry-preserving appearance augmentation for P1/P3 perception training.

This is the core robustness lever: by training the perception under a wide, purely synthetic
appearance distribution, the learned e_y/theta estimator becomes invariant to lighting/blur/
noise and generalizes to frames (and scenes) where the fixed-threshold CV label generator
would fail. The gain is from LEARNED INVARIANCE, not from the label source.

Discipline:
- ZERO target-domain priors. Magnitudes come from generic ranges only; no basement image or
  statistic is used to design them.
- Photometric ONLY. No flips/shifts/crops that would change the lateral geometry, so the
  `e_y`/`theta` labels stay valid.

Operates on BGR uint8 frames. cv2 is used for HSV jitter but imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import perturb


@dataclass
class AugmentConfig:
    p_apply: float = 0.9          # probability any augmentation is applied to a stack
    p_gamma: float = 0.5
    p_brightness: float = 0.5
    p_contrast: float = 0.4
    p_blur: float = 0.3
    p_noise: float = 0.3
    p_hue_sat: float = 0.5
    p_shadow: float = 0.3
    p_frame_drop: float = 0.15    # per-frame temporal dropout within a stack (hold previous)
    strength: float = 1.0         # global multiplier on magnitudes


class PhotometricAugmentor:
    def __init__(self, config: AugmentConfig | None = None, *, seed: int | None = None):
        self.cfg = config or AugmentConfig()
        self.rng = np.random.default_rng(seed)

    def _u(self, lo: float, hi: float) -> float:
        return float(self.rng.uniform(lo, hi))

    def _hue_sat_jitter(self, bgr: np.ndarray) -> np.ndarray:
        import cv2  # lazy

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        s = self.cfg.strength
        hsv[..., 0] = (hsv[..., 0] + self._u(-8, 8) * s) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * self._u(1 - 0.3 * s, 1 + 0.3 * s), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * self._u(1 - 0.25 * s, 1 + 0.25 * s), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _shadow(self, bgr: np.ndarray) -> np.ndarray:
        """Overlay a smooth random linear brightness gradient (fake shadow/highlight)."""
        h, w = bgr.shape[:2]
        angle = self._u(0, np.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        proj = np.cos(angle) * xx / max(1, w) + np.sin(angle) * yy / max(1, h)
        proj = (proj - proj.min()) / (float(np.ptp(proj)) + 1e-6)
        depth = self._u(0.15, 0.5) * self.cfg.strength
        gain = (1.0 - depth) + depth * proj[..., None]
        return np.clip(bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    def augment_frame(self, bgr: np.ndarray) -> np.ndarray:
        out = bgr
        s = self.cfg.strength
        if self.rng.random() < self.cfg.p_gamma:
            out = perturb.gamma(out, self._u(0.6, 1.8))
        if self.rng.random() < self.cfg.p_brightness:
            out = perturb.brightness(out, self._u(-0.25, 0.25) * s)
        if self.rng.random() < self.cfg.p_contrast:
            out = perturb.contrast(out, self._u(0.7, 1.4))
        if self.rng.random() < self.cfg.p_hue_sat:
            try:
                out = self._hue_sat_jitter(out)
            except Exception:
                pass
        if self.rng.random() < self.cfg.p_shadow:
            out = self._shadow(out)
        if self.rng.random() < self.cfg.p_blur:
            out = perturb.blur(out, self._u(0.6, 2.5) * s)
        if self.rng.random() < self.cfg.p_noise:
            out = perturb.gaussian_noise(out, self._u(0.01, 0.05) * s, seed=int(self.rng.integers(0, 2**31 - 1)))
        return out

    def augment_stack(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Apply consistent photometric params across a temporal stack, plus per-frame drop."""
        if self.rng.random() >= self.cfg.p_apply:
            return frames
        # One augmentor state shared across frames for photometric consistency.
        params_rng = np.random.default_rng(int(self.rng.integers(0, 2**31 - 1)))
        shared = PhotometricAugmentor(self.cfg, seed=int(params_rng.integers(0, 2**31 - 1)))
        out: list[np.ndarray] = []
        prev = None
        for frame in frames:
            if prev is not None and self.rng.random() < self.cfg.p_frame_drop:
                out.append(prev)  # temporal dropout: hold previous frame
                continue
            aug = shared.augment_frame(frame)
            out.append(aug)
            prev = aug
        return out


__all__ = ["AugmentConfig", "PhotometricAugmentor"]
