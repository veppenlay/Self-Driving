#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Turn a lane probability mask into geometric affordances (e_y, theta). Numpy-only.

Mirrors the band/left-right-center logic of the locked `_finalize_detection`, but operates on
a soft probability mask instead of a hard HSV mask. Used by P3 inference and reusable offline.
"""

from __future__ import annotations

import numpy as np

from .geometry import e_y_from_center, theta_from_centers


def _band_center(mask: np.ndarray, ref_y: int, *, band_half: int, threshold: float, width_prior: float | None):
    h, w = mask.shape
    lo = max(0, ref_y - band_half)
    hi = min(h, ref_y + band_half + 1)
    profile = mask[lo:hi, :].sum(axis=0)
    active = profile >= threshold
    if not active.any():
        return None, 0.0
    # connected runs
    idx = np.where(active)[0]
    splits = np.where(np.diff(idx) > 1)[0]
    groups = np.split(idx, splits + 1)
    runs = [(int(g[0]), int(g[-1])) for g in groups if (g[-1] - g[0] + 1) >= max(3, w // 80)]
    if not runs:
        return None, 0.0
    centers = []
    for a, b in runs:
        seg = profile[a:b + 1]
        xs = np.arange(a, b + 1)
        total = float(seg.sum())
        centers.append(float((xs * seg).sum() / total) if total > 1e-6 else float((a + b) / 2.0))
    center_x = float(w) / 2.0
    left = [(r, c) for r, c in zip(runs, centers) if c < center_x]
    right = [(r, c) for r, c in zip(runs, centers) if c >= center_x]
    if left and right:
        _, lx = max(left, key=lambda item: item[1])
        _, rx = min(right, key=lambda item: item[1])
        lane_width = rx - lx
        if lane_width > max(20.0, 0.12 * w):
            return (lx + rx) / 2.0, 1.0
    if width_prior is not None and len(centers) >= 1:
        # single-sided inference using the frozen width prior
        x = centers[int(np.argmax([profile[int(c)] for c in centers]))]
        if x >= center_x:
            lx = max(0.0, x - width_prior)
            rx = x
        else:
            lx = x
            rx = min(w - 1.0, x + width_prior)
        return (lx + rx) / 2.0, 0.6
    return None, 0.0


def mask_to_affordance(
    mask: np.ndarray,
    *,
    near_y_ratio: float = 0.54,
    far_y_ratio: float = 0.42,
    threshold: float = 0.5,
    width_prior: float | None = None,
) -> dict[str, float]:
    """mask: HxW float in [0,1]. Returns e_y/theta plus their qualities."""
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"mask must be HxW, got {mask.shape}")
    h, w = mask.shape
    band_half = max(2, int(round(h * 0.035)))
    near_y = int(round(h * near_y_ratio))
    far_y = int(round(h * far_y_ratio))

    near_center, near_q = _band_center(mask, near_y, band_half=band_half, threshold=threshold, width_prior=width_prior)
    far_center, far_q = _band_center(mask, far_y, band_half=band_half, threshold=threshold, width_prior=width_prior)

    if near_center is None:
        return {"e_y": 0.0, "theta": 0.0, "ey_quality": 0.0, "theta_quality": 0.0, "detected": False}

    e_y = e_y_from_center(near_center, w)
    theta = 0.0
    theta_q = 0.0
    if far_center is not None:
        theta = theta_from_centers(near_center, near_y, far_center, far_y)
        theta_q = float(min(near_q, far_q))
    return {"e_y": float(e_y), "theta": float(theta), "ey_quality": float(near_q),
            "theta_quality": float(theta_q), "detected": True}


__all__ = ["mask_to_affordance"]
