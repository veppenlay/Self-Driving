#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Purely synthetic appearance perturbations for the held-out robustness stress test.

Every perturbation here is a generic, hand-specified transform. It uses NO target-domain
(basement) image or statistic. Severities are fixed grids chosen from general priors so
that robustness is measured as *degradation under held-out perturbation*, not as a score
obtained by tuning to the target scene.

Operates on BGR uint8 images (HxWx3), numpy-only (no cv2), so it is deterministic and can
be shared by both the offline harness and the candidate inference scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def brightness(img: np.ndarray, delta: float) -> np.ndarray:
    """Additive brightness shift in [-255, 255] domain (delta as fraction of 255)."""
    return _to_u8(img.astype(np.float32) + delta * 255.0)


def gamma(img: np.ndarray, g: float) -> np.ndarray:
    """Gamma correction; g>1 darkens, g<1 brightens."""
    x = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)
    return _to_u8(np.power(x, max(1e-3, g)) * 255.0)


def contrast(img: np.ndarray, factor: float) -> np.ndarray:
    mean = float(img.astype(np.float32).mean())
    return _to_u8((img.astype(np.float32) - mean) * factor + mean)


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    return k / float(k.sum())


def _conv1d_axis(img: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    padded = np.pad(img, [(pad, pad) if a == axis else (0, 0) for a in range(img.ndim)], mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    for i, w in enumerate(kernel):
        sl = [slice(None)] * img.ndim
        sl[axis] = slice(i, i + img.shape[axis])
        out += w * padded[tuple(sl)].astype(np.float32)
    return out


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur (numpy)."""
    if sigma <= 1e-3:
        return img.copy()
    kernel = _gaussian_kernel1d(sigma)
    tmp = _conv1d_axis(img.astype(np.float32), kernel, axis=0)
    tmp = _conv1d_axis(tmp, kernel, axis=1)
    return _to_u8(tmp)


def gaussian_noise(img: np.ndarray, std: float, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, std * 255.0, size=img.shape).astype(np.float32)
    return _to_u8(img.astype(np.float32) + noise)


@dataclass(frozen=True)
class PerturbLevel:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PerturbAxis:
    """A family of increasing-severity perturbations plus the frame-drop flag."""

    name: str
    levels: tuple[PerturbLevel, ...]
    is_frame_drop: bool = False


def default_axes(*, noise_seed: int = 20260715) -> list[PerturbAxis]:
    """Fixed, generic severity grid (no target-domain tuning)."""
    return [
        PerturbAxis(
            "lighting_dark",
            (
                PerturbLevel("g1.3", lambda im: gamma(im, 1.3)),
                PerturbLevel("g1.7", lambda im: gamma(im, 1.7)),
                PerturbLevel("g2.2", lambda im: gamma(im, 2.2)),
            ),
        ),
        PerturbAxis(
            "lighting_bright",
            (
                PerturbLevel("g0.8", lambda im: gamma(im, 0.8)),
                PerturbLevel("b+0.15", lambda im: brightness(im, 0.15)),
                PerturbLevel("b+0.30", lambda im: brightness(im, 0.30)),
            ),
        ),
        PerturbAxis(
            "contrast",
            (
                PerturbLevel("c0.7", lambda im: contrast(im, 0.7)),
                PerturbLevel("c1.4", lambda im: contrast(im, 1.4)),
            ),
        ),
        PerturbAxis(
            "blur",
            (
                PerturbLevel("s1.0", lambda im: blur(im, 1.0)),
                PerturbLevel("s2.0", lambda im: blur(im, 2.0)),
                PerturbLevel("s3.5", lambda im: blur(im, 3.5)),
            ),
        ),
        PerturbAxis(
            "noise",
            (
                PerturbLevel("n0.03", lambda im: gaussian_noise(im, 0.03, seed=noise_seed)),
                PerturbLevel("n0.06", lambda im: gaussian_noise(im, 0.06, seed=noise_seed)),
            ),
        ),
        PerturbAxis(
            "frame_drop",
            (
                PerturbLevel("p0.2", lambda im: im),
                PerturbLevel("p0.4", lambda im: im),
            ),
            is_frame_drop=True,
        ),
    ]


def frame_drop_rate(level_name: str) -> float:
    """Parse a frame-drop level name like 'p0.2' -> 0.2."""
    try:
        return float(level_name.lstrip("p"))
    except ValueError:
        return 0.0


__all__ = [
    "brightness",
    "gamma",
    "contrast",
    "blur",
    "gaussian_noise",
    "PerturbLevel",
    "PerturbAxis",
    "default_axes",
    "frame_drop_rate",
]
