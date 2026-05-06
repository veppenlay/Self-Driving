#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug helpers for safe augmentation and steering-angle sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from sampler_utils import SamplerConfig, compute_sample_weights


def _tensor_to_display_rgb(tensor: torch.Tensor, *, input_color_space: str = "hsv") -> np.ndarray:
    """将 CxHxW tensor 转成可显示 RGB。默认当前工程 tensor 是 HSV [0,1]。"""
    arr = tensor.detach().cpu().float().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr, 0.0, 1.0)
    arr_u8 = (arr * 255.0).astype(np.uint8)
    if input_color_space == "hsv":
        return cv2.cvtColor(arr_u8, cv2.COLOR_HSV2RGB)
    return arr_u8


def visualize_augmented_samples(
    dataset,
    *,
    indices: Sequence[int] | None = None,
    num_samples: int = 8,
    input_color_space: str = "hsv",
    save_path: str | Path | None = None,
):
    """随机/指定可视化若干增强后的样本，人工检查增强是否安全。

    Dataset 需要返回 `(image_tensor, angle_tensor)`。此函数不会修改数据，只用于 debug。
    """
    if indices is None:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False).tolist()
    else:
        indices = list(indices)[:num_samples]

    n = len(indices)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.asarray(axes).reshape(-1)

    for ax, idx in zip(axes, indices):
        image, angle = dataset[idx]
        if not torch.is_tensor(image):
            raise TypeError("visualize_augmented_samples expects tensor images after transform")
        rgb = _tensor_to_display_rgb(image, input_color_space=input_color_space)
        angle_value = float(angle.reshape(-1)[0].item()) if torch.is_tensor(angle) else float(angle)
        ax.imshow(rgb)
        ax.set_title(f"idx={idx}, angle={angle_value:.4f}")
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def summarize_angle_distribution(
    angles: Sequence[float],
    *,
    sampler_config: SamplerConfig | None = None,
    save_path: str | Path | None = None,
):
    """画出原始 angle 分布与 sample weight 分布概览。"""
    weights, debug = compute_sample_weights(angles, sampler_config, return_debug=True)
    angles_arr = np.asarray(list(angles), dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(angles_arr, bins=40, color="#2f6f9f", alpha=0.85)
    axes[0].set_title("Steering angle distribution")
    axes[0].set_xlabel("angle")
    axes[0].set_ylabel("count")

    axes[1].hist(np.abs(angles_arr), bins=40, color="#b55d2a", alpha=0.85)
    axes[1].set_title("|angle| distribution")
    axes[1].set_xlabel("|angle|")

    axes[2].hist(weights, bins=40, color="#3f8f53", alpha=0.85)
    axes[2].set_title("Sample weight distribution")
    axes[2].set_xlabel("weight")

    fig.suptitle(
        f"weight min={debug['weightsMin']:.3f}, max={debug['weightsMax']:.3f}, mean={debug['weightsMean']:.3f}"
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig, debug