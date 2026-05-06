#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Weighted sampling utilities for steering-angle regression.

小数据集自动驾驶数据通常直行样本多、转弯样本少。直接 shuffle 会让模型更偏向预测
接近 0 的角度。这里通过 WeightedRandomSampler 提高少数角度区间的采样概率，缓解
长尾/不平衡问题，同时用权重裁剪避免极端过采样。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

BinMode = Literal["uniform", "quantile"]


@dataclass(slots=True)
class SamplerConfig:
    """采样权重配置。"""

    # uniform: 等宽分桶；quantile: 分位数分桶。小数据集优先建议 uniform，解释更直观。
    bin_mode: BinMode = "uniform"
    num_bins: int = 9

    # True 表示按 |angle| 分桶：重点解决直行多、转弯少的问题。
    use_abs_angle: bool = True

    # 平滑项避免某个 bin 频次为 0 或太小导致权重爆炸。
    smoothing: float = 1.0

    # 权重裁剪上限，避免少数样本被无限重复采样导致过拟合。
    max_weight: float = 8.0

    # 对接近 0 的直行样本额外降权，缓解直行样本压倒性占比。
    downweight_straight: bool = True
    straight_threshold: float = 0.08
    straight_weight_scale: float = 0.45

    # 最终归一化为均值 1，便于调试不同配置的量级。
    normalize_mean: bool = True

    # WeightedRandomSampler 的参数。
    replacement: bool = True
    num_samples: int | None = None


def _as_float_array(angles: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(angles), dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("angles must be a non-empty 1D sequence")
    if not np.all(np.isfinite(arr)):
        raise ValueError("angles contain NaN or Inf")
    return arr


def _make_edges(values: np.ndarray, config: SamplerConfig) -> np.ndarray:
    if config.num_bins < 2:
        raise ValueError("num_bins must be >= 2")

    v_min = float(values.min())
    v_max = float(values.max())
    if np.isclose(v_min, v_max):
        # 全部角度几乎相同时退化为一个很窄区间，避免 digitize 出错。
        eps = max(abs(v_min) * 1e-3, 1e-3)
        return np.linspace(v_min - eps, v_max + eps, config.num_bins + 1)

    if config.bin_mode == "uniform":
        return np.linspace(v_min, v_max, config.num_bins + 1)

    if config.bin_mode == "quantile":
        qs = np.linspace(0.0, 1.0, config.num_bins + 1)
        edges = np.quantile(values, qs)
        # 分位数在重复值多时可能重合，加一个极小扰动保证单调。
        edges = np.asarray(edges, dtype=np.float64)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-6
        return edges

    raise ValueError(f"unsupported bin_mode: {config.bin_mode}")


def compute_sample_weights(
    angles: Sequence[float],
    config: SamplerConfig | None = None,
    *,
    return_debug: bool = False,
):
    """根据转向角分布计算每个样本的采样权重。

    Args:
        angles: 每个样本的 steering angle。
        config: 采样配置。
        return_debug: True 时返回 `(weights, debug_info)`，便于画分布和排查。

    Returns:
        默认返回 `np.ndarray`，长度与 angles 一致。若 return_debug=True，则额外返回字典。
    """
    cfg = config or SamplerConfig()
    raw_angles = _as_float_array(angles)
    values = np.abs(raw_angles) if cfg.use_abs_angle else raw_angles.copy()
    edges = _make_edges(values, cfg)

    bin_ids = np.digitize(values, edges[1:-1], right=False)
    counts = np.bincount(bin_ids, minlength=cfg.num_bins).astype(np.float64)

    # 频次越低，权重越高；smoothing 防止极端权重。
    inv_freq = 1.0 / (counts + cfg.smoothing)
    weights = inv_freq[bin_ids]

    if cfg.downweight_straight:
        straight_mask = np.abs(raw_angles) <= cfg.straight_threshold
        weights[straight_mask] *= cfg.straight_weight_scale

    if cfg.max_weight is not None and cfg.max_weight > 0:
        weights = np.clip(weights, 0.0, cfg.max_weight)

    if cfg.normalize_mean:
        mean = float(weights.mean())
        if mean > 0:
            weights = weights / mean

    debug = {
        "edges": edges,
        "binIds": bin_ids,
        "counts": counts,
        "useAbsAngle": cfg.use_abs_angle,
        "weightsMin": float(weights.min()),
        "weightsMax": float(weights.max()),
        "weightsMean": float(weights.mean()),
    }
    return (weights.astype(np.float64), debug) if return_debug else weights.astype(np.float64)


def build_weighted_sampler(
    angles: Sequence[float],
    config: SamplerConfig | None = None,
    *,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    """构建 PyTorch WeightedRandomSampler。

    使用该 sampler 时，DataLoader 必须设置 shuffle=False。原因是 sampler 已经决定了样本
    抽取顺序和概率，再同时开启 shuffle 没有意义，PyTorch 也不允许二者同时指定。
    """
    cfg = config or SamplerConfig()
    weights = compute_sample_weights(angles, cfg)
    num_samples = cfg.num_samples if cfg.num_samples is not None else len(weights)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=cfg.replacement,
        generator=generator,
    )


SCHEME_NOTE = """
为什么适合转向角长尾/不平衡：
1. 直行样本通常集中在 0 附近，按 |angle| 分桶能把直行与不同幅度转弯分开。
2. 低频 bin 自动获得更高采样概率，训练时更常看到转弯样本。
3. max_weight 和 smoothing 防止极少数样本被过度重复采样，降低过拟合风险。
4. straight_weight_scale 可以进一步降低接近 0 的直行样本权重。
"""