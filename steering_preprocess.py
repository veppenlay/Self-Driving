from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    color_space: str = "hsv"
    input_size: tuple[int, int] = (120, 160)
    use_roi: bool = False


DEFAULT_PREPROCESS_CONFIG = PreprocessConfig()
DEFAULT_ROI_BOTTOM_RATIO = 0.6
ANGLE_KEY_DECIMALS = 6


def imread_bgr(path: str | Path) -> np.ndarray | None:
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def apply_bottom_roi(
    bgr: np.ndarray,
    *,
    config: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG,
    bottom_ratio: float = DEFAULT_ROI_BOTTOM_RATIO,
) -> np.ndarray:
    if not config.use_roi:
        return bgr

    ratio = float(bottom_ratio)
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"roi bottom_ratio must be in (0, 1], got {ratio}")

    height, width = bgr.shape[:2]
    crop_height = max(1, int(round(height * ratio)))
    y0 = max(0, height - crop_height)
    roi = bgr[y0:height, 0:width]
    if roi.size == 0:
        raise ValueError(f"roi is empty after crop: height={height}, width={width}, ratio={ratio}")
    return roi


def convert_color_space(image_rgb: np.ndarray, color_space: str) -> np.ndarray:
    if color_space == "rgb":
        return image_rgb
    if color_space == "hsv":
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    raise ValueError(f"unsupported color_space: {color_space}")


def preprocess_bgr_to_hwc_float(
    bgr: np.ndarray,
    *,
    config: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG,
    bottom_ratio: float = DEFAULT_ROI_BOTTOM_RATIO,
) -> np.ndarray:
    bgr = apply_bottom_roi(bgr, config=config, bottom_ratio=bottom_ratio)
    height, width = config.input_size
    resized = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    converted = convert_color_space(rgb, config.color_space)
    return converted.astype(np.float32) / 255.0


def preprocess_bgr_to_chw_float(
    bgr: np.ndarray,
    *,
    config: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG,
    bottom_ratio: float = DEFAULT_ROI_BOTTOM_RATIO,
) -> np.ndarray:
    hwc = preprocess_bgr_to_hwc_float(bgr, config=config, bottom_ratio=bottom_ratio)
    return np.transpose(hwc, (2, 0, 1)).astype(np.float32, copy=False)


def preprocess_bgr_to_tensor(
    bgr: np.ndarray,
    *,
    device: torch.device | None = None,
    config: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG,
    bottom_ratio: float = DEFAULT_ROI_BOTTOM_RATIO,
) -> torch.Tensor:
    chw = preprocess_bgr_to_chw_float(bgr, config=config, bottom_ratio=bottom_ratio)
    tensor = torch.from_numpy(chw).unsqueeze(0).float()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def preprocess_path_to_tensor(
    image_path: str | Path,
    *,
    device: torch.device | None = None,
    config: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG,
    bottom_ratio: float = DEFAULT_ROI_BOTTOM_RATIO,
) -> torch.Tensor:
    bgr = imread_bgr(image_path)
    if bgr is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    return preprocess_bgr_to_tensor(bgr, device=device, config=config, bottom_ratio=bottom_ratio)


def angle_to_key(angle: float) -> float:
    return round(float(angle), ANGLE_KEY_DECIMALS)


def build_angle_vocab(angles: Iterable[float]) -> list[float]:
    unique = sorted({angle_to_key(angle) for angle in angles})
    return [float(value) for value in unique]


def preprocess_config_to_dict(config: PreprocessConfig) -> dict[str, object]:
    return {
        "colorSpace": config.color_space,
        "inputSize": [int(config.input_size[0]), int(config.input_size[1])],
        "useRoi": bool(config.use_roi),
    }


def preprocess_config_from_dict(data: dict | None, *, fallback: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG) -> PreprocessConfig:
    if not isinstance(data, dict):
        return fallback

    color_space = str(data.get("colorSpace", data.get("color_space", fallback.color_space))).strip().lower()
    raw_size = data.get("inputSize", data.get("input_size", list(fallback.input_size)))
    if isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
        input_size = (int(raw_size[0]), int(raw_size[1]))
    else:
        input_size = fallback.input_size
    use_roi = bool(data.get("useRoi", data.get("use_roi", fallback.use_roi)))
    return PreprocessConfig(color_space=color_space, input_size=input_size, use_roi=use_roi)


def encode_angles_to_vocab(angles: torch.Tensor, angle_vocab: list[float], *, device: torch.device) -> torch.Tensor:
    if not angle_vocab:
        raise ValueError("angle_vocab must not be empty")
    vocab = torch.tensor(angle_vocab, dtype=torch.float32, device=device)
    angles = angles.view(-1, 1).float().to(device)
    distance = torch.abs(angles - vocab.view(1, -1))
    return torch.argmin(distance, dim=1)


def soft_encode_angles_to_vocab(
    angles: torch.Tensor,
    angle_vocab: list[float],
    *,
    device: torch.device,
    temperature: float = 0.03,
    neighbor_count: int = 3,
) -> torch.Tensor:
    if not angle_vocab:
        raise ValueError("angle_vocab must not be empty")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    vocab = torch.tensor(angle_vocab, dtype=torch.float32, device=device)
    angles = angles.view(-1, 1).float().to(device)
    distance = torch.abs(angles - vocab.view(1, -1))

    if neighbor_count > 0 and neighbor_count < vocab.numel():
        kth = torch.topk(distance, k=neighbor_count, dim=1, largest=False).values[:, -1:]
        mask = distance <= (kth + 1e-9)
        logits = torch.where(mask, -distance / temperature, torch.full_like(distance, -1e9))
    else:
        logits = -distance / temperature
    return torch.softmax(logits, dim=1)


def inverse_frequency_weights(indices: Iterable[int], num_classes: int, *, power: float = 1.0) -> torch.Tensor:
    counts = np.ones(num_classes, dtype=np.float32)
    for idx in indices:
        counts[int(idx)] += 1.0
    weights = counts.sum() / np.power(counts, power)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class NumpyRandomState:
    """Temporarily fix Python and NumPy RNG state for deterministic val_stress augmentation."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self._py_state = None
        self._np_state = None

    def __enter__(self):
        self._py_state = random.getstate()
        self._np_state = np.random.get_state()
        random.seed(self.seed)
        np.random.seed(self.seed % (2**32 - 1))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._py_state is not None:
            random.setstate(self._py_state)
        if self._np_state is not None:
            np.random.set_state(self._np_state)
        return False
