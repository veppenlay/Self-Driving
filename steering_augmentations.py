from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import inspect
import os

import cv2
import numpy as np

from steering_preprocess import DEFAULT_PREPROCESS_CONFIG, PreprocessConfig

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError as exc:  # pragma: no cover
    raise ImportError("Missing albumentations. Please install requirements before training.") from exc


@dataclass(slots=True)
class AugConfig:
    preprocess: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_pixel_value: float = 255.0
    style_mix_ratio: tuple[float, float, float] = (0.6, 0.25, 0.15)
    num_frames: int = 1


class StyleMixTransform:
    """Sample clean / moderate / strong branches and keep style metadata."""

    returns_dict = True

    def __init__(self, config: AugConfig):
        self.cfg = config
        weights = np.asarray(config.style_mix_ratio, dtype=np.float64)
        if weights.shape != (3,) or np.any(weights < 0):
            raise ValueError(f"invalid style_mix_ratio: {config.style_mix_ratio}")
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("style_mix_ratio must have a positive sum")
        self.branch_names = ("clean", "moderate", "strong")
        self.branch_probs = (weights / total).tolist()
        self.transforms = {
            "clean": build_clean_transforms(config),
            "moderate": build_moderate_transforms(config),
            "strong": build_strong_transforms(config),
        }

    def __call__(self, *, image: np.ndarray, **kwargs):
        branch_idx = int(np.random.choice(len(self.branch_names), p=self.branch_probs))
        branch_name = self.branch_names[branch_idx]
        result = self.transforms[branch_name](image=image, **kwargs)
        result["styleTag"] = branch_name
        result["isClean"] = branch_name == "clean"
        return result


def _convert_rgb_to_hsv(image: np.ndarray, **_: object) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def _build_color_convert(preprocess: PreprocessConfig):
    if preprocess.color_space == "rgb":
        return A.NoOp(p=1.0)
    if preprocess.color_space == "hsv":
        return A.Lambda(image=_convert_rgb_to_hsv, p=1.0)
    raise ValueError(f"unsupported color_space: {preprocess.color_space}")


def _build_gauss_noise(p: float):
    params = inspect.signature(A.GaussNoise).parameters
    if "var_limit" in params:
        return A.GaussNoise(var_limit=(4.0, 18.0), mean=0.0, p=p)
    std_low = (4.0**0.5) / 255.0
    std_high = (18.0**0.5) / 255.0
    return A.GaussNoise(std_range=(std_low, std_high), mean_range=(0.0, 0.0), p=p)


def _build_image_compression(p: float, low: int, high: int):
    params = inspect.signature(A.ImageCompression).parameters
    if "quality_lower" in params:
        return A.ImageCompression(quality_lower=low, quality_upper=high, p=p)
    return A.ImageCompression(quality_range=(low, high), p=p)


def _local_exposure(image: np.ndarray, *, brighten: bool, strength_range: tuple[float, float], **_: object) -> np.ndarray:
    h, w = image.shape[:2]
    center_x = np.random.uniform(0.25, 0.75) * w
    center_y = np.random.uniform(0.30, 0.80) * h
    sigma_x = np.random.uniform(0.18, 0.42) * w
    sigma_y = np.random.uniform(0.18, 0.42) * h
    strength = np.random.uniform(*strength_range)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = np.exp(-(((xx - center_x) ** 2) / (2.0 * sigma_x**2) + ((yy - center_y) ** 2) / (2.0 * sigma_y**2)))
    mask = mask[..., None].astype(np.float32)
    image_f = image.astype(np.float32)
    if brighten:
        out = image_f * (1.0 + mask * strength) + 255.0 * mask * (strength * 0.12)
    else:
        out = image_f * (1.0 - mask * strength)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _local_exposure_lambda(strength_range: tuple[float, float], brighten: bool):
    return A.Lambda(image=partial(_local_exposure, brighten=brighten, strength_range=strength_range), p=1.0)


def _resize(preprocess: PreprocessConfig):
    height, width = preprocess.input_size
    return A.Resize(height=height, width=width, interpolation=cv2.INTER_AREA, p=1.0)


def _additional_targets(cfg: AugConfig) -> dict[str, str]:
    if cfg.num_frames <= 1:
        return {}
    return {f"image{i}": "image" for i in range(1, int(cfg.num_frames))}


def _normalize_and_tensor(cfg: AugConfig):
    return [
        _build_color_convert(cfg.preprocess),
        A.Normalize(mean=cfg.mean, std=cfg.std, max_pixel_value=cfg.max_pixel_value, p=1.0),
        ToTensorV2(transpose_mask=False, p=1.0),
    ]


def _medium_style_ops() -> list:
    return [
        A.RandomBrightnessContrast(brightness_limit=0.14, contrast_limit=0.12, brightness_by_max=True, p=0.82),
        A.RandomGamma(gamma_limit=(90, 110), p=0.34),
        A.HueSaturationValue(hue_shift_limit=2, sat_shift_limit=8, val_shift_limit=8, p=0.18),
        A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.18),
        A.CLAHE(clip_limit=(1.0, 2.0), tile_grid_size=(8, 8), p=0.08),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                A.MotionBlur(blur_limit=(3, 3), p=1.0),
            ],
            p=0.10,
        ),
        _build_gauss_noise(0.08),
        A.ISONoise(color_shift=(0.005, 0.02), intensity=(0.04, 0.15), p=0.06),
        _build_image_compression(0.08, 84, 98),
        A.Sharpen(alpha=(0.04, 0.12), lightness=(0.9, 1.1), p=0.05),
    ]


def _strong_style_ops() -> list:
    return [
        A.RandomBrightnessContrast(brightness_limit=0.28, contrast_limit=0.24, brightness_by_max=True, p=0.96),
        A.RandomGamma(gamma_limit=(78, 126), p=0.60),
        A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=10, val_shift_limit=16, p=0.18),
        A.RGBShift(r_shift_limit=8, g_shift_limit=8, b_shift_limit=8, p=0.24),
        A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.18),
        A.OneOf(
            [
                _local_exposure_lambda((0.18, 0.42), brighten=True),
                _local_exposure_lambda((0.16, 0.34), brighten=False),
            ],
            p=0.38,
        ),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
            ],
            p=0.12,
        ),
        _build_gauss_noise(0.12),
        A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.08, 0.22), p=0.10),
        _build_image_compression(0.12, 72, 95),
        A.Sharpen(alpha=(0.05, 0.18), lightness=(0.85, 1.2), p=0.08),
    ]


def build_clean_transforms(config: AugConfig | None = None) -> A.Compose:
    cfg = config or AugConfig()
    return A.Compose(
        [
            _resize(cfg.preprocess),
            *_normalize_and_tensor(cfg),
        ],
        additional_targets=_additional_targets(cfg),
    )


def build_moderate_transforms(config: AugConfig | None = None) -> A.Compose:
    cfg = config or AugConfig()
    return A.Compose(
        [
            _resize(cfg.preprocess),
            *_medium_style_ops(),
            *_normalize_and_tensor(cfg),
        ],
        additional_targets=_additional_targets(cfg),
    )


def build_strong_transforms(config: AugConfig | None = None) -> A.Compose:
    cfg = config or AugConfig()
    return A.Compose(
        [
            _resize(cfg.preprocess),
            *_strong_style_ops(),
            *_normalize_and_tensor(cfg),
        ],
        additional_targets=_additional_targets(cfg),
    )


def build_train_transform_bundle(config: AugConfig | None = None) -> StyleMixTransform:
    cfg = config or AugConfig()
    return StyleMixTransform(cfg)


def build_train_transforms(config: AugConfig | None = None) -> A.Compose:
    cfg = config or AugConfig()
    clean_p, medium_p, strong_p = cfg.style_mix_ratio
    return A.Compose(
        [
            _resize(cfg.preprocess),
            A.OneOf(
                [
                    A.NoOp(p=max(clean_p, 1e-6)),
                    A.Sequential(_medium_style_ops(), p=max(medium_p, 1e-6)),
                    A.Sequential(_strong_style_ops(), p=max(strong_p, 1e-6)),
                ],
                p=1.0,
            ),
            *_normalize_and_tensor(cfg),
        ],
        additional_targets=_additional_targets(cfg),
    )


def build_eval_transforms(config: AugConfig | None = None) -> A.Compose:
    return build_clean_transforms(config)


def build_stress_transforms(config: AugConfig | None = None) -> A.Compose:
    return build_strong_transforms(config)
