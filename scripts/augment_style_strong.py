#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Strong photometric style augmentation for steering-angle datasets.

Generate exactly one aggressive style-augmented image per source image.
Only keeps augmented images in destination directory; source images are not copied.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _imread_rgb(path: Path) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        raise FileNotFoundError(f"无法读取图像: {path}")
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法解码图像: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _imwrite_rgb(path: Path, image_rgb: np.ndarray, jpeg_quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        ok, encoded = cv2.imencode(ext, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    else:
        ok, encoded = cv2.imencode(ext, bgr)
    if not ok:
        raise RuntimeError(f"图像编码失败: {path}")
    encoded.tofile(str(path))


def _build_transform() -> A.Compose:
    # 强烈风格扰动：亮度、对比度、gamma、CLAHE、局部曝光/阴影。
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.42,
                contrast_limit=0.38,
                brightness_by_max=True,
                p=0.95,
            ),
            A.RandomGamma(gamma_limit=(55, 155), p=0.85),
            A.CLAHE(clip_limit=(2.0, 4.5), tile_grid_size=(8, 8), p=0.55),
            A.OneOf(
                [
                    A.RandomShadow(
                        shadow_roi=(0.0, 0.0, 1.0, 1.0),
                        num_shadows_limit=(1, 3),
                        shadow_dimension=5,
                        p=1.0,
                    ),
                    A.RandomSunFlare(
                        flare_roi=(0.0, 0.0, 1.0, 0.7),
                        angle_range=(0.0, 1.0),
                        num_flare_circles_range=(4, 8),
                        src_radius=220,
                        src_color=(255, 255, 255),
                        p=1.0,
                    ),
                ],
                p=0.90,
            ),
            A.OneOf(
                [
                    A.ImageCompression(quality_range=(35, 70), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.15, 0.35), p=1.0),
                ],
                p=0.35,
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Generate one strong style-augmented image per source image.")
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.is_dir():
        raise NotADirectoryError(f"源目录不存在: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    transform = _build_transform()

    images = sorted([p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS], key=lambda p: p.name.lower())
    if not images:
        raise FileNotFoundError(f"源目录中未找到图像: {src}")

    for image_path in images:
        image_rgb = _imread_rgb(image_path)
        aug_rgb = transform(image=image_rgb)["image"]
        _imwrite_rgb(dst / image_path.name, aug_rgb, jpeg_quality=args.jpeg_quality)

    summary = {
        "srcDir": str(src),
        "dstDir": str(dst),
        "numImages": len(images),
        "policy": "one aggressive photometric-only augmentation per source image; no originals kept",
    }
    (dst / "augmentation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
