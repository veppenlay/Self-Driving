#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline dataset augmentation for steering-angle regression.

本脚本按用户要求生成完整增强数据集：
- 保留全部原始图片；
- 对源数据集中的每张图片额外生成 N 张安全增强图；
- 增强图命名严格为 `新序号_原转向角文本.jpg`，例如 `1743_-1.5800.jpg`；
- 同时重写 train.txt / val.txt / test.txt，使增强图进入其源图所属的同一划分；
- 所有增强都不修改 steering angle 标签值。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import albumentations as A
import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
NAME_RE = re.compile(r"^(?P<index>\d+)_(?P<angle>.+)$")


@dataclass(slots=True)
class OfflineAugConfig:
    copies_per_source_image: int = 10
    jpeg_quality: int = 95
    seed: int = 3407
    overwrite: bool = False



def build_safe_offline_transform() -> A.Compose:
    """构建尽可能丰富但仍安全的离线增强策略。

    只使用 photometric / image-quality 增强，不改变空间几何语义。
    """
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.20,
                contrast_limit=0.16,
                brightness_by_max=True,
                p=0.72,
            ),
            A.RandomGamma(gamma_limit=(86, 114), p=0.30),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=12,
                val_shift_limit=10,
                p=0.28,
            ),
            A.RGBShift(
                r_shift_limit=6,
                g_shift_limit=6,
                b_shift_limit=6,
                p=0.15,
            ),
            A.CLAHE(clip_limit=(1.0, 2.2), tile_grid_size=(8, 8), p=0.08),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                    A.MotionBlur(blur_limit=(3, 3), p=1.0),
                ],
                p=0.10,
            ),
            A.GaussNoise(std_range=(0.008, 0.022), mean_range=(0.0, 0.0), p=0.12),
            A.ISONoise(color_shift=(0.005, 0.02), intensity=(0.05, 0.18), p=0.08),
            A.ImageCompression(quality_range=(80, 98), p=0.10),
            A.Sharpen(alpha=(0.05, 0.15), lightness=(0.88, 1.12), p=0.08),
        ]
    )



def parse_data_list(list_path: Path) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_path, angle_str = line.rsplit(" ", 1)
            items.append((img_path, float(angle_str)))
    return items



def path_to_forward_slash(path: Path) -> str:
    return path.as_posix()



def _imread_rgb(path: Path) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        raise FileNotFoundError(f"无法读取图像: {path}")
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法解码图像: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)



def _imwrite_rgb(path: Path, image_rgb: np.ndarray, jpeg_quality: int) -> None:
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



def copy_original_images(src_dir: Path, dst_dir: Path, overwrite: bool) -> int:
    count = 0
    for src_path in src_dir.iterdir():
        if not src_path.is_file() or src_path.suffix.lower() not in IMG_EXTS:
            continue
        dst_path = dst_dir / src_path.name
        if dst_path.exists() and not overwrite:
            count += 1
            continue
        shutil.copy2(src_path, dst_path)
        count += 1
    return count



def parse_filename_metadata(src_name: str) -> tuple[int, str, str]:
    src_path = Path(src_name)
    match = NAME_RE.match(src_path.stem)
    if match is None:
        raise ValueError(f"文件名不符合 '序号_转向角' 规则: {src_name}")
    return int(match.group("index")), match.group("angle"), src_path.suffix.lower()



def build_augmented_name(next_index: int, angle_text: str, suffix: str) -> str:
    return f"{next_index}_{angle_text}{suffix}"



def augment_split(
    items: Iterable[tuple[str, float]],
    *,
    src_dir: Path,
    dst_dir: Path,
    transform: A.Compose,
    cfg: OfflineAugConfig,
    next_index_start: int,
) -> tuple[list[str], int, int]:
    output_lines: list[str] = []
    aug_count = 0
    next_index = next_index_start

    for src_path_str, angle in items:
        src_name = Path(src_path_str).name
        _, angle_text, suffix = parse_filename_metadata(src_name)

        output_lines.append(f"{path_to_forward_slash(dst_dir / src_name)} {angle:g}")

        image_rgb = _imread_rgb(src_dir / src_name)
        for _ in range(cfg.copies_per_source_image):
            aug_name = build_augmented_name(next_index, angle_text, suffix)
            dst_aug = dst_dir / aug_name
            aug_rgb = transform(image=image_rgb)["image"]
            _imwrite_rgb(dst_aug, aug_rgb, cfg.jpeg_quality)
            output_lines.append(f"{path_to_forward_slash(dst_aug)} {angle:g}")
            aug_count += 1
            next_index += 1

    return output_lines, aug_count, next_index



def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def collect_max_existing_index(src_dir: Path) -> int:
    max_index = -1
    for path in src_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
            continue
        idx, _, _ = parse_filename_metadata(path.name)
        max_index = max(max_index, idx)
    if max_index < 0:
        raise RuntimeError(f"在 {src_dir} 中未找到符合命名规则的图像")
    return max_index



def main() -> None:
    parser = argparse.ArgumentParser(description="安全离线增强自动驾驶转向角回归数据集")
    parser.add_argument("--src", required=True, help="原始数据集目录，例如 E:/桌面/data")
    parser.add_argument("--dst", required=True, help="增强后数据集目录，例如 E:/桌面/data_aug")
    parser.add_argument("--copies-per-source-image", type=int, default=10, help="每张源图额外生成多少张增强图")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG 输出质量")
    parser.add_argument("--seed", type=int, default=3407, help="随机种子")
    parser.add_argument("--overwrite", action="store_true", help="若目标已存在则覆盖增强结果")
    args = parser.parse_args()

    cfg = OfflineAugConfig(
        copies_per_source_image=args.copies_per_source_image,
        jpeg_quality=args.jpeg_quality,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    train_items = parse_data_list(src_dir / "train.txt")
    val_items = parse_data_list(src_dir / "val.txt")
    test_items = parse_data_list(src_dir / "test.txt")

    copied_originals = copy_original_images(src_dir, dst_dir, overwrite=cfg.overwrite)
    max_index = collect_max_existing_index(src_dir)
    next_index = max_index + 1

    transform = build_safe_offline_transform()
    train_lines, train_aug_count, next_index = augment_split(
        train_items,
        src_dir=src_dir,
        dst_dir=dst_dir,
        transform=transform,
        cfg=cfg,
        next_index_start=next_index,
    )
    val_lines, val_aug_count, next_index = augment_split(
        val_items,
        src_dir=src_dir,
        dst_dir=dst_dir,
        transform=transform,
        cfg=cfg,
        next_index_start=next_index,
    )
    test_lines, test_aug_count, next_index = augment_split(
        test_items,
        src_dir=src_dir,
        dst_dir=dst_dir,
        transform=transform,
        cfg=cfg,
        next_index_start=next_index,
    )

    write_lines(dst_dir / "train.txt", train_lines)
    write_lines(dst_dir / "val.txt", val_lines)
    write_lines(dst_dir / "test.txt", test_lines)

    summary = {
        "srcDir": str(src_dir),
        "dstDir": str(dst_dir),
        "copiedOriginalImages": copied_originals,
        "trainOriginalCount": len(train_items),
        "valOriginalCount": len(val_items),
        "testOriginalCount": len(test_items),
        "trainAugmentedCount": train_aug_count,
        "valAugmentedCount": val_aug_count,
        "testAugmentedCount": test_aug_count,
        "trainTotalCount": len(train_lines),
        "valTotalCount": len(val_lines),
        "testTotalCount": len(test_lines),
        "totalAugmentedCount": train_aug_count + val_aug_count + test_aug_count,
        "totalImageCountInDir": copied_originals + train_aug_count + val_aug_count + test_aug_count,
        "startAugmentedIndex": max_index + 1,
        "endAugmentedIndex": next_index - 1,
        "config": asdict(cfg),
        "policy": "copy all originals + augment every source image in its original split",
    }
    (dst_dir / "augmentation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
