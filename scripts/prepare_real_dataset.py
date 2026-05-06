#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare a steering dataset inside the project workspace.

Copies angle-suffixed images from a source folder and creates train/val/test list files.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_index(path: Path) -> int:
    if "_" not in path.stem:
        raise ValueError("missing frame index")
    return int(path.stem.split("_", 1)[0])


def parse_angle(path: Path) -> float:
    if "_" not in path.stem:
        raise ValueError("missing angle suffix")
    return float(path.stem.rsplit("_", 1)[-1])


def _sort_key(path: Path):
    try:
        return (0, parse_index(path), path.name.lower())
    except ValueError:
        return (1, path.name.lower())


def collect_images(src: Path) -> list[tuple[Path, int, float]]:
    items: list[tuple[Path, int, float]] = []
    for path in sorted(src.iterdir(), key=_sort_key):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            index = parse_index(path)
            angle = parse_angle(path)
        except ValueError:
            continue
        items.append((path, index, angle))
    return items


def build_supervised_items(
    items: list[tuple[Path, int, float]],
    *,
    label_shift: int,
) -> tuple[list[tuple[Path, int, float, str]], dict[str, int]]:
    if label_shift < 0:
        raise ValueError("label_shift must be >= 0")

    if label_shift == 0:
        rows = [(path, index, angle, path.name) for path, index, angle in items]
        return rows, {"droppedTailFrames": 0, "droppedNonConsecutive": 0}

    rows: list[tuple[Path, int, float, str]] = []
    dropped_non_consecutive = 0
    for pos in range(max(0, len(items) - label_shift)):
        src_path, src_index, _ = items[pos]
        _, future_index, future_angle = items[pos + label_shift]
        if future_index != src_index + label_shift:
            dropped_non_consecutive += 1
            continue
        new_name = f"{src_index}_{future_angle:.4f}{src_path.suffix.lower()}"
        rows.append((src_path, src_index, future_angle, new_name))

    dropped_tail = min(label_shift, len(items))
    return rows, {"droppedTailFrames": dropped_tail, "droppedNonConsecutive": dropped_non_consecutive}


def split_items(items: list[tuple[Path, float]], seed: int, train_ratio: float, val_ratio: float):
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    total = len(shuffled)
    n_train = max(1, int(total * train_ratio))
    n_val = max(1, int(total * val_ratio))
    n_test = total - n_train - n_val
    if n_test < 1:
        raise RuntimeError("dataset is too small to create non-empty train/val/test splits")
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy and split a steering image dataset.")
    parser.add_argument("--src", required=True, help="Source dataset folder, e.g. E:/桌面/data")
    parser.add_argument("--dst-root", required=True, help="Project dataset root, e.g. E:/桌面/项目/dataset")
    parser.add_argument("--name", default=None, help="Dataset directory name. Defaults to real_steering_data_<timestamp>.")
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--label-shift", type=int, default=0, help="Use steering label from t+shift for each frame t.")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst_root).expanduser().resolve()
    if not src.is_dir():
        raise NotADirectoryError(f"source dataset folder not found: {src}")
    dst_root.mkdir(parents=True, exist_ok=True)

    name = args.name or f"real_steering_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dst = dst_root / name
    if dst.exists():
        raise FileExistsError(f"destination already exists: {dst}")
    dst.mkdir(parents=True)

    items = collect_images(src)
    if len(items) < 3:
        raise RuntimeError(f"not enough valid angle-suffixed images in {src}")
    supervised_items, shift_stats = build_supervised_items(items, label_shift=args.label_shift)
    if len(supervised_items) < 3:
        raise RuntimeError(f"not enough valid supervised samples after label shift={args.label_shift}")

    copied: list[tuple[Path, float]] = []
    for src_path, _, angle, new_name in supervised_items:
        dst_path = dst / new_name
        shutil.copy2(src_path, dst_path)
        copied.append((dst_path, angle))

    splits = split_items(copied, args.seed, args.train_ratio, args.val_ratio)
    for mode, rows in splits.items():
        with (dst / f"{mode}.txt").open("w", encoding="utf-8") as f:
            for img_path, angle in rows:
                f.write(f"{img_path.as_posix()} {angle:.6f}\n")
        with (dst / f"{mode}_clean.txt").open("w", encoding="utf-8") as f:
            for img_path, angle in rows:
                f.write(f"{img_path.as_posix()} {angle:.6f}\n")

    summary = {
        "source": str(src),
        "destination": str(dst),
        "seed": args.seed,
        "trainRatio": args.train_ratio,
        "valRatio": args.val_ratio,
        "labelShiftFrames": args.label_shift,
        "numImages": len(copied),
        "numSourceImages": len(items),
        "droppedTailFrames": shift_stats["droppedTailFrames"],
        "droppedNonConsecutive": shift_stats["droppedNonConsecutive"],
        "splits": {mode: len(rows) for mode, rows in splits.items()},
        "uniqueAngles": sorted({round(angle, 6) for _, angle in copied}),
    }
    (dst / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_file = dst_root / "latest_real_steering_dataset.txt"
    latest_file.write_text(str(dst), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"latest_dataset_file={latest_file}")


if __name__ == "__main__":
    main()
