#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare a formal-view steering dataset using explicit run-based splits."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_angle(path: Path) -> float:
    if "_" not in path.stem:
        raise ValueError("missing angle suffix")
    return float(path.stem.rsplit("_", 1)[-1])


def collect_images(run_dir: Path) -> list[tuple[Path, float]]:
    items: list[tuple[Path, float]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            angle = parse_angle(path)
        except ValueError:
            continue
        items.append((path, angle))
    return items


def parse_run_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for value in values:
        if "=" in value:
            name, raw_path = value.split("=", 1)
            specs.append((name.strip(), Path(raw_path).expanduser().resolve()))
        else:
            run_dir = Path(value).expanduser().resolve()
            specs.append((run_dir.name, run_dir))
    return specs


def write_split(split_path: Path, rows: list[tuple[Path, float]]) -> None:
    with split_path.open("w", encoding="utf-8") as f:
        for img_path, angle in rows:
            f.write(f"{img_path.as_posix()} {angle:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a formal-view dataset with explicit run-based splits.")
    parser.add_argument("--dst-root", required=True, help="Project dataset root, e.g. E:/桌面/项目/dataset")
    parser.add_argument("--name", default=None, help="Dataset directory name. Defaults to formal_view_data_<timestamp>.")
    parser.add_argument("--train-run", action="append", default=[], help="Format: run_name=path. Can be repeated.")
    parser.add_argument("--val-run", action="append", default=[], help="Format: run_name=path. Can be repeated.")
    parser.add_argument("--test-run", action="append", default=[], help="Format: run_name=path. Can be repeated.")
    parser.add_argument("--val-style-run", action="append", default=[], help="Format: run_name=path. Can be repeated.")
    parser.add_argument("--copy-images", action="store_true", help="Copy images into the dataset directory instead of referencing original paths.")
    args = parser.parse_args()

    train_specs = parse_run_specs(args.train_run)
    val_specs = parse_run_specs(args.val_run)
    test_specs = parse_run_specs(args.test_run)
    val_style_specs = parse_run_specs(args.val_style_run)
    if not train_specs or not val_specs or not test_specs:
        raise RuntimeError("train/val/test runs must all be provided explicitly")

    dst_root = Path(args.dst_root).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)
    name = args.name or f"formal_view_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dst = dst_root / name
    if dst.exists():
        raise FileExistsError(f"destination already exists: {dst}")
    dst.mkdir(parents=True)

    split_specs = {
        "train_clean": train_specs,
        "val_clean": val_specs,
        "test_clean": test_specs,
        "val_style_real": val_style_specs,
    }
    copied_rows: dict[str, list[tuple[Path, float]]] = {name: [] for name in split_specs}
    summary_runs: dict[str, list[dict[str, object]]] = {}

    for split_name, specs in split_specs.items():
        rows: list[tuple[Path, float]] = []
        run_infos: list[dict[str, object]] = []
        for run_name, run_dir in specs:
            if not run_dir.is_dir():
                raise NotADirectoryError(f"run directory not found: {run_dir}")
            items = collect_images(run_dir)
            if not items:
                raise RuntimeError(f"no valid images found under {run_dir}")
            run_infos.append({"run": run_name, "path": str(run_dir), "count": len(items)})
            for src_path, angle in items:
                if args.copy_images:
                    rel_dir = dst / "images" / split_name / run_name
                    rel_dir.mkdir(parents=True, exist_ok=True)
                    dst_path = rel_dir / src_path.name
                    if dst_path.exists():
                        stem = src_path.stem
                        dst_path = rel_dir / f"{stem}_{len(rows):06d}{src_path.suffix.lower()}"
                    shutil.copy2(src_path, dst_path)
                    rows.append((dst_path, angle))
                else:
                    rows.append((src_path, angle))
        copied_rows[split_name] = rows
        summary_runs[split_name] = run_infos
        write_split(dst / f"{split_name}.txt", rows)

    # Compatibility splits for older training/eval scripts.
    write_split(dst / "train.txt", copied_rows["train_clean"])
    write_split(dst / "val.txt", copied_rows["val_clean"])
    write_split(dst / "test.txt", copied_rows["test_clean"])

    summary = {
        "destination": str(dst),
        "copyImages": bool(args.copy_images),
        "splits": {split_name: len(rows) for split_name, rows in copied_rows.items()},
        "runs": summary_runs,
        "uniqueAngles": sorted({round(angle, 6) for rows in copied_rows.values() for _, angle in rows}),
        "formalViewContract": {
            "colorSpace": "hsv",
            "inputSize": [144, 192],
            "useRoi": False,
        },
    }
    (dst / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_file = dst_root / "latest_formal_view_dataset.txt"
    latest_file.write_text(str(dst), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"latest_dataset_file={latest_file}")


if __name__ == "__main__":
    main()
