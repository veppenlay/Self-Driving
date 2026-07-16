#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
MERGED_NAME = "26\u5408\u5e76"
TEST_NAME = "25\u5730\u4e0b\u5ba4"


def parse_angle(path: Path) -> float:
    if "_" not in path.stem:
        raise ValueError(f"Cannot parse angle from filename: {path}")
    return float(path.stem.rsplit("_", 1)[1])


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = repo_root / "dataset"
    merged_root = dataset_root / MERGED_NAME
    test_root = dataset_root / TEST_NAME
    if not merged_root.is_dir():
        raise FileNotFoundError(merged_root)
    if not test_root.is_dir():
        raise FileNotFoundError(test_root)

    images = sorted(
        [
            path
            for path in test_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=lambda path: str(path.relative_to(test_root)).lower(),
    )
    if not images:
        raise FileNotFoundError(f"No test images found under {test_root}")

    lines = [f"{path.resolve().as_posix()} {parse_angle(path):.4f}" for path in images]
    text = "\n".join(lines) + "\n"
    for filename in ("test.txt", "test_clean.txt"):
        (merged_root / filename).write_text(text, encoding="utf-8")

    summary_path = merged_root / "dataset_summary.json"
    summary = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sample_counts = summary.setdefault("sampleCounts", {})
    sample_counts["test"] = len(images)
    summary["testSourceOverride"] = {
        "name": TEST_NAME,
        "path": str(test_root.resolve()),
        "images": len(images),
        "labelPolicy": "Parse the steering angle from the filename suffix after the last underscore.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"test_images={len(images)}")
    print(f"merged_root={merged_root}")
    print(f"test_root={test_root}")


if __name__ == "__main__":
    main()
