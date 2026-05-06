#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a merged dataset split from base/aug training roots and a dedicated test root."
    )
    parser.add_argument("--base-root", required=True, help="Training dataset root with split txt files.")
    parser.add_argument("--aug-root", required=True, help="Augmented dataset root with mirrored filenames.")
    parser.add_argument("--test-root", required=True, help="Test dataset root.")
    parser.add_argument("--output-root", required=True, help="Output dataset directory for generated split files.")
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument(
        "--val-groups",
        type=int,
        default=None,
        help="Number of filename groups reserved for validation. Defaults to the count from base val.txt.",
    )
    return parser.parse_args()


def _read_split_entries(split_path: Path) -> list[tuple[str, float]]:
    entries: list[tuple[str, float]] = []
    with split_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            path_text, angle_text = line.rsplit(" ", 1)
            entries.append((path_text, float(angle_text)))
    return entries


def _load_base_labels(base_root: Path) -> tuple[dict[str, float], dict[str, int]]:
    labels: dict[str, float] = {}
    split_counts: dict[str, int] = {}
    for split_name in ("train.txt", "val.txt", "test.txt"):
        split_path = base_root / split_name
        if not split_path.is_file():
            continue
        split_entries = _read_split_entries(split_path)
        split_counts[split_name] = len(split_entries)
        for raw_path, angle in split_entries:
            basename = Path(raw_path).name
            labels[basename] = angle
    if not labels:
        raise FileNotFoundError(f"no base split txt files found under {base_root}")
    return labels, split_counts


def _resolve_image(root: Path, basename: str) -> Path:
    candidate = root / basename
    if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
        return candidate.resolve()
    raise FileNotFoundError(f"missing mirrored image: {candidate}")


def _write_split(path: Path, rows: list[tuple[Path, float]]) -> None:
    lines = [f"{row_path.as_posix()} {angle:.4f}" for row_path, angle in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _resolve_test_image(test_root: Path, raw_path: str) -> Path:
    raw = Path(raw_path)
    direct = (test_root / raw).resolve()
    if direct.is_file():
        return direct

    parts = raw.parts
    for start_idx in range(1, len(parts)):
        candidate = test_root.joinpath(*parts[start_idx:]).resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing test image for split entry: {raw_path}")


def _collect_test_entries(test_root: Path) -> list[tuple[Path, float]]:
    rows: list[tuple[Path, float]] = []
    for split_name in ("train.txt", "val.txt", "test.txt", "test_clean.txt"):
        split_path = test_root / split_name
        if not split_path.is_file():
            continue
        for raw_path, angle in _read_split_entries(split_path):
            resolved = _resolve_test_image(test_root, raw_path)
            rows.append((resolved, angle))
    if not rows:
        raise FileNotFoundError(f"no test split txt files found under {test_root}")
    rows.sort(key=lambda item: str(item[0]))
    return rows


def main() -> None:
    args = _parse_args()

    base_root = Path(args.base_root).expanduser().resolve()
    aug_root = Path(args.aug_root).expanduser().resolve()
    test_root = Path(args.test_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    labels_by_name, split_counts = _load_base_labels(base_root)
    filenames = sorted(labels_by_name.keys())
    val_groups = args.val_groups if args.val_groups is not None else split_counts.get("val.txt", max(1, round(len(filenames) * 0.15)))
    if val_groups <= 0 or val_groups >= len(filenames):
        raise ValueError(f"val_groups must be in (0, {len(filenames)}), got {val_groups}")

    rng = random.Random(args.seed)
    shuffled = filenames[:]
    rng.shuffle(shuffled)
    val_names = set(shuffled[:val_groups])
    train_names = [name for name in filenames if name not in val_names]

    train_rows: list[tuple[Path, float]] = []
    val_rows: list[tuple[Path, float]] = []
    for basename in train_names:
        angle = labels_by_name[basename]
        train_rows.append((_resolve_image(base_root, basename), angle))
        train_rows.append((_resolve_image(aug_root, basename), angle))
    for basename in sorted(val_names):
        angle = labels_by_name[basename]
        val_rows.append((_resolve_image(base_root, basename), angle))
        val_rows.append((_resolve_image(aug_root, basename), angle))

    train_rows.sort(key=lambda item: str(item[0]))
    val_rows.sort(key=lambda item: str(item[0]))
    test_rows = _collect_test_entries(test_root)

    for split_name, rows in (
        ("train_clean.txt", train_rows),
        ("val_clean.txt", val_rows),
        ("test_clean.txt", test_rows),
        ("train.txt", train_rows),
        ("val.txt", val_rows),
        ("test.txt", test_rows),
    ):
        _write_split(output_root / split_name, rows)

    summary = {
        "seed": int(args.seed),
        "baseRoot": str(base_root),
        "augRoot": str(aug_root),
        "testRoot": str(test_root),
        "groupCounts": {
            "all": len(filenames),
            "train": len(train_names),
            "val": len(val_names),
        },
        "sampleCounts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "splitPolicy": {
            "trainSources": ["base-root", "aug-root"],
            "validation": "same-group holdout from merged training pool",
            "test": "dedicated test-root only",
        },
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"output_root={output_root}")
    print(
        f"group_counts all={summary['groupCounts']['all']} train={summary['groupCounts']['train']} val={summary['groupCounts']['val']}"
    )
    print(
        f"sample_counts train={summary['sampleCounts']['train']} val={summary['sampleCounts']['val']} test={summary['sampleCounts']['test']}"
    )


if __name__ == "__main__":
    main()
