#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 20260625

SOURCE_NAMES = [
    "26\u5317\u7406\u5de5l",
    "26\u5317\u7406\u5de5r",
    "25\u8302\u534e",
    "25\u676d\u5dde",
]
OUTPUT_NAME = "26\u5408\u5e76"


def parse_angle(path: Path) -> float:
    if "_" not in path.stem:
        raise ValueError(f"Cannot parse angle from filename: {path}")
    return float(path.stem.rsplit("_", 1)[1])


def read_split(split_path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with split_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            image_text, angle_text = line.rsplit(" ", 1)
            rows.append((image_text, float(angle_text)))
    return rows


def collect_images(source_root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=lambda path: str(path.relative_to(source_root)).lower(),
    )


def resolve_split_image(
    source_root: Path,
    split_file: Path,
    image_text: str,
    basename_map: dict[str, list[Path]],
    relative_map: dict[str, Path],
) -> Path:
    normalized = image_text.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    raw = Path(normalized)
    candidates = [source_root / raw, split_file.parent / raw]
    if raw.is_absolute():
        candidates.insert(0, raw)

    cleaned_parts = [part for part in raw.parts if part != "."]
    if cleaned_parts:
        candidates.append(source_root.joinpath(*cleaned_parts))
        relative_match = relative_map.get("/".join(cleaned_parts).lower())
        if relative_match is not None:
            return relative_match

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = basename_map.get(raw.name, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        sample_keys = list(relative_map.keys())[:5]
        raise FileNotFoundError(
            f"Ambiguous basename {raw.name!r} in {source_root}; "
            f"normalized={normalized!r}; cleaned_parts={cleaned_parts!r}; "
            f"relative_key={'/'.join(cleaned_parts).lower()!r}; sample_keys={sample_keys!r}"
        )
    raise FileNotFoundError(f"Missing image for split entry {image_text!r} in {split_file}")


def write_split(output_root: Path, split_name: str, rows: list[tuple[Path, float]]) -> None:
    rows = sorted(rows, key=lambda row: str(row[0]))
    text = "".join(f"{path.as_posix()} {angle:.4f}\n" for path, angle in rows)
    for filename in (f"{split_name}.txt", f"{split_name}_clean.txt"):
        (output_root / filename).write_text(text, encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = repo_root / "dataset"
    output_root = dataset_root / OUTPUT_NAME
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise RuntimeError(f"Output directory is not empty; refusing to overwrite: {output_root}")

    rng = random.Random(SEED)
    merged_splits: dict[str, list[tuple[Path, float]]] = {"train": [], "val": [], "test": []}
    summary: dict[str, object] = {
        "seed": SEED,
        "outputRoot": str(output_root.resolve()),
        "sources": [],
        "sampleCounts": {},
        "labelPolicy": "Parse the steering angle from the filename suffix after the last underscore.",
        "layoutPolicy": "Copy each source into its own subdirectory to avoid filename collisions.",
    }

    for source_name in SOURCE_NAMES:
        source_root = dataset_root / source_name
        if not source_root.is_dir():
            raise FileNotFoundError(source_root)

        images = collect_images(source_root)
        basename_map: dict[str, list[Path]] = {}
        relative_map: dict[str, Path] = {}
        for image in images:
            basename_map.setdefault(image.name, []).append(image.resolve())
            relative_key = image.relative_to(source_root).as_posix().lower()
            relative_map[relative_key] = image.resolve()

        copied: dict[Path, Path] = {}
        source_output = output_root / source_name
        for image in images:
            relative = image.relative_to(source_root)
            destination = source_output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, destination)
            copied[image.resolve()] = destination.resolve()

        source_counts: dict[str, object] = {"name": source_name, "images": len(images), "splits": {}}

        rows = [(copied[image.resolve()], parse_angle(image)) for image in images]
        rng.shuffle(rows)
        val_count = max(1, round(len(rows) * 0.15)) if len(rows) >= 3 else 0
        test_count = max(1, round(len(rows) * 0.15)) if len(rows) >= 3 else 0
        generated = {
            "val": sorted(rows[:val_count], key=lambda row: str(row[0])),
            "test": sorted(rows[val_count : val_count + test_count], key=lambda row: str(row[0])),
            "train": sorted(rows[val_count + test_count :], key=lambda row: str(row[0])),
        }
        for split_name, split_rows in generated.items():
            merged_splits[split_name].extend(split_rows)
            source_counts["splits"][split_name] = len(split_rows)

        summary["sources"].append(source_counts)

    for split_name, rows in merged_splits.items():
        write_split(output_root, split_name, rows)
        summary["sampleCounts"][split_name] = len(rows)

    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
