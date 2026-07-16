#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate 2D labels: steering angle + lane-center e_y pseudo-label."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from steering_preprocess import imread_bgr  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_REFERENCE_Y_RATIO = 0.54
STATUS_QUALITY = {
    "ok": 1.0,
    "inferred_missing_left": 0.6,
    "inferred_missing_right": 0.6,
    "insufficient_yellow_edges": 0.0,
    "narrow_lane_width": 0.0,
}


@dataclass(frozen=True)
class ImageItem:
    path: Path
    steering: float
    sequence: str
    frame: int
    split: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate steering + e_y label sidecar CSV for 2D training.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-y-ratio", type=float, default=DEFAULT_REFERENCE_Y_RATIO)
    parser.add_argument("--split-file", action="append", default=[], help="Format: split=path. Can be repeated.")
    parser.add_argument("--test-root", default=None, help="Optional external dataset root scanned entirely as split=test.")
    parser.add_argument("--test-split-name", default="test", help="Split name used for --test-root rows.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--copy-visualization", action="store_true", help="Write preview images for calibration.")
    parser.add_argument("--visualization-limit", type=int, default=160)
    return parser.parse_args()


def _sort_key(path: Path) -> tuple[int, str]:
    prefix = path.stem.split("_", 1)[0]
    return (int(prefix) if prefix.isdigit() else 10**12, path.name.lower())


def _parse_steering(path: Path) -> float:
    if "_" not in path.stem:
        raise ValueError(f"missing steering suffix: {path}")
    return float(path.stem.rsplit("_", 1)[1])


def _parse_frame(path: Path) -> int:
    match = re.match(r"^(\d+)_", path.name)
    if not match:
        raise ValueError(f"missing frame prefix: {path}")
    return int(match.group(1))


def _sequence_name(path: Path, dataset_root: Path) -> str:
    try:
        rel = path.relative_to(dataset_root)
        return rel.parts[0] if len(rel.parts) > 1 else "."
    except ValueError:
        return path.parent.name


def _build_basename_index(dataset_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            index.setdefault(path.name, []).append(path)
    return index


def _resolve_split_image(image_text: str, split_file: Path, dataset_root: Path, basename_index: dict[str, list[Path]]) -> Path | None:
    raw = Path(image_text)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([split_file.parent / raw, dataset_root / raw])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = basename_index.get(raw.name, [])
    if len(matches) == 1:
        return matches[0].resolve()
    return None


def _collect_from_split_files(dataset_root: Path, specs: list[str]) -> list[ImageItem]:
    basename_index = _build_basename_index(dataset_root)
    items: list[ImageItem] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"invalid --split-file spec, expected split=path: {spec}")
        split, raw_path = spec.split("=", 1)
        split_name = split.strip()
        split_file = Path(raw_path).expanduser().resolve()
        for line_no, line in enumerate(split_file.read_text(encoding="utf-8-sig").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            try:
                image_text, steering_text = text.rsplit(maxsplit=1)
                steering = float(steering_text)
            except ValueError:
                image_text = text
                steering = None
            image_path = _resolve_split_image(image_text, split_file, dataset_root, basename_index)
            if image_path is None:
                print(f"WARNING missing split image {image_text!r} at {split_file}:{line_no}")
                continue
            if steering is None:
                steering = _parse_steering(image_path)
            items.append(
                ImageItem(
                    path=image_path,
                    steering=float(steering),
                    sequence=_sequence_name(image_path, dataset_root),
                    frame=_parse_frame(image_path),
                    split=split_name,
                )
            )
    return sorted(items, key=lambda item: (item.split or "", item.sequence, item.frame, str(item.path)))


def _collect_from_images(dataset_root: Path, *, seed: int, train_ratio: float, val_ratio: float) -> list[ImageItem]:
    grouped: dict[str, list[Path]] = {}
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            try:
                _parse_steering(path)
                _parse_frame(path)
            except ValueError:
                continue
            grouped.setdefault(_sequence_name(path, dataset_root), []).append(path.resolve())

    rng = random.Random(seed)
    items: list[ImageItem] = []
    for sequence, paths in sorted(grouped.items()):
        paths = sorted(paths, key=_sort_key)
        n = len(paths)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(max(1, n_train), max(1, n - 2)) if n >= 3 else n
        n_val = min(max(1, n_val), max(0, n - n_train - 1)) if n >= 3 else 0
        # Keep temporal order inside each split. Shuffle only sequence order via deterministic no-op seed hook.
        rng.random()
        splits = (
            [("train", path) for path in paths[:n_train]]
            + [("val", path) for path in paths[n_train : n_train + n_val]]
            + [("test", path) for path in paths[n_train + n_val :]]
        )
        for split, path in splits:
            items.append(
                ImageItem(
                    path=path,
                    steering=_parse_steering(path),
                    sequence=sequence,
                    frame=_parse_frame(path),
                    split=split,
                )
            )
    return sorted(items, key=lambda item: (item.split or "", item.sequence, item.frame, str(item.path)))


def _collect_external_test_root(test_root: Path, split_name: str) -> list[ImageItem]:
    items: list[ImageItem] = []
    for path in test_root.rglob("*"):
        if not (path.is_file() and path.suffix.lower() in IMAGE_EXTS):
            continue
        try:
            steering = _parse_steering(path)
            frame = _parse_frame(path)
        except ValueError:
            continue
        items.append(
            ImageItem(
                path=path.resolve(),
                steering=steering,
                sequence=_sequence_name(path.resolve(), test_root),
                frame=frame,
                split=split_name,
            )
        )
    return sorted(items, key=lambda item: (item.split or "", item.sequence, item.frame, str(item.path)))


def _yellow_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([14, 55, 95], dtype=np.uint8), np.array([42, 255, 255], dtype=np.uint8))
    mask[: int(mask.shape[0] * 0.18), :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _runs_from_profile(profile: np.ndarray, min_width: int, gap: int) -> list[list[int]]:
    active = profile > 0
    runs: list[list[int]] = []
    start: int | None = None
    last: int | None = None
    for idx, value in enumerate(active):
        if value:
            if start is None:
                start = idx
            last = idx
        elif start is not None and last is not None:
            if last - start + 1 >= min_width:
                runs.append([start, last])
            start = None
            last = None
    if start is not None and last is not None and last - start + 1 >= min_width:
        runs.append([start, last])

    merged: list[list[int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= gap:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return merged


def _weighted_center(profile: np.ndarray, run: list[int]) -> float:
    start, stop = int(run[0]), int(run[1])
    xs = np.arange(start, stop + 1, dtype=np.float32)
    weights = profile[start : stop + 1].astype(np.float32)
    total = float(weights.sum())
    if total <= 1e-6:
        return float((start + stop) / 2.0)
    return float((xs * weights).sum() / total)


def _raw_detection(image_bgr: np.ndarray, reference_y_ratio: float) -> dict[str, Any]:
    height, width = image_bgr.shape[:2]
    mask = _yellow_mask(image_bgr)
    reference_y = int(round(height * reference_y_ratio))
    band_half = max(10, int(round(height * 0.035)))
    band = mask[max(0, reference_y - band_half) : min(height, reference_y + band_half + 1), :]
    col_count = band.sum(axis=0) / 255.0
    threshold = max(3.0, 0.18 * band.shape[0])
    profile = np.where(col_count >= threshold, col_count, 0.0)
    runs = _runs_from_profile(profile, min_width=max(5, width // 80), gap=max(3, width // 160))
    centers = [_weighted_center(col_count, run) for run in runs]
    return {
        "imageWidth": width,
        "imageHeight": height,
        "referenceY": reference_y,
        "mask": mask,
        "runs": runs,
        "centers": centers,
    }


def _finalize_detection(raw: dict[str, Any], width_prior: float | None) -> dict[str, Any]:
    width = float(raw["imageWidth"])
    image_center_x = width / 2.0
    runs = raw["runs"]
    centers = raw["centers"]
    left = [(run, x) for run, x in zip(runs, centers) if x < image_center_x]
    right = [(run, x) for run, x in zip(runs, centers) if x >= image_center_x]
    result: dict[str, Any] = {
        "status": "insufficient_yellow_edges",
        "eY": None,
        "eyQuality": 0.0,
        "referenceY": raw["referenceY"],
        "imageCenterX": image_center_x,
        "leftX": None,
        "rightX": None,
        "laneCenterX": None,
        "laneWidthPx": None,
        "inferredLaneWidthPx": None,
        "yellowRunCount": len(runs),
        "yellowRuns": runs,
    }
    if left and right:
        left_run, left_x = max(left, key=lambda item: item[1])
        right_run, right_x = min(right, key=lambda item: item[1])
        lane_width = float(right_x - left_x)
        if lane_width > max(20.0, 0.12 * width):
            lane_center = (left_x + right_x) / 2.0
            result.update(
                {
                    "status": "ok",
                    "eY": (lane_center - image_center_x) / (width / 2.0),
                    "eyQuality": STATUS_QUALITY["ok"],
                    "leftX": left_x,
                    "rightX": right_x,
                    "laneCenterX": lane_center,
                    "laneWidthPx": lane_width,
                    "leftRun": left_run,
                    "rightRun": right_run,
                }
            )
        else:
            result.update({"status": "narrow_lane_width", "leftX": left_x, "rightX": right_x, "laneWidthPx": lane_width})
    elif width_prior is not None and len(runs) == 1:
        run = runs[0]
        x = centers[0]
        if x >= image_center_x:
            right_x = x
            left_x = max(0.0, right_x - width_prior)
            status = "inferred_missing_left"
        else:
            left_x = x
            right_x = min(width - 1.0, left_x + width_prior)
            status = "inferred_missing_right"
        lane_center = (left_x + right_x) / 2.0
        result.update(
            {
                "status": status,
                "eY": (lane_center - image_center_x) / (width / 2.0),
                "eyQuality": STATUS_QUALITY[status],
                "leftX": left_x,
                "rightX": right_x,
                "laneCenterX": lane_center,
                "laneWidthPx": right_x - left_x,
                "inferredLaneWidthPx": width_prior,
            }
        )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_preview(path: Path, image: np.ndarray, detection: dict[str, Any]) -> None:
    vis = image.copy()
    y = int(detection["referenceY"])
    h, w = vis.shape[:2]
    cv2.line(vis, (0, y), (w - 1, y), (0, 255, 255), 2)
    cv2.line(vis, (w // 2, 0), (w // 2, h - 1), (255, 255, 255), 1)
    for key, color in (("leftX", (255, 80, 0)), ("rightX", (0, 80, 255)), ("laneCenterX", (255, 0, 255))):
        if detection.get(key) is not None:
            x = int(round(float(detection[key])))
            cv2.line(vis, (x, max(0, y - 24)), (x, min(h - 1, y + 24)), color, 2)
    label = f"e_y={detection.get('eY')} status={detection.get('status')} quality={detection.get('eyQuality')}"
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, vis)
    if not ok:
        raise RuntimeError(f"failed to encode preview: {path}")
    encoded.tofile(str(path))


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = (
        _collect_from_split_files(dataset_root, args.split_file)
        if args.split_file
        else _collect_from_images(dataset_root, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    )
    if args.test_root:
        test_root = Path(args.test_root).resolve()
        external_items = _collect_external_test_root(test_root, args.test_split_name)
        seen_paths = {item.path.resolve() for item in items}
        added = 0
        for item in external_items:
            if item.path.resolve() in seen_paths:
                continue
            items.append(item)
            seen_paths.add(item.path.resolve())
            added += 1
        print(f"externalTestRoot={test_root} added={added} split={args.test_split_name}")
        items = sorted(items, key=lambda item: (item.split or "", item.sequence, item.frame, str(item.path)))
    if not items:
        raise FileNotFoundError(f"no labelable images found under {dataset_root}")

    raw_cache: list[tuple[ImageItem, np.ndarray, dict[str, Any]]] = []
    direct_widths: list[float] = []
    for item in items:
        image = imread_bgr(item.path)
        if image is None:
            raise FileNotFoundError(f"failed to read image: {item.path}")
        raw = _raw_detection(image, args.reference_y_ratio)
        preliminary = _finalize_detection(raw, width_prior=None)
        if preliminary["status"] == "ok" and preliminary["laneWidthPx"] is not None:
            direct_widths.append(float(preliminary["laneWidthPx"]))
        raw_cache.append((item, image, raw))

    width_prior = float(np.median(direct_widths)) if direct_widths else None
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    preview_dir = output_dir / "preview"
    for idx, (item, image, raw) in enumerate(raw_cache):
        detection = _finalize_detection(raw, width_prior=width_prior)
        status_counts[detection["status"]] = status_counts.get(detection["status"], 0) + 1
        split_name = item.split or "unspecified"
        split_counts[split_name] = split_counts.get(split_name, 0) + 1
        row = {
            "index": idx,
            "split": split_name,
            "sequence": item.sequence,
            "frame": item.frame,
            "image": str(item.path),
            "steering": item.steering,
            "referenceYRatio": float(args.reference_y_ratio),
            "laneWidthPriorPx": width_prior,
            "imageWidth": raw["imageWidth"],
            "imageHeight": raw["imageHeight"],
            **{key: value for key, value in detection.items() if key != "yellowRuns"},
            "yellowRunsJson": json.dumps(detection.get("yellowRuns", []), ensure_ascii=False),
        }
        if args.copy_visualization and idx < args.visualization_limit:
            preview_path = preview_dir / f"{idx:06d}_{item.path.stem}.jpg"
            _write_preview(preview_path, image, detection)
            row["preview"] = str(preview_path)
        rows.append(row)

    csv_path = output_dir / "labels_2d.csv"
    jsonl_path = output_dir / "labels_2d.jsonl"
    summary_path = output_dir / "labels_2d_summary.json"
    _write_csv(csv_path, rows)
    _write_jsonl(jsonl_path, rows)
    summary = {
        "datasetRoot": str(dataset_root),
        "testRoot": str(Path(args.test_root).resolve()) if args.test_root else None,
        "referenceYRatio": float(args.reference_y_ratio),
        "labelCount": len(rows),
        "splitCounts": split_counts,
        "statusCounts": status_counts,
        "laneWidthPriorPx": width_prior,
        "outputs": {"csv": str(csv_path), "jsonl": str(jsonl_path), "summary": str(summary_path)},
        "labelSchema": {
            "target": ["steering", "eY"],
            "eyQuality": STATUS_QUALITY,
            "note": "Rows with eyQuality=0 keep steering labels but should not contribute to e_y loss.",
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
