#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate a 2D checkpoint on labels_2d.csv."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from datasets import AutoDrive2DDataset  # noqa: E402
from models import build_model_for_checkpoint  # noqa: E402
from steering_preprocess import DEFAULT_PREPROCESS_CONFIG, preprocess_config_from_dict  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 2D steering/e_y checkpoint.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--label-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def _device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(ckpt_path: Path, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    variant = checkpoint.get("modelVariant") if isinstance(checkpoint, dict) else None
    model = build_model_for_checkpoint(state, variant).to(device)
    model.eval()
    preprocess = preprocess_config_from_dict(checkpoint.get("preprocess") if isinstance(checkpoint, dict) else None, fallback=DEFAULT_PREPROCESS_CONFIG)
    num_frames = int(checkpoint.get("numFrames", getattr(model, "num_frames", 3))) if isinstance(checkpoint, dict) else int(getattr(model, "num_frames", 3))
    frame_stride = int(checkpoint.get("frameStride", 1)) if isinstance(checkpoint, dict) else 1
    return model, preprocess, num_frames, frame_stride, checkpoint


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mae(values: np.ndarray, target: np.ndarray, weight: np.ndarray | None = None) -> float:
    err = np.abs(values - target)
    if weight is None:
        return float(err.mean()) if err.size else float("nan")
    denom = float(weight.sum())
    if denom <= 1e-8:
        return float("nan")
    return float((err * weight).sum() / denom)


def _plot_predictions(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    if not rows:
        return
    x = np.arange(len(rows))
    steering_gt = np.asarray([float(row["steeringGT"]) for row in rows])
    steering_pred = np.asarray([float(row["steeringPred"]) for row in rows])
    ey_gt = np.asarray([float(row["eyGT"]) for row in rows])
    ey_pred = np.asarray([float(row["eyPred"]) for row in rows])
    ey_quality = np.asarray([float(row["eyQuality"]) for row in rows])

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(x, steering_gt, label="GT steering", color="#111111", linewidth=2.0)
    axes[0].plot(x, steering_pred, label="Pred steering", color="#1f77b4", linewidth=1.6)
    axes[0].set_ylabel("Steering")
    axes[0].set_title(f"{title} steering MAE={_mae(steering_pred, steering_gt):.6f}")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(x, ey_gt, label="GT e_y pseudo-label", color="#111111", linewidth=2.0)
    axes[1].plot(x, ey_pred, label="Pred e_y", color="#d62728", linewidth=1.6)
    invalid = np.where(ey_quality <= 0)[0]
    if invalid.size:
        axes[1].scatter(invalid, ey_gt[invalid], label="e_y quality=0", color="#ff7f0e", s=12, alpha=0.75)
    axes[1].set_ylabel("e_y")
    axes[1].set_xlabel("Sample index")
    axes[1].set_title(f"{title} e_y weighted MAE={_mae(ey_pred, ey_gt, ey_quality):.6f}")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    device = _device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, preprocess, num_frames, frame_stride, checkpoint = _load_model(Path(args.ckpt).resolve(), device)
    dataset = AutoDrive2DDataset(
        args.label_csv,
        split=args.split,
        preprocess=preprocess,
        num_frames=num_frames,
        frame_stride=frame_stride,
        include_zero_quality=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    rows: list[dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for images, labels, meta in loader:
            images = images.to(device, non_blocking=True)
            pred = model(images).detach().cpu().numpy()
            labels_np = labels.numpy()
            quality = meta["eyQuality"].numpy()
            paths = meta["path"]
            sequences = meta["sequence"]
            frames = meta["frame"].numpy()
            status = meta["status"]
            for batch_idx in range(pred.shape[0]):
                rows.append(
                    {
                        "index": offset,
                        "split": args.split,
                        "sequence": str(sequences[batch_idx]),
                        "frame": int(frames[batch_idx]),
                        "image": str(paths[batch_idx]),
                        "status": str(status[batch_idx]),
                        "eyQuality": float(quality[batch_idx]),
                        "steeringGT": float(labels_np[batch_idx, 0]),
                        "eyGT": float(labels_np[batch_idx, 1]),
                        "steeringPred": float(pred[batch_idx, 0]),
                        "eyPred": float(pred[batch_idx, 1]),
                        "steeringAbsError": abs(float(pred[batch_idx, 0] - labels_np[batch_idx, 0])),
                        "eyAbsError": abs(float(pred[batch_idx, 1] - labels_np[batch_idx, 1])),
                    }
                )
                offset += 1

    steering_gt = np.asarray([float(row["steeringGT"]) for row in rows])
    steering_pred = np.asarray([float(row["steeringPred"]) for row in rows])
    ey_gt = np.asarray([float(row["eyGT"]) for row in rows])
    ey_pred = np.asarray([float(row["eyPred"]) for row in rows])
    ey_quality = np.asarray([float(row["eyQuality"]) for row in rows])
    summary = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "labelCsv": str(Path(args.label_csv).resolve()),
        "split": args.split,
        "count": len(rows),
        "steeringMAE": _mae(steering_pred, steering_gt),
        "eyWeightedMAE": _mae(ey_pred, ey_gt, ey_quality),
        "eyQualitySum": float(ey_quality.sum()),
        "preprocess": checkpoint.get("preprocess") if isinstance(checkpoint, dict) else None,
        "numFrames": num_frames,
        "frameStride": frame_stride,
    }
    _write_csv(output_dir / f"predictions_{args.split}.csv", rows)
    (output_dir / f"summary_{args.split}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot_predictions(output_dir / f"predictions_{args.split}.png", rows, f"2D evaluation split={args.split}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
