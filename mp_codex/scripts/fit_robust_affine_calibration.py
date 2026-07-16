#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fit a robust steering affine calibration on 26-merge train and gate on val."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def read_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    prediction = np.asarray([float(row["steeringPred"]) for row in rows], dtype=np.float64)
    truth = np.asarray([float(row["steeringGT"]) for row in rows], dtype=np.float64)
    return prediction, truth


def robust_affine(x: np.ndarray, y: np.ndarray, iterations: int = 20) -> tuple[float, float]:
    design = np.column_stack([x, np.ones_like(x)])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(iterations):
        residual = y - design @ coefficients
        center = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - center))) + 1e-8
        cutoff = 1.345 * scale
        absolute = np.abs(residual)
        weights = np.ones_like(absolute)
        mask = absolute > cutoff
        weights[mask] = cutoff / absolute[mask]
        weighted = design * np.sqrt(weights[:, None])
        target = y * np.sqrt(weights)
        coefficients = np.linalg.lstsq(weighted, target, rcond=None)[0]
    return float(coefficients[0]), float(coefficients[1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-predictions", required=True)
    p.add_argument("--val-predictions", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-relative-val-gain", type=float, default=0.03)
    args = p.parse_args()
    train_x, train_y = read_xy(Path(args.train_predictions))
    val_x, val_y = read_xy(Path(args.val_predictions))
    slope, intercept = robust_affine(train_x, train_y)
    raw_mae = float(np.mean(np.abs(val_x - val_y)))
    calibrated_mae = float(np.mean(np.abs(slope * val_x + intercept - val_y)))
    relative_gain = 1.0 - calibrated_mae / raw_mae
    accepted = relative_gain >= args.min_relative_val_gain
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_out = output / "best_seq_cfc_temporal3_2d.pth"
    if accepted:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state = checkpoint["model"]
        state["reg_head.weight"][0].mul_(slope)
        state["reg_head.bias"][0].mul_(slope).add_(intercept)
        checkpoint["experimentVariant"] = "robust_affine_steering_calibration"
        checkpoint["steeringCalibration"] = {"slope": slope, "intercept": intercept, "foldedIntoRegHead": True}
        torch.save(checkpoint, checkpoint_out)
    summary = {
        "fitData": "dataset/26合并:train",
        "gateData": "dataset/26合并:val",
        "heldOutTest": "dataset/25地下室",
        "method": "Huber IRLS affine calibration folded into steering head",
        "slope": slope,
        "intercept": intercept,
        "rawValMAE": raw_mae,
        "calibratedValMAE": calibrated_mae,
        "relativeValGain": relative_gain,
        "minimumRelativeValGain": args.min_relative_val_gain,
        "acceptedForExternalTest": accepted,
        "checkpoint": str(checkpoint_out) if accepted else None,
    }
    (output / "calibration_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
