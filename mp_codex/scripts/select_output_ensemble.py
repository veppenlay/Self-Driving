#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Select an output ensemble on 26-merge val, then materialize one held-out prediction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def aligned(base: list[dict[str, str]], tuned: list[dict[str, str]]) -> None:
    if len(base) != len(tuned):
        raise ValueError("prediction row counts differ")
    for index, (left, right) in enumerate(zip(base, tuned)):
        keys = ("split", "sequence", "frame", "steeringGT")
        if any(left.get(key) != right.get(key) for key in keys):
            raise ValueError(f"prediction rows are not aligned at index {index}")


def mae(rows_a: list[dict[str, str]], rows_b: list[dict[str, str]], alpha: float) -> float:
    truth = np.asarray([float(row["steeringGT"]) for row in rows_a])
    pred_a = np.asarray([float(row["steeringPred"]) for row in rows_a])
    pred_b = np.asarray([float(row["steeringPred"]) for row in rows_b])
    return float(np.mean(np.abs((1.0 - alpha) * pred_a + alpha * pred_b - truth)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-val", required=True)
    p.add_argument("--tuned-val", required=True)
    p.add_argument("--base-test", required=True)
    p.add_argument("--tuned-test", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-relative-val-gain", type=float, default=0.05)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.125, 0.25, 0.5, 0.75, 1.0])
    args = p.parse_args()
    base_val, tuned_val = read(Path(args.base_val)), read(Path(args.tuned_val))
    aligned(base_val, tuned_val)
    grid = [{"alpha": alpha, "valSteeringMAE": mae(base_val, tuned_val, alpha)} for alpha in [0.0, *args.alphas]]
    threshold = grid[0]["valSteeringMAE"] * (1.0 - args.min_relative_val_gain)
    eligible = [row for row in grid[1:] if row["valSteeringMAE"] <= threshold]
    selected = min(eligible, key=lambda row: row["alpha"]) if eligible else min(grid, key=lambda row: row["valSteeringMAE"])
    alpha = float(selected["alpha"])

    # The held-out rows are opened only after the validation-only selection is fixed.
    base_test, tuned_test = read(Path(args.base_test)), read(Path(args.tuned_test))
    aligned(base_test, tuned_test)
    output_rows = []
    for left, right in zip(base_test, tuned_test):
        row = dict(left)
        prediction = (1.0 - alpha) * float(left["steeringPred"]) + alpha * float(right["steeringPred"])
        row["steeringPred"] = prediction
        row["steeringAbsError"] = abs(prediction - float(row["steeringGT"]))
        row["ensembleAlpha"] = alpha
        output_rows.append(row)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "predictions_test.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "selectionData": "dataset/26合并:val",
        "heldOutTest": "dataset/25地下室",
        "selectionRule": "smallest alpha reaching 5% validation gain",
        "grid": grid,
        "threshold": threshold,
        "selectedAlpha": alpha,
        "selectedValSteeringMAE": selected["valSteeringMAE"],
        "testPredictions": str(csv_path),
    }
    (output / "ensemble_selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
