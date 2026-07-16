#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Select the smallest locked->fine-tuned weight interpolation with useful val gain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_DIR = REPO_ROOT / "locked_model"
if str(LOCKED_DIR) not in sys.path:
    sys.path.insert(0, str(LOCKED_DIR))

import train as locked_train  # type: ignore  # noqa: E402
from datasets import AutoDrive2DDataset  # type: ignore  # noqa: E402
from models import AutoDriveNetSeqCfC  # type: ignore  # noqa: E402
from steering_preprocess import preprocess_config_from_dict  # type: ignore  # noqa: E402


def blend_states(base: dict[str, torch.Tensor], tuned: dict[str, torch.Tensor], alpha: float) -> dict[str, torch.Tensor]:
    if base.keys() != tuned.keys():
        raise ValueError("checkpoint state keys differ")
    result = {}
    for key in base:
        left, right = base[key], tuned[key]
        if left.shape != right.shape:
            raise ValueError(f"shape mismatch for {key}: {left.shape} vs {right.shape}")
        result[key] = torch.lerp(left, right, alpha) if left.is_floating_point() else left.clone()
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--locked", required=True)
    p.add_argument("--tuned", required=True)
    p.add_argument("--label-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-relative-val-gain", type=float, default=0.05)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.125, 0.25, 0.5, 0.75, 1.0])
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    locked = torch.load(args.locked, map_location="cpu")
    tuned = torch.load(args.tuned, map_location="cpu")
    preprocess = preprocess_config_from_dict(locked.get("preprocess"))
    val_ds = AutoDrive2DDataset(args.label_csv, split="val", preprocess=preprocess, num_frames=3, frame_stride=1)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    model = AutoDriveNetSeqCfC(num_frames=3).to(device)

    records = []
    for alpha in [0.0, *args.alphas]:
        state = blend_states(locked["model"], tuned["model"], float(alpha))
        model.load_state_dict(state, strict=True)
        metric = locked_train._run_epoch(model, val_loader, device=device, optimizer=None, ey_loss_weight=0.35)
        records.append({"alpha": float(alpha), "valSteeringMAE": float(metric["steeringMAE"]), "valEyMAE": float(metric["eyMAE"])})
        print(f"alpha={alpha:.3f} val={metric['steeringMAE']:.6f}")

    initial = records[0]["valSteeringMAE"]
    threshold = initial * (1.0 - args.min_relative_val_gain)
    eligible = [row for row in records[1:] if row["valSteeringMAE"] <= threshold]
    selected = min(eligible, key=lambda row: row["alpha"]) if eligible else min(records, key=lambda row: row["valSteeringMAE"])
    selected_alpha = float(selected["alpha"])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(locked)
    payload["model"] = blend_states(locked["model"], tuned["model"], selected_alpha)
    payload["epoch"] = -1
    payload["bestValSteeringMAE"] = float(selected["valSteeringMAE"])
    payload["experimentVariant"] = "locked_warm_weight_interpolation"
    payload["interpolationAlpha"] = selected_alpha
    checkpoint_path = output / "best_seq_cfc_temporal3_2d.pth"
    torch.save(payload, checkpoint_path)
    summary = {
        "selectionData": "dataset/26合并:val",
        "heldOutTest": "dataset/25地下室",
        "selectionRule": "smallest alpha reaching configured relative validation gain; otherwise best validation MAE",
        "minRelativeValGain": args.min_relative_val_gain,
        "threshold": threshold,
        "selectedAlpha": selected_alpha,
        "selectedValSteeringMAE": selected["valSteeringMAE"],
        "grid": records,
        "checkpoint": str(checkpoint_path),
    }
    (output / "interpolation_selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
