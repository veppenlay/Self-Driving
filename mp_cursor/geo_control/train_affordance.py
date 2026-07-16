#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1 training: regress geometric affordances [steering(aux), e_y, theta] with strong augmentation.

- Trains ONLY on `dataset/26合并` (splits train/val inside labels_geo.csv).
- e_y and theta are the primary quality-weighted losses; steering is a weak aux head.
- Model selection uses val e_y MAE (the primary control channel), NOT steering MAE.
- Strong synthetic augmentation (zero target-domain priors) is applied to train frames only.

Run on the GPU environment. Example:
  python -m mp_cursor.geo_control.train_affordance \
    --label-csv mp_cursor/geo_control/labels/labels_geo.csv \
    --output-dir mp_cursor/exp_P1_affordance/checkpoints --epochs 150
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_DIR = REPO_ROOT / "locked_model"
if str(LOCKED_DIR) not in sys.path:
    sys.path.insert(0, str(LOCKED_DIR))

from steering_preprocess import PreprocessConfig, preprocess_config_to_dict  # type: ignore  # noqa: E402

from .augment import AugmentConfig, PhotometricAugmentor
from .datasets_geo import AffordanceDataset
from .models_affordance import AFFORDANCE_OUTPUT_NAMES, AffordanceCfCNet


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train P1 affordance model (e_y/theta) with augmentation.")
    p.add_argument("--label-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-frames", type=int, default=3)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--steer-loss-weight", type=float, default=0.1)
    p.add_argument("--ey-loss-weight", type=float, default=1.0)
    p.add_argument("--theta-loss-weight", type=float, default=0.5)
    p.add_argument("--aug-strength", type=float, default=1.0)
    p.add_argument("--no-augment", action="store_true", help="Ablation: disable augmentation (expect worse robustness).")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260629)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--best-save-name", default="best_affordance_p1.pth")
    p.add_argument("--save-name", default="affordance_p1.pth")
    p.add_argument("--early-stop-patience-epochs", type=int, default=12)
    p.add_argument("--early-stop-min-delta", type=float, default=0.002)
    p.add_argument("--max-train-batches", type=int, default=0, help="Smoke test: cap batches/epoch (0=all).")
    return p.parse_args()


def _device(req: str) -> torch.device:
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_epoch(model, loader, *, device, optimizer, weights, max_batches: int = 0) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    crit = nn.SmoothL1Loss(reduction="none")
    w_steer, w_ey, w_theta = weights
    totals = {"loss": 0.0, "steer": 0.0, "ey": 0.0, "theta": 0.0}
    ey_wsum = 0.0
    th_wsum = 0.0
    count = 0
    for bi, (images, labels, meta) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        q_ey = meta["eyQuality"].to(device).view(-1)
        q_th = meta["thetaQuality"].to(device).view(-1)
        with torch.set_grad_enabled(training):
            pred = model(images)
            if pred.shape[-1] != 3:
                raise RuntimeError(f"affordance model must output [B,3], got {tuple(pred.shape)}")
            per = crit(pred, labels)
            steer_loss = per[:, 0].mean()
            ey_loss = (per[:, 1] * q_ey).sum() / torch.clamp(q_ey.sum(), min=1.0)
            theta_loss = (per[:, 2] * q_th).sum() / torch.clamp(q_th.sum(), min=1.0)
            loss = w_steer * steer_loss + w_ey * ey_loss + w_theta * theta_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        b = labels.size(0)
        totals["loss"] += float(loss.item()) * b
        totals["steer"] += float((pred[:, 0] - labels[:, 0]).abs().sum().item())
        totals["ey"] += float(((pred[:, 1] - labels[:, 1]).abs() * q_ey).sum().item())
        totals["theta"] += float(((pred[:, 2] - labels[:, 2]).abs() * q_th).sum().item())
        ey_wsum += float(q_ey.sum().item())
        th_wsum += float(q_th.sum().item())
        count += b
    return {
        "loss": totals["loss"] / max(1, count),
        "steeringMAE": totals["steer"] / max(1, count),
        "eyMAE": totals["ey"] / max(1e-6, ey_wsum),
        "thetaMAE": totals["theta"] / max(1e-6, th_wsum),
    }


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    device = _device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess = PreprocessConfig(color_space="hsv", input_size=(144, 192), use_roi=True, illumination_profile="none")
    aug_cfg = AugmentConfig(strength=float(args.aug_strength))
    augmentor = None if args.no_augment else PhotometricAugmentor(aug_cfg, seed=args.seed)

    common = dict(preprocess=preprocess, num_frames=args.num_frames, frame_stride=args.frame_stride)
    train_ds = AffordanceDataset(args.label_csv, split="train", augmentor=augmentor, **common)
    val_ds = AffordanceDataset(args.label_csv, split="val", **common)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = AffordanceCfCNet(num_frames=args.num_frames).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    weights = (args.steer_loss_weight, args.ey_loss_weight, args.theta_loss_weight)

    best_val_ey = float("inf")
    best_path = output_dir / args.best_save_name
    latest_path = output_dir / args.save_name
    summary_path = output_dir / "training_summary_p1.json"
    history: list[dict[str, Any]] = []
    epochs_no_improve = 0
    stopped_epoch = None

    def _payload(epoch: int) -> dict[str, Any]:
        return {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "modelVariant": "affordance_cfc_temporal3",
            "outputDim": 3,
            "outputNames": list(AFFORDANCE_OUTPUT_NAMES),
            "targetNames": list(AFFORDANCE_OUTPUT_NAMES),
            "preprocess": preprocess_config_to_dict(preprocess),
            "numFrames": int(args.num_frames),
            "frameStride": int(args.frame_stride),
            "lossWeights": {"steering": weights[0], "e_y": weights[1], "theta": weights[2]},
            "augment": {"enabled": augmentor is not None, "strength": float(args.aug_strength)},
            "bestValEyMAE": float(best_val_ey),
        }

    for epoch in range(1, args.epochs + 1):
        tr = _run_epoch(model, train_loader, device=device, optimizer=optimizer, weights=weights, max_batches=args.max_train_batches)
        va = _run_epoch(model, val_loader, device=device, optimizer=None, weights=weights, max_batches=args.max_train_batches)
        scheduler.step()
        history.append({"epoch": epoch, "train": tr, "val": va, "lr": float(scheduler.get_last_lr()[0])})
        print(f"epoch={epoch:03d} train_loss={tr['loss']:.5f} val_ey_mae={va['eyMAE']:.5f} "
              f"val_theta_mae={va['thetaMAE']:.5f} val_steer_mae={va['steeringMAE']:.5f}")
        torch.save(_payload(epoch), latest_path)
        if va["eyMAE"] < best_val_ey - args.early_stop_min_delta:
            best_val_ey = va["eyMAE"]
            torch.save(_payload(epoch), best_path)
            epochs_no_improve = 0
        else:
            if va["eyMAE"] < best_val_ey:
                best_val_ey = va["eyMAE"]
                torch.save(_payload(epoch), best_path)
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience_epochs:
                stopped_epoch = epoch
                print(f"early_stop at epoch={epoch} best_val_ey_mae={best_val_ey:.6f}")
                break

    summary = {
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "labelCsv": str(Path(args.label_csv).resolve()),
        "outputDir": str(output_dir),
        "bestCheckpoint": str(best_path),
        "bestValEyMAE": best_val_ey,
        "augment": {"enabled": augmentor is not None, "strength": float(args.aug_strength)},
        "lossWeights": {"steering": weights[0], "e_y": weights[1], "theta": weights[2]},
        "stoppedEpoch": stopped_epoch,
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"best={best_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
