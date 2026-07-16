#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paired low-LR fine-tuning from the locked checkpoint, with optional augmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_DIR = REPO_ROOT / "locked_model"
SRC_DIR = REPO_ROOT / "mp_codex" / "src"
for path in (LOCKED_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train as locked_train  # type: ignore  # noqa: E402
from models import AutoDriveNetSeqCfC  # type: ignore  # noqa: E402
from steering_preprocess import PreprocessConfig  # type: ignore  # noqa: E402
from warm_dataset import WarmAutoDrive2DDataset  # noqa: E402
from warm_photometric import WarmPhotometricAugmentor  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--label-csv", required=True)
    p.add_argument("--init-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--freeze-backbone-bn", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--ey-loss-weight", type=float, default=0.35)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260715)
    p.add_argument("--early-stop-patience", type=float, default=0.002)
    p.add_argument("--early-stop-patience-epochs", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    preprocess = PreprocessConfig(color_space="hsv", input_size=(144, 192), use_roi=True, illumination_profile="none")
    augmentor = WarmPhotometricAugmentor(seed=args.seed) if args.augment else None
    train_ds = WarmAutoDrive2DDataset(args.label_csv, split="train", preprocess=preprocess, num_frames=3, frame_stride=1, augmentor=augmentor)
    val_ds = WarmAutoDrive2DDataset(args.label_csv, split="val", preprocess=preprocess, num_frames=3, frame_stride=1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = AutoDriveNetSeqCfC(num_frames=3).to(device)
    checkpoint = torch.load(Path(args.init_checkpoint).resolve(), map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if args.freeze_backbone_bn:
        original_train = model.train

        def train_with_frozen_backbone_bn(mode: bool = True):
            result = original_train(mode)
            if mode:
                for module in model.backbone.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
            return result

        model.train = train_with_frozen_backbone_bn  # type: ignore[method-assign]
    initial = locked_train._run_epoch(model, val_loader, device=device, optimizer=None, ey_loss_weight=args.ey_loss_weight)
    best_val = float(initial["steeringMAE"])

    # Compatibility fields consumed by locked_train checkpoint/report helpers.
    args.model_variant = "seq_cfc_temporal3"
    args.num_frames = 3
    args.frame_stride = 1
    args.save_name = "seq_cfc_temporal3_2d.pth"
    args.best_save_name = "best_seq_cfc_temporal3_2d.pth"
    args.early_stop = True
    args.use_roi = True
    latest_path = output / args.save_name
    best_path = output / args.best_save_name
    history_path = output / "training_log_2d.csv"
    summary_path = output / "training_summary_2d.json"
    torch.save(locked_train._make_payload(model, epoch=0, best_val=best_val, args=args, preprocess=preprocess), best_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    history = []
    stale = 0
    min_delta = max(0.0, args.early_stop_patience)
    stopped_epoch = None
    print(f"device={device} augment={args.augment} freeze_backbone_bn={args.freeze_backbone_bn} initial_val={best_val:.6f}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = locked_train._run_epoch(model, train_loader, device=device, optimizer=optimizer, ey_loss_weight=args.ey_loss_weight)
        val_metrics = locked_train._run_epoch(model, val_loader, device=device, optimizer=None, ey_loss_weight=args.ey_loss_weight)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": float(scheduler.get_last_lr()[0])})
        val_mae = float(val_metrics["steeringMAE"])
        print(f"epoch={epoch:03d} train={train_metrics['steeringMAE']:.6f} val={val_mae:.6f}")
        torch.save(locked_train._make_payload(model, epoch=epoch, best_val=best_val, args=args, preprocess=preprocess), latest_path)
        meaningful = val_mae < best_val - min_delta
        if val_mae < best_val:
            best_val = val_mae
            torch.save(locked_train._make_payload(model, epoch=epoch, best_val=best_val, args=args, preprocess=preprocess), best_path)
        stale = 0 if meaningful else stale + 1
        locked_train._write_history_csv(history_path, history)
        if stale >= args.early_stop_patience_epochs:
            stopped_epoch = epoch
            print(f"early_stop epoch={epoch} best_val={best_val:.6f}")
            break

    summary = {
        "protocol": "warm_start_paired",
        "trainSplit": "dataset/26合并:train",
        "valSplit": "dataset/26合并:val",
        "heldOutTest": "dataset/25地下室",
        "augmentation": bool(args.augment),
        "freezeBackboneBnRunningStats": bool(args.freeze_backbone_bn),
        "initialCheckpoint": str(Path(args.init_checkpoint).resolve()),
        "initialValSteeringMAE": float(initial["steeringMAE"]),
        "bestValSteeringMAE": best_val,
        "bestCheckpoint": str(best_path),
        "stoppedEpoch": stopped_epoch,
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"best={best_path} best_val={best_val:.6f}")


if __name__ == "__main__":
    main()
