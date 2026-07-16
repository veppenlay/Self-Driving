#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 training: lane segmentation from automatic CV pseudo-masks + strong augmentation.

Trains ONLY on `dataset/26合并` (train/val). Loss = BCE + soft Dice on the yellow pseudo-mask.
Model selection uses val Dice. Augmentation (zero target-domain priors) drives robustness.

Run on the GPU environment. Example:
  python -m mp_cursor.geo_control.train_segmentation \
    --label-csv locked_model/labels/current/labels_2d.csv \
    --output-dir mp_cursor/exp_P3_seg/checkpoints --epochs 80
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_DIR = REPO_ROOT / "locked_model"
if str(LOCKED_DIR) not in sys.path:
    sys.path.insert(0, str(LOCKED_DIR))

from .augment import AugmentConfig, PhotometricAugmentor
from .datasets_seg import INPUT_H, INPUT_W, LaneSegDataset
from .models_seg import LaneSegNet


def _device(req: str) -> torch.device:
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dice(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = 2.0 * (prob * target).sum(dim=(1, 2, 3)) + eps
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (num / den).mean()


def _run_epoch(model, loader, *, device, optimizer, max_batches: int = 0) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_dice = 0.0
    count = 0
    for bi, (images, masks, _meta) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            prob = torch.sigmoid(logits)
            loss = bce(logits, masks) + (1.0 - _dice(prob, masks))
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        b = images.size(0)
        total_loss += float(loss.item()) * b
        total_dice += float(_dice((prob > 0.5).float(), masks).item()) * b
        count += b
    return {"loss": total_loss / max(1, count), "dice": total_dice / max(1, count)}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train P3 lane segmentation on CV pseudo-masks.")
    p.add_argument("--label-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--aug-strength", type=float, default=1.0)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260629)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--best-save-name", default="best_laneseg_p3.pth")
    p.add_argument("--early-stop-patience-epochs", type=int, default=10)
    p.add_argument("--max-train-batches", type=int, default=0, help="Smoke test: cap batches/epoch (0=all).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    device = _device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    augmentor = None if args.no_augment else PhotometricAugmentor(AugmentConfig(strength=float(args.aug_strength)), seed=args.seed)
    train_ds = LaneSegDataset(args.label_csv, split="train", augmentor=augmentor)
    val_ds = LaneSegDataset(args.label_csv, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = LaneSegNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_dice = -1.0
    best_path = output_dir / args.best_save_name
    summary_path = output_dir / "training_summary_p3.json"
    history: list[dict[str, Any]] = []
    epochs_no_improve = 0
    stopped_epoch = None

    def _payload(epoch: int) -> dict[str, Any]:
        return {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "modelVariant": "laneseg_mobilenetv3",
            "inputSize": [INPUT_H, INPUT_W],
            "colorSpace": "hsv",
            "useRoi": True,
            "augment": {"enabled": augmentor is not None, "strength": float(args.aug_strength)},
            "bestValDice": float(best_dice),
        }

    for epoch in range(1, args.epochs + 1):
        tr = _run_epoch(model, train_loader, device=device, optimizer=optimizer, max_batches=args.max_train_batches)
        va = _run_epoch(model, val_loader, device=device, optimizer=None, max_batches=args.max_train_batches)
        scheduler.step()
        history.append({"epoch": epoch, "train": tr, "val": va, "lr": float(scheduler.get_last_lr()[0])})
        print(f"epoch={epoch:03d} train_loss={tr['loss']:.5f} train_dice={tr['dice']:.4f} val_dice={va['dice']:.4f}")
        if va["dice"] > best_dice:
            best_dice = va["dice"]
            torch.save(_payload(epoch), best_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience_epochs:
                stopped_epoch = epoch
                print(f"early_stop at epoch={epoch} best_val_dice={best_dice:.4f}")
                break

    summary = {
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "labelCsv": str(Path(args.label_csv).resolve()),
        "outputDir": str(output_dir),
        "bestCheckpoint": str(best_path),
        "bestValDice": best_dice,
        "augment": {"enabled": augmentor is not None, "strength": float(args.aug_strength)},
        "stoppedEpoch": stopped_epoch,
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"best={best_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
