#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
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
LOCKED_MODEL_DIR = REPO_ROOT / "locked_model"
SRC_DIR = REPO_ROOT / "mp_codex" / "src"
for path in (LOCKED_MODEL_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from datasets import AutoDrive2DDataset  # noqa: E402
from e4_disc import E4DiscCfC  # noqa: E402
from steering_preprocess import PreprocessConfig, preprocess_config_to_dict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train E4-C rescue candidate.")
    parser.add_argument("--label-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ey-loss-weight", type=float, default=0.35)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.005)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def device_for(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if requested in ("auto", "cuda") and torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, *, device, optimizer, ey_loss_weight):
    training = optimizer is not None
    model.train(training)
    criterion = nn.SmoothL1Loss(reduction="none")
    total_loss = steering_abs = ey_abs = ey_weight_sum = 0.0
    count = 0
    for images, labels, meta in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        quality = meta["eyQuality"].to(device, non_blocking=True).view(-1)
        with torch.set_grad_enabled(training):
            prediction = model(images)
            per_dim = criterion(prediction, labels)
            steering_loss = per_dim[:, 0].mean()
            ey_loss = (per_dim[:, 1] * quality).sum() / quality.sum().clamp_min(1.0)
            loss = steering_loss + float(ey_loss_weight) * ey_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        batch = labels.size(0)
        total_loss += float(loss.item()) * batch
        steering_abs += float((prediction[:, 0] - labels[:, 0]).abs().sum().item())
        ey_abs += float(((prediction[:, 1] - labels[:, 1]).abs() * quality).sum().item())
        ey_weight_sum += float(quality.sum().item())
        count += batch
    return {
        "loss": total_loss / max(1, count),
        "steeringMAE": steering_abs / max(1, count),
        "eyMAE": ey_abs / max(1e-6, ey_weight_sum),
    }


def checkpoint(model, *, epoch, best_val, args, preprocess) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "modelVariant": "e4_c_frozen_backbone_bn_projection_ln",
        "experimentVariant": "E4-C",
        "outputDim": 2,
        "outputNames": ["steering", "e_y"],
        "preprocess": preprocess_config_to_dict(preprocess),
        "numFrames": 3,
        "frameStride": 1,
        "eyLossWeight": float(args.ey_loss_weight),
        "bestValSteeringMAE": float(best_val),
    }


def write_outputs(output_dir: Path, args, history, best_val, best_path, stopped_epoch) -> None:
    with (output_dir / "training_log_2d.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["epoch", "lr", "trainLoss", "trainSteeringMAE", "trainEyMAE", "valLoss", "valSteeringMAE", "valEyMAE"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({
                "epoch": row["epoch"], "lr": row["lr"],
                "trainLoss": row["train"]["loss"], "trainSteeringMAE": row["train"]["steeringMAE"], "trainEyMAE": row["train"]["eyMAE"],
                "valLoss": row["val"]["loss"], "valSteeringMAE": row["val"]["steeringMAE"], "valEyMAE": row["val"]["eyMAE"],
            })
    summary = {
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "experimentVariant": "E4-C_frozen_backbone_bn_projection_ln",
        "labelCsv": str(Path(args.label_csv).resolve()),
        "epochsRequested": args.epochs,
        "seed": args.seed,
        "batchSize": args.batch_size,
        "optimizer": {"name": "AdamW", "lr": args.lr, "weightDecay": args.weight_decay},
        "eyLossWeight": args.ey_loss_weight,
        "bestValSteeringMAE": best_val,
        "bestCheckpoint": str(best_path),
        "stoppedEpoch": stopped_epoch,
        "history": history,
    }
    (output_dir / "training_summary_2d.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = device_for(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocess = PreprocessConfig(color_space="hsv", input_size=(144, 192), use_roi=True, illumination_profile="none")
    train_ds = AutoDrive2DDataset(args.label_csv, split="train", preprocess=preprocess, num_frames=3, frame_stride=1)
    val_ds = AutoDrive2DDataset(args.label_csv, split="val", preprocess=preprocess, num_frames=3, frame_stride=1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    model = E4DiscCfC(num_frames=3, hidden_dim=32, use_pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best_val = float("inf")
    stale = 0
    stopped_epoch = None
    history = []
    best_path = output_dir / "best_e4_c_frozen_bn_projection_ln.pth"
    latest_path = output_dir / "e4_c_frozen_bn_projection_ln.pth"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device=device, optimizer=optimizer, ey_loss_weight=args.ey_loss_weight)
        val_metrics = run_epoch(model, val_loader, device=device, optimizer=None, ey_loss_weight=args.ey_loss_weight)
        scheduler.step()
        history.append({"epoch": epoch, "lr": float(scheduler.get_last_lr()[0]), "train": train_metrics, "val": val_metrics})
        print(f"epoch={epoch:03d} train_steer_mae={train_metrics['steeringMAE']:.6f} val_steer_mae={val_metrics['steeringMAE']:.6f} val_ey_mae={val_metrics['eyMAE']:.6f}", flush=True)
        val_mae = float(val_metrics["steeringMAE"])
        meaningful = val_mae < best_val - args.early_stop_min_delta
        torch.save(checkpoint(model, epoch=epoch, best_val=min(best_val, val_mae), args=args, preprocess=preprocess), latest_path)
        if val_mae < best_val:
            best_val = val_mae
            torch.save(checkpoint(model, epoch=epoch, best_val=best_val, args=args, preprocess=preprocess), best_path)
        stale = 0 if meaningful else stale + 1
        write_outputs(output_dir, args, history, best_val, best_path, stopped_epoch)
        if stale >= args.early_stop_patience:
            stopped_epoch = epoch
            print(f"early_stop epoch={epoch} best_val={best_val:.6f}", flush=True)
            break
    write_outputs(output_dir, args, history, best_val, best_path, stopped_epoch)
    print(f"best={best_path}", flush=True)


if __name__ == "__main__":
    main()

