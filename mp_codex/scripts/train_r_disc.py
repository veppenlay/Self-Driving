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

from r_disc import LaneGeometryDataset, RDiscCfC, lane_heatmap_targets, weighted_lane_heatmap_loss  # noqa: E402
from steering_preprocess import PreprocessConfig, preprocess_config_to_dict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train R-DISC with a training-only lane heatmap head.")
    parser.add_argument("--label-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ey-loss-weight", type=float, default=0.35)
    parser.add_argument("--lane-loss-weight", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.005)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if requested in ("auto", "cuda") and torch.cuda.is_available() else "cpu")


def run_epoch(
    model: RDiscCfC,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    ey_loss_weight: float,
    lane_loss_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    regression_loss = nn.SmoothL1Loss(reduction="none")
    totals = {"loss": 0.0, "steering_abs": 0.0, "ey_abs": 0.0, "ey_weight": 0.0, "lane_loss": 0.0, "lane_quality": 0.0}
    count = 0
    for images, labels, meta in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        ey_quality = meta["eyQuality"].to(device, non_blocking=True).view(-1)
        lane_geometry = meta["laneGeometry"].to(device, non_blocking=True)
        lane_quality = meta["laneQuality"].to(device, non_blocking=True).view(-1)
        with torch.set_grad_enabled(training):
            prediction, lane_logits = model(images, return_lane=True)
            per_dim = regression_loss(prediction, labels)
            steering_loss = per_dim[:, 0].mean()
            ey_loss = (per_dim[:, 1] * ey_quality).sum() / ey_quality.sum().clamp_min(1.0)
            lane_target = lane_heatmap_targets(
                lane_geometry,
                height=lane_logits.shape[-2],
                width=lane_logits.shape[-1],
            )
            lane_loss = weighted_lane_heatmap_loss(lane_logits, lane_target, lane_quality)
            loss = steering_loss + float(ey_loss_weight) * ey_loss + float(lane_loss_weight) * lane_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        batch = labels.size(0)
        totals["loss"] += float(loss.item()) * batch
        totals["steering_abs"] += float((prediction[:, 0] - labels[:, 0]).abs().sum().item())
        totals["ey_abs"] += float(((prediction[:, 1] - labels[:, 1]).abs() * ey_quality).sum().item())
        totals["ey_weight"] += float(ey_quality.sum().item())
        totals["lane_loss"] += float(lane_loss.item()) * float(lane_quality.sum().item())
        totals["lane_quality"] += float(lane_quality.sum().item())
        count += batch
    return {
        "loss": totals["loss"] / max(1, count),
        "steeringMAE": totals["steering_abs"] / max(1, count),
        "eyMAE": totals["ey_abs"] / max(1e-6, totals["ey_weight"]),
        "laneLoss": totals["lane_loss"] / max(1e-6, totals["lane_quality"]),
        "laneQualitySum": totals["lane_quality"],
    }


def checkpoint_payload(model: RDiscCfC, *, epoch: int, best_val: float, args: argparse.Namespace, preprocess: PreprocessConfig) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model": model.deployment_state_dict(),
        "modelFull": model.state_dict(),
        "modelVariant": "seq_cfc_temporal3",
        "experimentVariant": "R-DISC_lane_heatmap_lambda_0p1",
        "outputDim": 2,
        "outputNames": ["steering", "e_y"],
        "preprocess": preprocess_config_to_dict(preprocess),
        "numFrames": 3,
        "frameStride": 1,
        "eyLossWeight": float(args.ey_loss_weight),
        "laneLossWeight": float(args.lane_loss_weight),
        "bestValSteeringMAE": float(best_val),
    }


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    fields = ["epoch", "lr", "trainLoss", "trainSteeringMAE", "trainEyMAE", "trainLaneLoss", "valLoss", "valSteeringMAE", "valEyMAE", "valLaneLoss"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "lr": row["lr"],
                    "trainLoss": row["train"]["loss"],
                    "trainSteeringMAE": row["train"]["steeringMAE"],
                    "trainEyMAE": row["train"]["eyMAE"],
                    "trainLaneLoss": row["train"]["laneLoss"],
                    "valLoss": row["val"]["loss"],
                    "valSteeringMAE": row["val"]["steeringMAE"],
                    "valEyMAE": row["val"]["eyMAE"],
                    "valLaneLoss": row["val"]["laneLoss"],
                }
            )


def write_summary(path: Path, *, args: argparse.Namespace, history: list[dict[str, Any]], best_val: float, best_path: Path, stopped_epoch: int | None) -> None:
    payload = {
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "experimentVariant": "R-DISC_lane_heatmap_lambda_0p1",
        "labelCsv": str(Path(args.label_csv).resolve()),
        "outputDir": str(path.parent),
        "epochsRequested": args.epochs,
        "seed": args.seed,
        "batchSize": args.batch_size,
        "optimizer": {"name": "AdamW", "lr": args.lr, "weightDecay": args.weight_decay},
        "eyLossWeight": args.ey_loss_weight,
        "laneLossWeight": args.lane_loss_weight,
        "bestValSteeringMAE": best_val,
        "bestCheckpoint": str(best_path),
        "stoppedEpoch": stopped_epoch,
        "history": history,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocess = PreprocessConfig(color_space="hsv", input_size=(144, 192), use_roi=True, illumination_profile="none")
    train_dataset = LaneGeometryDataset(args.label_csv, split="train", preprocess=preprocess, num_frames=3, frame_stride=1)
    val_dataset = LaneGeometryDataset(args.label_csv, split="val", preprocess=preprocess, num_frames=3, frame_stride=1)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    model = RDiscCfC(num_frames=3, use_pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best_val = float("inf")
    stale_epochs = 0
    stopped_epoch: int | None = None
    history: list[dict[str, Any]] = []
    best_path = output_dir / "best_r_disc_seq_cfc_temporal3_2d.pth"
    latest_path = output_dir / "r_disc_seq_cfc_temporal3_2d.pth"
    summary_path = output_dir / "training_summary_2d.json"
    history_path = output_dir / "training_log_2d.csv"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device=device, optimizer=optimizer, ey_loss_weight=args.ey_loss_weight, lane_loss_weight=args.lane_loss_weight)
        val_metrics = run_epoch(model, val_loader, device=device, optimizer=None, ey_loss_weight=args.ey_loss_weight, lane_loss_weight=args.lane_loss_weight)
        scheduler.step()
        row = {"epoch": epoch, "lr": float(scheduler.get_last_lr()[0]), "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"epoch={epoch:03d} train_steer_mae={train_metrics['steeringMAE']:.6f} "
            f"val_steer_mae={val_metrics['steeringMAE']:.6f} val_lane_loss={val_metrics['laneLoss']:.6f}",
            flush=True,
        )
        val_mae = float(val_metrics["steeringMAE"])
        meaningful_improve = val_mae < best_val - args.early_stop_min_delta
        torch.save(checkpoint_payload(model, epoch=epoch, best_val=min(best_val, val_mae), args=args, preprocess=preprocess), latest_path)
        if val_mae < best_val:
            best_val = val_mae
            torch.save(checkpoint_payload(model, epoch=epoch, best_val=best_val, args=args, preprocess=preprocess), best_path)
        if meaningful_improve:
            stale_epochs = 0
        else:
            stale_epochs += 1
        write_history(history_path, history)
        write_summary(summary_path, args=args, history=history, best_val=best_val, best_path=best_path, stopped_epoch=stopped_epoch)
        if stale_epochs >= args.early_stop_patience:
            stopped_epoch = epoch
            print(f"early_stop epoch={epoch} best_val={best_val:.6f}", flush=True)
            break
    write_summary(summary_path, args=args, history=history, best_val=best_val, best_path=best_path, stopped_epoch=stopped_epoch)
    print(f"best={best_path}", flush=True)


if __name__ == "__main__":
    main()
