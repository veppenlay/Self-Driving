#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train a 2D steering model: output [steering, e_y]."""

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


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from datasets import AutoDrive2DDataset  # noqa: E402
from models import MODEL_OUTPUT_NAMES, AutoDriveLegacyNet, AutoDriveNetSeqCfC, AutoDriveNetSeqGRU, AutoDriveNetTemporal  # noqa: E402
from steering_preprocess import PreprocessConfig, preprocess_config_to_dict  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 2D steering/e_y regression model.")
    parser.add_argument("--label-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-variant", default="seq_cfc_temporal3", choices=["legacy", "temporal3", "seq_gru_temporal3", "seq_cfc_temporal3"])
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ey-loss-weight", type=float, default=0.35)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-name", default="seq_cfc_temporal3_2d.pth")
    parser.add_argument("--best-save-name", default="best_seq_cfc_temporal3_2d.pth")
    parser.add_argument("--resume-checkpoint", default=None, help="Continue training from an existing 2D checkpoint.")
    parser.add_argument("--init-checkpoint", default=None, help="Initialize weights from an existing 2D checkpoint without resuming epoch/best metrics.")
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--use-roi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable bottom-70%% ROI crop before resize. Use --no-use-roi for full-frame training.",
    )
    parser.add_argument(
        "--early-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop when val steering MAE stops improving. Use --no-early-stop to disable.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=float,
        default=0.005,
        help="Minimum val steering MAE improvement required to reset the no-improve counter (min_delta).",
    )
    parser.add_argument(
        "--early-stop-patience-epochs",
        type=int,
        default=10,
        help="Stop after this many epochs without improvement >= --early-stop-patience.",
    )
    return parser.parse_args()


def _device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(variant: str, num_frames: int) -> nn.Module:
    if variant == "legacy":
        return AutoDriveLegacyNet()
    if variant == "temporal3":
        return AutoDriveNetTemporal(num_frames=num_frames)
    if variant == "seq_gru_temporal3":
        return AutoDriveNetSeqGRU(num_frames=num_frames)
    if variant == "seq_cfc_temporal3":
        return AutoDriveNetSeqCfC(num_frames=num_frames)
    raise ValueError(f"unsupported model variant: {variant}")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    ey_loss_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.SmoothL1Loss(reduction="none")
    total_loss = 0.0
    steering_abs = 0.0
    ey_abs = 0.0
    ey_weight_sum = 0.0
    count = 0

    for images, labels, meta in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        ey_quality = meta["eyQuality"].to(device, non_blocking=True).view(-1)

        with torch.set_grad_enabled(training):
            pred = model(images)
            if pred.shape[-1] != 2:
                raise RuntimeError(f"2D model must output shape [B,2], got {tuple(pred.shape)}")
            per_dim_loss = criterion(pred, labels)
            steering_loss = per_dim_loss[:, 0].mean()
            ey_loss_raw = per_dim_loss[:, 1]
            ey_den = torch.clamp(ey_quality.sum(), min=1.0)
            ey_loss = (ey_loss_raw * ey_quality).sum() / ey_den
            loss = steering_loss + float(ey_loss_weight) * ey_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        batch = labels.size(0)
        total_loss += float(loss.item()) * batch
        steering_abs += float((pred[:, 0] - labels[:, 0]).abs().sum().item())
        ey_abs += float(((pred[:, 1] - labels[:, 1]).abs() * ey_quality).sum().item())
        ey_weight_sum += float(ey_quality.sum().item())
        count += batch

    return {
        "loss": total_loss / max(1, count),
        "steeringMAE": steering_abs / max(1, count),
        "eyMAE": ey_abs / max(1e-6, ey_weight_sum),
        "eyWeightSum": ey_weight_sum,
    }


def _make_payload(
    model: nn.Module,
    *,
    epoch: int,
    best_val: float,
    args: argparse.Namespace,
    preprocess: PreprocessConfig,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "modelVariant": args.model_variant,
        "outputDim": 2,
        "outputNames": list(MODEL_OUTPUT_NAMES),
        "targetNames": ["steering", "e_y"],
        "preprocess": preprocess_config_to_dict(preprocess),
        "numFrames": int(args.num_frames),
        "frameStride": int(args.frame_stride),
        "eyLossWeight": float(args.ey_loss_weight),
        "bestValSteeringMAE": float(best_val),
    }


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    fieldnames = [
        "epoch",
        "lr",
        "trainLoss",
        "trainSteeringMAE",
        "trainEyMAE",
        "trainEyWeightSum",
        "valLoss",
        "valSteeringMAE",
        "valEyMAE",
        "valEyWeightSum",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            train = row["train"]
            val = row["val"]
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "lr": row["lr"],
                    "trainLoss": train["loss"],
                    "trainSteeringMAE": train["steeringMAE"],
                    "trainEyMAE": train["eyMAE"],
                    "trainEyWeightSum": train["eyWeightSum"],
                    "valLoss": val["loss"],
                    "valSteeringMAE": val["steeringMAE"],
                    "valEyMAE": val["eyMAE"],
                    "valEyWeightSum": val["eyWeightSum"],
                }
            )


def _write_progress_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    preprocess: PreprocessConfig,
    best_path: Path,
    latest_path: Path,
    best_val: float,
    history: list[dict[str, Any]],
    test_metrics: dict[str, float] | None = None,
    early_stopped: bool = False,
    stopped_epoch: int | None = None,
) -> None:
    summary = {
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "labelCsv": str(Path(args.label_csv).resolve()),
        "outputDir": str(output_dir),
        "modelVariant": args.model_variant,
        "outputNames": list(MODEL_OUTPUT_NAMES),
        "preprocess": preprocess_config_to_dict(preprocess),
        "numFrames": args.num_frames,
        "frameStride": args.frame_stride,
        "eyLossWeight": args.ey_loss_weight,
        "bestCheckpoint": str(best_path),
        "latestCheckpoint": str(latest_path),
        "bestValSteeringMAE": best_val,
        "earlyStop": {
            "enabled": bool(args.early_stop),
            "patienceMinDelta": float(args.early_stop_patience),
            "patienceEpochs": int(args.early_stop_patience_epochs),
            "triggered": bool(early_stopped),
            "stoppedEpoch": stopped_epoch,
        },
        "testMetrics": test_metrics,
        "history": history,
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    device = _device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_variant == "legacy":
        args.num_frames = 1
        args.frame_stride = 1
        preprocess = PreprocessConfig(
            color_space="hsv",
            input_size=(120, 160),
            use_roi=bool(args.use_roi),
            illumination_profile="none",
        )
    else:
        preprocess = PreprocessConfig(
            color_space="hsv",
            input_size=(144, 192),
            use_roi=bool(args.use_roi),
            illumination_profile="none",
        )
    train_ds = AutoDrive2DDataset(args.label_csv, split="train", preprocess=preprocess, num_frames=args.num_frames, frame_stride=args.frame_stride)
    val_ds = AutoDrive2DDataset(args.label_csv, split="val", preprocess=preprocess, num_frames=args.num_frames, frame_stride=args.frame_stride)
    test_ds = AutoDrive2DDataset(args.label_csv, split="test", preprocess=preprocess, num_frames=args.num_frames, frame_stride=args.frame_stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = _build_model(args.model_variant, args.num_frames).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    latest_path = output_dir / args.save_name
    best_path = output_dir / args.best_save_name
    history_csv_path = output_dir / "training_log_2d.csv"
    summary_path = output_dir / "training_summary_2d.json"
    start_epoch = 1
    epochs_no_improve = 0
    early_stopped = False
    stopped_epoch: int | None = None
    min_delta = max(0.0, float(args.early_stop_patience))
    patience_epochs = max(1, int(args.early_stop_patience_epochs))

    if args.init_checkpoint and args.resume_checkpoint:
        raise ValueError("--init-checkpoint and --resume-checkpoint are mutually exclusive")

    if args.init_checkpoint:
        checkpoint_path = Path(args.init_checkpoint).resolve()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state, strict=True)
        print(f"init={checkpoint_path}")

    if args.resume_checkpoint:
        checkpoint_path = Path(args.resume_checkpoint).resolve()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        best_val = float(checkpoint.get("bestValSteeringMAE", best_val))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if summary_path.is_file():
            try:
                previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                history = list(previous_summary.get("history", []))
            except (OSError, json.JSONDecodeError, TypeError):
                history = []
        print(f"resume={checkpoint_path} start_epoch={start_epoch} best_val={best_val:.6f}")

    if args.early_stop:
        print(
            f"early_stop=on min_delta={min_delta:.6f} patience_epochs={patience_epochs} "
            f"max_epochs={args.epochs}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device=device, optimizer=optimizer, ey_loss_weight=args.ey_loss_weight)
        val_metrics = _run_epoch(model, val_loader, device=device, optimizer=None, ey_loss_weight=args.ey_loss_weight)
        scheduler.step()
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": float(scheduler.get_last_lr()[0])}
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"train_steer_mae={train_metrics['steeringMAE']:.6f} val_steer_mae={val_metrics['steeringMAE']:.6f} "
            f"val_ey_mae={val_metrics['eyMAE']:.6f}"
        )
        payload = _make_payload(model, epoch=epoch, best_val=best_val, args=args, preprocess=preprocess)
        torch.save(payload, latest_path)

        val_mae = float(val_metrics["steeringMAE"])
        meaningful_improve = val_mae < (best_val - min_delta)
        if val_mae < best_val:
            best_val = val_mae
            payload = _make_payload(model, epoch=epoch, best_val=best_val, args=args, preprocess=preprocess)
            torch.save(payload, best_path)
        if args.early_stop:
            if meaningful_improve:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"early_stop_stale={epochs_no_improve}/{patience_epochs}")
                if epochs_no_improve >= patience_epochs:
                    early_stopped = True
                    stopped_epoch = epoch
                    print(
                        f"early_stop triggered at epoch={epoch} "
                        f"best_val_steer_mae={best_val:.6f} min_delta={min_delta:.6f}"
                    )
                    _write_history_csv(history_csv_path, history)
                    _write_progress_summary(
                        summary_path,
                        args=args,
                        output_dir=output_dir,
                        preprocess=preprocess,
                        best_path=best_path,
                        latest_path=latest_path,
                        best_val=best_val,
                        history=history,
                        early_stopped=early_stopped,
                        stopped_epoch=stopped_epoch,
                    )
                    break

        _write_history_csv(history_csv_path, history)
        _write_progress_summary(
            summary_path,
            args=args,
            output_dir=output_dir,
            preprocess=preprocess,
            best_path=best_path,
            latest_path=latest_path,
            best_val=best_val,
            history=history,
            early_stopped=early_stopped,
            stopped_epoch=stopped_epoch,
        )

    if not best_path.is_file():
        raise RuntimeError(f"best checkpoint was not saved: {best_path}")
    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model"], strict=True)
    test_metrics = _run_epoch(model, test_loader, device=device, optimizer=None, ey_loss_weight=args.ey_loss_weight)
    _write_progress_summary(
        summary_path,
        args=args,
        output_dir=output_dir,
        preprocess=preprocess,
        best_path=best_path,
        latest_path=latest_path,
        best_val=best_val,
        history=history,
        test_metrics=test_metrics,
        early_stopped=early_stopped,
        stopped_epoch=stopped_epoch,
    )
    print(f"best={best_path}")
    print(f"summary={summary_path}")
    if early_stopped:
        print(f"early_stopped_epoch={stopped_epoch}")


if __name__ == "__main__":
    main()
