#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A2: LayerNorm on temporal projection / CfC input (no GN)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
LOCKED = ROOT / "locked_model"
sys.path[:0] = [str(ROOT), str(LOCKED)]

from mp_cursor.challengers.common import (  # noqa: E402
    apply_hmm,
    device_of,
    evaluate_predictions,
    make_loaders,
    train_loop,
    write_verdict,
)
from datasets import AutoDrive2DDataset  # noqa: E402
from steering_models import RegressionSequenceCfCSteeringNet  # noqa: E402
from steering_preprocess import PreprocessConfig  # noqa: E402


class SeqCfCTemporalLN(RegressionSequenceCfCSteeringNet):
    def __init__(self, *, num_frames: int = 3, use_pretrained: bool = True):
        super().__init__(num_frames=num_frames, use_pretrained=use_pretrained)
        self.temporal_projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.Dropout(0.10),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.projection_dim),
        )


def smooth_l1_loss(pred, labels, ey_quality, ey_loss_weight: float):
    criterion = nn.SmoothL1Loss(reduction="none")
    per = criterion(pred[:, :2], labels)
    steering = per[:, 0].mean()
    ey_den = torch.clamp(ey_quality.sum(), min=1.0)
    ey = (per[:, 1] * ey_quality).sum() / ey_den
    return steering + float(ey_loss_weight) * ey


def metrics(pred, labels, ey_quality):
    p = pred[:, :2]
    return {
        "steering_abs": float((p[:, 0] - labels[:, 0]).abs().sum().item()),
        "ey_abs": float(((p[:, 1] - labels[:, 1]).abs() * ey_quality).sum().item()),
        "ey_w": float(ey_quality.sum().item()),
    }


def parse_args():
    p = argparse.ArgumentParser(description="A2 temporal LN challenger")
    p.add_argument("--label-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--ey-loss-weight", type=float, default=0.35)
    p.add_argument("--device", default="auto")
    p.add_argument("--init-checkpoint", default=None)
    p.add_argument("--no-early-stop", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = device_of(args.device)
    preprocess = PreprocessConfig(color_space="hsv", input_size=(144, 192), use_roi=True, illumination_profile="none")
    out = Path(args.output_dir).resolve()
    train_loader, val_loader, test_loader = make_loaders(
        AutoDrive2DDataset, Path(args.label_csv), preprocess, batch_size=args.batch_size, num_workers=args.num_workers
    )
    model = SeqCfCTemporalLN(num_frames=3).to(device)
    init_ck = Path(args.init_checkpoint).resolve() if args.init_checkpoint else None
    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=out,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ey_loss_weight=args.ey_loss_weight,
        early_stop=not args.no_early_stop,
        min_delta=0.005,
        patience_epochs=10,
        experiment_name="A2_temporal_ln",
        preprocess=preprocess,
        loss_fn=smooth_l1_loss,
        metrics_fn=metrics,
        init_checkpoint=init_ck,
    )
    raw = evaluate_predictions(model, test_loader, device=device, output_dir=out / "evaluation")
    hmm = apply_hmm(Path(raw["predictionsCsv"]), out / "evaluation" / "hmm")
    write_verdict(out / "verdict.md", experiment="A2_temporal_ln", hmm_mae=hmm["hmmSteeringMAE"], raw_mae=hmm["rawSteeringMAE"])


if __name__ == "__main__":
    main()
