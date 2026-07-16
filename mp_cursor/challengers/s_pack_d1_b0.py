#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S-pack: FrameDiff (D1) + steering NLL (B0) combined for gate fallback."""

from __future__ import annotations

import argparse
import math
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
from mp_cursor.challengers.d1_framediff import FrameDiffDataset, SeqCfCFrameDiff  # noqa: E402
from steering_preprocess import PreprocessConfig  # noqa: E402


class SeqCfCFrameDiffNLL(SeqCfCFrameDiff):
    def __init__(self, *, num_frames: int = 3, use_pretrained: bool = True):
        super().__init__(num_frames=num_frames, use_pretrained=use_pretrained)
        self.reg_head = nn.Linear(self.hidden_dim, 3)


def nll_loss(pred, labels, ey_quality, ey_loss_weight: float):
    steering = pred[:, 0]
    e_y = pred[:, 1]
    log_sigma = pred[:, 2].clamp(-6.0, 2.0)
    inv_var = torch.exp(-2.0 * log_sigma)
    nll = 0.5 * ((labels[:, 0] - steering) ** 2 * inv_var + 2.0 * log_sigma + math.log(2.0 * math.pi))
    steering_loss = nll.mean()
    ey_raw = nn.functional.smooth_l1_loss(e_y, labels[:, 1], reduction="none")
    ey_den = torch.clamp(ey_quality.sum(), min=1.0)
    ey_loss = (ey_raw * ey_quality).sum() / ey_den
    return steering_loss + float(ey_loss_weight) * ey_loss


def metrics(pred, labels, ey_quality):
    p = pred[:, :2]
    return {
        "steering_abs": float((p[:, 0] - labels[:, 0]).abs().sum().item()),
        "ey_abs": float(((p[:, 1] - labels[:, 1]).abs() * ey_quality).sum().item()),
        "ey_w": float(ey_quality.sum().item()),
    }


def parse_args():
    p = argparse.ArgumentParser()
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
        FrameDiffDataset, Path(args.label_csv), preprocess, batch_size=args.batch_size, num_workers=args.num_workers
    )
    model = SeqCfCFrameDiffNLL(num_frames=3).to(device)
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
        experiment_name="S_pack_D1_B0",
        preprocess=preprocess,
        loss_fn=nll_loss,
        metrics_fn=metrics,
        init_checkpoint=init_ck,
    )
    raw = evaluate_predictions(model, test_loader, device=device, output_dir=out / "evaluation")
    hmm = apply_hmm(Path(raw["predictionsCsv"]), out / "evaluation" / "hmm")
    write_verdict(out / "verdict.md", experiment="S_pack_D1_B0", hmm_mae=hmm["hmmSteeringMAE"], raw_mae=hmm["rawSteeringMAE"])


if __name__ == "__main__":
    main()
