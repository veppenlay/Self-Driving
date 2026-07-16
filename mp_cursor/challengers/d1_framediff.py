#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D1: append V-channel frame diffs to each Temporal3 frame (4ch/frame)."""

from __future__ import annotations

import argparse
import sys
from itertools import chain
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
from steering_models import RegressionSteeringNet, _CfCLiteCell  # noqa: E402
from steering_preprocess import PreprocessConfig, imread_bgr, preprocess_bgr_to_tensor  # noqa: E402


class FrameDiffDataset(AutoDrive2DDataset):
    def __getitem__(self, index: int):
        row = self.rows[index]
        paths = self._resolve_frame_paths(row["image"])
        chw_list = []
        prev_v = None
        for path in paths:
            bgr = imread_bgr(path)
            if bgr is None:
                raise FileNotFoundError(path)
            chw = preprocess_bgr_to_tensor(bgr, config=self.preprocess).squeeze(0)
            v = chw[2:3]
            dv = torch.zeros_like(v) if prev_v is None else (v - prev_v)
            prev_v = v
            chw_list.append(torch.cat([chw, dv], dim=0))
        image = torch.cat(chw_list, dim=0)
        label = torch.tensor([row["steering"], row["e_y"]], dtype=torch.float32)
        meta = {
            "eyQuality": torch.tensor(row["ey_quality"], dtype=torch.float32),
            "path": str(row["image"]),
            "status": row["status"],
            "sequence": row["sequence"],
            "frame": row["frame"],
        }
        return image, label, meta


class SeqCfCFrameDiff(RegressionSteeringNet):
    def __init__(self, *, num_frames: int = 3, use_pretrained: bool = True):
        super().__init__(num_aux_classes=11, use_pretrained=use_pretrained, in_channels=4)
        if num_frames < 2:
            raise ValueError(num_frames)
        self.num_frames = int(num_frames)
        self.temporal_input_channels = 4 * self.num_frames
        self.projection_dim = 64
        self.hidden_dim = 32
        self.temporal_projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.projection_dim),
            nn.Dropout(0.10),
            nn.ReLU(inplace=True),
        )
        self.cfc_cell = _CfCLiteCell(self.projection_dim, self.hidden_dim, recurrent_fanin=6, mask_seed=20260626)
        self.reg_head = nn.Linear(self.hidden_dim, 2)

    def head_parameters(self):
        return chain(
            self.mid_reduce.parameters(),
            self.final_reduce.parameters(),
            self.fuse.parameters(),
            self.spatial_pool.parameters(),
            self.temporal_projection.parameters(),
            self.cfc_cell.parameters(),
            self.reg_head.parameters(),
            self.aux_head.parameters(),
        )

    def extract_feature_sequence(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.dim() != 4 or input_tensor.size(1) != self.temporal_input_channels:
            raise ValueError(f"expected [B,{self.temporal_input_channels},H,W], got {tuple(input_tensor.shape)}")
        b, _, h, w = input_tensor.shape
        flat = input_tensor.reshape(b, self.num_frames, 4, h, w).reshape(b * self.num_frames, 4, h, w)
        feats = super().extract_features(flat)
        return feats.reshape(b, self.num_frames, self.feature_dim)

    def forward(self, input_tensor: torch.Tensor, return_aux: bool = False):
        feat_seq = self.extract_feature_sequence(input_tensor)
        projected = self.temporal_projection(feat_seq)
        hidden = projected.new_zeros(projected.size(0), self.hidden_dim)
        for t in range(self.num_frames):
            hidden = self.cfc_cell(projected[:, t, :], hidden)
        angle = self.reg_head(hidden)
        if return_aux:
            return angle, self.aux_head(feat_seq[:, -1, :])
        return angle


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
    model = SeqCfCFrameDiff(num_frames=3).to(device)
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
        experiment_name="D1_framediff",
        preprocess=preprocess,
        loss_fn=smooth_l1_loss,
        metrics_fn=metrics,
        init_checkpoint=init_ck,
    )
    raw = evaluate_predictions(model, test_loader, device=device, output_dir=out / "evaluation")
    hmm = apply_hmm(Path(raw["predictionsCsv"]), out / "evaluation" / "hmm")
    write_verdict(out / "verdict.md", experiment="D1_framediff", hmm_mae=hmm["hmmSteeringMAE"], raw_mae=hmm["rawSteeringMAE"])


if __name__ == "__main__":
    main()
