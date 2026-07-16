#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Eval-only helper: load challenger checkpoint and write basement RAW+HMM+verdict."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
LOCKED = ROOT / "locked_model"
sys.path[:0] = [str(ROOT), str(LOCKED)]

from mp_cursor.challengers.common import apply_hmm, device_of, evaluate_predictions, make_loaders, write_verdict  # noqa: E402
from mp_cursor.challengers.d1_framediff import FrameDiffDataset, SeqCfCFrameDiff  # noqa: E402
from mp_cursor.challengers.b0_nll import SeqCfCNLL  # noqa: E402
from mp_cursor.challengers.a2_temporal_ln import SeqCfCTemporalLN  # noqa: E402
from datasets import AutoDrive2DDataset  # noqa: E402
from steering_preprocess import PreprocessConfig  # noqa: E402


MODELS = {
    "D1_framediff": (SeqCfCFrameDiff, FrameDiffDataset),
    "B0_nll": (SeqCfCNLL, AutoDrive2DDataset),
    "A2_temporal_ln": (SeqCfCTemporalLN, AutoDrive2DDataset),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True, choices=list(MODELS))
    p.add_argument("--label-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = device_of(args.device)
    out = Path(args.output_dir).resolve()
    ckpt = out / "checkpoints" / "best.pth"
    model_cls, ds_cls = MODELS[args.experiment]
    preprocess = PreprocessConfig(color_space="hsv", input_size=(144, 192), use_roi=True, illumination_profile="none")
    _, _, test_loader = make_loaders(ds_cls, Path(args.label_csv), preprocess, batch_size=32, num_workers=4)
    model = model_cls(num_frames=3).to(device)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    raw = evaluate_predictions(model, test_loader, device=device, output_dir=out / "evaluation")
    hmm = apply_hmm(Path(raw["predictionsCsv"]), out / "evaluation" / "hmm")
    write_verdict(out / "verdict.md", experiment=args.experiment, hmm_mae=hmm["hmmSteeringMAE"], raw_mae=hmm["rawSteeringMAE"])


if __name__ == "__main__":
    main()
