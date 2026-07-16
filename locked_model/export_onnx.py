#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export a 2D checkpoint to ONNX. Output shape is [B, 2] = [steering, e_y]."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from models import build_model_for_checkpoint  # noqa: E402
from steering_preprocess import DEFAULT_PREPROCESS_CONFIG, preprocess_config_from_dict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 2D steering/e_y checkpoint to ONNX.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    variant = checkpoint.get("modelVariant") if isinstance(checkpoint, dict) else None
    model = build_model_for_checkpoint(state, variant).to(device)
    model.eval()
    preprocess = preprocess_config_from_dict(checkpoint.get("preprocess") if isinstance(checkpoint, dict) else None, fallback=DEFAULT_PREPROCESS_CONFIG)
    num_frames = int(checkpoint.get("numFrames", getattr(model, "num_frames", 3))) if isinstance(checkpoint, dict) else int(getattr(model, "num_frames", 3))
    height, width = preprocess.input_size
    dummy = torch.randn(1, 3 * max(1, num_frames), height, width, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=args.opset,
        dynamo=False,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["steering_ey"],
        dynamic_axes={"input": {0: "batch_size"}, "steering_ey": {0: "batch_size"}},
    )
    print(f"ckpt: {ckpt_path}")
    print(f"out: {out_path}")
    print(f"output: [steering, e_y]")
    print(f"input: num_frames={num_frames} shape=(1,{3 * max(1, num_frames)},{height},{width})")


if __name__ == "__main__":
    main()
