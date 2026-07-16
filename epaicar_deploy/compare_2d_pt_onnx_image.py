#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_gt(path: Path) -> float | None:
    match = re.search(r"_(-?\d+(?:\.\d+)?)$", path.stem)
    return float(match.group(1)) if match else None


def parse_frame_index(path: Path) -> int:
    prefix = path.stem.split("_", 1)[0]
    if not prefix.isdigit():
        raise ValueError("image filename does not start with numeric frame index: {}".format(path))
    return int(prefix)


def frame_candidate(image_path: Path, index: int) -> Path | None:
    for ext in [image_path.suffix, *IMAGE_EXTS]:
        matches = sorted(image_path.parent.glob("{}_*{}".format(index, ext)))
        if matches:
            return matches[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=r"E:\桌面\v-Net\locked_model")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=r"E:\桌面\v-Net\locked_model\checkpoints\best_seq_cfc_temporal3_2d.pth")
    parser.add_argument("--onnx", default=r"E:\桌面\v-Net\epaicar_deploy\best_seq_cfc_temporal3_2d_correct.onnx")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    from models import build_model_for_checkpoint
    from steering_preprocess import DEFAULT_PREPROCESS_CONFIG, imread_bgr, preprocess_bgr_to_tensor, preprocess_config_from_dict

    image_path = Path(args.image).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    onnx_path = Path(args.onnx).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    variant = ckpt.get("modelVariant") if isinstance(ckpt, dict) else None
    model = build_model_for_checkpoint(state, variant).to(device)
    model.eval()

    preprocess = preprocess_config_from_dict(ckpt.get("preprocess") if isinstance(ckpt, dict) else None, fallback=DEFAULT_PREPROCESS_CONFIG)
    num_frames = int(ckpt.get("numFrames", getattr(model, "num_frames", 1))) if isinstance(ckpt, dict) else int(getattr(model, "num_frames", 1))
    frame_stride = int(ckpt.get("frameStride", 1)) if isinstance(ckpt, dict) else 1
    current_index = parse_frame_index(image_path)

    frame_paths: list[Path] = []
    frame_tensors: list[torch.Tensor] = []
    last_valid = image_path
    for offset in range(num_frames - 1, -1, -1):
        target_index = current_index - offset * frame_stride
        candidate = frame_candidate(image_path, target_index) if target_index >= 0 else None
        if candidate is None:
            candidate = last_valid
        bgr = imread_bgr(candidate)
        if bgr is None:
            raise FileNotFoundError("failed to read image frame: {}".format(candidate))
        frame_paths.append(candidate)
        frame_tensors.append(preprocess_bgr_to_tensor(bgr, config=preprocess).squeeze(0))
        last_valid = candidate

    tensor = torch.cat(frame_tensors, dim=0).unsqueeze(0).to(device)
    batch = tensor.detach().cpu().numpy().astype(np.float32)

    with torch.no_grad():
        pt_out = model(tensor).detach().cpu().numpy().reshape(-1)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_out = np.asarray(session.run([output_name], {input_name: batch})[0]).reshape(-1)

    gt = parse_gt(image_path)
    print("image:", image_path)
    print("checkpoint:", ckpt_path)
    print("onnx:", onnx_path)
    print("device:", device)
    print("model_variant:", variant)
    print("gt_from_filename:", "None" if gt is None else "{:.6f}".format(gt))
    print("preprocess:", "color_space={} input_size={} use_roi={} illumination={}".format(
        preprocess.color_space,
        preprocess.input_size,
        preprocess.use_roi,
        preprocess.illumination_profile,
    ))
    print("temporal:", "num_frames={} frame_stride={} frame_paths={}".format(
        num_frames,
        frame_stride,
        [p.name for p in frame_paths],
    ))
    print("input_shape:", batch.shape)
    print("input_dtype:", batch.dtype)
    print("onnx_input_name:", input_name)
    print("onnx_output_name:", output_name)
    print("pt_output_vector:", "[" + ", ".join("{:.8f}".format(float(x)) for x in pt_out) + "]")
    print("onnx_output_vector:", "[" + ", ".join("{:.8f}".format(float(x)) for x in onnx_out) + "]")
    print("pt_steering:", "{:.8f}".format(float(pt_out[0])))
    print("onnx_steering:", "{:.8f}".format(float(onnx_out[0])))
    if len(pt_out) > 1:
        print("pt_e_y:", "{:.8f}".format(float(pt_out[1])))
    if len(onnx_out) > 1:
        print("onnx_e_y:", "{:.8f}".format(float(onnx_out[1])))
    print("pt_onnx_steering_abs_diff:", "{:.8f}".format(abs(float(pt_out[0]) - float(onnx_out[0]))))
    if gt is not None:
        print("pt_steering_abs_error_vs_gt:", "{:.8f}".format(abs(float(pt_out[0]) - gt)))
        print("onnx_steering_abs_error_vs_gt:", "{:.8f}".format(abs(float(onnx_out[0]) - gt)))


if __name__ == "__main__":
    main()
