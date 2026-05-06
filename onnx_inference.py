#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run ONNX inference for a regression model using explicit preprocess settings."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from steering_preprocess import DEFAULT_PREPROCESS_CONFIG, preprocess_bgr_to_chw_float, preprocess_config_from_dict


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _parse_frame_index(path: Path) -> int | None:
    match = re.match(r"^(\d+)_", path.name)
    if not match:
        return None
    return int(match.group(1))


def _frame_candidate(image_path: Path, target_index: int) -> Path | None:
    suffix = image_path.suffix
    parent = image_path.parent
    for candidate in parent.glob(f"{target_index}_*{suffix}"):
        if candidate.is_file():
            return candidate
    for ext in IMAGE_EXTS:
        for candidate in parent.glob(f"{target_index}_*{ext}"):
            if candidate.is_file():
                return candidate
    return None


def _load_frame_stack(image_path: Path, *, num_frames: int, frame_stride: int) -> list[np.ndarray]:
    current_index = _parse_frame_index(image_path)
    frames: list[np.ndarray] = []
    for offset in range(num_frames - 1, -1, -1):
        candidate = image_path
        if current_index is not None:
            target_index = current_index - offset * frame_stride
            resolved = _frame_candidate(image_path, target_index)
            if resolved is not None:
                candidate = resolved
        img = cv2.imread(str(candidate))
        if img is None:
            raise FileNotFoundError(f"failed to read temporal frame: {candidate}")
        frames.append(img)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ONNX inference for a steering regression model.")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--color-space", default=None)
    args = parser.parse_args()

    onnx_path = Path(args.onnx).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve()

    preprocess = DEFAULT_PREPROCESS_CONFIG
    if args.height and args.width:
        preprocess = preprocess_config_from_dict(
            {
                "colorSpace": args.color_space or preprocess.color_space,
                "inputSize": [args.height, args.width],
                "useRoi": False,
            },
            fallback=preprocess,
        )
    elif args.color_space:
        preprocess = preprocess_config_from_dict({"colorSpace": args.color_space}, fallback=preprocess)

    frames = _load_frame_stack(
        image_path,
        num_frames=max(1, args.num_frames),
        frame_stride=max(1, args.frame_stride),
    )
    chw_frames = [preprocess_bgr_to_chw_float(img, config=preprocess) for img in frames]
    chw = np.concatenate(chw_frames, axis=0)
    batch = np.expand_dims(chw.astype(np.float32), axis=0)

    session = ort.InferenceSession(str(onnx_path))
    output = session.run(None, {session.get_inputs()[0].name: batch})
    print(f"onnx: {onnx_path}")
    print(f"image: {image_path}")
    print(f"preprocess: color_space={preprocess.color_space} input_size={preprocess.input_size} use_roi={preprocess.use_roi}")
    print(f"temporal: num_frames={len(frames)} frame_stride={max(1, args.frame_stride)} input_shape={batch.shape}")
    print(f"prediction: {float(np.asarray(output[0]).reshape(-1)[0]):.6f}")


if __name__ == "__main__":
    main()
