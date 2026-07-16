#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run ONNX inference for a 2D model on one image."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from steering_preprocess import DEFAULT_PREPROCESS_CONFIG, imread_bgr, preprocess_bgr_to_chw_float, preprocess_config_from_dict  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _frame_index(path: Path) -> int | None:
    match = re.match(r"^(\d+)_", path.name)
    return int(match.group(1)) if match else None


def _frame_candidate(parent: Path, frame_index: int) -> Path | None:
    for ext in IMAGE_EXTS:
        matches = sorted(parent.glob(f"{frame_index}_*{ext}"))
        if matches:
            return matches[0]
    return None


def _load_stack(image_path: Path, num_frames: int, frame_stride: int):
    current = _frame_index(image_path)
    frames = []
    last_valid = image_path
    for offset in range(num_frames - 1, -1, -1):
        candidate = None
        if current is not None:
            candidate = _frame_candidate(image_path.parent, current - offset * frame_stride)
        frame_path = candidate if candidate is not None else last_valid
        image = imread_bgr(frame_path)
        if image is None:
            raise FileNotFoundError(f"failed to read image: {frame_path}")
        frames.append(image)
        last_valid = frame_path
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2D ONNX inference.")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--color-space", default="hsv")
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("onnxruntime is required for ONNX inference") from exc

    preprocess = preprocess_config_from_dict(
        {"colorSpace": args.color_space, "inputSize": [args.height, args.width], "useRoi": False},
        fallback=DEFAULT_PREPROCESS_CONFIG,
    )
    frames = _load_stack(Path(args.image).resolve(), max(1, args.num_frames), max(1, args.frame_stride))
    chw = np.concatenate([preprocess_bgr_to_chw_float(frame, config=preprocess) for frame in frames], axis=0)
    batch = np.expand_dims(chw.astype(np.float32), axis=0)
    session = ort.InferenceSession(str(Path(args.onnx).resolve()))
    output = session.run(None, {session.get_inputs()[0].name: batch})[0].reshape(-1)
    if output.size < 2:
        raise RuntimeError(f"expected ONNX output with 2 values, got {output.shape}")
    print(f"steering: {float(output[0]):.6f}")
    print(f"e_y: {float(output[1]):.6f}")


if __name__ == "__main__":
    main()
