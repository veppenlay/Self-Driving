#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from epaicar_temporal3_2d_trt_udp_server import (
    NUM_FRAMES,
    TensorRTInfer,
    imread_bgr,
    preprocess_frame_bgr,
    stack_chw_frames,
)


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def parse_index_label(path):
    stem = Path(path).stem
    parts = stem.split("_", 1)
    index = None
    label = None
    if parts and parts[0].isdigit():
        index = int(parts[0])
    if len(parts) > 1:
        try:
            label = float(parts[1])
        except ValueError:
            label = None
    return index, label


def sort_key(path):
    index, _ = parse_index_label(path)
    if index is None:
        return (1, Path(path).name)
    return (0, index, Path(path).name)


def list_images(seq_dir):
    seq_dir = Path(seq_dir)
    files = [p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files, key=sort_key)


def build_temporal_input(paths, current_pos, frame_stride):
    frames = []
    used_paths = []
    for slot in range(NUM_FRAMES):
        offset = (NUM_FRAMES - 1 - slot) * frame_stride
        pos = max(0, current_pos - offset)
        src_path = paths[pos]
        bgr = imread_bgr(src_path)
        if bgr is None:
            raise RuntimeError("failed to read image: {}".format(src_path))
        frames.append(preprocess_frame_bgr(bgr))
        used_paths.append(src_path.name)
    return stack_chw_frames(frames), used_paths


def summarize(rows):
    labeled = [r for r in rows if r["gt"] is not None]
    if not labeled:
        return {"count": len(rows), "labeled_count": 0}
    errors = np.array([r["error"] for r in labeled], dtype=np.float64)
    abs_errors = np.abs(errors)
    preds = np.array([r["steering"] for r in labeled], dtype=np.float64)
    gts = np.array([r["gt"] for r in labeled], dtype=np.float64)
    worst = max(labeled, key=lambda r: r["abs_error"])
    return {
        "count": len(rows),
        "labeled_count": len(labeled),
        "mae": float(abs_errors.mean()),
        "rmse": float(math.sqrt(np.mean(errors * errors))),
        "max_abs_error": float(abs_errors.max()),
        "max_abs_error_frame": worst["frame"],
        "max_abs_error_gt": worst["gt"],
        "max_abs_error_pred": worst["steering"],
        "mean_gt": float(gts.mean()),
        "mean_pred": float(preds.mean()),
        "min_pred": float(preds.min()),
        "max_pred": float(preds.max()),
    }


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "index",
                "gt",
                "steering",
                "e_y",
                "error",
                "abs_error",
                "frames",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate temporal3 2D TensorRT engine on an image sequence")
    parser.add_argument("--engine", default="/home/epaicar/results/0703_temporal3_2d_fp16.engine")
    parser.add_argument("--seq-dir", required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--print-rows", type=int, default=12)
    return parser


def main():
    args = build_parser().parse_args()
    paths = list_images(args.seq_dir)
    if not paths:
        raise RuntimeError("no images found in {}".format(args.seq_dir))

    out_csv = args.out_csv or os.path.join(args.seq_dir, "temporal3_2d_trt_results.csv")
    out_json = args.out_json or os.path.join(args.seq_dir, "temporal3_2d_trt_summary.json")

    infer = TensorRTInfer(args.engine)
    rows = []
    try:
        for pos, path in enumerate(paths):
            inp, used = build_temporal_input(paths, pos, args.frame_stride)
            output = infer.infer(inp).reshape(-1)
            steering = float(output[0])
            e_y = float(output[1]) if output.size > 1 else 0.0
            index, gt = parse_index_label(path)
            if gt is None:
                error = None
                abs_error = None
            else:
                error = steering - gt
                abs_error = abs(error)
            rows.append({
                "frame": path.name,
                "index": index,
                "gt": gt,
                "steering": steering,
                "e_y": e_y,
                "error": error,
                "abs_error": abs_error,
                "frames": "|".join(used),
            })
    finally:
        infer.close()

    summary = summarize(rows)
    write_csv(out_csv, rows)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("engine={}".format(args.engine))
    print("seq_dir={}".format(args.seq_dir))
    print("frame_stride={}".format(args.frame_stride))
    print("input_shape=(1, 9, 144, 192)")
    print("count={count} labeled_count={labeled_count}".format(**summary))
    if summary.get("labeled_count", 0):
        print(
            "mae={mae:.6f} rmse={rmse:.6f} max_abs_error={max_abs_error:.6f} "
            "worst={max_abs_error_frame} gt={max_abs_error_gt:.6f} pred={max_abs_error_pred:.6f}".format(**summary)
        )
        print(
            "mean_gt={mean_gt:.6f} mean_pred={mean_pred:.6f} "
            "min_pred={min_pred:.6f} max_pred={max_pred:.6f}".format(**summary)
        )

    print("csv={}".format(out_csv))
    print("json={}".format(out_json))

    head = rows[: max(0, args.print_rows)]
    if head:
        print("first_rows:")
        for row in head:
            print(
                "{frame} gt={gt} steering={steering:.6f} e_y={e_y:.6f} "
                "abs_error={abs_error} frames={frames}".format(**row)
            )

    labeled = [r for r in rows if r["abs_error"] is not None]
    if labeled:
        print("worst_rows:")
        for row in sorted(labeled, key=lambda r: r["abs_error"], reverse=True)[: max(0, args.print_rows)]:
            print(
                "{frame} gt={gt:.6f} steering={steering:.6f} e_y={e_y:.6f} "
                "abs_error={abs_error:.6f} frames={frames}".format(**row)
            )


if __name__ == "__main__":
    main()
