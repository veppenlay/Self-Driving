#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect offline baseline metrics without changing the mainline pipeline."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_utils import (  # noqa: E402
    benchmark_single_sample_latency,
    checkpoint_file_size,
    collect_samples,
    count_parameters,
    load_model_spec,
    resolve_device,
    summarize_dataset,
    write_json,
)


def _build_dataset_record(
    *,
    name: str,
    path_text: str | None,
    split_name: str | None,
    recursive: bool,
    model_spec,
    device,
) -> tuple[dict[str, Any] | None, list[tuple[Path, float]]]:
    if not path_text:
        return None, []
    path = Path(path_text).expanduser().resolve()
    samples = collect_samples(path, split_name=split_name, recursive=recursive)
    summary = summarize_dataset(name, samples, model_spec, device, split_name=split_name, recursive=recursive)
    summary["source"] = str(path)
    return summary, samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect frozen baseline metrics for a steering checkpoint.")
    parser.add_argument("--ckpt", required=True, help="Path to the checkpoint file.")
    parser.add_argument("--output", required=True, help="JSON output path.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--baseline-name", default="baseline_3frame_temporal_v2")

    parser.add_argument("--test-clean-root", required=True, help="Dataset root that contains test_clean.txt.")
    parser.add_argument("--test-clean-split", default="test_clean")

    parser.add_argument("--data1-path", default=None, help="Optional flat or split-backed dataset path.")
    parser.add_argument("--data1-split", default=None, help="Optional split name for data1 path.")
    parser.add_argument("--data1-recursive", action="store_true", help="Treat data1 path as a recursive dataset.")

    parser.add_argument("--external-path", default=None, help="Optional external evaluation dataset path.")
    parser.add_argument("--external-split", default=None, help="Optional split name for the external dataset.")
    parser.add_argument("--external-recursive", action="store_true", help="Treat external path as a recursive dataset.")
    parser.add_argument("--external-name", default="external_test")

    parser.add_argument("--benchmark-source", default="test_clean", choices=["test_clean", "data1", "external"])
    parser.add_argument("--benchmark-warmup", type=int, default=30)
    parser.add_argument("--benchmark-iters", type=int, default=200)
    parser.add_argument("--benchmark-samples", type=int, default=16)
    args = parser.parse_args()

    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    device = resolve_device(args.device)
    model_spec = load_model_spec(checkpoint_path, device)

    test_clean_summary, test_clean_samples = _build_dataset_record(
        name="test_clean",
        path_text=args.test_clean_root,
        split_name=args.test_clean_split,
        recursive=False,
        model_spec=model_spec,
        device=device,
    )
    data1_summary, data1_samples = _build_dataset_record(
        name="data1",
        path_text=args.data1_path,
        split_name=args.data1_split,
        recursive=args.data1_recursive,
        model_spec=model_spec,
        device=device,
    )
    external_summary, external_samples = _build_dataset_record(
        name=args.external_name,
        path_text=args.external_path,
        split_name=args.external_split,
        recursive=args.external_recursive,
        model_spec=model_spec,
        device=device,
    )

    benchmark_pool = {
        "test_clean": test_clean_samples,
        "data1": data1_samples,
        "external": external_samples,
    }
    benchmark_samples = benchmark_pool[args.benchmark_source]
    if not benchmark_samples:
        raise RuntimeError(f"benchmark source '{args.benchmark_source}' does not contain any samples")

    payload = {
        "baselineName": args.baseline_name,
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path),
        "model": {
            "variant": model_spec.model_variant,
            "numFrames": model_spec.num_frames,
            "frameStride": model_spec.frame_stride,
            "preprocess": {
                "colorSpace": model_spec.preprocess.color_space,
                "inputSize": list(model_spec.preprocess.input_size),
                "useRoi": model_spec.preprocess.use_roi,
            },
            "parameters": count_parameters(model_spec.model),
            "fileSize": checkpoint_file_size(checkpoint_path),
        },
        "datasets": {
            "test_clean": test_clean_summary,
            "data1": data1_summary,
            "external": external_summary,
        },
        "latencyBenchmark": benchmark_single_sample_latency(
            model_spec,
            benchmark_samples,
            device,
            warmup=args.benchmark_warmup,
            iterations=args.benchmark_iters,
            sample_count=args.benchmark_samples,
        ),
    }
    write_json(output_path, payload)

    print(f"output={output_path}")
    if test_clean_summary is not None:
        print(f"test_clean mae={test_clean_summary['mae']:.6f} rmse={test_clean_summary['rmse']:.6f}")
    if data1_summary is not None:
        print(f"data1 mae={data1_summary['mae']:.6f} rmse={data1_summary['rmse']:.6f}")
    if external_summary is not None:
        print(f"{external_summary['name']} mae={external_summary['mae']:.6f} rmse={external_summary['rmse']:.6f}")
    print(
        "latency "
        f"device={payload['latencyBenchmark']['device']} "
        f"ms={payload['latencyBenchmark']['latencyMs']:.4f} "
        f"fps={payload['latencyBenchmark']['fps']:.4f}"
    )


if __name__ == "__main__":
    main()
