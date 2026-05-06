#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare temporal steering checkpoints with unified offline metrics."""

from __future__ import annotations

import argparse
import json
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


def _parse_model_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"invalid --model spec: {spec}")
    name, raw_path = spec.split("=", 1)
    return name.strip(), Path(raw_path).expanduser().resolve()


def _parse_dataset_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"invalid dataset spec: {spec}")
    name, raw_path = spec.split("=", 1)
    return name.strip(), Path(raw_path).expanduser().resolve()


def _parse_split_dataset_spec(spec: str) -> tuple[str, Path, str]:
    name, payload = _parse_dataset_spec(spec)
    if "@" not in str(payload):
        raise ValueError(f"split dataset spec must use name=path@split, got: {spec}")
    raw_path, split_name = str(payload).rsplit("@", 1)
    return name, Path(raw_path).expanduser().resolve(), split_name.strip()


def _safe_ratio(new_value: float, base_value: float) -> float | None:
    if abs(base_value) < 1e-12:
        return None
    return float((new_value - base_value) / base_value)


def _format_delta(delta: float | None) -> str:
    if delta is None:
        return "NA"
    return f"{delta * 100.0:+.2f}%"


def _evaluate_model(
    *,
    model_name: str,
    checkpoint_path: Path,
    datasets: list[dict[str, Any]],
    device,
    benchmark_source: str,
    benchmark_warmup: int,
    benchmark_iters: int,
    benchmark_samples: int,
) -> dict[str, Any]:
    model_spec = load_model_spec(checkpoint_path, device)
    dataset_summaries: dict[str, dict[str, Any]] = {}
    sample_pools: dict[str, list[tuple[Path, float]]] = {}

    for item in datasets:
        samples = collect_samples(
            item["path"],
            split_name=item.get("split"),
            recursive=bool(item.get("recursive", False)),
        )
        sample_pools[item["name"]] = samples
        dataset_summaries[item["name"]] = summarize_dataset(
            item["name"],
            samples,
            model_spec,
            device,
            split_name=item.get("split"),
            recursive=bool(item.get("recursive", False)),
        )
        dataset_summaries[item["name"]]["source"] = str(item["path"])

    benchmark_samples_pool = sample_pools.get(benchmark_source)
    if not benchmark_samples_pool:
        raise RuntimeError(f"benchmark source dataset '{benchmark_source}' is missing or empty")

    return {
        "name": model_name,
        "checkpoint": str(checkpoint_path),
        "modelVariant": model_spec.model_variant,
        "numFrames": model_spec.num_frames,
        "frameStride": model_spec.frame_stride,
        "preprocess": {
            "colorSpace": model_spec.preprocess.color_space,
            "inputSize": list(model_spec.preprocess.input_size),
            "useRoi": model_spec.preprocess.use_roi,
        },
        "parameters": count_parameters(model_spec.model),
        "fileSize": checkpoint_file_size(checkpoint_path),
        "datasets": dataset_summaries,
        "latencyBenchmark": benchmark_single_sample_latency(
            model_spec,
            benchmark_samples_pool,
            device,
            warmup=benchmark_warmup,
            iterations=benchmark_iters,
            sample_count=benchmark_samples,
        ),
    }


def _build_comparison(models: list[dict[str, Any]], dataset_names: list[str]) -> dict[str, Any] | None:
    if len(models) != 2:
        return None
    reference = models[0]
    candidate = models[1]
    comparison: dict[str, Any] = {
        "referenceModel": reference["name"],
        "candidateModel": candidate["name"],
        "datasets": {},
        "latency": {},
        "size": {},
    }
    for dataset_name in dataset_names:
        ref_metrics = reference["datasets"][dataset_name]
        cand_metrics = candidate["datasets"][dataset_name]
        comparison["datasets"][dataset_name] = {
            "referenceMae": ref_metrics["mae"],
            "candidateMae": cand_metrics["mae"],
            "maeDelta": cand_metrics["mae"] - ref_metrics["mae"],
            "maeDeltaRatio": _safe_ratio(cand_metrics["mae"], ref_metrics["mae"]),
            "referenceRmse": ref_metrics["rmse"],
            "candidateRmse": cand_metrics["rmse"],
            "rmseDelta": cand_metrics["rmse"] - ref_metrics["rmse"],
            "rmseDeltaRatio": _safe_ratio(cand_metrics["rmse"], ref_metrics["rmse"]),
        }

    ref_latency = reference["latencyBenchmark"]
    cand_latency = candidate["latencyBenchmark"]
    comparison["latency"] = {
        "referenceLatencyMs": ref_latency["latencyMs"],
        "candidateLatencyMs": cand_latency["latencyMs"],
        "latencyDeltaMs": cand_latency["latencyMs"] - ref_latency["latencyMs"],
        "latencyDeltaRatio": _safe_ratio(cand_latency["latencyMs"], ref_latency["latencyMs"]),
        "referenceFps": ref_latency["fps"],
        "candidateFps": cand_latency["fps"],
        "fpsDelta": cand_latency["fps"] - ref_latency["fps"],
        "fpsDeltaRatio": _safe_ratio(cand_latency["fps"], ref_latency["fps"]),
    }
    comparison["size"] = {
        "referenceFileSizeMiB": reference["fileSize"]["miB"],
        "candidateFileSizeMiB": candidate["fileSize"]["miB"],
        "fileSizeDeltaMiB": candidate["fileSize"]["miB"] - reference["fileSize"]["miB"],
        "fileSizeDeltaRatio": _safe_ratio(candidate["fileSize"]["miB"], reference["fileSize"]["miB"]),
        "referenceParams": reference["parameters"]["total"],
        "candidateParams": candidate["parameters"]["total"],
        "paramDelta": candidate["parameters"]["total"] - reference["parameters"]["total"],
        "paramDeltaRatio": _safe_ratio(candidate["parameters"]["total"], reference["parameters"]["total"]),
    }
    return comparison


def _write_markdown_report(
    output_path: Path,
    models: list[dict[str, Any]],
    dataset_names: list[str],
    comparison: dict[str, Any] | None,
) -> None:
    lines: list[str] = []
    lines.append("# 2-frame vs 3-frame Temporal V2")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## Models")
    lines.append("")
    for model in models:
        lines.append(
            f"- `{model['name']}`: ckpt=`{model['checkpoint']}`, variant=`{model['modelVariant']}`, "
            f"num_frames=`{model['numFrames']}`, frame_stride=`{model['frameStride']}`, "
            f"params=`{model['parameters']['total']:,}`, file_size_mib=`{model['fileSize']['miB']:.3f}`"
        )
    lines.append("")
    lines.append("## Accuracy")
    lines.append("")
    header = ["dataset"]
    for model in models:
        header.extend([f"{model['name']} MAE", f"{model['name']} RMSE"])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for dataset_name in dataset_names:
        row = [dataset_name]
        for model in models:
            metrics = model["datasets"][dataset_name]
            row.extend([f"{metrics['mae']:.6f}", f"{metrics['rmse']:.6f}"])
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Latency")
    lines.append("")
    lines.append("| model | latency_ms | fps | file_size_mib | params |")
    lines.append("|---|---:|---:|---:|---:|")
    for model in models:
        bench = model["latencyBenchmark"]
        lines.append(
            f"| {model['name']} | {bench['latencyMs']:.6f} | {bench['fps']:.6f} | "
            f"{model['fileSize']['miB']:.6f} | {model['parameters']['total']} |"
        )
    lines.append("")
    lines.append("## Memory")
    lines.append("")
    for model in models:
        memory = model["latencyBenchmark"].get("memory")
        lines.append(f"- `{model['name']}`: `{json.dumps(memory, ensure_ascii=False)}`")
    lines.append("")
    if comparison is not None:
        lines.append("## Delta (candidate vs reference)")
        lines.append("")
        lines.append(
            f"Reference=`{comparison['referenceModel']}` Candidate=`{comparison['candidateModel']}`"
        )
        lines.append("")
        lines.append("| dataset | MAE delta | MAE delta % | RMSE delta | RMSE delta % |")
        lines.append("|---|---:|---:|---:|---:|")
        for dataset_name, metrics in comparison["datasets"].items():
            lines.append(
                f"| {dataset_name} | {metrics['maeDelta']:+.6f} | {_format_delta(metrics['maeDeltaRatio'])} | "
                f"{metrics['rmseDelta']:+.6f} | {_format_delta(metrics['rmseDeltaRatio'])} |"
            )
        latency = comparison["latency"]
        size = comparison["size"]
        lines.append("")
        lines.append(
            f"- Latency delta: `{latency['latencyDeltaMs']:+.6f} ms` ({_format_delta(latency['latencyDeltaRatio'])})"
        )
        lines.append(f"- FPS delta: `{latency['fpsDelta']:+.6f}` ({_format_delta(latency['fpsDeltaRatio'])})")
        lines.append(
            f"- File size delta: `{size['fileSizeDeltaMiB']:+.6f} MiB` ({_format_delta(size['fileSizeDeltaRatio'])})"
        )
        lines.append(f"- Parameter delta: `{size['paramDelta']:+d}` ({_format_delta(size['paramDeltaRatio'])})")
        lines.append("")
        lines.append("## Deployment Recommendation")
        lines.append("")
        lines.append(
            "Fill this section after reviewing `data_test` and `test_clean` degradation against the measured latency/FPS gain."
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare temporal checkpoints with unified MAE/RMSE/latency metrics.")
    parser.add_argument("--model", action="append", required=True, help="Format: name=checkpoint_path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split-dataset", action="append", default=[], help="Format: name=path@split")
    parser.add_argument("--flat-dataset", action="append", default=[], help="Format: name=path")
    parser.add_argument("--recursive-dataset", action="append", default=[], help="Format: name=path")
    parser.add_argument("--benchmark-source", required=True, help="Dataset name used for latency benchmark.")
    parser.add_argument("--benchmark-warmup", type=int, default=30)
    parser.add_argument("--benchmark-iters", type=int, default=200)
    parser.add_argument("--benchmark-samples", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    dataset_specs: list[dict[str, Any]] = []
    for spec in args.split_dataset:
        name, path, split_name = _parse_split_dataset_spec(spec)
        dataset_specs.append({"name": name, "path": path, "split": split_name, "recursive": False, "kind": "split"})
    for spec in args.flat_dataset:
        name, path = _parse_dataset_spec(spec)
        dataset_specs.append({"name": name, "path": path, "split": None, "recursive": False, "kind": "flat"})
    for spec in args.recursive_dataset:
        name, path = _parse_dataset_spec(spec)
        dataset_specs.append({"name": name, "path": path, "split": None, "recursive": True, "kind": "recursive"})
    if not dataset_specs:
        raise ValueError("at least one dataset must be provided")

    models: list[dict[str, Any]] = []
    for spec in args.model:
        name, checkpoint_path = _parse_model_spec(spec)
        print(f"Evaluate model={name} checkpoint={checkpoint_path}")
        models.append(
            _evaluate_model(
                model_name=name,
                checkpoint_path=checkpoint_path,
                datasets=dataset_specs,
                device=device,
                benchmark_source=args.benchmark_source,
                benchmark_warmup=args.benchmark_warmup,
                benchmark_iters=args.benchmark_iters,
                benchmark_samples=args.benchmark_samples,
            )
        )

    dataset_names = [item["name"] for item in dataset_specs]
    comparison = _build_comparison(models, dataset_names)
    payload = {
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "datasets": [
            {
                **item,
                "path": str(item["path"]),
            }
            for item in dataset_specs
        ],
        "models": models,
        "comparison": comparison,
    }

    json_path = output_dir / "temporal_ablation_summary.json"
    md_path = output_dir / "TEMPORAL_ABLATION_REPORT.md"
    write_json(json_path, payload)
    _write_markdown_report(md_path, models, dataset_names, comparison)

    print(f"summary={json_path}")
    print(f"report={md_path}")


if __name__ == "__main__":
    main()
