#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate one or more regression steering checkpoints on flat and recursive datasets.

This script supports mixed checkpoints that may use different preprocessing contracts.
Each model is evaluated with its own checkpoint-provided preprocess config.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from steering_preprocess import (  # noqa: E402
    DEFAULT_PREPROCESS_CONFIG,
    PreprocessConfig,
    imread_bgr,
    preprocess_bgr_to_tensor,
    preprocess_config_from_dict,
    preprocess_path_to_tensor,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    ckpt: Path
    model: torch.nn.Module
    preprocess: PreprocessConfig
    num_frames: int
    frame_stride: int


def _load_net_module():
    import importlib.util

    module_path = REPO_ROOT / "models.py"
    spec = importlib.util.spec_from_file_location("compare_regression_net_models", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_model(name: str, ckpt_path: Path, device: torch.device) -> ModelSpec:
    module = _load_net_module()
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = module.build_model_for_checkpoint(state).to(device)
    model.eval()
    preprocess = DEFAULT_PREPROCESS_CONFIG
    num_frames = 1
    frame_stride = 1
    if isinstance(ckpt, dict):
        preprocess = preprocess_config_from_dict(ckpt.get("preprocess"), fallback=DEFAULT_PREPROCESS_CONFIG)
        num_frames = max(1, int(ckpt.get("numFrames", getattr(model, "num_frames", 1))))
        frame_stride = max(1, int(ckpt.get("frameStride", 1)))
    else:
        num_frames = max(1, int(getattr(model, "num_frames", 1)))
    return ModelSpec(
        name=name,
        ckpt=ckpt_path,
        model=model,
        preprocess=preprocess,
        num_frames=num_frames,
        frame_stride=frame_stride,
    )


def _parse_angle_from_name(image_path: Path) -> float:
    stem = image_path.stem
    if "_" not in stem:
        raise ValueError("image filename does not contain an angle suffix")
    return float(stem.rsplit("_", 1)[-1])


def _sort_key(path: Path):
    prefix = path.stem.split("_", 1)[0]
    if prefix.isdigit():
        return (0, int(prefix), path.name.lower())
    return (1, path.name.lower())


def _collect_images(folder: Path) -> list[Path]:
    images: list[Path] = []
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            _parse_angle_from_name(path)
        except ValueError:
            continue
        images.append(path)
    return sorted(images, key=_sort_key)


def _find_image_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    for directory in sorted([p for p in root.rglob("*") if p.is_dir()]):
        if _collect_images(directory):
            dirs.append(directory)
    return dirs


def _safe_name(path: Path, root: Path | None = None) -> str:
    rel = path.relative_to(root) if root is not None else path
    return "__".join(rel.parts).replace(":", "")


def _config_key(config: PreprocessConfig) -> tuple[str, tuple[int, int], bool]:
    return (config.color_space, tuple(config.input_size), config.use_roi)


def _parse_frame_index(path: Path) -> int:
    prefix = path.stem.split("_", 1)[0]
    if not prefix.isdigit():
        raise ValueError(f"image filename does not start with numeric frame index: {path}")
    return int(prefix)


def _build_temporal_tensor(image_path: Path, model_spec: ModelSpec, device: torch.device) -> torch.Tensor:
    if model_spec.num_frames <= 1:
        return preprocess_path_to_tensor(image_path, device=device, config=model_spec.preprocess)

    current_index = _parse_frame_index(image_path)
    parent = image_path.parent
    frames: list[torch.Tensor] = []
    last_valid_path = image_path
    for offset in range(model_spec.num_frames - 1, -1, -1):
        target_index = current_index - offset * model_spec.frame_stride
        candidate = last_valid_path
        if target_index >= 0:
            matches = sorted(parent.glob(f"{target_index}_*"))
            if matches:
                candidate = matches[0]
        bgr = imread_bgr(candidate)
        if bgr is None:
            raise FileNotFoundError(f"failed to read temporal frame: {candidate}")
        frames.append(preprocess_bgr_to_tensor(bgr, config=model_spec.preprocess).squeeze(0))
        last_valid_path = candidate
    stacked = torch.cat(frames, dim=0).unsqueeze(0)
    return stacked.to(device)


def _predict_batch(image_paths: list[Path], models: list[ModelSpec], device: torch.device) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    preds_by_model: dict[str, list[float]] = {model.name: [] for model in models}
    gt_values: list[float] = []
    for idx, image_path in enumerate(image_paths):
        gt = _parse_angle_from_name(image_path)
        row: dict[str, Any] = {"index": idx, "image": str(image_path), "groundTruth": gt}
        gt_values.append(gt)
        tensor_cache: dict[tuple[str, tuple[int, int], bool, int, int], torch.Tensor] = {}
        with torch.no_grad():
            for model_spec in models:
                key = (*_config_key(model_spec.preprocess), model_spec.num_frames, model_spec.frame_stride)
                tensor = tensor_cache.get(key)
                if tensor is None:
                    tensor = _build_temporal_tensor(image_path, model_spec, device)
                    tensor_cache[key] = tensor
                pred = model_spec.model(tensor)
                value = float(pred.reshape(-1)[0].item())
                preds_by_model[model_spec.name].append(value)
                row[model_spec.name] = value
                row[f"{model_spec.name}_abs_error"] = abs(value - gt)
        rows.append(row)

    gt_arr = np.asarray(gt_values, dtype=np.float64)
    pred_arrays = {name: np.asarray(values, dtype=np.float64) for name, values in preds_by_model.items()}
    mae = {name: float(np.mean(np.abs(values - gt_arr))) for name, values in pred_arrays.items()}
    return {"rows": rows, "groundTruth": gt_arr, "predictions": pred_arrays, "mae": mae}


def _angle_group_summary(gt_arr: np.ndarray, pred_arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for angle in sorted(set(float(v) for v in gt_arr.tolist())):
        mask = np.isclose(gt_arr, angle, atol=1e-6)
        row: dict[str, Any] = {"groundTruth": angle, "count": int(mask.sum())}
        for name, pred in pred_arrays.items():
            row[f"{name}_mae"] = float(np.abs(pred[mask] - gt_arr[mask]).mean()) if mask.any() else 0.0
            row[f"{name}_mean_pred"] = float(pred[mask].mean()) if mask.any() else 0.0
            row[f"{name}_mean_bias"] = float((pred[mask] - gt_arr[mask]).mean()) if mask.any() else 0.0
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pick_colors(count: int) -> list[str]:
    base = ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e", "#8c564b", "#17becf"]
    if count <= len(base):
        return base[:count]
    cmap = plt.cm.get_cmap("tab20", count)
    return [cmap(i) for i in range(count)]


def _plot_flat(
    output_plot: Path,
    dataset_name: str,
    gt_arr: np.ndarray,
    pred_arrays: dict[str, np.ndarray],
    mae: dict[str, float],
) -> None:
    names = list(pred_arrays.keys())
    colors = _pick_colors(len(names))
    abs_errors = {name: np.abs(pred_arrays[name] - gt_arr) for name in names}
    angle_rows = _angle_group_summary(gt_arr, pred_arrays)
    unique_angles = [row["groundTruth"] for row in angle_rows]
    counts = [row["count"] for row in angle_rows]
    x = np.arange(len(unique_angles))
    width = 0.78 / max(1, len(names))

    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    fig.suptitle(f"Regression Review - {dataset_name} ({len(gt_arr)} images)", fontsize=15)

    ax = axes[0, 0]
    values = [mae[name] for name in names]
    ax.bar(names, values, color=colors)
    ax.set_title("Overall MAE")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    ax = axes[0, 1]
    boxplot_kwargs = {"showfliers": False}
    if "tick_labels" in inspect.signature(ax.boxplot).parameters:
        boxplot_kwargs["tick_labels"] = names
    else:
        boxplot_kwargs["labels"] = names
    ax.boxplot([abs_errors[name] for name in names], **boxplot_kwargs)
    ax.set_title("Absolute Error Distribution")
    ax.set_ylabel("Absolute Error")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    ax = axes[1, 0]
    start = -0.5 * width * (len(names) - 1)
    for i, name in enumerate(names):
        values = [row[f"{name}_mae"] for row in angle_rows]
        ax.bar(x + start + i * width, values, width=width, label=name, color=colors[i], alpha=0.88)
    ax.set_title("MAE by Ground-Truth Angle")
    ax.set_xlabel("Ground-Truth Angle")
    ax.set_ylabel("MAE")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{angle:g}\n(n={count})" for angle, count in zip(unique_angles, counts)], rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(unique_angles, unique_angles, color="#111111", linewidth=2.0, label="Ideal")
    for i, name in enumerate(names):
        values = [row[f"{name}_mean_pred"] for row in angle_rows]
        ax.plot(unique_angles, values, marker="o", linewidth=1.8, label=name, color=colors[i])
    ax.set_title("Mean Prediction by Ground-Truth Angle")
    ax.set_xlabel("Ground-Truth Angle")
    ax.set_ylabel("Mean Prediction")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_plot, dpi=170)
    plt.close(fig)


def _test_flat_dataset(name: str, folder: Path, models: list[ModelSpec], output_dir: Path, device: torch.device) -> dict[str, Any]:
    images = _collect_images(folder)
    if not images:
        raise FileNotFoundError(f"no angle-suffixed images found: {folder}")
    result = _predict_batch(images, models, device)
    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    json_path = dataset_dir / "regression_compare.json"
    csv_path = dataset_dir / "regression_compare.csv"
    angle_csv_path = dataset_dir / "regression_compare_by_angle.csv"
    plot_path = dataset_dir / "regression_compare.png"

    angle_rows = _angle_group_summary(result["groundTruth"], result["predictions"])
    _write_csv(csv_path, result["rows"])
    _write_csv(angle_csv_path, angle_rows)
    _plot_flat(plot_path, name, result["groundTruth"], result["predictions"], result["mae"])

    summary = {
        "name": name,
        "kind": "flat",
        "folder": str(folder),
        "numImages": len(images),
        "mae": result["mae"],
        "json": str(json_path),
        "csv": str(csv_path),
        "angleCsv": str(angle_csv_path),
        "plot": str(plot_path),
    }
    json_path.write_text(json.dumps({**summary, "results": result["rows"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _test_recursive_dataset(name: str, root: Path, models: list[ModelSpec], output_dir: Path, device: torch.device) -> dict[str, Any]:
    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    image_dirs = _find_image_dirs(root)
    folder_records: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for folder in image_dirs:
        rel_name = _safe_name(folder, root)
        summary = _test_flat_dataset(rel_name, folder, models, dataset_dir, device)
        folder_records.append(
            {
                "folder": str(folder),
                "relativeFolder": folder.relative_to(root).as_posix(),
                "numImages": summary["numImages"],
                "mae": summary["mae"],
                "plot": summary["plot"],
                "json": summary["json"],
            }
        )
        folder_json = json.loads(Path(summary["json"]).read_text(encoding="utf-8"))
        for row in folder_json.get("results", []):
            row["relativeFolder"] = folder.relative_to(root).as_posix()
            all_rows.append(row)

    if not all_rows:
        raise FileNotFoundError(f"no recursive image folders found: {root}")

    model_names = [model.name for model in models]
    total_images = sum(int(record["numImages"]) for record in folder_records)
    weighted_mae: dict[str, float] = {}
    for model_name in model_names:
        weighted_mae[model_name] = sum(float(record["mae"][model_name]) * int(record["numImages"]) for record in folder_records) / total_images

    summary_csv = dataset_dir / "recursive_regression_folder_summary.csv"
    _write_csv(
        summary_csv,
        [
            {
                "folder": record["folder"],
                "relativeFolder": record["relativeFolder"],
                "numImages": record["numImages"],
                **{f"{name}_mae": record["mae"][name] for name in model_names},
                "plot": record["plot"],
                "json": record["json"],
            }
            for record in folder_records
        ],
    )

    summary_plot = dataset_dir / "recursive_regression_folder_mae.png"
    ordered = sorted(folder_records, key=lambda r: min(float(r["mae"][name]) for name in model_names))
    y = np.arange(len(ordered))
    height = 0.78 / max(1, len(model_names))
    colors = _pick_colors(len(model_names))
    fig_height = max(6.0, min(28.0, 0.55 * len(ordered) + 2.5))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    start = -0.5 * height * (len(model_names) - 1)
    for i, model_name in enumerate(model_names):
        values = [float(record["mae"][model_name]) for record in ordered]
        ax.barh(y + start + i * height, values, height=height, label=model_name, color=colors[i], alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([record["relativeFolder"] for record in ordered], fontsize=9)
    ax.set_xlabel("MAE")
    ax.set_title(f"Folder-Level MAE - {name}")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(summary_plot, dpi=170)
    plt.close(fig)

    index_path = dataset_dir / "recursive_regression_index.json"
    summary = {
        "name": name,
        "kind": "recursive",
        "root": str(root),
        "numFolders": len(folder_records),
        "numImages": total_images,
        "mae": weighted_mae,
        "folderSummaryCsv": str(summary_csv),
        "folderSummaryPlot": str(summary_plot),
        "index": str(index_path),
        "folders": folder_records,
    }
    index_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _write_report(output_dir: Path, records: list[dict[str, Any]], models: list[ModelSpec]) -> Path:
    report_path = output_dir / "REGRESSION_COMPARE_REVIEW_CN.md"
    model_names = [model.name for model in models]

    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    lines: list[str] = []
    lines.append("# 回归模型对比评测")
    lines.append("")
    lines.append("## 评测模型")
    lines.append("")
    for model in models:
        lines.append(
            f"- `{model.name}`: `{model.ckpt}` | preprocess=`{model.preprocess.color_space}` / "
            f"`{model.preprocess.input_size[0]}x{model.preprocess.input_size[1]}` / ROI=`{model.preprocess.use_roi}` / "
            f"num_frames=`{model.num_frames}` / frame_stride=`{model.frame_stride}`"
        )
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    header = ["dataset", "kind", "images", "folders", *[f"{name} MAE" for name in model_names], "winner"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for record in records:
        mae = record.get("mae") or {}
        winner = min(model_names, key=lambda name: float(mae.get(name, float("inf"))))
        row = [
            record["name"],
            record["kind"],
            fmt(record.get("numImages")),
            fmt(record.get("numFolders")),
            *[fmt(mae.get(name)) for name in model_names],
            winner,
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## 结果文件")
    lines.append("")
    for record in records:
        lines.append(f"- `{record['name']}`: `{json.dumps(record, ensure_ascii=False)}`")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _parse_model_specs(args: argparse.Namespace, device: torch.device) -> list[ModelSpec]:
    models: list[ModelSpec] = []
    for spec in args.model:
        if "=" not in spec:
            raise ValueError(f"invalid --model spec: {spec}")
        name, raw_path = spec.split("=", 1)
        models.append(_load_model(name.strip(), Path(raw_path).resolve(), device))
    if models:
        return models
    if args.improved_ckpt and args.legacy_ckpt:
        return [
            _load_model("improved_mobilenet", Path(args.improved_ckpt).resolve(), device),
            _load_model("legacy_original_cnn", Path(args.legacy_ckpt).resolve(), device),
        ]
    raise ValueError("provide --model name=ckpt at least twice, or use --improved-ckpt with --legacy-ckpt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare regression checkpoints with per-checkpoint preprocess contracts.")
    parser.add_argument("--model", action="append", default=[], help="Format: name=checkpoint_path; may be repeated")
    parser.add_argument("--improved-ckpt", default=None)
    parser.add_argument("--legacy-ckpt", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--flat-dataset", action="append", default=[], help="Format: name=path")
    parser.add_argument("--recursive-dataset", action="append", default=[], help="Format: name=path")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = _parse_model_specs(args, device)
    records: list[dict[str, Any]] = []

    for spec in args.flat_dataset:
        name, raw_path = spec.split("=", 1)
        print(f"TEST flat {name}: {raw_path}")
        records.append(_test_flat_dataset(name, Path(raw_path).resolve(), models, output_dir, device))

    for spec in args.recursive_dataset:
        name, raw_path = spec.split("=", 1)
        print(f"TEST recursive {name}: {raw_path}")
        records.append(_test_recursive_dataset(name, Path(raw_path).resolve(), models, output_dir, device))

    index_path = output_dir / "regression_compare_index.json"
    index = {
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "models": [
            {
                "name": model.name,
                "ckpt": str(model.ckpt),
                "preprocess": {
                    "colorSpace": model.preprocess.color_space,
                    "inputSize": list(model.preprocess.input_size),
                    "useRoi": model.preprocess.use_roi,
                },
                "numFrames": model.num_frames,
                "frameStride": model.frame_stride,
            }
            for model in models
        ],
        "records": records,
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_report(output_dir, records, models)

    print(f"REPORT={report_path}")
    print(f"INDEX={index_path}")
    for record in records:
        print(f"{record['name']}: {record.get('mae')}")


if __name__ == "__main__":
    main()
