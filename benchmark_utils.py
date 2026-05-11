#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standalone helpers for offline baseline metric collection.

This module is intentionally kept separate from the training and evaluation
entrypoints so baseline freezing can evolve without changing the mainline
behavior.
"""

from __future__ import annotations

import json
import math
import os
import time
from ctypes import Structure, byref, c_size_t, c_ulong, sizeof
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from datasets import TEMPORAL_PADDING_DESCRIPTION, TEMPORAL_PADDING_MODE
from models import build_model_for_checkpoint
from steering_preprocess import DEFAULT_PREPROCESS_CONFIG, PreprocessConfig, imread_bgr, preprocess_bgr_to_tensor, preprocess_config_from_dict


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class LoadedModel:
    checkpoint: Path
    model: torch.nn.Module
    preprocess: PreprocessConfig
    num_frames: int
    frame_stride: int
    model_variant: str | None


def resolve_device(device_name: str = "auto") -> torch.device:
    requested = device_name.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(requested)


def load_model_spec(checkpoint_path: Path, device: torch.device) -> LoadedModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_variant = checkpoint.get("modelVariant") if isinstance(checkpoint, dict) else None
    model = build_model_for_checkpoint(state, model_variant).to(device)
    model.eval()

    preprocess = DEFAULT_PREPROCESS_CONFIG
    num_frames = max(1, int(getattr(model, "num_frames", 1)))
    frame_stride = 1
    if isinstance(checkpoint, dict):
        preprocess = preprocess_config_from_dict(checkpoint.get("preprocess"), fallback=DEFAULT_PREPROCESS_CONFIG)
        num_frames = max(1, int(checkpoint.get("numFrames", num_frames)))
        frame_stride = max(1, int(checkpoint.get("frameStride", 1)))

    return LoadedModel(
        checkpoint=checkpoint_path,
        model=model,
        preprocess=preprocess,
        num_frames=num_frames,
        frame_stride=frame_stride,
        model_variant=model_variant,
    )


def parse_angle_from_name(image_path: Path) -> float:
    stem = image_path.stem
    if "_" not in stem:
        raise ValueError(f"image filename does not contain an angle suffix: {image_path}")
    return float(stem.rsplit("_", 1)[-1])


def parse_frame_index(image_path: Path) -> int:
    stem = image_path.stem
    if "_" not in stem:
        raise ValueError(f"image filename does not contain a numeric frame prefix: {image_path}")
    prefix = stem.split("_", 1)[0]
    if not prefix.isdigit():
        raise ValueError(f"image filename does not start with a numeric frame prefix: {image_path}")
    return int(prefix)


def _image_sort_key(image_path: Path) -> tuple[Any, ...]:
    prefix = image_path.stem.split("_", 1)[0]
    if prefix.isdigit():
        return (image_path.parent.as_posix().lower(), 0, int(prefix), image_path.name.lower())
    return (image_path.parent.as_posix().lower(), 1, image_path.name.lower())


def collect_flat_samples(folder: Path) -> list[tuple[Path, float]]:
    samples: list[tuple[Path, float]] = []
    for image_path in sorted(folder.iterdir(), key=_image_sort_key):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            angle = parse_angle_from_name(image_path)
        except ValueError:
            continue
        samples.append((image_path, angle))
    return samples


def collect_recursive_samples(root: Path) -> list[tuple[Path, float]]:
    samples: list[tuple[Path, float]] = []
    for image_path in sorted(root.rglob("*"), key=_image_sort_key):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            angle = parse_angle_from_name(image_path)
        except ValueError:
            continue
        samples.append((image_path, angle))
    return samples


def collect_split_samples(data_root: Path, split_name: str) -> list[tuple[Path, float]]:
    split_path = data_root / f"{split_name}.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"split file not found: {split_path}")

    filename_lookup: dict[str, Path] = {}
    for image_path in data_root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        filename_lookup.setdefault(image_path.name, image_path)

    def _resolve_sample_path(raw_text: str) -> Path:
        raw_path = Path(raw_text)
        if raw_path.is_file():
            return raw_path
        relative_candidate = data_root / raw_path
        if relative_candidate.is_file():
            return relative_candidate
        basename_candidate = filename_lookup.get(raw_path.name)
        if basename_candidate is not None:
            return basename_candidate
        return raw_path

    samples: list[tuple[Path, float]] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        image_text, angle_text = text.rsplit(" ", 1)
        samples.append((_resolve_sample_path(image_text), float(angle_text)))
    return samples


def collect_samples(path: Path, *, split_name: str | None = None, recursive: bool = False) -> list[tuple[Path, float]]:
    if split_name:
        return collect_split_samples(path, split_name)
    if recursive:
        return collect_recursive_samples(path)
    return collect_flat_samples(path)


def build_temporal_tensor(image_path: Path, model_spec: LoadedModel, device: torch.device) -> torch.Tensor:
    if model_spec.num_frames <= 1:
        bgr = imread_bgr(image_path)
        if bgr is None:
            raise FileNotFoundError(f"failed to read image: {image_path}")
        return preprocess_bgr_to_tensor(bgr, device=device, config=model_spec.preprocess)

    current_index = parse_frame_index(image_path)
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
    return torch.cat(frames, dim=0).unsqueeze(0).to(device)


def predict_samples(
    samples: list[tuple[Path, float]],
    model_spec: LoadedModel,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[float] = []
    targets: list[float] = []
    with torch.no_grad():
        for image_path, target in samples:
            tensor = build_temporal_tensor(image_path, model_spec, device)
            pred = model_spec.model(tensor)
            predictions.append(float(pred.reshape(-1)[0].item()))
            targets.append(float(target))
    return np.asarray(predictions, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def compute_error_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    errors = predictions - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(math.sqrt(float(np.mean(np.square(errors)))))
    return {"mae": mae, "rmse": rmse}


def summarize_dataset(
    name: str,
    samples: list[tuple[Path, float]],
    model_spec: LoadedModel,
    device: torch.device,
    *,
    split_name: str | None = None,
    recursive: bool = False,
) -> dict[str, Any]:
    if not samples:
        raise RuntimeError(f"dataset '{name}' does not contain any valid samples")
    predictions, targets = predict_samples(samples, model_spec, device)
    return {
        "name": name,
        "numImages": len(samples),
        "split": split_name,
        "recursive": bool(recursive),
        "temporalPaddingMode": TEMPORAL_PADDING_MODE if model_spec.num_frames > 1 else "single_frame",
        "temporalPaddingDescription": TEMPORAL_PADDING_DESCRIPTION if model_spec.num_frames > 1 else "Temporal padding is not used.",
        **compute_error_metrics(predictions, targets),
    }


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(int(parameter.numel()) for parameter in model.parameters()),
        "trainable": sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad),
    }


def checkpoint_file_size(checkpoint_path: Path) -> dict[str, float | int]:
    size_bytes = int(checkpoint_path.stat().st_size)
    return {
        "bytes": size_bytes,
        "miB": float(size_bytes / (1024.0 * 1024.0)),
    }


class _ProcessMemoryCounters(Structure):
    _fields_ = [
        ("cb", c_ulong),
        ("PageFaultCount", c_ulong),
        ("PeakWorkingSetSize", c_size_t),
        ("WorkingSetSize", c_size_t),
        ("QuotaPeakPagedPoolUsage", c_size_t),
        ("QuotaPagedPoolUsage", c_size_t),
        ("QuotaPeakNonPagedPoolUsage", c_size_t),
        ("QuotaNonPagedPoolUsage", c_size_t),
        ("PagefileUsage", c_size_t),
        ("PeakPagefileUsage", c_size_t),
    ]


def _process_memory_snapshot_mib() -> dict[str, float] | None:
    if os.name == "nt":
        try:
            from ctypes import WinDLL

            kernel32 = WinDLL("kernel32", use_last_error=True)
            psapi = WinDLL("psapi", use_last_error=True)
            counters = _ProcessMemoryCounters()
            counters.cb = sizeof(_ProcessMemoryCounters)
            process = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(process, byref(counters), counters.cb)
            if not ok:
                return None
            scale = 1024.0 * 1024.0
            return {
                "rssMiB": float(counters.WorkingSetSize / scale),
                "peakRssMiB": float(counters.PeakWorkingSetSize / scale),
                "pagefileMiB": float(counters.PagefileUsage / scale),
                "peakPagefileMiB": float(counters.PeakPagefileUsage / scale),
            }
        except Exception:
            return None
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        factor = 1024.0 if os.name != "darwin" else 1024.0 * 1024.0
        peak_rss_mib = float(usage.ru_maxrss / factor)
        return {"rssMiB": peak_rss_mib, "peakRssMiB": peak_rss_mib}
    except Exception:
        return None


def benchmark_single_sample_latency(
    model_spec: LoadedModel,
    samples: list[tuple[Path, float]],
    device: torch.device,
    *,
    warmup: int = 30,
    iterations: int = 200,
    sample_count: int = 16,
) -> dict[str, float | int | str]:
    if not samples:
        raise RuntimeError("latency benchmark requires at least one sample")

    pool = samples[: max(1, min(sample_count, len(samples)))]
    tensors = [build_temporal_tensor(image_path, model_spec, device) for image_path, _ in pool]
    process_before = _process_memory_snapshot_mib()

    with torch.no_grad():
        for idx in range(max(0, warmup)):
            model_spec.model(tensors[idx % len(tensors)])

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for idx in range(max(1, iterations)):
            model_spec.model(tensors[idx % len(tensors)])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    avg_latency_ms = float(elapsed * 1000.0 / max(1, iterations))
    fps = float(max(1, iterations) / elapsed) if elapsed > 0 else float("inf")
    process_after = _process_memory_snapshot_mib()
    memory: dict[str, float] = {}
    if process_before is not None:
        memory.update({f"processBefore{key[0].upper()}{key[1:]}": value for key, value in process_before.items()})
    if process_after is not None:
        memory.update({f"processAfter{key[0].upper()}{key[1:]}": value for key, value in process_after.items()})
    if device.type == "cuda":
        memory["devicePeakAllocatedMiB"] = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        memory["devicePeakReservedMiB"] = float(torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0))
    return {
        "runtime": "pytorch",
        "device": str(device),
        "batchSize": 1,
        "warmupIterations": int(max(0, warmup)),
        "measuredIterations": int(max(1, iterations)),
        "latencyMs": avg_latency_ms,
        "fps": fps,
        "memory": memory or None,
    }


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
