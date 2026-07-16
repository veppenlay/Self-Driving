#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared train/eval helpers for exploration challengers (26合并 → 25地下室)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED = REPO_ROOT / "locked_model"
if str(LOCKED) not in sys.path:
    sys.path.insert(0, str(LOCKED))

from steering_preprocess import PreprocessConfig, preprocess_config_to_dict  # noqa: E402

BASELINE_HMM_MAE = 0.1672482219835827
HMM_JSON = REPO_ROOT / "epaicar_deploy" / "steering_only_hmm_online_model.json"
HMM_SCRIPT = LOCKED / "apply_current_hmm_postprocess.py"


def device_of(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(
    dataset_cls: type,
    label_csv: Path,
    preprocess: PreprocessConfig,
    *,
    batch_size: int,
    num_workers: int,
    num_frames: int = 3,
    **dataset_kwargs: Any,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    common = dict(preprocess=preprocess, num_frames=num_frames, frame_stride=1, **dataset_kwargs)
    pin = torch.cuda.is_available()
    return (
        DataLoader(dataset_cls(label_csv, split="train", **common), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin),
        DataLoader(dataset_cls(label_csv, split="val", **common), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin),
        DataLoader(dataset_cls(label_csv, split="test", **common), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin),
    )


def train_loop(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    ey_loss_weight: float,
    early_stop: bool,
    min_delta: float,
    patience_epochs: int,
    experiment_name: str,
    preprocess: PreprocessConfig,
    loss_fn: Callable[..., torch.Tensor],
    metrics_fn: Callable[..., dict[str, float]],
    init_checkpoint: Path | None = None,
    seed: int = 20260629,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    best_path = output_dir / "checkpoints" / "best.pth"
    latest_path = output_dir / "checkpoints" / "latest.pth"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    epochs_no_improve = 0
    early_stopped = False
    stopped_epoch = None

    if init_checkpoint is not None and init_checkpoint.is_file():
        payload = torch.load(init_checkpoint, map_location=device, weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model_sd = model.state_dict()
        filtered = {
            k: v
            for k, v in state.items()
            if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)
        }
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        skipped = len(state) - len(filtered)
        print(f"init={init_checkpoint} loaded={len(filtered)} skipped_shape={skipped} missing={len(missing)} unexpected={len(unexpected)}")

    print(f"exp={experiment_name} early_stop={early_stop} min_delta={min_delta} patience={patience_epochs} max_epochs={epochs}")

    def run_epoch(loader: DataLoader, train: bool) -> dict[str, float]:
        model.train(train)
        totals = {"loss": 0.0, "steeringMAE": 0.0, "eyMAE": 0.0, "eyWeightSum": 0.0, "count": 0.0}
        for images, labels, meta in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            ey_quality = meta["eyQuality"].to(device, non_blocking=True).view(-1)
            with torch.set_grad_enabled(train):
                pred = model(images)
                loss = loss_fn(pred, labels, ey_quality, ey_loss_weight)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            batch = float(labels.size(0))
            m = metrics_fn(pred.detach(), labels, ey_quality)
            totals["loss"] += float(loss.item()) * batch
            totals["steeringMAE"] += m["steering_abs"]
            totals["eyMAE"] += m["ey_abs"]
            totals["eyWeightSum"] += m["ey_w"]
            totals["count"] += batch
        count = max(1.0, totals["count"])
        ey_w = max(1e-6, totals["eyWeightSum"])
        return {
            "loss": totals["loss"] / count,
            "steeringMAE": totals["steeringMAE"] / count,
            "eyMAE": totals["eyMAE"] / ey_w,
            "eyWeightSum": totals["eyWeightSum"],
        }

    for epoch in range(1, epochs + 1):
        train_m = run_epoch(train_loader, True)
        val_m = run_epoch(val_loader, False)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_m, "val": val_m, "lr": float(scheduler.get_last_lr()[0])})
        print(
            f"epoch={epoch:03d} train_loss={train_m['loss']:.6f} "
            f"train_steer_mae={train_m['steeringMAE']:.6f} val_steer_mae={val_m['steeringMAE']:.6f} "
            f"val_ey_mae={val_m['eyMAE']:.6f}"
        )
        payload = {
            "model": model.state_dict(),
            "epoch": epoch,
            "bestValSteeringMAE": best_val,
            "experiment": experiment_name,
            "preprocess": preprocess_config_to_dict(preprocess),
            "numFrames": 3,
            "frameStride": 1,
        }
        torch.save(payload, latest_path)
        val_mae = float(val_m["steeringMAE"])
        meaningful = val_mae < (best_val - min_delta)
        if val_mae < best_val:
            best_val = val_mae
            payload["bestValSteeringMAE"] = best_val
            torch.save(payload, best_path)
        if early_stop:
            if meaningful:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"early_stop_stale={epochs_no_improve}/{patience_epochs}")
                if epochs_no_improve >= patience_epochs:
                    early_stopped = True
                    stopped_epoch = epoch
                    print(f"early_stop triggered at epoch={epoch} best_val={best_val:.6f}")
                    break

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    test_m = run_epoch(test_loader, False)
    summary = {
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment_name,
        "labelDataBoundary": {"trainVal": "dataset/26合并", "test": "dataset/25地下室"},
        "bestValSteeringMAE": best_val,
        "testMetrics": test_m,
        "earlyStop": {
            "enabled": early_stop,
            "patienceMinDelta": min_delta,
            "patienceEpochs": patience_epochs,
            "triggered": early_stopped,
            "stoppedEpoch": stopped_epoch,
        },
        "preprocess": preprocess_config_to_dict(preprocess),
        "history": history,
        "bestCheckpoint": str(best_path),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"best={best_path} test_steer_mae={test_m['steeringMAE']:.6f}")
    return summary


@torch.no_grad()
def evaluate_predictions(model: nn.Module, loader: DataLoader, *, device: torch.device, output_dir: Path) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    steering_abs = 0.0
    ey_abs = 0.0
    ey_w = 0.0
    count = 0
    for images, labels, meta in loader:
        images = images.to(device, non_blocking=True)
        pred = model(images)
        if pred.shape[-1] > 2:
            pred = pred[:, :2]
        pred_np = pred.cpu().numpy()
        labels_np = labels.numpy()
        qualities = meta["eyQuality"].numpy()
        paths = meta["path"]
        statuses = meta["status"]
        sequences = meta["sequence"]
        frames = meta["frame"]
        for i in range(labels_np.shape[0]):
            rows.append(
                {
                    "index": len(rows),
                    "sequence": sequences[i],
                    "frame": int(frames[i]),
                    "path": paths[i],
                    "status": statuses[i],
                    "eyQuality": float(qualities[i]),
                    "steeringGT": float(labels_np[i, 0]),
                    "eyGT": float(labels_np[i, 1]),
                    "steeringPred": float(pred_np[i, 0]),
                    "eyPred": float(pred_np[i, 1]),
                }
            )
            steering_abs += abs(pred_np[i, 0] - labels_np[i, 0])
            ey_abs += abs(pred_np[i, 1] - labels_np[i, 1]) * float(qualities[i])
            ey_w += float(qualities[i])
            count += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = output_dir / "predictions_test.csv"
    with pred_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "count": int(count),
        "steeringMAE": float(steering_abs / max(1, count)),
        "eyWeightedMAE": float(ey_abs / max(1e-6, ey_w)),
        "predictionsCsv": str(pred_csv),
    }
    (output_dir / "summary_test.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def apply_hmm(pred_csv: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(HMM_SCRIPT),
        "--model-json",
        str(HMM_JSON),
        "--input-pred-csv",
        str(pred_csv),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.check_call(cmd)
    summary_path = output_dir / "current_postprocess_summary.json"
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    hmm_mae = float(raw["currentPostSteeringMAE"])
    raw_mae = float(raw["rawSteeringMAE"])
    out = {
        "rawSteeringMAE": raw_mae,
        "hmmSteeringMAE": hmm_mae,
        "baselineHmmMae": BASELINE_HMM_MAE,
        "deltaVsBaseline": hmm_mae - BASELINE_HMM_MAE,
        "sourceSummary": str(summary_path),
    }
    (output_dir / "summary_test_hmm.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def write_verdict(path: Path, *, experiment: str, hmm_mae: float, raw_mae: float) -> str:
    delta = hmm_mae - BASELINE_HMM_MAE
    verdict = "晋级" if hmm_mae < BASELINE_HMM_MAE else "淘汰"
    text = (
        f"数据：train/val=26合并；test=25地下室\n"
        f"实验：{experiment}\n"
        f"RAW MAE = {raw_mae:.6f}\n"
        f"HMM MAE = {hmm_mae:.6f}\n"
        f"相对锁存 Δ = {delta:+.6f} (baseline HMM {BASELINE_HMM_MAE:.6f})\n"
        f"结论：{verdict}\n"
    )
    path.write_text(text, encoding="utf-8")
    print(text)
    return verdict
