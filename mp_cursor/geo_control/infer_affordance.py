#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the P1 affordance model on a split and (optionally) the held-out perturbation stress test.

Produces per-frame prediction CSVs (sequence, frame, steeringPred, eyPred, thetaPred) and, when
a frozen controller + label CSV are given, scores each via the offline harness and writes a
clean-vs-perturbed degradation table.

Run on the GPU/dev environment. Example (clean + stress on basement test):
  python -m mp_cursor.geo_control.infer_affordance \
    --ckpt mp_cursor/exp_P1_affordance/checkpoints/best_affordance_p1.pth \
    --label-csv mp_cursor/geo_control/labels/labels_geo.csv --split test \
    --controller mp_cursor/geo_control/frozen/controller_pid.json \
    --output-dir mp_cursor/exp_P1_affordance/eval --stress
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_DIR = REPO_ROOT / "locked_model"
if str(LOCKED_DIR) not in sys.path:
    sys.path.insert(0, str(LOCKED_DIR))

from steering_preprocess import preprocess_config_from_dict, DEFAULT_PREPROCESS_CONFIG  # type: ignore  # noqa: E402

from . import perturb
from .datasets_geo import AffordanceDataset
from .eval_harness import compare_degradation, score_predictions
from .models_affordance import AffordanceCfCNet


def _device(req: str) -> torch.device:
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    num_frames = int(ckpt.get("numFrames", 3)) if isinstance(ckpt, dict) else 3
    frame_stride = int(ckpt.get("frameStride", 1)) if isinstance(ckpt, dict) else 1
    preprocess = preprocess_config_from_dict(ckpt.get("preprocess") if isinstance(ckpt, dict) else None, fallback=DEFAULT_PREPROCESS_CONFIG)
    model = AffordanceCfCNet(num_frames=num_frames).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, preprocess, num_frames, frame_stride


def _predict(model, loader, device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, _labels, meta in loader:
            images = images.to(device, non_blocking=True)
            pred = model(images).detach().cpu().numpy()
            seqs = meta["sequence"]
            frames = meta["frame"].numpy()
            for i in range(pred.shape[0]):
                rows.append({
                    "sequence": str(seqs[i]),
                    "frame": int(frames[i]),
                    "steeringPred": float(pred[i, 0]),
                    "eyPred": float(pred[i, 1]),
                    "thetaPred": float(pred[i, 2]),
                })
    return rows


def _write_pred_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sequence", "frame", "steeringPred", "eyPred", "thetaPred"])
        writer.writeheader()
        writer.writerows(rows)


def _make_loader(label_csv, split, preprocess, num_frames, frame_stride, *, perturb_fn=None, frame_drop=0.0, batch_size=16, num_workers=0):
    ds = AffordanceDataset(
        label_csv, split=split, preprocess=preprocess, num_frames=num_frames, frame_stride=frame_stride,
        augmentor=None, perturb_fn=perturb_fn, frame_drop_rate=frame_drop, include_zero_quality=True,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P1 affordance inference + stress test.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--label-csv", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--controller", default=None, help="Frozen controller JSON for inline scoring.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stress", action="store_true", help="Also run the held-out perturbation axes.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = _device(args.device)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model, preprocess, num_frames, frame_stride = _load_model(Path(args.ckpt).resolve(), device)
    label_csv = Path(args.label_csv).resolve()
    controller_path = Path(args.controller).resolve() if args.controller else None

    def _score(pred_csv: Path, tag: str) -> Path | None:
        if controller_path is None:
            return None
        metrics = score_predictions(label_csv=label_csv, pred_csv=pred_csv, controller_path=controller_path, split=args.split)
        mpath = out_dir / f"metrics_{tag}.json"
        mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return mpath

    # clean
    clean_rows = _predict(model, _make_loader(label_csv, args.split, preprocess, num_frames, frame_stride,
                                              batch_size=args.batch_size, num_workers=args.num_workers), device)
    clean_csv = out_dir / "predictions_clean.csv"
    _write_pred_csv(clean_csv, clean_rows)
    clean_metrics = _score(clean_csv, "clean")
    print(f"clean predictions -> {clean_csv}")

    if args.stress:
        stress_dir = out_dir / "stress"
        stress_dir.mkdir(parents=True, exist_ok=True)
        stress_pairs: list[tuple[str, Path]] = []
        for axis in perturb.default_axes():
            for level in axis.levels:
                tag = f"{axis.name}_{level.name}"
                if axis.is_frame_drop:
                    loader = _make_loader(label_csv, args.split, preprocess, num_frames, frame_stride,
                                          frame_drop=perturb.frame_drop_rate(level.name),
                                          batch_size=args.batch_size, num_workers=args.num_workers)
                else:
                    fn: Callable[[np.ndarray], np.ndarray] = level.fn
                    loader = _make_loader(label_csv, args.split, preprocess, num_frames, frame_stride,
                                          perturb_fn=fn, batch_size=args.batch_size, num_workers=args.num_workers)
                rows = _predict(model, loader, device)
                pcsv = stress_dir / f"predictions_{tag}.csv"
                _write_pred_csv(pcsv, rows)
                mpath = _score(pcsv, f"stress_{tag}")
                if mpath is not None:
                    stress_pairs.append((tag, mpath))
                print(f"stress {tag} -> {pcsv}")

        if clean_metrics is not None and stress_pairs:
            degradation = compare_degradation(clean_metrics, stress_pairs)
            (out_dir / "stress_degradation.json").write_text(
                json.dumps(degradation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"degradation -> {out_dir / 'stress_degradation.json'}")


if __name__ == "__main__":
    main()
