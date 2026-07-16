#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the P3 segmentation model on a split: mask -> e_y/theta -> predictions CSV (+ stress).

Reuses the offline harness for scoring and clean-vs-perturbed degradation, mirroring P1 so the
two paradigms are compared on identical proxy metrics.

Run on the GPU/dev environment. Example:
  python -m mp_cursor.geo_control.infer_seg \
    --ckpt mp_cursor/exp_P3_seg/checkpoints/best_laneseg_p3.pth \
    --label-csv mp_cursor/geo_control/labels/labels_geo.csv --split test \
    --controller mp_cursor/geo_control/frozen/controller_pid.json \
    --output-dir mp_cursor/exp_P3_seg/eval --stress
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import perturb
from .datasets_seg import INPUT_W, LaneSegDataset
from .eval_harness import compare_degradation, score_predictions
from .models_seg import LaneSegNet
from .seg_geometry import mask_to_affordance


def _device(req: str) -> torch.device:
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _predict(model, loader, device, *, width_prior: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, _masks, meta in loader:
            images = images.to(device, non_blocking=True)
            prob = torch.sigmoid(model(images)).detach().cpu().numpy()[:, 0]  # (B,H,W)
            seqs = meta["sequence"]
            frames = meta["frame"].numpy()
            for i in range(prob.shape[0]):
                aff = mask_to_affordance(prob[i], width_prior=width_prior)
                rows.append({
                    "sequence": str(seqs[i]),
                    "frame": int(frames[i]),
                    "steeringPred": 0.0,
                    "eyPred": aff["e_y"] if aff["detected"] else "",
                    "thetaPred": aff["theta"] if aff["detected"] else "",
                })
    return rows


def _write_pred_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sequence", "frame", "steeringPred", "eyPred", "thetaPred"])
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P3 segmentation inference + stress test.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--label-csv", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--controller", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stress", action="store_true")
    p.add_argument("--lane-width-frac", type=float, default=0.64, help="Frozen source-domain lane width as fraction of image width.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = _device(args.device)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    label_csv = Path(args.label_csv).resolve()
    controller_path = Path(args.controller).resolve() if args.controller else None
    width_prior = float(args.lane_width_frac) * INPUT_W

    ckpt = torch.load(Path(args.ckpt).resolve(), map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = LaneSegNet().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    def _loader(perturb_fn=None):
        ds = LaneSegDataset(label_csv, split=args.split, augmentor=None, perturb_fn=perturb_fn)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    def _score(pred_csv: Path, tag: str) -> Path | None:
        if controller_path is None:
            return None
        metrics = score_predictions(label_csv=label_csv, pred_csv=pred_csv, controller_path=controller_path, split=args.split)
        mpath = out_dir / f"metrics_{tag}.json"
        mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return mpath

    clean_rows = _predict(model, _loader(), device, width_prior=width_prior)
    clean_csv = out_dir / "predictions_clean.csv"
    _write_pred_csv(clean_csv, clean_rows)
    clean_metrics = _score(clean_csv, "clean")
    print(f"clean predictions -> {clean_csv}")

    if args.stress:
        stress_dir = out_dir / "stress"
        stress_dir.mkdir(parents=True, exist_ok=True)
        stress_pairs: list[tuple[str, Path]] = []
        for axis in perturb.default_axes():
            if axis.is_frame_drop:
                continue  # single-frame perception: temporal drop is not applicable
            for level in axis.levels:
                tag = f"{axis.name}_{level.name}"
                fn: Callable[[np.ndarray], np.ndarray] = level.fn
                rows = _predict(model, _loader(perturb_fn=fn), device, width_prior=width_prior)
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
