#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calibrate + freeze lateral controller gains on the SOURCE domain only.

We fit the controller so that `controller(e_y, theta)` best reproduces the recorded
human `steering` on `dataset/26合并` (split=train). This is a legitimate source-domain
calibration; the resulting gains are then frozen and used read-only on the basement
test / on-car, per the exploration discipline. No target-domain data is touched here.

Input CSV = a labels CSV with at least: split, sequence, frame, steering, eY, eyQuality.
Optional heading columns: theta, thetaQuality (from `generate_geo_labels.py`).

Usage:
  python -m mp_cursor.geo_control.calibrate_controller \
    --label-csv locked_model/labels/current/labels_2d.csv \
    --output-dir mp_cursor/geo_control/frozen \
    --nominal-speed 0.5 --fps 30
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .controller import (
    ControllerLimits,
    PIDLateralController,
    PurePursuitController,
    StanleyController,
    save_controller,
)


def _read_rows(label_csv: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with label_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        for raw in csv.DictReader(fh):
            if split != "all" and str(raw.get("split", "")).lower() != split.lower():
                continue
            rows.append(raw)
    return rows


def _get_float(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        val = row.get(key, None)
        if val not in (None, "", "None"):
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return float(default)


def _ordered_by_sequence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]):
        return (str(row.get("sequence", "")), int(_get_float(row, "frame")))

    return sorted(rows, key=key)


def _build_features(rows: list[dict[str, Any]], *, fps: float) -> dict[str, np.ndarray]:
    rows = _ordered_by_sequence(rows)
    dt = 1.0 / max(1e-6, fps)
    e_y, d_ey, theta, steering, w_ey, w_th, has_theta = [], [], [], [], [], [], []
    prev_seq = None
    prev_ey = None
    for row in rows:
        seq = str(row.get("sequence", ""))
        ey = _get_float(row, "eY", "e_y")
        th = _get_float(row, "theta")
        st = _get_float(row, "steering")
        q_ey = _get_float(row, "eyQuality", default=0.0)
        q_th = _get_float(row, "thetaQuality", default=(1.0 if "theta" in row else 0.0))
        if seq != prev_seq or prev_ey is None:
            d = 0.0
        else:
            d = (ey - prev_ey) / dt
        e_y.append(ey)
        d_ey.append(d)
        theta.append(th)
        steering.append(st)
        w_ey.append(q_ey)
        w_th.append(q_th)
        has_theta.append(1.0 if ("theta" in row and row.get("theta") not in (None, "", "None")) else 0.0)
        prev_seq = seq
        prev_ey = ey
    return {
        "e_y": np.asarray(e_y, dtype=np.float64),
        "d_ey": np.asarray(d_ey, dtype=np.float64),
        "theta": np.asarray(theta, dtype=np.float64),
        "steering": np.asarray(steering, dtype=np.float64),
        "w_ey": np.asarray(w_ey, dtype=np.float64),
        "w_th": np.asarray(w_th, dtype=np.float64),
        "has_theta": np.asarray(has_theta, dtype=np.float64),
    }


def _weighted_lstsq(design: np.ndarray, target: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, float]:
    w = np.clip(weight, 0.0, None)
    sw = np.sqrt(w)[:, None]
    a = design * sw
    b = target * sw[:, 0]
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    pred = design @ coef
    err = np.abs(pred - target)
    denom = float(w.sum())
    wmae = float((err * w).sum() / denom) if denom > 1e-8 else float("nan")
    return coef, wmae


def _fit_pid(feat: dict[str, np.ndarray], *, use_theta: bool) -> tuple[dict[str, float], float]:
    cols = [feat["e_y"], feat["d_ey"]]
    names = ["kp", "kd"]
    weight = feat["w_ey"].copy()
    if use_theta:
        cols.append(feat["theta"])
        names.append("ktheta")
        weight = np.minimum(weight, np.where(feat["has_theta"] > 0, feat["w_th"], weight))
    cols.append(np.ones_like(feat["e_y"]))
    names.append("bias")
    design = np.stack(cols, axis=1)
    coef, wmae = _weighted_lstsq(design, feat["steering"], weight)
    gains = {name: float(value) for name, value in zip(names, coef)}
    gains.setdefault("ktheta", 0.0)
    gains.setdefault("ki", 0.0)
    return gains, wmae


def _fit_stanley(feat: dict[str, np.ndarray], *, nominal_speed: float, soft: float, use_theta: bool) -> tuple[dict[str, float], float]:
    cross = np.arctan2(feat["e_y"], nominal_speed + soft)  # k folded into scale below
    cols = [cross]
    names = ["scale"]
    weight = feat["w_ey"].copy()
    if use_theta:
        cols.append(feat["theta"])
        names.append("scale_theta")
        weight = np.minimum(weight, np.where(feat["has_theta"] > 0, feat["w_th"], weight))
    cols.append(np.ones_like(feat["e_y"]))
    names.append("bias")
    design = np.stack(cols, axis=1)
    coef, wmae = _weighted_lstsq(design, feat["steering"], weight)
    fitted = {name: float(value) for name, value in zip(names, coef)}
    scale = fitted.get("scale", 1.0)
    gains = {
        "k": 1.0,
        "soft": float(soft),
        "scale": scale,
        "ktheta": (fitted.get("scale_theta", 0.0) / scale) if abs(scale) > 1e-9 else 0.0,
        "bias": fitted.get("bias", 0.0),
    }
    return gains, wmae


def _fit_pure_pursuit(feat: dict[str, np.ndarray], *, lookahead: float, use_theta: bool) -> tuple[dict[str, float], float]:
    curvature = 2.0 * feat["e_y"] / (lookahead * lookahead)
    cols = [curvature]
    names = ["scale"]
    weight = feat["w_ey"].copy()
    if use_theta:
        cols.append(feat["theta"])
        names.append("ktheta")
        weight = np.minimum(weight, np.where(feat["has_theta"] > 0, feat["w_th"], weight))
    cols.append(np.ones_like(feat["e_y"]))
    names.append("bias")
    design = np.stack(cols, axis=1)
    coef, wmae = _weighted_lstsq(design, feat["steering"], weight)
    fitted = {name: float(value) for name, value in zip(names, coef)}
    gains = {
        "lookahead": float(lookahead),
        "scale": fitted.get("scale", 1.0),
        "ktheta": fitted.get("ktheta", 0.0),
        "bias": fitted.get("bias", 0.0),
    }
    return gains, wmae


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate + freeze lateral controllers on the source domain.")
    parser.add_argument("--label-csv", required=True, help="Labels CSV (uses split=train as source domain).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--nominal-speed", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stanley-soft", type=float, default=0.2)
    parser.add_argument("--pursuit-lookahead", type=float, default=1.0)
    parser.add_argument("--max-abs", type=float, default=3.0)
    parser.add_argument("--slew-rate", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    label_csv = Path(args.label_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(label_csv, args.source_split)
    if not rows:
        raise SystemExit(f"no rows for split={args.source_split!r} in {label_csv}")
    feat = _build_features(rows, fps=args.fps)
    use_theta = bool(feat["has_theta"].max() > 0)

    limits = ControllerLimits(max_abs=float(args.max_abs), slew_rate=float(args.slew_rate))

    pid_gains, pid_mae = _fit_pid(feat, use_theta=use_theta)
    stanley_gains, stanley_mae = _fit_stanley(
        feat, nominal_speed=args.nominal_speed, soft=args.stanley_soft, use_theta=use_theta
    )
    pp_gains, pp_mae = _fit_pure_pursuit(feat, lookahead=args.pursuit_lookahead, use_theta=use_theta)

    meta_common = {
        "sourceLabelCsv": str(label_csv),
        "sourceSplit": args.source_split,
        "sampleCount": int(len(rows)),
        "usesTheta": use_theta,
        "nominalSpeed": float(args.nominal_speed),
        "fps": float(args.fps),
        "note": "Frozen on source domain; read-only at target/on-car. sourceImitationMAE is informational only.",
    }

    controllers = {
        "pid": (PIDLateralController(limits=limits, **pid_gains), pid_mae),
        "stanley": (StanleyController(limits=limits, **stanley_gains), stanley_mae),
        "pure_pursuit": (PurePursuitController(limits=limits, **pp_gains), pp_mae),
    }

    summary: dict[str, Any] = {"controllers": {}, "meta": meta_common}
    for name, (controller, mae) in controllers.items():
        out_path = output_dir / f"controller_{name}.json"
        save_controller(controller, out_path, meta={**meta_common, "sourceImitationMAE": mae})
        summary["controllers"][name] = {
            "path": str(out_path),
            "sourceImitationMAE": mae,
            "gains": controller.to_dict()["gains"],
        }

    best = min(summary["controllers"].items(), key=lambda kv: (kv[1]["sourceImitationMAE"] if kv[1]["sourceImitationMAE"] == kv[1]["sourceImitationMAE"] else float("inf")))
    summary["recommended"] = best[0]
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
