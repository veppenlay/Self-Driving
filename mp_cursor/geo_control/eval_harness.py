#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline proxy evaluation harness for geometry-decoupled candidates.

The final verdict is on-car closed loop; this harness only *shortlists* candidates
offline by replaying the `25地下室` sequences (open loop) and scoring geometric proxies:

- perception fidelity : predicted `e_y` (and `theta`) vs the CV geometric reference,
                        quality-weighted (frames with quality<=0 are excluded).
- control agreement   : the FROZEN controller is applied to (a) the candidate's
                        perception and (b) the CV reference perception; we report how
                        well the candidate-driven command tracks the reference-driven
                        command (Pearson r + sign-match rate).
- smoothness (jerk)   : mean |Δangular| within a sequence.
- coverage            : fraction of frames with a finite prediction (CV can be blind).

`compare` tabulates clean-vs-perturbed degradation (robustness evidence).

Proxy caveat: the reference is the same CV detector used to make labels, so clean
perception MAE is somewhat circular. The informative signals are coverage, smoothness,
and — above all — the *degradation* under held-out perturbation.

Pure numpy + csv (no torch/cv2), so it runs anywhere.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .controller import BaseController, load_controller


REF_EY_KEYS = ("eY", "e_y", "eyGT")
PRED_EY_KEYS = ("eyPred", "e_y_pred", "eY", "ey")
REF_THETA_KEYS = ("theta", "thetaGT")
PRED_THETA_KEYS = ("thetaPred", "theta_pred", "theta")
ANGULAR_KEYS = ("angular", "cmd", "steeringPred", "steering_cmd")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _get(row: dict[str, Any], keys: tuple[str, ...], default: float | None = None) -> float | None:
    for key in keys:
        val = row.get(key, None)
        if val not in (None, "", "None"):
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return default


def _key(row: dict[str, Any]) -> tuple[str, int]:
    frame = row.get("frame", 0)
    try:
        frame_i = int(float(frame))
    except (TypeError, ValueError):
        frame_i = 0
    return (str(row.get("sequence", "")), frame_i)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / denom) if denom > 1e-12 else float("nan")


def _apply_controller_per_sequence(
    controller: BaseController,
    keys: list[tuple[str, int]],
    e_y: np.ndarray,
    theta: np.ndarray,
    *,
    fps: float,
    speed: float,
) -> np.ndarray:
    dt = 1.0 / max(1e-6, fps)
    out = np.zeros(len(keys), dtype=np.float64)
    prev_seq = None
    for i, (seq, _frame) in enumerate(keys):
        if seq != prev_seq:
            controller.reset()
            prev_seq = seq
        out[i] = controller.step(float(e_y[i]), float(theta[i]), v=speed, dt=dt)
    return out


def _mean_abs_diff_per_sequence(keys: list[tuple[str, int]], values: np.ndarray) -> float:
    diffs: list[float] = []
    prev_seq = None
    prev_val = None
    for (seq, _frame), val in zip(keys, values):
        if seq != prev_seq:
            prev_seq = seq
            prev_val = val
            continue
        if prev_val is not None and math.isfinite(val) and math.isfinite(prev_val):
            diffs.append(abs(val - prev_val))
        prev_val = val
    return float(np.mean(diffs)) if diffs else float("nan")


def score_predictions(
    *,
    label_csv: Path,
    pred_csv: Path,
    controller_path: Path | None,
    split: str = "test",
    fps: float = 30.0,
    speed: float = 0.5,
    departure_tau: float = 0.35,
) -> dict[str, Any]:
    labels = {_key(r): r for r in _read_csv(label_csv) if split in ("all", "") or str(r.get("split", "")).lower() == split.lower()}
    preds = {_key(r): r for r in _read_csv(pred_csv)}
    common = [k for k in labels.keys() if k in preds]
    common.sort(key=lambda k: (k[0], k[1]))
    if not common:
        raise SystemExit(f"no overlapping (sequence,frame) rows between {label_csv} and {pred_csv} for split={split}")

    controller = load_controller(controller_path) if controller_path else None

    ref_ey, ref_th, q_ey, q_th = [], [], [], []
    pr_ey, pr_th, angular, has_pred, has_angular = [], [], [], [], []
    for k in common:
        lrow = labels[k]
        prow = preds[k]
        rey = _get(lrow, REF_EY_KEYS, 0.0)
        rth = _get(lrow, REF_THETA_KEYS, 0.0)
        qey = _get(lrow, ("eyQuality",), 0.0) or 0.0
        qth = _get(lrow, ("thetaQuality",), 1.0 if "theta" in lrow else 0.0) or 0.0
        pey = _get(prow, PRED_EY_KEYS, None)
        pth = _get(prow, PRED_THETA_KEYS, None)
        ang = _get(prow, ANGULAR_KEYS, None)
        ref_ey.append(rey)
        ref_th.append(rth)
        q_ey.append(qey)
        q_th.append(qth)
        has_pred.append(1.0 if pey is not None else 0.0)
        pr_ey.append(pey if pey is not None else 0.0)
        pr_th.append(pth if pth is not None else 0.0)
        has_angular.append(1.0 if ang is not None else 0.0)
        angular.append(ang if ang is not None else float("nan"))

    ref_ey = np.asarray(ref_ey); ref_th = np.asarray(ref_th)
    q_ey = np.asarray(q_ey); q_th = np.asarray(q_th)
    pr_ey = np.asarray(pr_ey); pr_th = np.asarray(pr_th)
    has_pred = np.asarray(has_pred); has_angular = np.asarray(has_angular)
    angular = np.asarray(angular)

    # Angular command: prefer explicit angular; else derive via frozen controller on predicted perception.
    ref_cmd = None
    cand_cmd = angular.copy()
    if controller is not None:
        ref_cmd = _apply_controller_per_sequence(controller, common, ref_ey, ref_th, fps=fps, speed=speed)
        need = ~np.isfinite(cand_cmd)
        if need.any():
            derived = _apply_controller_per_sequence(controller, common, pr_ey, pr_th, fps=fps, speed=speed)
            cand_cmd = np.where(need, derived, cand_cmd)

    def _wmae(pred: np.ndarray, ref: np.ndarray, w: np.ndarray) -> float:
        denom = float(w.sum())
        return float((np.abs(pred - ref) * w).sum() / denom) if denom > 1e-8 else float("nan")

    metrics: dict[str, Any] = {
        "labelCsv": str(label_csv),
        "predCsv": str(pred_csv),
        "controller": str(controller_path) if controller_path else None,
        "split": split,
        "count": len(common),
        "coverage": float(has_pred.mean()) if has_pred.size else float("nan"),
        "perception": {
            "eyWeightedMAE": _wmae(pr_ey, ref_ey, q_ey * has_pred),
            "eyQualitySum": float(q_ey.sum()),
        },
        "referenceLaneDeparture": {
            "tau": departure_tau,
            "fraction": float((np.abs(ref_ey[q_ey > 0]) > departure_tau).mean()) if (q_ey > 0).any() else float("nan"),
            "note": "dataset-level property (recorded trajectory), identical across candidates",
        },
    }

    if float(q_th.sum()) > 0 and has_pred.sum() > 0:
        metrics["perception"]["thetaWeightedMAE"] = _wmae(pr_th, ref_th, q_th * has_pred)

    if ref_cmd is not None:
        valid = (q_ey > 0) & np.isfinite(cand_cmd) & np.isfinite(ref_cmd)
        if valid.any():
            a = cand_cmd[valid]; b = ref_cmd[valid]
            metrics["control"] = {
                "commandSource": "explicit_angular" if has_angular.mean() > 0.5 else "controller_on_prediction",
                "agreementPearson": _pearson(a, b),
                "signMatchRate": float(np.mean(np.sign(a) == np.sign(b))),
                "commandVsRefMAE": float(np.mean(np.abs(a - b))),
                "candidateJerk": _mean_abs_diff_per_sequence(common, cand_cmd),
                "referenceJerk": _mean_abs_diff_per_sequence(common, ref_cmd),
            }
    return metrics


def compare_degradation(clean_json: Path, stress_jsons: list[tuple[str, Path]]) -> dict[str, Any]:
    def _load(p: Path) -> dict[str, Any]:
        return json.loads(Path(p).read_text(encoding="utf-8"))

    clean = _load(clean_json)

    def _pick(m: dict[str, Any]) -> dict[str, float]:
        return {
            "eyWeightedMAE": float(m.get("perception", {}).get("eyWeightedMAE", float("nan"))),
            "coverage": float(m.get("coverage", float("nan"))),
            "candidateJerk": float(m.get("control", {}).get("candidateJerk", float("nan"))),
            "agreementPearson": float(m.get("control", {}).get("agreementPearson", float("nan"))),
        }

    clean_pick = _pick(clean)
    rows = []
    for label, path in stress_jsons:
        m = _pick(_load(path))
        rows.append({
            "perturbation": label,
            "clean": clean_pick,
            "perturbed": m,
            "delta": {k: (m[k] - clean_pick[k]) for k in clean_pick},
        })
    # worst-case degradation summary (higher eyMAE / lower coverage = worse)
    ey_deltas = [r["delta"]["eyWeightedMAE"] for r in rows if math.isfinite(r["delta"]["eyWeightedMAE"])]
    cov_deltas = [r["delta"]["coverage"] for r in rows if math.isfinite(r["delta"]["coverage"])]
    return {
        "cleanMetricsJson": str(clean_json),
        "clean": clean_pick,
        "perturbations": rows,
        "worst": {
            "maxEyMAEIncrease": (max(ey_deltas) if ey_deltas else float("nan")),
            "maxCoverageDrop": (min(cov_deltas) if cov_deltas else float("nan")),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline proxy harness for geometry-decoupled candidates.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="Score one candidate prediction CSV.")
    s.add_argument("--label-csv", required=True)
    s.add_argument("--pred-csv", required=True)
    s.add_argument("--controller", default=None, help="Frozen controller JSON (optional).")
    s.add_argument("--split", default="test")
    s.add_argument("--fps", type=float, default=30.0)
    s.add_argument("--speed", type=float, default=0.5)
    s.add_argument("--departure-tau", type=float, default=0.35)
    s.add_argument("--output-json", required=True)

    c = sub.add_parser("compare", help="Tabulate clean-vs-perturbed degradation.")
    c.add_argument("--clean-json", required=True)
    c.add_argument("--stress-glob", required=True, help="Glob for perturbed metric JSONs; label = filename stem.")
    c.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cmd == "score":
        metrics = score_predictions(
            label_csv=Path(args.label_csv).resolve(),
            pred_csv=Path(args.pred_csv).resolve(),
            controller_path=Path(args.controller).resolve() if args.controller else None,
            split=args.split,
            fps=args.fps,
            speed=args.speed,
            departure_tau=args.departure_tau,
        )
        out = Path(args.output_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    elif args.cmd == "compare":
        stress = [(Path(p).stem, Path(p)) for p in sorted(glob.glob(args.stress_glob))]
        result = compare_degradation(Path(args.clean_json).resolve(), stress)
        out = Path(args.output_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
