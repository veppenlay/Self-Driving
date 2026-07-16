#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P2: run the online classical-CV perception on a split; quantify its fragility.

This is NOT a robustness candidate. It serves two purposes from the plan:
1. Diagnostic: measure how brittle the fixed-threshold CV detector is (detection rate on the
   basement, and how fast detection collapses under held-out perturbation).
2. Safety fallback source: the same `CVLanePerception` is the temporal-hold / blind fallback for
   P1/P3 when learned perception is unavailable.

HSV thresholds are inherited from the locked detector and are NOT tuned here. `width_prior` is a
frozen source-domain constant.

Needs cv2. Example:
  python -m mp_cursor.geo_control.diagnose_cv \
    --label-csv mp_cursor/geo_control/labels/labels_geo.csv --split test \
    --width-prior 411.0 --controller mp_cursor/geo_control/frozen/controller_pid.json \
    --output-dir mp_cursor/exp_P2_cv/eval --stress
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_DIR = REPO_ROOT / "locked_model"
if str(LOCKED_DIR) not in sys.path:
    sys.path.insert(0, str(LOCKED_DIR))

from steering_preprocess import imread_bgr  # type: ignore  # noqa: E402

from . import perturb
from .cv_perception import CVLanePerception
from .eval_harness import compare_degradation, score_predictions


def _read_rows(label_csv: Path, split: str) -> list[dict[str, Any]]:
    rows = []
    with label_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        for raw in csv.DictReader(fh):
            if split != "all" and str(raw.get("split", "")).lower() != split.lower():
                continue
            rows.append(raw)
    rows.sort(key=lambda r: (str(r.get("sequence", "")), int(float(r.get("frame") or 0))))
    return rows


def _run(rows: list[dict[str, Any]], *, width_prior: float | None, hold_frames: int, perturb_fn=None) -> tuple[list[dict[str, Any]], dict[str, float]]:
    perception = CVLanePerception(width_prior_px=width_prior, hold_frames=hold_frames)
    out_rows: list[dict[str, Any]] = []
    prev_seq = None
    n = 0
    n_cv = 0
    n_hold = 0
    for row in rows:
        seq = str(row.get("sequence", ""))
        if seq != prev_seq:
            perception.reset()
            prev_seq = seq
        bgr = imread_bgr(Path(row["image"]))
        if bgr is None:
            continue
        if perturb_fn is not None:
            bgr = perturb_fn(bgr)
        res = perception.process(bgr)
        n += 1
        usable = res["source"] in ("cv", "hold")
        if res["source"] == "cv":
            n_cv += 1
        elif res["source"] == "hold":
            n_hold += 1
        out_rows.append({
            "sequence": seq,
            "frame": int(float(row.get("frame") or 0)),
            "eyPred": (round(res["e_y"], 6) if usable else ""),
            "thetaPred": (round(res["theta"], 6) if usable else ""),
            "source": res["source"],
        })
    stats = {
        "count": n,
        "detectionRate": (n_cv / n) if n else 0.0,
        "usableRate": ((n_cv + n_hold) / n) if n else 0.0,
        "holdRate": (n_hold / n) if n else 0.0,
    }
    return out_rows, stats


def _write_pred_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sequence", "frame", "eyPred", "thetaPred", "source"])
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P2 online CV perception diagnostic + fallback source.")
    p.add_argument("--label-csv", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--width-prior", type=float, default=None, help="Frozen source-domain lane width prior (px).")
    p.add_argument("--hold-frames", type=int, default=5)
    p.add_argument("--controller", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    label_csv = Path(args.label_csv).resolve()
    controller_path = Path(args.controller).resolve() if args.controller else None
    rows = _read_rows(label_csv, args.split)
    if not rows:
        raise SystemExit(f"no rows for split={args.split!r}")

    def _score(pred_csv: Path, tag: str) -> Path | None:
        if controller_path is None:
            return None
        metrics = score_predictions(label_csv=label_csv, pred_csv=pred_csv, controller_path=controller_path, split=args.split)
        mpath = out_dir / f"metrics_{tag}.json"
        mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return mpath

    clean_rows, clean_stats = _run(rows, width_prior=args.width_prior, hold_frames=args.hold_frames)
    clean_csv = out_dir / "predictions_clean.csv"
    _write_pred_csv(clean_csv, clean_rows)
    clean_metrics = _score(clean_csv, "clean")
    diagnostics: dict[str, Any] = {"clean": clean_stats}
    print(f"clean CV detectionRate={clean_stats['detectionRate']:.3f} usableRate={clean_stats['usableRate']:.3f}")

    if args.stress:
        stress_dir = out_dir / "stress"
        stress_dir.mkdir(parents=True, exist_ok=True)
        stress_pairs: list[tuple[str, Path]] = []
        stress_stats: dict[str, Any] = {}
        for axis in perturb.default_axes():
            if axis.is_frame_drop:
                continue
            for level in axis.levels:
                tag = f"{axis.name}_{level.name}"
                fn: Callable[[np.ndarray], np.ndarray] = level.fn
                rws, st = _run(rows, width_prior=args.width_prior, hold_frames=args.hold_frames, perturb_fn=fn)
                pcsv = stress_dir / f"predictions_{tag}.csv"
                _write_pred_csv(pcsv, rws)
                stress_stats[tag] = st
                mpath = _score(pcsv, f"stress_{tag}")
                if mpath is not None:
                    stress_pairs.append((tag, mpath))
                print(f"stress {tag} detectionRate={st['detectionRate']:.3f}")
        diagnostics["stress"] = stress_stats
        if clean_metrics is not None and stress_pairs:
            degradation = compare_degradation(clean_metrics, stress_pairs)
            (out_dir / "stress_degradation.json").write_text(
                json.dumps(degradation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    (out_dir / "cv_fragility.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"fragility -> {out_dir / 'cv_fragility.json'}")


if __name__ == "__main__":
    main()
