#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Numpy-only smoke test for the geometry-control core (no torch/cv2 needed).

Run from repo root:  python -m mp_cursor.geo_control._smoke
"""

from __future__ import annotations

import csv
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from .calibrate_controller import main as calibrate_main
from .controller import PIDLateralController, load_controller
from .eval_harness import compare_degradation, score_predictions
from .seg_geometry import mask_to_affordance
from .closed_loop_score import _events_metrics, _telemetry_metrics
from . import perturb


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _make_labels(path: Path, *, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    true_kp, true_kth, true_bias = 1.7, 0.6, 0.02
    rows = []
    for split, n_seq in (("train", 6), ("test", 2)):
        for s in range(n_seq):
            ey_prev = 0.0
            for f in range(40):
                ey = float(np.clip(0.85 * ey_prev + rng.normal(0, 0.15), -0.9, 0.9))
                th = float(np.clip(0.5 * ey + rng.normal(0, 0.05), -0.6, 0.6))
                steer = true_kp * ey + true_kth * th + true_bias + rng.normal(0, 0.01)
                rows.append({
                    "split": split, "sequence": f"{split}{s}", "frame": f,
                    "steering": round(steer, 6), "eY": round(ey, 6), "eyQuality": 1.0,
                    "theta": round(th, 6), "thetaQuality": 1.0,
                    "status": "ok",
                })
                ey_prev = ey
    _write_csv(path, rows, ["split", "sequence", "frame", "steering", "eY", "eyQuality", "theta", "thetaQuality", "status"])
    return true_kp, true_kth


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="geo_smoke_"))
    print(f"[smoke] workdir={tmp}")

    # 1) controller basic behavior + json round-trip
    ctrl = PIDLateralController(kp=1.5, kd=0.1, ktheta=0.5, bias=0.0)
    out_right = ctrl.step(0.4, 0.1)
    ctrl.reset()
    out_left = ctrl.step(-0.4, -0.1)
    assert out_right > 0 > out_left, (out_right, out_left)
    d = ctrl.to_dict()
    assert load_controller_roundtrip(d) == "pid"
    print(f"[smoke] controller ok: right={out_right:.3f} left={out_left:.3f}")

    # 2) calibration recovers approximately true gains
    label_csv = tmp / "labels.csv"
    true_kp, true_kth = _make_labels(label_csv)
    frozen_dir = tmp / "frozen"
    import sys
    argv_bak = sys.argv[:]
    sys.argv = ["calibrate", "--label-csv", str(label_csv), "--output-dir", str(frozen_dir), "--fps", "30"]
    try:
        calibrate_main()
    finally:
        sys.argv = argv_bak
    pid = load_controller(frozen_dir / "controller_pid.json")
    assert abs(pid.g.kp - true_kp) < 0.25, pid.g.kp
    assert abs(pid.g.ktheta - true_kth) < 0.35, pid.g.ktheta
    print(f"[smoke] calibration ok: fit kp={pid.g.kp:.3f} (true {true_kp}) ktheta={pid.g.ktheta:.3f} (true {true_kth})")

    # 3) harness scoring: candidate = reference eY + small error
    labels = list(csv.DictReader(open(label_csv, encoding="utf-8-sig")))
    pred_rows = []
    rng = np.random.default_rng(1)
    for r in labels:
        if r["split"] != "test":
            continue
        pred_rows.append({
            "sequence": r["sequence"], "frame": r["frame"],
            "eyPred": round(float(r["eY"]) + float(rng.normal(0, 0.03)), 6),
            "thetaPred": round(float(r["theta"]) + float(rng.normal(0, 0.02)), 6),
        })
    pred_csv = tmp / "pred_clean.csv"
    _write_csv(pred_csv, pred_rows, ["sequence", "frame", "eyPred", "thetaPred"])
    clean = score_predictions(
        label_csv=label_csv, pred_csv=pred_csv,
        controller_path=frozen_dir / "controller_pid.json", split="test",
    )
    assert clean["coverage"] == 1.0
    assert clean["perception"]["eyWeightedMAE"] < 0.1
    assert clean["control"]["signMatchRate"] > 0.8
    print(f"[smoke] harness score ok: eyMAE={clean['perception']['eyWeightedMAE']:.4f} "
          f"signMatch={clean['control']['signMatchRate']:.3f} jerk={clean['control']['candidateJerk']:.4f}")

    # 4) perturbation functions produce valid images + degrade a noisy candidate
    img = (rng.integers(0, 255, size=(48, 64, 3))).astype(np.uint8)
    for axis in perturb.default_axes():
        if axis.is_frame_drop:
            continue
        for lvl in axis.levels:
            out = lvl.fn(img)
            assert out.shape == img.shape and out.dtype == np.uint8
    print(f"[smoke] perturbations ok: {sum(len(a.levels) for a in perturb.default_axes())} levels")

    # 5) compare_degradation
    clean_json = tmp / "clean.json"
    clean_json.write_text(json.dumps(clean), encoding="utf-8")
    worse = dict(clean)
    worse = json.loads(json.dumps(clean))
    worse["perception"]["eyWeightedMAE"] += 0.2
    worse["coverage"] = 0.7
    stress_json = tmp / "stress_blur.json"
    stress_json.write_text(json.dumps(worse), encoding="utf-8")
    cmp = compare_degradation(clean_json, [("blur", stress_json)])
    assert abs(cmp["worst"]["maxEyMAEIncrease"] - 0.2) < 1e-6
    assert abs(cmp["worst"]["maxCoverageDrop"] - (-0.3)) < 1e-6
    print(f"[smoke] compare ok: worstEyIncrease={cmp['worst']['maxEyMAEIncrease']:.3f} "
          f"worstCovDrop={cmp['worst']['maxCoverageDrop']:.3f}")

    # 6) seg_geometry: synthetic mask with two vertical lane stripes -> e_y near 0
    mask = np.zeros((144, 192), dtype=np.float32)
    mask[:, 40:48] = 1.0   # left stripe
    mask[:, 144:152] = 1.0  # right stripe (center ~96 = image center)
    aff = mask_to_affordance(mask, width_prior=0.64 * 192)
    assert aff["detected"] and abs(aff["e_y"]) < 0.15, aff
    # shift both stripes right -> e_y should become positive (lane center to the right)
    mask_r = np.zeros((144, 192), dtype=np.float32)
    mask_r[:, 80:88] = 1.0
    mask_r[:, 170:178] = 1.0
    aff_r = mask_to_affordance(mask_r, width_prior=0.64 * 192)
    assert aff_r["detected"] and aff_r["e_y"] > aff["e_y"], (aff, aff_r)
    print(f"[smoke] seg_geometry ok: centered e_y={aff['e_y']:.3f} shifted e_y={aff_r['e_y']:.3f}")

    # 7) closed_loop_score building blocks
    tel = tmp / "telemetry.csv"
    _write_csv(tel, [
        {"recv_time": i * 0.033, "seq": i, "stamp": i * 0.033, "raw": 0.0, "hmm": 0.0,
         "steering": 0.1 * np.sin(i / 5.0), "angular": 0.1 * np.sin(i / 5.0), "e_y": 0.0,
         "theta": 0.0, "control_source": "geometry"}
        for i in range(300)
    ], ["recv_time", "seq", "stamp", "raw", "hmm", "steering", "angular", "e_y", "theta", "control_source"])
    tm = _telemetry_metrics(tel)
    assert tm["count"] == 300 and math.isfinite(tm["meanJerk"])
    ev_path = tmp / "events.json"
    ev_path.write_text(json.dumps({"laps_completed": 3, "out_of_bounds": 1, "interventions": 0, "duration_s": 120}), encoding="utf-8")
    em = _events_metrics(ev_path)
    assert abs(em["oobPerLap"] - (1 / 3)) < 1e-6
    print(f"[smoke] closed_loop_score ok: meanJerk={tm['meanJerk']:.4f} oobPerLap={em['oobPerLap']:.3f}")

    print("[smoke] ALL PASSED")


def load_controller_roundtrip(d: dict) -> str:
    from .controller import BaseController
    return BaseController.from_dict(d).kind


if __name__ == "__main__":
    main()
