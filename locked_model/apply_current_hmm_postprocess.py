#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Current deployment post-process: steering-only HMM online filter.

This script intentionally ignores e_y. It fits/applies a causal HMM filter from
steering labels and raw steering predictions only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from evaluate_hmm_state_control import (  # noqa: E402
    HmmParams,
    _build_transition_model,
    _fit_emission_model,
    _online_filter,
    _read_csv,
    _sort_label_rows,
    _sort_prediction_rows,
    _state_values,
)


SCHEME_NAME = "steering_only_hmm_online_filter"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the current steering-only HMM online post-process.")
    parser.add_argument("--input-pred-csv", required=True, help="Prediction CSV containing sequence/frame/steeringPred.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-json", default=None, help="Existing HMM model JSON. If omitted, fit from label/train/fit CSVs.")
    parser.add_argument("--label-csv", default=None, help="labels_2d.csv for fitting states/transitions.")
    parser.add_argument("--train-pred-csv", default=None, help="Train prediction CSV for emission fitting.")
    parser.add_argument("--fit-pred-csv", default=None, help="Validation prediction CSV for emission fitting.")
    parser.add_argument("--state-model-out", default=None)
    parser.add_argument("--state-round-decimals", type=int, default=3)
    parser.add_argument("--min-sigma", type=float, default=0.08)
    parser.add_argument("--transition-smoothing", type=float, default=0.01)
    parser.add_argument("--sigma-scale", type=float, default=5.0)
    parser.add_argument("--switch-penalty", type=float, default=0.5)
    parser.add_argument("--clip-min", type=float, default=-2.5)
    parser.add_argument("--clip-max", type=float, default=2.5)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ensure_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        out = dict(row)
        out.setdefault("index", index)
        out.setdefault("sequence", ".")
        out.setdefault("frame", index)
        out.setdefault("eyPred", 0.0)
        if "steeringPred" not in out:
            raise ValueError("input CSV must contain steeringPred")
        output.append(out)
    return output


def _mae(rows: list[dict[str, Any]], pred_key: str) -> float | None:
    if not rows or "steeringGT" not in rows[0]:
        return None
    truth = np.asarray([float(row["steeringGT"]) for row in rows], dtype=np.float64)
    pred = np.asarray([float(row[pred_key]) for row in rows], dtype=np.float64)
    return float(np.mean(np.abs(pred - truth)))


def _large_error_counts(rows: list[dict[str, Any]], pred_key: str) -> dict[str, int] | None:
    if not rows or "steeringGT" not in rows[0]:
        return None
    truth = np.asarray([float(row["steeringGT"]) for row in rows], dtype=np.float64)
    pred = np.asarray([float(row[pred_key]) for row in rows], dtype=np.float64)
    err = np.abs(pred - truth)
    return {
        "absErrorGt0p5": int(np.sum(err > 0.5)),
        "absErrorGt1p0": int(np.sum(err > 1.0)),
        "absErrorGt2p0": int(np.sum(err > 2.0)),
    }


def _fit_model(args: argparse.Namespace) -> dict[str, Any]:
    if not args.label_csv or not args.fit_pred_csv:
        raise ValueError("fitting requires --label-csv and --fit-pred-csv")
    label_rows = _sort_label_rows(_read_csv(Path(args.label_csv).resolve()))
    fit_rows = _sort_prediction_rows(_ensure_columns(_read_csv(Path(args.fit_pred_csv).resolve())))
    emission_rows = list(fit_rows)
    if args.train_pred_csv:
        emission_rows = _sort_prediction_rows(_ensure_columns(_read_csv(Path(args.train_pred_csv).resolve()))) + emission_rows

    states = _state_values(label_rows, int(args.state_round_decimals))
    means, sigmas, _ey_means, _ey_sigmas, emission_details = _fit_emission_model(
        emission_rows,
        states,
        min_sigma=float(args.min_sigma),
    )
    transition, start_prob = _build_transition_model(
        label_rows,
        states,
        decimals=int(args.state_round_decimals),
        smoothing=float(args.transition_smoothing),
    )
    return {
        "scheme": SCHEME_NAME,
        "usesEy": False,
        "description": "Causal online HMM filter using only raw steering predictions and train/val steering labels.",
        "labelCsv": str(Path(args.label_csv).resolve()),
        "trainPredCsv": str(Path(args.train_pred_csv).resolve()) if args.train_pred_csv else None,
        "fitPredCsv": str(Path(args.fit_pred_csv).resolve()),
        "stateRoundDecimals": int(args.state_round_decimals),
        "minSigma": float(args.min_sigma),
        "params": {
            "transition_smoothing": float(args.transition_smoothing),
            "sigma_scale": float(args.sigma_scale),
            "switch_penalty": float(args.switch_penalty),
            "ey_weight": 0.0,
        },
        "states": [float(x) for x in states],
        "transition": transition.tolist(),
        "startProb": start_prob.tolist(),
        "emissionMeans": [float(x) for x in means],
        "emissionSigmas": [float(x) for x in sigmas],
        "emissionDetails": emission_details,
    }


def _load_model(path: Path) -> dict[str, Any]:
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("scheme") != SCHEME_NAME:
        raise ValueError(f"unexpected postprocess scheme: {model.get('scheme')}")
    if bool(model.get("usesEy")):
        raise ValueError("current postprocess model must not use e_y")
    params = model.get("params", {})
    if abs(float(params.get("ey_weight", 0.0))) > 1e-12:
        raise ValueError("current postprocess requires ey_weight=0")
    return model


def _apply_model(rows: list[dict[str, Any]], model: dict[str, Any], *, clip_min: float, clip_max: float) -> list[dict[str, Any]]:
    rows = _sort_prediction_rows(_ensure_columns(rows))
    states = np.asarray(model["states"], dtype=np.float64)
    transition = np.asarray(model["transition"], dtype=np.float64)
    start_prob = np.asarray(model["startProb"], dtype=np.float64)
    means = np.asarray(model["emissionMeans"], dtype=np.float64)
    sigmas = np.asarray(model["emissionSigmas"], dtype=np.float64)
    zeros = np.zeros_like(states)
    ones = np.ones_like(states)
    raw_params = model["params"]
    params = HmmParams(
        transition_smoothing=float(raw_params["transition_smoothing"]),
        sigma_scale=float(raw_params["sigma_scale"]),
        switch_penalty=float(raw_params["switch_penalty"]),
        ey_weight=0.0,
    )

    output: list[dict[str, Any]] = []
    grouped: dict[str, list[int]] = {}
    for row_index, row in enumerate(rows):
        grouped.setdefault(str(row["sequence"]), []).append(row_index)

    pred_values = np.zeros(len(rows), dtype=np.float64)
    state_indices = np.zeros(len(rows), dtype=np.int64)
    for row_indices in grouped.values():
        observations = np.asarray([float(rows[idx]["steeringPred"]) for idx in row_indices], dtype=np.float64)
        ey_observations = np.zeros(len(row_indices), dtype=np.float64)
        seq_pred, seq_state = _online_filter(
            observations,
            ey_observations,
            states=states,
            transition=transition,
            start_prob=start_prob,
            means=means,
            sigmas=sigmas,
            ey_means=zeros,
            ey_sigmas=ones,
            params=params,
        )
        pred_values[row_indices] = np.clip(seq_pred, float(clip_min), float(clip_max))
        state_indices[row_indices] = seq_state

    for idx, row in enumerate(rows):
        raw = float(row["steeringPred"])
        pred = float(pred_values[idx])
        out = dict(row)
        out["currentPostprocess"] = SCHEME_NAME
        out["currentPostSteeringPred"] = pred
        out["currentPostStateIndex"] = int(state_indices[idx])
        if "steeringGT" in out:
            truth = float(out["steeringGT"])
            out["rawSteeringAbsError"] = abs(raw - truth)
            out["currentPostSteeringAbsError"] = abs(pred - truth)
        output.append(out)
    return output


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_json:
        model_path = Path(args.model_json).resolve()
        model = _load_model(model_path)
    else:
        model = _fit_model(args)
        model_path = Path(args.state_model_out).resolve() if args.state_model_out else output_dir / "current_steering_only_hmm_online_model.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    input_rows = _read_csv(Path(args.input_pred_csv).resolve())
    output_rows = _apply_model(input_rows, model, clip_min=args.clip_min, clip_max=args.clip_max)
    output_csv = output_dir / "current_postprocess_predictions.csv"
    _write_csv(output_csv, output_rows)

    raw_mae = _mae(output_rows, "steeringPred")
    post_mae = _mae(output_rows, "currentPostSteeringPred")
    summary = {
        "scheme": SCHEME_NAME,
        "usesEy": False,
        "inputPredCsv": str(Path(args.input_pred_csv).resolve()),
        "stateModelJson": str(model_path),
        "outputCsv": str(output_csv),
        "params": model["params"],
        "count": len(output_rows),
        "rawSteeringMAE": raw_mae,
        "currentPostSteeringMAE": post_mae,
        "deltaMAE": (post_mae - raw_mae) if raw_mae is not None and post_mae is not None else None,
        "rawLargeErrorCounts": _large_error_counts(output_rows, "steeringPred"),
        "currentPostLargeErrorCounts": _large_error_counts(output_rows, "currentPostSteeringPred"),
    }
    summary_path = output_dir / "current_postprocess_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
