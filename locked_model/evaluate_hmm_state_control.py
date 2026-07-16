#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate HMM steering-state fusion without test-answer leakage.

Training/fitting inputs:
- train/val labels: learn discrete steering states and transition probabilities.
- train/val predictions: learn p(raw steering prediction | state) and tune HMM parameters.

Test-time inputs:
- sequence boundary, frame order, and raw steering predictions only.
- ground truth is used only after inference for metrics and plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class HmmParams:
    transition_smoothing: float
    sigma_scale: float
    switch_penalty: float
    ey_weight: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HMM/HSMM-style steering state fusion evaluation.")
    parser.add_argument("--label-csv", required=True, help="labels_2d.csv containing train/val labels.")
    parser.add_argument("--train-pred-csv", default=None, help="Optional train prediction CSV for emission fitting.")
    parser.add_argument("--fit-pred-csv", required=True, help="Validation prediction CSV for fitting/tuning.")
    parser.add_argument("--test-pred-csv", required=True, help="Held-out test prediction CSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--state-round-decimals", type=int, default=3)
    parser.add_argument("--min-sigma", type=float, default=0.08)
    parser.add_argument("--objective", choices=["mae", "large_error_weighted"], default="mae")
    parser.add_argument("--ey-weights", default="0,0.1,0.25,0.5,1,2", help="Comma-separated e_y emission weights to tune on validation; use 0 for steering-only.")
    return parser.parse_args()


def _parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("--ey-weights must contain at least one float")
    return values


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty csv: {path}")
    return rows


def _sort_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (str(row["sequence"]), int(float(row["frame"])), int(float(row["index"]))))


def _sort_label_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (str(row["split"]), str(row["sequence"]), int(float(row["frame"])), int(float(row["index"]))))


def _round_state(value: float, decimals: int) -> float:
    rounded = round(float(value), decimals)
    return 0.0 if abs(rounded) < 0.5 * 10 ** (-decimals) else rounded


def _mae(values: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(values - target))) if values.size else float("nan")


def _objective_score(values: np.ndarray, target: np.ndarray, objective: str) -> float:
    err = np.abs(values - target)
    if not err.size:
        return float("nan")
    if objective == "large_error_weighted":
        weights = 1.0 + 3.0 * (err > 0.5) + 5.0 * (err > 1.0)
        return float(np.mean(err * weights))
    return float(np.mean(err))


def _nearest_state_indices(values: np.ndarray, states: np.ndarray) -> np.ndarray:
    return np.argmin(np.abs(values[:, None] - states[None, :]), axis=1)


def _state_values(label_rows: list[dict[str, Any]], decimals: int) -> np.ndarray:
    values = sorted({_round_state(float(row["steering"]), decimals) for row in label_rows if row.get("split") in {"train", "val"}})
    if not values:
        raise ValueError("no train/val states found in label CSV")
    return np.asarray(values, dtype=np.float64)


def _build_transition_model(
    label_rows: list[dict[str, Any]],
    states: np.ndarray,
    *,
    decimals: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    state_to_index = {float(state): idx for idx, state in enumerate(states)}
    count = np.ones((len(states), len(states)), dtype=np.float64) * float(smoothing)
    start = np.ones(len(states), dtype=np.float64) * float(smoothing)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in label_rows:
        if row.get("split") not in {"train", "val"}:
            continue
        grouped.setdefault((str(row["split"]), str(row["sequence"])), []).append(row)

    for rows in grouped.values():
        rows = sorted(rows, key=lambda row: int(float(row["frame"])))
        previous: int | None = None
        for row_index, row in enumerate(rows):
            state = _round_state(float(row["steering"]), decimals)
            index = state_to_index.get(state)
            if index is None:
                index = int(np.argmin(np.abs(states - state)))
            if row_index == 0:
                start[index] += 1.0
            if previous is not None:
                count[previous, index] += 1.0
            previous = index

    transition = count / np.clip(count.sum(axis=1, keepdims=True), 1e-12, None)
    start_prob = start / np.clip(start.sum(), 1e-12, None)
    return transition, start_prob


def _fit_emission_model(
    prediction_rows: list[dict[str, Any]],
    states: np.ndarray,
    *,
    min_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    raw = np.asarray([float(row["steeringPred"]) for row in prediction_rows], dtype=np.float64)
    ey = np.asarray([float(row["eyPred"]) for row in prediction_rows], dtype=np.float64)
    truth = np.asarray([float(row["steeringGT"]) for row in prediction_rows], dtype=np.float64)
    nearest = _nearest_state_indices(truth, states)
    residual = raw - truth
    global_sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual)))) if residual.size else 0.25
    global_sigma = max(float(min_sigma), global_sigma)

    means = states.copy()
    sigmas = np.ones(len(states), dtype=np.float64) * global_sigma
    ey_global_sigma = 1.4826 * float(np.median(np.abs(ey - np.median(ey)))) if ey.size else 0.25
    ey_global_sigma = max(float(min_sigma), ey_global_sigma)
    ey_means = np.zeros(len(states), dtype=np.float64)
    ey_sigmas = np.ones(len(states), dtype=np.float64) * ey_global_sigma
    counts = np.zeros(len(states), dtype=np.int64)
    for idx in range(len(states)):
        values = raw[nearest == idx]
        ey_values = ey[nearest == idx]
        counts[idx] = int(values.size)
        if values.size:
            means[idx] = float(np.median(values))
            centered = values - means[idx]
            sigma = 1.4826 * float(np.median(np.abs(centered))) if values.size > 2 else global_sigma
            sigmas[idx] = max(float(min_sigma), sigma)
        if ey_values.size:
            ey_means[idx] = float(np.median(ey_values))
            ey_centered = ey_values - ey_means[idx]
            ey_sigma = 1.4826 * float(np.median(np.abs(ey_centered))) if ey_values.size > 2 else ey_global_sigma
            ey_sigmas[idx] = max(float(min_sigma), ey_sigma)

    details = {
        "globalSigma": global_sigma,
        "eyGlobalSigma": ey_global_sigma,
        "stateSampleCounts": {f"{state:.6f}": int(count) for state, count in zip(states, counts)},
    }
    return means, sigmas, ey_means, ey_sigmas, details


def _apply_switch_penalty(transition: np.ndarray, penalty: float) -> np.ndarray:
    adjusted = transition.copy()
    if penalty > 0:
        factor = math.exp(-float(penalty))
        off_diag = ~np.eye(adjusted.shape[0], dtype=bool)
        adjusted[off_diag] *= factor
    adjusted = adjusted / np.clip(adjusted.sum(axis=1, keepdims=True), 1e-300, None)
    return adjusted


def _log_emission(
    observations: np.ndarray,
    ey_observations: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    ey_means: np.ndarray,
    ey_sigmas: np.ndarray,
    params: HmmParams,
) -> np.ndarray:
    sigma = np.clip(sigmas * float(params.sigma_scale), 1e-6, None)
    raw_log = -0.5 * ((observations[:, None] - means[None, :]) / sigma[None, :]) ** 2 - np.log(sigma[None, :])
    if params.ey_weight <= 0:
        return raw_log
    ey_sigma = np.clip(ey_sigmas, 1e-6, None)
    ey_log = -0.5 * ((ey_observations[:, None] - ey_means[None, :]) / ey_sigma[None, :]) ** 2 - np.log(ey_sigma[None, :])
    return raw_log + float(params.ey_weight) * ey_log


def _viterbi(
    observations: np.ndarray,
    ey_observations: np.ndarray,
    *,
    states: np.ndarray,
    transition: np.ndarray,
    start_prob: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    ey_means: np.ndarray,
    ey_sigmas: np.ndarray,
    params: HmmParams,
) -> tuple[np.ndarray, np.ndarray]:
    transition = _apply_switch_penalty(transition, params.switch_penalty)
    log_transition = np.log(np.clip(transition, 1e-300, None))
    log_start = np.log(np.clip(start_prob, 1e-300, None))
    emission = _log_emission(observations, ey_observations, means, sigmas, ey_means, ey_sigmas, params)
    length = len(observations)
    score = np.zeros((length, len(states)), dtype=np.float64)
    back = np.zeros((length, len(states)), dtype=np.int64)
    score[0] = log_start + emission[0]
    for t in range(1, length):
        candidate = score[t - 1][:, None] + log_transition
        back[t] = np.argmax(candidate, axis=0)
        score[t] = candidate[back[t], np.arange(len(states))] + emission[t]
    path = np.zeros(length, dtype=np.int64)
    path[-1] = int(np.argmax(score[-1]))
    for t in range(length - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return states[path], path


def _online_filter(
    observations: np.ndarray,
    ey_observations: np.ndarray,
    *,
    states: np.ndarray,
    transition: np.ndarray,
    start_prob: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    ey_means: np.ndarray,
    ey_sigmas: np.ndarray,
    params: HmmParams,
) -> tuple[np.ndarray, np.ndarray]:
    transition = _apply_switch_penalty(transition, params.switch_penalty)
    sigma = np.clip(sigmas * float(params.sigma_scale), 1e-6, None)
    ey_sigma = np.clip(ey_sigmas, 1e-6, None)
    posterior = start_prob.astype(np.float64).copy()
    outputs: list[float] = []
    indices: list[int] = []
    for observation, ey_observation in zip(observations, ey_observations):
        predicted = posterior @ transition
        emission = np.exp(-0.5 * ((observation - means) / sigma) ** 2) / sigma
        if params.ey_weight > 0:
            ey_emission = np.exp(-0.5 * ((ey_observation - ey_means) / ey_sigma) ** 2) / ey_sigma
            emission = emission * np.power(ey_emission, float(params.ey_weight))
        posterior = predicted * emission
        posterior = posterior / np.clip(posterior.sum(), 1e-300, None)
        index = int(np.argmax(posterior))
        outputs.append(float(states[index]))
        indices.append(index)
    return np.asarray(outputs, dtype=np.float64), np.asarray(indices, dtype=np.int64)


def _infer_grouped(
    rows: list[dict[str, Any]],
    *,
    states: np.ndarray,
    transition: np.ndarray,
    start_prob: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    ey_means: np.ndarray,
    ey_sigmas: np.ndarray,
    params: HmmParams,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    outputs = np.zeros(len(rows), dtype=np.float64)
    indices = np.zeros(len(rows), dtype=np.int64)
    by_sequence: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_sequence.setdefault(str(row["sequence"]), []).append(idx)

    for row_indices in by_sequence.values():
        observations = np.asarray([float(rows[idx]["steeringPred"]) for idx in row_indices], dtype=np.float64)
        ey_observations = np.asarray([float(rows[idx]["eyPred"]) for idx in row_indices], dtype=np.float64)
        if method == "viterbi":
            sequence_outputs, sequence_indices = _viterbi(
                observations,
                ey_observations,
                states=states,
                transition=transition,
                start_prob=start_prob,
                means=means,
                sigmas=sigmas,
                ey_means=ey_means,
                ey_sigmas=ey_sigmas,
                params=params,
            )
        elif method == "filter":
            sequence_outputs, sequence_indices = _online_filter(
                observations,
                ey_observations,
                states=states,
                transition=transition,
                start_prob=start_prob,
                means=means,
                sigmas=sigmas,
                ey_means=ey_means,
                ey_sigmas=ey_sigmas,
                params=params,
            )
        else:
            raise ValueError(f"unknown method: {method}")
        outputs[row_indices] = sequence_outputs
        indices[row_indices] = sequence_indices
    return outputs, indices


def _fit_params(
    rows: list[dict[str, Any]],
    *,
    states: np.ndarray,
    transition_models: dict[float, tuple[np.ndarray, np.ndarray]],
    means: np.ndarray,
    sigmas: np.ndarray,
    ey_means: np.ndarray,
    ey_sigmas: np.ndarray,
    method: str,
    objective: str,
    ey_weights: list[float],
) -> tuple[HmmParams, float, list[dict[str, Any]]]:
    truth = np.asarray([float(row["steeringGT"]) for row in rows], dtype=np.float64)
    records: list[dict[str, Any]] = []
    best_score = float("inf")
    best_params: HmmParams | None = None

    for smoothing, (transition, start_prob) in transition_models.items():
        for sigma_scale in [0.8, 1.0, 1.25, 1.5, 2.0, 2.8, 3.6, 5.0]:
            for switch_penalty in [0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0, 12.0]:
                for ey_weight in ey_weights:
                    params = HmmParams(smoothing, sigma_scale, switch_penalty, ey_weight)
                    pred, _ = _infer_grouped(
                        rows,
                        states=states,
                        transition=transition,
                        start_prob=start_prob,
                        means=means,
                        sigmas=sigmas,
                        ey_means=ey_means,
                        ey_sigmas=ey_sigmas,
                        params=params,
                        method=method,
                    )
                    score = _objective_score(pred, truth, objective)
                    mae = _mae(pred, truth)
                    record = {**asdict(params), "method": method, "objectiveScore": score, "mae": mae}
                    records.append(record)
                    if score < best_score:
                        best_score = score
                        best_params = params
    if best_params is None:
        raise RuntimeError(f"failed to fit HMM params for method={method}")
    return best_params, best_score, records


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


def _metrics(rows: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    truth = np.asarray([float(row["steeringGT"]) for row in rows], dtype=np.float64)
    raw = np.asarray([float(row["steeringPred"]) for row in rows], dtype=np.float64)
    pred = np.asarray([float(row[pred_key]) for row in rows], dtype=np.float64)
    raw_err = np.abs(raw - truth)
    pred_err = np.abs(pred - truth)
    return {
        "count": len(rows),
        "rawSteeringMAE": _mae(raw, truth),
        f"{pred_key}MAE": _mae(pred, truth),
        "deltaMAE": _mae(pred, truth) - _mae(raw, truth),
        "rawAbsErrorGt0p5Count": int(np.sum(raw_err > 0.5)),
        f"{pred_key}AbsErrorGt0p5Count": int(np.sum(pred_err > 0.5)),
        "rawAbsErrorGt1p0Count": int(np.sum(raw_err > 1.0)),
        f"{pred_key}AbsErrorGt1p0Count": int(np.sum(pred_err > 1.0)),
    }


def _sequence_metrics(rows: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sequence"]), []).append(row)
    return {sequence: _metrics(seq_rows, pred_key) for sequence, seq_rows in sorted(grouped.items())}


def _add_predictions(
    rows: list[dict[str, Any]],
    *,
    states: np.ndarray,
    transition_models: dict[float, tuple[np.ndarray, np.ndarray]],
    means: np.ndarray,
    sigmas: np.ndarray,
    ey_means: np.ndarray,
    ey_sigmas: np.ndarray,
    viterbi_params: HmmParams,
    filter_params: HmmParams,
) -> list[dict[str, Any]]:
    v_transition, v_start = transition_models[viterbi_params.transition_smoothing]
    f_transition, f_start = transition_models[filter_params.transition_smoothing]
    v_pred, v_state = _infer_grouped(
        rows,
        states=states,
        transition=v_transition,
        start_prob=v_start,
        means=means,
        sigmas=sigmas,
        ey_means=ey_means,
        ey_sigmas=ey_sigmas,
        params=viterbi_params,
        method="viterbi",
    )
    f_pred, f_state = _infer_grouped(
        rows,
        states=states,
        transition=f_transition,
        start_prob=f_start,
        means=means,
        sigmas=sigmas,
        ey_means=ey_means,
        ey_sigmas=ey_sigmas,
        params=filter_params,
        method="filter",
    )

    output: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        truth = float(row["steeringGT"])
        raw = float(row["steeringPred"])
        out = dict(row)
        out.update(
            {
                "hmmViterbiPred": float(v_pred[idx]),
                "hmmViterbiStateIndex": int(v_state[idx]),
                "hmmFilterPred": float(f_pred[idx]),
                "hmmFilterStateIndex": int(f_state[idx]),
                "rawSteeringAbsError": abs(raw - truth),
                "hmmViterbiAbsError": abs(float(v_pred[idx]) - truth),
                "hmmFilterAbsError": abs(float(f_pred[idx]) - truth),
            }
        )
        output.append(out)
    return output


def _plot(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    x = np.arange(len(rows))
    truth = np.asarray([float(row["steeringGT"]) for row in rows], dtype=np.float64)
    raw = np.asarray([float(row["steeringPred"]) for row in rows], dtype=np.float64)
    vit = np.asarray([float(row["hmmViterbiPred"]) for row in rows], dtype=np.float64)
    filt = np.asarray([float(row["hmmFilterPred"]) for row in rows], dtype=np.float64)
    marker_step = max(1, len(rows) // 45)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(x, truth, label="Ground truth steering label", color="#111111", linewidth=2.0, zorder=4)
    axes[0].plot(
        x,
        raw,
        label="Raw model steering prediction",
        color="#0066ff",
        linestyle="--",
        marker="s",
        markersize=2.6,
        markevery=marker_step,
        linewidth=1.35,
        alpha=0.9,
        zorder=2,
    )
    axes[0].plot(
        x,
        vit,
        label="HMM Viterbi state output (offline, no GT)",
        color="#d62728",
        linestyle="-.",
        marker="o",
        markersize=2.4,
        markevery=marker_step,
        linewidth=1.35,
        alpha=0.9,
        zorder=3,
    )
    axes[0].plot(
        x,
        filt,
        label="HMM online filter state output (causal)",
        color="#00a651",
        linestyle=":",
        marker="^",
        markersize=2.5,
        markevery=marker_step,
        linewidth=1.45,
        alpha=0.9,
        zorder=3,
    )
    axes[0].set_ylabel("Steering")
    axes[0].set_title(
        f"{title} | raw={_mae(raw, truth):.6f}, "
        f"Viterbi={_mae(vit, truth):.6f}, online={_mae(filt, truth):.6f}"
    )
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(x, np.abs(raw - truth), label="Raw abs error", color="#0066ff", linewidth=1.25)
    axes[1].plot(x, np.abs(vit - truth), label="HMM Viterbi abs error", color="#d62728", linewidth=1.25)
    axes[1].plot(x, np.abs(filt - truth), label="HMM online abs error", color="#00a651", linewidth=1.25)
    axes[1].axhline(0.5, label="0.5 large-error threshold", color="#888888", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("Abs error")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_all(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    _plot(output_dir / "test_truth_raw_hmm_state_control.png", rows, "25basement HMM state control")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sequence"]), []).append(row)
    for sequence, seq_rows in sorted(grouped.items()):
        _plot(output_dir / f"sequence_{sequence}_truth_raw_hmm_state_control.png", seq_rows, f"25basement sequence {sequence}")


def _write_note(output_dir: Path) -> None:
    text = """# HMM 状态控制图例说明

- 黑色 `Ground truth steering label`：真实转向角，只用于最终评测和画图，不参与测试时 HMM 推断。
- 蓝色 `Raw model steering prediction`：二维网络原始 steering 输出。
- 红色 `HMM Viterbi state output`：整段序列 Viterbi 解码后的离散状态输出；使用未来 raw 观测，但不使用 GT，适合离线分析。
- 绿色 `HMM online filter state output`：在线因果滤波输出，只使用当前和历史 raw 观测，更接近实车部署。
- 下半图：三种输出相对 GT 的绝对误差。

参数学习只使用 train/val 标签和 train/val 预测；测试集 GT 只在推断结束后计算 MAE。
"""
    (output_dir / "PLOT_LEGEND_CN.md").write_text(text, encoding="utf-8")


def _state_model_payload(
    *,
    states: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    ey_means: np.ndarray,
    ey_sigmas: np.ndarray,
    emission_details: dict[str, Any],
    viterbi_params: HmmParams,
    filter_params: HmmParams,
) -> dict[str, Any]:
    return {
        "states": [float(x) for x in states],
        "emissionMeans": [float(x) for x in means],
        "emissionSigmas": [float(x) for x in sigmas],
        "eyEmissionMeans": [float(x) for x in ey_means],
        "eyEmissionSigmas": [float(x) for x in ey_sigmas],
        "emissionDetails": emission_details,
        "viterbiParams": asdict(viterbi_params),
        "onlineFilterParams": asdict(filter_params),
    }


def main() -> None:
    args = _parse_args()
    ey_weights = _parse_float_list(args.ey_weights)
    label_rows = _sort_label_rows(_read_csv(Path(args.label_csv).resolve()))
    fit_rows = _sort_prediction_rows(_read_csv(Path(args.fit_pred_csv).resolve()))
    test_rows = _sort_prediction_rows(_read_csv(Path(args.test_pred_csv).resolve()))
    emission_rows = list(fit_rows)
    if args.train_pred_csv:
        emission_rows = _sort_prediction_rows(_read_csv(Path(args.train_pred_csv).resolve())) + emission_rows

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    states = _state_values(label_rows, args.state_round_decimals)
    means, sigmas, ey_means, ey_sigmas, emission_details = _fit_emission_model(emission_rows, states, min_sigma=args.min_sigma)
    transition_models = {
        smoothing: _build_transition_model(
            label_rows,
            states,
            decimals=args.state_round_decimals,
            smoothing=smoothing,
        )
        for smoothing in [0.001, 0.01, 0.1, 1.0]
    }

    viterbi_params, v_score, v_records = _fit_params(
        fit_rows,
        states=states,
        transition_models=transition_models,
        means=means,
        sigmas=sigmas,
        ey_means=ey_means,
        ey_sigmas=ey_sigmas,
        method="viterbi",
        objective=args.objective,
        ey_weights=ey_weights,
    )
    filter_params, f_score, f_records = _fit_params(
        fit_rows,
        states=states,
        transition_models=transition_models,
        means=means,
        sigmas=sigmas,
        ey_means=ey_means,
        ey_sigmas=ey_sigmas,
        method="filter",
        objective=args.objective,
        ey_weights=ey_weights,
    )

    fit_output_rows = _add_predictions(
        fit_rows,
        states=states,
        transition_models=transition_models,
        means=means,
        sigmas=sigmas,
        ey_means=ey_means,
        ey_sigmas=ey_sigmas,
        viterbi_params=viterbi_params,
        filter_params=filter_params,
    )
    test_output_rows = _add_predictions(
        test_rows,
        states=states,
        transition_models=transition_models,
        means=means,
        sigmas=sigmas,
        ey_means=ey_means,
        ey_sigmas=ey_sigmas,
        viterbi_params=viterbi_params,
        filter_params=filter_params,
    )

    summary = {
        "labelCsv": str(Path(args.label_csv).resolve()),
        "trainPredictionCsv": str(Path(args.train_pred_csv).resolve()) if args.train_pred_csv else None,
        "fitPredictionCsv": str(Path(args.fit_pred_csv).resolve()),
        "testPredictionCsv": str(Path(args.test_pred_csv).resolve()),
        "objective": args.objective,
        "noAnswerLeakage": (
            "HMM states/transitions/emissions are learned from train/val. "
            "During test inference, only sequence boundaries and steeringPred observations are used. "
            "Test ground truth is used only after inference for metrics and plots."
        ),
        "stateCount": int(len(states)),
        "states": [float(x) for x in states],
        "viterbi": {
            "params": asdict(viterbi_params),
            "fitObjectiveScore": v_score,
            "fitMetrics": _metrics(fit_output_rows, "hmmViterbiPred"),
            "testMetrics": _metrics(test_output_rows, "hmmViterbiPred"),
            "testBySequence": _sequence_metrics(test_output_rows, "hmmViterbiPred"),
        },
        "onlineFilter": {
            "params": asdict(filter_params),
            "fitObjectiveScore": f_score,
            "fitMetrics": _metrics(fit_output_rows, "hmmFilterPred"),
            "testMetrics": _metrics(test_output_rows, "hmmFilterPred"),
            "testBySequence": _sequence_metrics(test_output_rows, "hmmFilterPred"),
        },
        "outputs": {
            "fitPredictionsCsv": str(output_dir / "fit_predictions_hmm_state_control.csv"),
            "testPredictionsCsv": str(output_dir / "test_predictions_hmm_state_control.csv"),
            "summaryJson": str(output_dir / "hmm_state_control_summary.json"),
            "plot": str(output_dir / "test_truth_raw_hmm_state_control.png"),
        },
    }

    _write_csv(output_dir / "fit_predictions_hmm_state_control.csv", fit_output_rows)
    _write_csv(output_dir / "test_predictions_hmm_state_control.csv", test_output_rows)
    _write_csv(output_dir / "hmm_param_search.csv", v_records + f_records)
    (output_dir / "hmm_state_model.json").write_text(
        json.dumps(
            _state_model_payload(
                states=states,
                means=means,
                sigmas=sigmas,
                ey_means=ey_means,
                ey_sigmas=ey_sigmas,
                emission_details=emission_details,
                viterbi_params=viterbi_params,
                filter_params=filter_params,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "hmm_state_control_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot_all(output_dir, test_output_rows)
    _write_note(output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
