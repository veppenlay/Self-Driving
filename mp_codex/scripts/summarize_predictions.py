#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metrics(rows: list[dict[str, str]], prediction_key: str) -> dict[str, float | int]:
    truth = np.asarray([float(row["steeringGT"]) for row in rows], dtype=np.float64)
    prediction = np.asarray([float(row[prediction_key]) for row in rows], dtype=np.float64)
    error = np.abs(prediction - truth)
    return {
        "count": int(error.size),
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "p50": float(np.percentile(error, 50)),
        "p90": float(np.percentile(error, 90)),
        "p95": float(np.percentile(error, 95)),
        "p99": float(np.percentile(error, 99)),
        "max": float(error.max()),
        "absErrorGt0p5": int(np.sum(error > 0.5)),
        "absErrorGt1p0": int(np.sum(error > 1.0)),
        "absErrorGt2p0": int(np.sum(error > 2.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RAW and fixed-HMM steering predictions.")
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--hmm-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    raw_rows = read_rows(Path(args.raw_csv).resolve())
    hmm_rows = read_rows(Path(args.hmm_csv).resolve())
    payload = {
        "experimentId": args.experiment_id,
        "testSet": "dataset/25地下室",
        "raw": metrics(raw_rows, "steeringPred"),
        "fixedHmm": metrics(hmm_rows, "currentPostSteeringPred"),
        "lockedBaseline": {
            "rawMae": 0.20461061395469163,
            "hmmMae": 0.1672482219835827,
            "rawAbsErrorGt1p0": 26,
            "hmmAbsErrorGt1p0": 28,
            "hmmAbsErrorGt2p0": 6,
        },
    }
    payload["deltaVsLocked"] = {
        "rawMaePct": 100.0 * (payload["raw"]["mae"] / payload["lockedBaseline"]["rawMae"] - 1.0),
        "hmmMaePct": 100.0 * (payload["fixedHmm"]["mae"] / payload["lockedBaseline"]["hmmMae"] - 1.0),
        "hmmAbsErrorGt1p0": payload["fixedHmm"]["absErrorGt1p0"] - payload["lockedBaseline"]["hmmAbsErrorGt1p0"],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
