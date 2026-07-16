#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Score an on-car closed-loop run and compare a candidate against the locked baseline.

Final verdict of the exploration. Combines:
- Telemetry CSV (from `epaicar_deploy/record_telemetry.py`): command smoothness / saturation /
  effective rate — computed automatically from the angular stream.
- Events JSON (manually tallied during the run): laps completed, out-of-bounds count, human
  interventions, duration. This is what actually decides success on the car.

Discipline reminder: the candidate must run with SOURCE-DOMAIN-FROZEN params (no basement
retuning). This script only reports; it never tunes anything.

numpy-only. Example:
  python -m mp_cursor.geo_control.closed_loop_score \
    --candidate-telemetry run_P1.csv --candidate-events events_P1.json \
    --baseline-events events_baseline.json \
    --output-json mp_cursor/exp_P1_affordance/closed_loop_verdict.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _telemetry_metrics(csv_path: Path | None) -> dict[str, Any]:
    if csv_path is None:
        return {}
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    if not rows:
        return {"count": 0}

    def col(name: str) -> np.ndarray:
        out = []
        for r in rows:
            v = r.get(name, "")
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(float("nan"))
        return np.asarray(out, dtype=np.float64)

    angular = col("angular")
    stamp = col("stamp")
    finite = angular[np.isfinite(angular)]
    jerk = np.abs(np.diff(finite)) if finite.size > 1 else np.asarray([])
    valid_stamp = stamp[np.isfinite(stamp)]
    dur = float(valid_stamp[-1] - valid_stamp[0]) if valid_stamp.size > 1 else 0.0
    max_abs = float(np.nanmax(np.abs(angular))) if finite.size else float("nan")
    sat_thresh = 0.98 * max_abs if max_abs and math.isfinite(max_abs) and max_abs > 0 else float("inf")
    return {
        "count": len(rows),
        "durationSeconds": dur,
        "effectiveHz": (len(rows) / dur) if dur > 0 else float("nan"),
        "angularMeanAbs": float(np.mean(np.abs(finite))) if finite.size else float("nan"),
        "angularMaxAbs": max_abs,
        "meanJerk": float(np.mean(jerk)) if jerk.size else float("nan"),
        "p95Jerk": float(np.percentile(jerk, 95)) if jerk.size else float("nan"),
        "saturationRate": float(np.mean(np.abs(finite) >= sat_thresh)) if finite.size and math.isfinite(sat_thresh) else float("nan"),
    }


def _events_metrics(json_path: Path | None) -> dict[str, Any]:
    if json_path is None:
        return {}
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    laps = float(data.get("laps_completed", 0) or 0)
    oob = float(data.get("out_of_bounds", 0) or 0)
    interventions = float(data.get("interventions", 0) or 0)
    duration = float(data.get("duration_s", 0) or 0)
    minutes = duration / 60.0 if duration > 0 else float("nan")
    return {
        "lapsCompleted": laps,
        "outOfBounds": oob,
        "interventions": interventions,
        "durationSeconds": duration,
        "oobPerLap": (oob / laps) if laps > 0 else float("nan"),
        "interventionsPerLap": (interventions / laps) if laps > 0 else float("nan"),
        "interventionsPerMinute": (interventions / minutes) if minutes and math.isfinite(minutes) and minutes > 0 else float("nan"),
        "notes": data.get("notes", ""),
    }


def _summarize(*, telemetry: Path | None, events: Path | None) -> dict[str, Any]:
    return {"telemetry": _telemetry_metrics(telemetry), "events": _events_metrics(events)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score + compare a closed-loop run against the locked baseline.")
    parser.add_argument("--candidate-telemetry", default=None)
    parser.add_argument("--candidate-events", default=None)
    parser.add_argument("--baseline-telemetry", default=None)
    parser.add_argument("--baseline-events", default=None)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    candidate = _summarize(
        telemetry=Path(args.candidate_telemetry).resolve() if args.candidate_telemetry else None,
        events=Path(args.candidate_events).resolve() if args.candidate_events else None,
    )
    baseline = _summarize(
        telemetry=Path(args.baseline_telemetry).resolve() if args.baseline_telemetry else None,
        events=Path(args.baseline_events).resolve() if args.baseline_events else None,
    )

    cand_ev = candidate.get("events", {})
    base_ev = baseline.get("events", {})
    verdict: dict[str, Any] = {"candidate": candidate, "baseline": baseline}
    if cand_ev and base_ev:
        verdict["comparison"] = {
            "deltaOobPerLap": _sub(cand_ev.get("oobPerLap"), base_ev.get("oobPerLap")),
            "deltaInterventionsPerLap": _sub(cand_ev.get("interventionsPerLap"), base_ev.get("interventionsPerLap")),
            "candidateBetter": _better(cand_ev, base_ev),
            "criterion": "fewer out-of-bounds and interventions per lap than the locked baseline, params frozen",
        }
    out = Path(args.output_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


def _sub(a: Any, b: Any) -> float:
    try:
        av, bv = float(a), float(b)
        if math.isfinite(av) and math.isfinite(bv):
            return av - bv
    except (TypeError, ValueError):
        pass
    return float("nan")


def _better(cand: dict[str, Any], base: dict[str, Any]) -> bool | None:
    d_oob = _sub(cand.get("oobPerLap"), base.get("oobPerLap"))
    d_int = _sub(cand.get("interventionsPerLap"), base.get("interventionsPerLap"))
    if math.isnan(d_oob) and math.isnan(d_int):
        return None
    oob_ok = math.isnan(d_oob) or d_oob <= 0
    int_ok = math.isnan(d_int) or d_int <= 0
    return bool(oob_ok and int_ok)


if __name__ == "__main__":
    main()
