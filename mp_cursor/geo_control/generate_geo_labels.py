#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extend the existing 2D labels with an automatic heading label `theta` (zero manual work).

This does NOT modify the locked `locked_model/generate_2d_labels.py`; it reuses its CV
detection primitives and adds a SECOND reference band. For every image we detect the lane
center at a NEAR band (the existing 0.54 reference) and a FAR band (default 0.42), then
derive `theta` from the two centers. `thetaQuality` is 0 unless BOTH bands detect cleanly,
mirroring the `eyQuality` gating so low-confidence frames never enter the loss.

Runs on the GPU/dev environment (needs cv2). Output: a `labels_geo.csv` that is a superset
of the base labels CSV with added `theta`, `thetaQuality`, plus far-band diagnostics.

Usage:
  python -m mp_cursor.geo_control.generate_geo_labels \
    --base-label-csv locked_model/labels/current/labels_2d.csv \
    --output-csv mp_cursor/geo_control/labels/labels_geo.csv \
    --far-y-ratio 0.42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_DIR = REPO_ROOT / "locked_model"
for _p in (str(LOCKED_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .geometry import theta_from_centers  # noqa: E402


def _lazy_locked_primitives():
    """Import cv2-backed detection primitives from the locked label generator."""
    import generate_2d_labels as g2d  # type: ignore
    from steering_preprocess import imread_bgr  # type: ignore

    return g2d, imread_bgr


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add automatic heading label theta to base 2D labels.")
    parser.add_argument("--base-label-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--far-y-ratio", type=float, default=0.42)
    parser.add_argument("--near-y-ratio", type=float, default=0.54, help="Should match the base labels reference ratio.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_csv = Path(args.base_label_csv).resolve()
    out_csv = Path(args.output_csv).resolve()
    g2d, imread_bgr = _lazy_locked_primitives()

    rows = _read_csv(base_csv)
    if not rows:
        raise SystemExit(f"empty base labels: {base_csv}")

    width_prior = _float_or_none(rows[0].get("laneWidthPriorPx"))
    out_rows: list[dict[str, Any]] = []
    theta_quality_sum = 0.0
    detected = 0
    for row in rows:
        image_path = Path(row["image"])
        image = imread_bgr(image_path)
        near_center = _float_or_none(row.get("laneCenterX"))
        near_quality = _float_or_none(row.get("eyQuality")) or 0.0
        height = _float_or_none(row.get("imageHeight"))

        theta = 0.0
        theta_quality = 0.0
        far_center = None
        far_status = "unread"
        if image is not None:
            h = image.shape[0]
            far_raw = g2d._raw_detection(image, args.far_y_ratio)
            far_det = g2d._finalize_detection(far_raw, width_prior=width_prior)
            far_status = str(far_det.get("status"))
            far_center = far_det.get("laneCenterX")
            far_quality = float(far_det.get("eyQuality") or 0.0)
            near_y = float(row.get("referenceY") or h * args.near_y_ratio)
            far_y = float(far_raw.get("referenceY") or h * args.far_y_ratio)
            if near_center is not None and far_center is not None and near_quality > 0 and far_quality > 0:
                theta = theta_from_centers(near_center, near_y, float(far_center), far_y)
                theta_quality = float(min(near_quality, far_quality))
                detected += 1

        theta_quality_sum += theta_quality
        out = dict(row)
        out["theta"] = round(theta, 6)
        out["thetaQuality"] = theta_quality
        out["farStatus"] = far_status
        out["farLaneCenterX"] = ("" if far_center is None else round(float(far_center), 4))
        out["farYRatio"] = args.far_y_ratio
        out_rows.append(out)

    _write_csv(out_csv, out_rows)
    summary = {
        "baseLabelCsv": str(base_csv),
        "outputCsv": str(out_csv),
        "count": len(out_rows),
        "thetaDetected": detected,
        "thetaDetectRate": (detected / len(out_rows)) if out_rows else 0.0,
        "thetaQualitySum": theta_quality_sum,
        "farYRatio": args.far_y_ratio,
        "nearYRatio": args.near_y_ratio,
    }
    (out_csv.parent / "labels_geo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
