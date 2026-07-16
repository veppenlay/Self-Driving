#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Confirmation mini-OOD: dark / blur / mid-frame-drop on 25basement test."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
LOCKED = ROOT / "locked_model"
sys.path[:0] = [str(ROOT), str(LOCKED)]

from mp_cursor.challengers.a2_temporal_ln import SeqCfCTemporalLN  # noqa: E402
from mp_cursor.challengers.b0_nll import SeqCfCNLL  # noqa: E402
from mp_cursor.challengers.common import BASELINE_HMM_MAE, apply_hmm, device_of  # noqa: E402
from mp_cursor.challengers.d1_framediff import FrameDiffDataset, SeqCfCFrameDiff  # noqa: E402
from mp_cursor.challengers.s_pack_d1_b0 import SeqCfCFrameDiffNLL  # noqa: E402
from mp_cursor.geo_control.perturb import blur, brightness  # noqa: E402
from datasets import AutoDrive2DDataset  # noqa: E402
from steering_preprocess import PreprocessConfig, imread_bgr, preprocess_bgr_to_tensor  # noqa: E402

MODEL_MAP = {
    "D1_framediff": (SeqCfCFrameDiff, True),
    "B0_nll": (SeqCfCNLL, False),
    "A2_temporal_ln": (SeqCfCTemporalLN, False),
    "S_pack_D1_B0": (SeqCfCFrameDiffNLL, True),
}


def build_tensor(paths, preprocess, *, use_framediff: bool, force_mid_drop: bool = False):
    paths = list(paths)
    if force_mid_drop and len(paths) >= 2:
        paths[len(paths) // 2] = paths[0]
    chw_list = []
    prev_v = None
    for path in paths:
        bgr = imread_bgr(path)
        if bgr is None:
            raise FileNotFoundError(path)
        chw = preprocess_bgr_to_tensor(bgr, config=preprocess).squeeze(0)
        if use_framediff:
            v = chw[2:3]
            dv = torch.zeros_like(v) if prev_v is None else (v - prev_v)
            prev_v = v
            chw_list.append(torch.cat([chw, dv], dim=0))
        else:
            chw_list.append(chw)
    return torch.cat(chw_list, dim=0)


def run_case(model, base_ds, preprocess, device, *, use_framediff: bool, transform=None, force_mid_drop=False):
    rows = []
    steering_abs = 0.0
    with torch.no_grad():
        for i in range(len(base_ds)):
            row = base_ds.rows[i]
            paths = base_ds._resolve_frame_paths(row["image"])
            if transform is not None:
                # apply on each frame image before preprocess
                tensors = []
                prev_v = None
                paths = list(paths)
                if force_mid_drop and len(paths) >= 2:
                    paths[len(paths) // 2] = paths[0]
                for path in paths:
                    bgr = transform(imread_bgr(path))
                    chw = preprocess_bgr_to_tensor(bgr, config=preprocess).squeeze(0)
                    if use_framediff:
                        v = chw[2:3]
                        dv = torch.zeros_like(v) if prev_v is None else (v - prev_v)
                        prev_v = v
                        tensors.append(torch.cat([chw, dv], dim=0))
                    else:
                        tensors.append(chw)
                image = torch.cat(tensors, dim=0)
            else:
                image = build_tensor(paths, preprocess, use_framediff=use_framediff, force_mid_drop=force_mid_drop)
            pred = model(image.unsqueeze(0).to(device))
            if pred.shape[-1] > 2:
                pred = pred[:, :2]
            gt = float(row["steering"])
            sp = float(pred[0, 0].item())
            steering_abs += abs(sp - gt)
            rows.append(
                {
                    "index": len(rows),
                    "sequence": row["sequence"],
                    "frame": row["frame"],
                    "path": str(row["image"]),
                    "status": row["status"],
                    "eyQuality": row["ey_quality"],
                    "steeringGT": gt,
                    "eyGT": row["e_y"],
                    "steeringPred": sp,
                    "eyPred": float(pred[0, 1].item()),
                }
            )
    return rows, steering_abs / max(1, len(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True, choices=list(MODEL_MAP))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--label-csv", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = device_of(args.device)
    out_dir = Path(args.output_dir).resolve()
    model_cls, use_framediff = MODEL_MAP[args.experiment]
    preprocess = PreprocessConfig(color_space="hsv", input_size=(144, 192), use_roi=True, illumination_profile="none")
    ds_cls = FrameDiffDataset if use_framediff else AutoDrive2DDataset
    base_ds = ds_cls(Path(args.label_csv), split="test", preprocess=preprocess, num_frames=3, frame_stride=1)
    model = model_cls(num_frames=3).to(device)
    ckpt = out_dir / "checkpoints" / "best.pth"
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    clean_hmm_path = out_dir / "evaluation" / "hmm" / "summary_test_hmm.json"
    clean = json.loads(clean_hmm_path.read_text(encoding="utf-8")) if clean_hmm_path.is_file() else {}
    results = {"clean": clean, "stress": {}}

    cases = [
        ("dark", lambda bgr: brightness(bgr, -0.25), False),
        ("blur", lambda bgr: blur(bgr, 1.2), False),
        ("drop_mid", None, True),
    ]
    for name, transform, drop in cases:
        rows, raw_mae = run_case(
            model, base_ds, preprocess, device, use_framediff=use_framediff, transform=transform, force_mid_drop=drop
        )
        stress_dir = out_dir / "confirmation" / "ood_mini" / name
        stress_dir.mkdir(parents=True, exist_ok=True)
        pred_csv = stress_dir / "predictions_test.csv"
        with pred_csv.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        hmm = apply_hmm(pred_csv, stress_dir / "hmm")
        results["stress"][name] = {"rawSteeringMAE": raw_mae, **hmm}
        print(f"stress[{name}] raw={raw_mae:.4f} hmm={hmm['hmmSteeringMAE']:.4f}")

    out_json = out_dir / "confirmation" / "ood_mini" / "summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    clean_hmm = float(clean.get("hmmSteeringMAE", BASELINE_HMM_MAE))
    lines = [f"# Mini OOD — {args.experiment}", f"clean HMM={clean_hmm:.6f}", ""]
    for k, v in results["stress"].items():
        lines.append(f"- {k}: HMM={v['hmmSteeringMAE']:.6f} (Δclean={v['hmmSteeringMAE'] - clean_hmm:+.6f})")
    (out_dir / "confirmation" / "ood_mini" / "NOTES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
