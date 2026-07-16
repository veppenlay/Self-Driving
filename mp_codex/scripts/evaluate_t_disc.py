#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_MODEL_DIR = REPO_ROOT / "locked_model"
SRC_DIR = REPO_ROOT / "mp_codex" / "src"
for path in (LOCKED_MODEL_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from datasets import AutoDrive2DDataset  # noqa: E402
from steering_preprocess import preprocess_config_from_dict  # noqa: E402
from t_disc import TDiscCfC  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a T-DISC checkpoint.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--label-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(Path(args.ckpt).resolve(), map_location=device)
    model = TDiscCfC(num_frames=8, hidden_dim=32, use_pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    preprocess = preprocess_config_from_dict(checkpoint.get("preprocess"))
    dataset = AutoDrive2DDataset(args.label_csv, split=args.split, preprocess=preprocess, num_frames=8, frame_stride=1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    rows: list[dict[str, object]] = []
    offset = 0
    with torch.no_grad():
        for images, labels, meta in loader:
            prediction = model(images.to(device, non_blocking=True)).cpu().numpy()
            labels_np = labels.numpy()
            for index in range(prediction.shape[0]):
                rows.append(
                    {
                        "index": offset,
                        "split": args.split,
                        "sequence": str(meta["sequence"][index]),
                        "frame": int(meta["frame"][index]),
                        "image": str(meta["path"][index]),
                        "status": str(meta["status"][index]),
                        "eyQuality": float(meta["eyQuality"][index]),
                        "steeringGT": float(labels_np[index, 0]),
                        "eyGT": float(labels_np[index, 1]),
                        "steeringPred": float(prediction[index, 0]),
                        "eyPred": float(prediction[index, 1]),
                    }
                )
                offset += 1
    truth = np.asarray([row["steeringGT"] for row in rows], dtype=np.float64)
    prediction = np.asarray([row["steeringPred"] for row in rows], dtype=np.float64)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"predictions_{args.split}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "split": args.split,
        "count": len(rows),
        "steeringMAE": float(np.mean(np.abs(prediction - truth))),
        "numFrames": 8,
        "hiddenDim": 32,
        "deltaT": 1.0 / 30.0,
    }
    (output_dir / f"summary_{args.split}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

