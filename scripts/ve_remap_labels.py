#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Remap locked Windows label paths to the ve Linux dataset layout (CPU-safe)."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    repo = Path("/root/autodl-tmp/v-Net").resolve()
    src = repo / "locked_model/labels/current/labels_2d.csv"
    out_dir = repo / "locked_model/labels/remote"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "labels_2d.csv"
    marker = "dataset/"

    rows: list[dict[str, str]] = []
    missing: list[str] = []
    with src.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        for raw in reader:
            image_text = raw["image"].replace("\\", "/")
            idx = image_text.lower().rfind(marker)
            if idx < 0:
                raise RuntimeError(f"bad image path: {raw['image']!r}")
            rel = image_text[idx + len(marker) :]
            new_path = (repo / "dataset" / rel).resolve()
            if not new_path.is_file():
                missing.append(str(new_path))
            raw["image"] = str(new_path)
            if raw.get("preview"):
                raw["preview"] = ""
            rows.append(raw)

    if missing:
        print(f"MISSING_COUNT {len(missing)}")
        print("\n".join(missing[:20]))
        raise SystemExit(1)

    with dst.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    split_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    summary = {
        "sourceCsv": str(src),
        "datasetRoot": str(repo / "dataset" / "26合并"),
        "testRoot": str(repo / "dataset" / "25地下室"),
        "labelCount": len(rows),
        "splitCounts": split_counts,
        "statusCounts": status_counts,
        "note": "Paths remapped from locked_model/labels/current; label values unchanged.",
        "outputs": {"csv": str(dst)},
    }
    (out_dir / "labels_2d_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("REMAP_OK", len(rows), split_counts)
    print("sample", rows[0]["image"])


if __name__ == "__main__":
    main()
