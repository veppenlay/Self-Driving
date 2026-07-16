#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


ROW_PATTERN = re.compile(r"^(.*)\s+([+-]?\d+(?:\.\d+)?)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_paths(path: Path) -> set[str]:
    result: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        match = ROW_PATTERN.match(line.strip())
        if not match:
            raise ValueError(f"cannot parse {path}:{line_number}: {line!r}")
        result.add(str(Path(match.group(1)).resolve()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit fixed 26merge train/val and 25basement test boundaries.")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--label-csv", default="locked_model/labels/remote/labels_2d.csv")
    parser.add_argument("--output", default="mp_codex/runs/p0_lite/data_leakage_check.json")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    list_dir = repo / "dataset" / "26合并"
    lists = {split: list_dir / f"{split}.txt" for split in ("train", "val", "test")}
    paths = {split: list_paths(path) for split, path in lists.items()}
    label_sets = {split: set() for split in paths}
    label_csv = (repo / args.label_csv).resolve()
    with label_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split = str(row.get("split") or "").lower()
            if split in label_sets:
                label_sets[split].add(str(Path(row["image"]).resolve()))
    train_root = str((repo / "dataset" / "26合并").resolve())
    test_root = str((repo / "dataset" / "25地下室").resolve())
    checks = {
        "counts": {split: len(values) for split, values in paths.items()},
        "expected_counts": {"train": 2654, "val": 569, "test": 423},
        "split_hashes": {split: sha256(path) for split, path in lists.items()},
        "overlap": {
            "train_val": len(paths["train"] & paths["val"]),
            "train_test": len(paths["train"] & paths["test"]),
            "val_test": len(paths["val"] & paths["test"]),
        },
        "outside_train_root": sum(not value.startswith(train_root + str(Path('/'))) for value in paths["train"] | paths["val"]),
        "outside_test_root": sum(not value.startswith(test_root + str(Path('/'))) for value in paths["test"]),
        "label_csv_counts": {split: len(values) for split, values in label_sets.items()},
        "label_csv_matches_lists": {split: label_sets[split] == paths[split] for split in paths},
    }
    expected = checks["expected_counts"]
    passed = (
        checks["counts"] == expected
        and all(value == 0 for value in checks["overlap"].values())
        and checks["outside_train_root"] == 0
        and checks["outside_test_root"] == 0
        and all(checks["label_csv_matches_lists"].values())
    )
    payload = {"passed": passed, "repo": str(repo), "label_csv": str(label_csv), **checks}
    output = (repo / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

