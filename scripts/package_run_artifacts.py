#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Package the essential artifacts from a cloud training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PATTERNS = (
    "training_summary.json",
    "dataset_summary.json",
    "command.txt",
    "run_command.sh",
    "git_commit.txt",
    "pip_freeze.txt",
    "env.txt",
    "train.log",
    "stdout.log",
    "stderr.log",
    "*.json",
    "*.csv",
    "*.md",
)

CHECKPOINT_PATTERNS = {
    "best": ("best*.pth", "best_*.pth"),
    "all": ("*.pth",),
    "none": (),
}

TENSORBOARD_PATTERNS = ("events.out.tfevents*",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package cloud training artifacts for local analysis.")
    parser.add_argument("--run-dir", required=True, help="Training run directory, for example runs/exp001.")
    parser.add_argument("--output-dir", required=True, help="Artifact package output directory.")
    parser.add_argument(
        "--include-checkpoints",
        choices=["best", "all", "none"],
        default="best",
        help="Which checkpoint files to copy. Default: best.",
    )
    parser.add_argument("--include-tensorboard", action="store_true", help="Copy TensorBoard event files.")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Additional file or directory to copy. Can be repeated.",
    )
    parser.add_argument("--zip", action="store_true", help="Also create a .zip archive next to output-dir.")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(src: Path, dst_root: Path, base_root: Path, manifest: list[dict[str, Any]]) -> None:
    src = src.resolve()
    if not src.is_file():
        return
    rel = src.relative_to(base_root) if src.is_relative_to(base_root) else Path("extra") / src.name
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append(
        {
            "source": str(src),
            "packagePath": rel.as_posix(),
            "bytes": src.stat().st_size,
            "sha256": file_sha256(src),
        }
    )


def collect_pattern_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(set(files), key=lambda path: path.as_posix())


def copy_extra(src: Path, dst_root: Path, manifest: list[dict[str, Any]]) -> None:
    src = src.expanduser().resolve()
    if src.is_file():
        copy_file(src, dst_root, src.parent, manifest)
        return
    if not src.is_dir():
        raise FileNotFoundError(f"extra path not found: {src}")
    for path in sorted(src.rglob("*")):
        if path.is_file():
            copy_file(path, dst_root / "extra" / src.name, src, manifest)


def try_command(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(command, cwd=str(cwd), check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip()


def write_if_missing(path: Path, content: str | None) -> None:
    if not content or path.exists():
        return
    path.write_text(content + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not run_dir.is_dir():
        raise FileNotFoundError(f"run-dir not found: {run_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    write_if_missing(run_dir / "git_commit.txt", try_command(["git", "rev-parse", "HEAD"], repo_root))

    manifest: list[dict[str, Any]] = []
    patterns = list(DEFAULT_PATTERNS)
    patterns.extend(CHECKPOINT_PATTERNS[args.include_checkpoints])
    if args.include_tensorboard:
        patterns.extend(TENSORBOARD_PATTERNS)

    for path in collect_pattern_files(run_dir, tuple(patterns)):
        copy_file(path, output_dir, run_dir, manifest)

    for extra in args.extra:
        copy_extra(Path(extra), output_dir, manifest)

    manifest_doc = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "runDir": str(run_dir),
        "outputDir": str(output_dir),
        "includeCheckpoints": args.include_checkpoints,
        "includeTensorboard": bool(args.include_tensorboard),
        "fileCount": len(manifest),
        "totalBytes": sum(item["bytes"] for item in manifest),
        "files": manifest,
    }
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.zip:
        archive_base = output_dir.with_suffix("")
        shutil.make_archive(str(archive_base), "zip", root_dir=str(output_dir))
        print(f"zip={archive_base}.zip")

    print(f"artifact_dir={output_dir}")
    print(f"file_count={manifest_doc['fileCount']}")
    print(f"total_mib={manifest_doc['totalBytes'] / 1024 / 1024:.3f}")


if __name__ == "__main__":
    main()
