#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from steering_preprocess import NumpyRandomState, PreprocessConfig, preprocess_bgr_to_tensor


TEMPORAL_PADDING_MODE = "replicate_current_frame_for_missing_history"
TEMPORAL_PADDING_DESCRIPTION = (
    "When a requested history frame is missing, fall back to the current frame path for that missing slot. "
    "This preserves the existing project behavior and keeps sequence-start samples trainable."
)


def _imread_bgr(path: str):
    p = os.fspath(path)
    buf = np.fromfile(p, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _is_albumentations_transform(transform: Any) -> bool:
    return hasattr(transform, "__call__") and transform.__class__.__module__.startswith("albumentations")


def _parse_frame_index(path: str | Path) -> int:
    stem = Path(path).stem
    if "_" not in stem:
        raise ValueError(f"missing frame index in filename: {path}")
    return int(stem.split("_", 1)[0])


class AutoDriveDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        mode: str,
        transform=None,
        deterministic_seed: int | None = None,
        *,
        split_name: str | None = None,
        return_meta: bool = False,
        teacher_config: PreprocessConfig | None = None,
        num_frames: int = 1,
        frame_stride: int = 1,
        cache_images: bool = True,
    ):
        self.data_folder = data_folder
        self.mode = mode.lower()
        self.transform = transform
        self.deterministic_seed = deterministic_seed
        self.return_meta = return_meta
        self.teacher_config = teacher_config
        self.num_frames = max(1, int(num_frames))
        self.frame_stride = max(1, int(frame_stride))
        self.cache_images = bool(cache_images)
        self._image_cache: dict[str, np.ndarray] = {}
        assert self.mode in {"train", "val", "test", "val_style_real", "train_clean", "val_clean", "test_clean"}

        self.frame_lookup: dict[tuple[str, int], str] = {}
        self.filename_lookup: dict[str, str] = {}
        for image_file in Path(self.data_folder).rglob("*"):
            if not image_file.is_file() or image_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                key = (str(image_file.parent), _parse_frame_index(image_file))
            except ValueError:
                continue
            self.frame_lookup[key] = str(image_file)
            self.filename_lookup.setdefault(image_file.name, str(image_file))

        file_path = self._resolve_split_file(split_name)
        self.file_list: list[tuple[str, float]] = []
        self.remapped_sample_paths = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    img_path, angle_str = line.rsplit(" ", 1)
                    angle = float(angle_str)
                except ValueError:
                    continue
                resolved_img_path = self._resolve_sample_image_path(img_path)
                if resolved_img_path != img_path:
                    self.remapped_sample_paths += 1
                self.file_list.append((resolved_img_path, angle))

        self._index_frames_for_sample_parents()
        self.angles: list[float] = [angle for _, angle in self.file_list]
        self.temporal_summary = self._build_temporal_summary()

    def _read_bgr(self, image_path: str) -> np.ndarray:
        if self.cache_images:
            cached = self._image_cache.get(image_path)
            if cached is not None:
                return cached.copy()
        frame = _imread_bgr(image_path)
        if frame is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        if self.cache_images:
            self._image_cache[image_path] = frame
            return frame.copy()
        return frame

    def _resolve_split_file(self, split_name: str | None) -> str:
        def _candidate_names(name: str) -> list[str]:
            key = name.lower()
            if key == "train":
                return ["train_clean.txt", "train.txt"]
            if key == "val":
                return ["val_clean.txt", "val.txt"]
            if key == "test":
                return ["test_clean.txt", "test.txt"]
            if key == "train_clean":
                return ["train_clean.txt", "train.txt"]
            if key == "val_clean":
                return ["val_clean.txt", "val.txt"]
            if key == "test_clean":
                return ["test_clean.txt", "test.txt"]
            if key == "val_style_real":
                return ["val_style_real.txt", "val_style.txt", "val.txt"]
            return [f"{key}.txt"]

        requested = split_name or self.mode
        requested_path = Path(requested)
        if requested_path.is_file():
            return str(requested_path)

        for candidate in _candidate_names(requested):
            file_path = Path(self.data_folder) / candidate
            if file_path.is_file():
                return str(file_path)
        tried = ", ".join(str(Path(self.data_folder) / name) for name in _candidate_names(requested))
        raise FileNotFoundError(f"split file not found for mode={self.mode} split_name={split_name}; tried: {tried}")

    def _resolve_sample_image_path(self, image_path: str) -> str:
        raw_path = Path(image_path)
        if raw_path.is_file():
            return str(raw_path)
        relative_candidate = Path(self.data_folder) / raw_path
        if relative_candidate.is_file():
            return str(relative_candidate)
        basename_candidate = self.filename_lookup.get(raw_path.name)
        if basename_candidate is not None:
            return basename_candidate
        return image_path

    def _index_frames_for_sample_parents(self) -> None:
        parent_dirs = sorted({Path(image_path).parent for image_path, _ in self.file_list})
        for parent_dir in parent_dirs:
            if not parent_dir.is_dir():
                continue
            for image_file in parent_dir.iterdir():
                if not image_file.is_file() or image_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                try:
                    key = (str(image_file.parent), _parse_frame_index(image_file))
                except ValueError:
                    continue
                self.frame_lookup[key] = str(image_file)

    def _resolve_frame_stack_paths(self, image_path: str) -> tuple[list[str], int]:
        current_path = Path(image_path)
        current_index = _parse_frame_index(current_path)
        folder_key = str(current_path.parent)
        frame_paths: list[str] = []
        last_valid_path = image_path
        missing_history_count = 0
        for offset in range(self.num_frames - 1, -1, -1):
            target_index = current_index - offset * self.frame_stride
            candidate_path = self.frame_lookup.get((folder_key, target_index))
            if candidate_path is None:
                candidate_path = last_valid_path
                if offset > 0:
                    missing_history_count += 1
            frame_paths.append(candidate_path)
            last_valid_path = candidate_path
        return frame_paths, missing_history_count

    def _build_temporal_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "numFrames": int(self.num_frames),
            "frameStride": int(self.frame_stride),
            "paddingMode": TEMPORAL_PADDING_MODE if self.num_frames > 1 else "single_frame",
            "paddingDescription": TEMPORAL_PADDING_DESCRIPTION if self.num_frames > 1 else "Temporal padding is not used.",
            "remappedSamplePaths": int(self.remapped_sample_paths),
            "samplesWithMissingHistory": 0,
            "totalMissingHistoryFrames": 0,
            "maxMissingHistoryFramesPerSample": 0,
        }
        if self.num_frames <= 1:
            return summary

        samples_with_missing = 0
        total_missing = 0
        max_missing = 0
        for image_path, _ in self.file_list:
            _, missing_history_count = self._resolve_frame_stack_paths(image_path)
            if missing_history_count <= 0:
                continue
            samples_with_missing += 1
            total_missing += missing_history_count
            max_missing = max(max_missing, missing_history_count)

        summary["samplesWithMissingHistory"] = int(samples_with_missing)
        summary["totalMissingHistoryFrames"] = int(total_missing)
        summary["maxMissingHistoryFramesPerSample"] = int(max_missing)
        return summary

    def _load_frame_stack(self, image_path: str) -> list[np.ndarray]:
        frame_paths, _ = self._resolve_frame_stack_paths(image_path)
        frames: list[np.ndarray] = []
        for candidate_path in frame_paths:
            frame = self._read_bgr(candidate_path)
            frames.append(frame)
        return frames

    def _pack_default_frames(self, frames: list[np.ndarray]) -> torch.Tensor:
        tensors = []
        for frame in frames:
            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(np.transpose(img_hsv, (2, 0, 1))).float())
        return torch.cat(tensors, dim=0)

    def _apply_transform(self, img_bgr: np.ndarray | list[np.ndarray], idx: int):
        meta: dict[str, Any] = {"styleTag": "clean", "isClean": True}
        frames = img_bgr if isinstance(img_bgr, list) else [img_bgr]
        if self.transform is None:
            return self._pack_default_frames(frames), meta

        if _is_albumentations_transform(self.transform) or getattr(self.transform, "returns_dict", False):
            rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
            payload: dict[str, Any] = {"image": rgb_frames[0]}
            for frame_idx, frame in enumerate(rgb_frames[1:], start=1):
                payload[f"image{frame_idx}"] = frame
            if self.deterministic_seed is not None:
                with NumpyRandomState(self.deterministic_seed + idx):
                    result = self.transform(**payload)
            else:
                result = self.transform(**payload)

            if isinstance(result, dict):
                tensor_list = [result["image"]]
                for frame_idx in range(1, len(frames)):
                    tensor_list.append(result[f"image{frame_idx}"])
                image_tensor = torch.cat(tensor_list, dim=0)
                meta = {**meta, **{k: v for k, v in result.items() if not str(k).startswith("image")}}
                meta["isClean"] = bool(meta.get("isClean", meta.get("styleTag", "clean") == "clean"))
                meta["styleTag"] = str(meta.get("styleTag", "clean"))
                return image_tensor, meta
            if torch.is_tensor(result):
                return result, meta
            raise TypeError(f"unsupported transform result type: {type(result)!r}")

        if len(frames) > 1:
            raise RuntimeError("temporal stacking requires an albumentations-compatible transform")
        img_hsv = cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV)
        img_pil = Image.fromarray(img_hsv)
        return self.transform(img_pil), meta

    def __getitem__(self, i: int):
        img_path, angle = self.file_list[i]
        frames = self._load_frame_stack(img_path)
        current_img = frames[-1]
        teacher_img = None
        if self.teacher_config is not None:
            teacher_img = preprocess_bgr_to_tensor(current_img, config=self.teacher_config).squeeze(0)
        img, meta = self._apply_transform(frames if self.num_frames > 1 else current_img, i)
        label = torch.as_tensor([angle], dtype=torch.float32)
        if not self.return_meta:
            return img, label

        sample_meta: dict[str, Any] = {
            "styleTag": str(meta.get("styleTag", "clean")),
            "isClean": torch.tensor(bool(meta.get("isClean", True)), dtype=torch.bool),
            "path": img_path,
        }
        if teacher_img is not None:
            sample_meta["teacherImage"] = teacher_img
        return img, label, sample_meta

    def __len__(self) -> int:
        return len(self.file_list)
