from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_MODEL_DIR = REPO_ROOT / "locked_model"
if str(LOCKED_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCKED_MODEL_DIR))

from datasets import AutoDrive2DDataset  # noqa: E402
from steering_models import RegressionSequenceCfCSteeringNet  # noqa: E402


STATUS_WEIGHTS = {
    "ok": 1.0,
    "inferred_missing_left": 0.4,
    "inferred_missing_right": 0.4,
    "insufficient_yellow_edges": 0.0,
    "narrow_lane_width": 0.0,
}


def _optional_float(value: Any) -> float:
    if value in (None, "", "None"):
        return float("nan")
    return float(value)


class LaneGeometryDataset(AutoDrive2DDataset):
    """Locked temporal dataset plus training-only lane-point geometry."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        geometry: dict[str, dict[str, float]] = {}
        with self.label_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                status = str(raw.get("status") or "")
                geometry[str(Path(raw["image"]))] = {
                    "left_x": _optional_float(raw.get("leftX")),
                    "right_x": _optional_float(raw.get("rightX")),
                    "reference_y": _optional_float(raw.get("referenceY")),
                    "image_width": _optional_float(raw.get("imageWidth")),
                    "image_height": _optional_float(raw.get("imageHeight")),
                    "quality": float(STATUS_WEIGHTS.get(status, 0.0)),
                }
        self._geometry = geometry

    def __getitem__(self, index: int):
        images, labels, meta = super().__getitem__(index)
        values = self._geometry.get(str(Path(meta["path"])))
        if values is None:
            raise KeyError(f"lane geometry missing for {meta['path']}")
        geometry = torch.tensor(
            [
                values["left_x"],
                values["right_x"],
                values["reference_y"],
                values["image_width"],
                values["image_height"],
            ],
            dtype=torch.float32,
        )
        quality = values["quality"]
        if not torch.isfinite(geometry).all():
            quality = 0.0
            geometry = torch.nan_to_num(geometry, nan=0.0)
        meta["laneGeometry"] = geometry
        meta["laneQuality"] = torch.tensor(quality, dtype=torch.float32)
        return images, labels, meta


class RDiscCfC(RegressionSequenceCfCSteeringNet):
    """Locked CfC main path with a training-only two-channel lane heatmap head."""

    def __init__(self, *, num_frames: int = 3, use_pretrained: bool = True):
        super().__init__(num_frames=num_frames, use_pretrained=use_pretrained)
        self.lane_head = nn.Sequential(
            nn.Conv2d(self.feature_dim, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
            nn.Conv2d(32, 2, kernel_size=1, bias=True),
        )

    def _feature_sequence_and_last_map(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat_frames, batch_size = self._flatten_temporal_input(input_tensor)
        mid_map, final_map = self.extract_feature_maps(flat_frames)
        mid_map = self.mid_reduce(mid_map)
        final_map = self.final_reduce(final_map)
        final_map = F.interpolate(final_map, size=mid_map.shape[-2:], mode="bilinear", align_corners=False)
        fused_map = self.fuse(torch.cat([mid_map, final_map], dim=1))
        frame_features = self.spatial_pool(fused_map)
        feature_seq = frame_features.reshape(batch_size, self.num_frames, self.feature_dim)
        _, channels, height, width = fused_map.shape
        last_map = fused_map.reshape(batch_size, self.num_frames, channels, height, width)[:, -1]
        return feature_seq, last_map

    def forward(self, input_tensor: torch.Tensor, *, return_lane: bool = False):
        feature_seq, last_map = self._feature_sequence_and_last_map(input_tensor)
        projected_seq = self.temporal_projection(feature_seq)
        hidden = self._run_cfc(projected_seq)
        prediction = self.reg_head(hidden)
        if return_lane:
            return prediction, self.lane_head(last_map)
        return prediction

    def deployment_state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value for key, value in self.state_dict().items() if not key.startswith("lane_head.")}


def lane_heatmap_targets(
    geometry: torch.Tensor,
    *,
    height: int,
    width: int,
    roi_bottom_ratio: float = 0.7,
    sigma_cells: float = 1.0,
) -> torch.Tensor:
    """Create left/right Gaussian point heatmaps at the labelled reference row."""

    left_x, right_x, reference_y, image_width, image_height = geometry.unbind(dim=1)
    image_width = image_width.clamp_min(1.0)
    image_height = image_height.clamp_min(1.0)
    crop_top = (1.0 - float(roi_bottom_ratio)) * image_height
    crop_height = float(roi_bottom_ratio) * image_height
    center_y = ((reference_y - crop_top) / crop_height.clamp_min(1.0)).clamp(0.0, 1.0) * max(0, height - 1)
    centers_x = torch.stack(
        [
            (left_x / image_width).clamp(0.0, 1.0) * max(0, width - 1),
            (right_x / image_width).clamp(0.0, 1.0) * max(0, width - 1),
        ],
        dim=1,
    )
    grid_y = torch.arange(height, device=geometry.device, dtype=geometry.dtype).view(1, 1, height, 1)
    grid_x = torch.arange(width, device=geometry.device, dtype=geometry.dtype).view(1, 1, 1, width)
    dx2 = (grid_x - centers_x[:, :, None, None]).square()
    dy2 = (grid_y - center_y[:, None, None, None]).square()
    denominator = 2.0 * max(float(sigma_cells), math.sqrt(1e-6)) ** 2
    return torch.exp(-(dx2 + dy2) / denominator)


def weighted_lane_heatmap_loss(logits: torch.Tensor, target: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    pixel_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pixel_weight = 1.0 + 4.0 * target
    per_sample = (pixel_loss * pixel_weight).mean(dim=(1, 2, 3))
    quality = quality.view(-1).clamp_min(0.0)
    return (per_sample * quality).sum() / quality.sum().clamp_min(1.0)

