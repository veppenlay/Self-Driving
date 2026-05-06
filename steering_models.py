from __future__ import annotations

from itertools import chain
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class _MobileNetFeatureBase(nn.Module):
    def __init__(self, *, use_pretrained: bool = True, in_channels: int = 3):
        super().__init__()
        weights = None
        self.pretrained_loaded = False
        self.input_channels = int(in_channels)
        self.num_frames = max(1, self.input_channels // 3) if self.input_channels % 3 == 0 else 1
        if use_pretrained:
            try:
                weights = MobileNet_V3_Small_Weights.DEFAULT
            except Exception:
                weights = None
        try:
            backbone = mobilenet_v3_small(weights=weights)
            self.pretrained_loaded = weights is not None
        except Exception:
            backbone = mobilenet_v3_small(weights=None)
            self.pretrained_loaded = False
        self.backbone = backbone.features
        if self.input_channels != 3:
            self._patch_stem_conv(self.input_channels)

    def _patch_stem_conv(self, in_channels: int) -> None:
        stem = self.backbone[0][0]
        if not isinstance(stem, nn.Conv2d):
            raise TypeError(f"unexpected MobileNet stem type: {type(stem)!r}")
        new_stem = nn.Conv2d(
            in_channels=in_channels,
            out_channels=stem.out_channels,
            kernel_size=stem.kernel_size,
            stride=stem.stride,
            padding=stem.padding,
            dilation=stem.dilation,
            groups=stem.groups,
            bias=stem.bias is not None,
            padding_mode=stem.padding_mode,
        )
        with torch.no_grad():
            if in_channels % stem.in_channels == 0:
                repeat = in_channels // stem.in_channels
                weight = stem.weight.repeat(1, repeat, 1, 1) / float(repeat)
            else:
                weight = stem.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
                weight *= float(stem.in_channels) / float(in_channels)
            new_stem.weight.copy_(weight)
            if stem.bias is not None and new_stem.bias is not None:
                new_stem.bias.copy_(stem.bias)
        self.backbone[0][0] = new_stem

    def _ensure_nchw(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.dim() != 4 or input_tensor.size(1) != self.input_channels:
            raise ValueError(f"expected input shape [B,{self.input_channels},H,W], got {tuple(input_tensor.shape)}")
        return input_tensor

    def extract_feature_maps(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._ensure_nchw(input_tensor)
        mid = None
        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            if idx == 8:
                mid = x
        if mid is None:
            raise RuntimeError("failed to capture MobileNet intermediate feature map")
        return mid, x

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def backbone_parameters(self):
        return self.backbone.parameters()


class _DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Hardswish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _SpatialAttentionPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(16, channels // 4)
        self.attention = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feature_map.shape
        logits = self.attention(feature_map).view(b, 1, h * w)
        weights = torch.softmax(logits, dim=-1).view(b, 1, h, w)
        return (feature_map * weights).sum(dim=(2, 3))


class RegressionSteeringNetV1(_MobileNetFeatureBase):
    """Current MobileNet regression baseline with pure global average pooling."""

    def __init__(self, *, num_aux_classes: int = 11, use_pretrained: bool = True, in_channels: int = 3):
        super().__init__(use_pretrained=use_pretrained, in_channels=in_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.15)
        self.feature_dim = 576
        self.reg_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Hardswish(),
            nn.Dropout(0.15),
            nn.Linear(128, 32),
            nn.Hardswish(),
            nn.Linear(32, 1),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Hardswish(),
            nn.Dropout(0.20),
            nn.Linear(128, num_aux_classes),
        )

    def head_parameters(self):
        return chain(self.reg_head.parameters(), self.aux_head.parameters())

    def extract_features(self, input_tensor: torch.Tensor) -> torch.Tensor:
        _, final_map = self.extract_feature_maps(input_tensor)
        feat = self.pool(final_map).flatten(1)
        return self.dropout(feat)

    def forward(self, input_tensor: torch.Tensor, return_aux: bool = False):
        feat = self.extract_features(input_tensor)
        angle = self.reg_head(feat)
        if return_aux:
            return angle, self.aux_head(feat)
        return angle


class RegressionSteeringNet(_MobileNetFeatureBase):
    """MobileNet regression v2: fuse lane geometry from high-res features with semantic features."""

    def __init__(self, *, num_aux_classes: int = 11, use_pretrained: bool = True, in_channels: int = 3):
        super().__init__(use_pretrained=use_pretrained, in_channels=in_channels)
        self.mid_reduce = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.Hardswish(),
        )
        self.final_reduce = nn.Sequential(
            nn.Conv2d(576, 80, kernel_size=1, bias=False),
            nn.BatchNorm2d(80),
            nn.Hardswish(),
        )
        self.fuse = _DepthwiseSeparableBlock(48 + 80, 112)
        self.spatial_pool = _SpatialAttentionPool(112)
        self.feature_dim = 112
        self.reg_head = nn.Sequential(
            nn.Linear(self.feature_dim, 96),
            nn.Hardswish(),
            nn.Dropout(0.12),
            nn.Linear(96, 24),
            nn.Hardswish(),
            nn.Linear(24, 1),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(self.feature_dim, 96),
            nn.Hardswish(),
            nn.Dropout(0.15),
            nn.Linear(96, num_aux_classes),
        )

    def head_parameters(self):
        return chain(
            self.mid_reduce.parameters(),
            self.final_reduce.parameters(),
            self.fuse.parameters(),
            self.spatial_pool.parameters(),
            self.reg_head.parameters(),
            self.aux_head.parameters(),
        )

    def extract_features(self, input_tensor: torch.Tensor) -> torch.Tensor:
        mid_map, final_map = self.extract_feature_maps(input_tensor)
        mid_map = self.mid_reduce(mid_map)
        final_map = self.final_reduce(final_map)
        final_map = F.interpolate(final_map, size=mid_map.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([mid_map, final_map], dim=1))
        return self.spatial_pool(fused)

    def forward(self, input_tensor: torch.Tensor, return_aux: bool = False):
        feat = self.extract_features(input_tensor)
        angle = self.reg_head(feat)
        if return_aux:
            return angle, self.aux_head(feat)
        return angle


class RegressionTemporalSteeringNet(RegressionSteeringNet):
    """Lightweight temporal variant: fuse stacked frames with a tiny grouped 1x1 adapter."""

    def __init__(self, *, num_aux_classes: int = 11, use_pretrained: bool = True, num_frames: int = 3):
        if num_frames < 2:
            raise ValueError(f"temporal model expects num_frames >= 2, got {num_frames}")
        self.num_frames = int(num_frames)
        self.temporal_input_channels = 3 * self.num_frames
        super().__init__(num_aux_classes=num_aux_classes, use_pretrained=use_pretrained, in_channels=3)
        self.temporal_adapter = nn.Sequential(
            nn.Conv2d(self.temporal_input_channels, 3, kernel_size=1, groups=3, bias=False),
            nn.BatchNorm2d(3),
            nn.Hardswish(),
        )
        self._init_temporal_adapter()

    def _init_temporal_adapter(self) -> None:
        conv = self.temporal_adapter[0]
        with torch.no_grad():
            conv.weight.zero_()
            frame_weights = torch.linspace(0.2, 0.6, steps=self.num_frames, dtype=conv.weight.dtype)
            frame_weights = frame_weights / frame_weights.sum()
            for channel in range(3):
                for frame_idx, weight in enumerate(frame_weights.tolist()):
                    conv.weight[channel, frame_idx, 0, 0] = weight

    def head_parameters(self):
        return chain(self.temporal_adapter.parameters(), super().head_parameters())

    def _adapt_temporal_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.dim() != 4 or input_tensor.size(1) != self.temporal_input_channels:
            raise ValueError(f"expected input shape [B,{self.temporal_input_channels},H,W], got {tuple(input_tensor.shape)}")
        return self.temporal_adapter(input_tensor)

    def extract_features(self, input_tensor: torch.Tensor) -> torch.Tensor:
        adapted = self._adapt_temporal_input(input_tensor)
        return super().extract_features(adapted)


class PrototypeResidualNet(_MobileNetFeatureBase):
    def __init__(self, *, num_classes: int, use_pretrained: bool = True):
        super().__init__(use_pretrained=use_pretrained, in_channels=3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.10)
        self.feature_dim = 576
        self.shared_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Hardswish(),
            nn.Dropout(0.20),
        )
        self.classifier = nn.Linear(128, num_classes)
        self.delta_head = nn.Linear(128, 1)

    def head_parameters(self):
        return chain(self.shared_head.parameters(), self.classifier.parameters(), self.delta_head.parameters())

    def extract_features(self, input_tensor: torch.Tensor) -> torch.Tensor:
        _, final_map = self.extract_feature_maps(input_tensor)
        feat = self.pool(final_map).flatten(1)
        return self.dropout(feat)

    def forward(self, input_tensor: torch.Tensor):
        feat = self.extract_features(input_tensor)
        hidden = self.shared_head(feat)
        logits = self.classifier(hidden)
        raw_delta = self.delta_head(hidden)
        return torch.cat([logits, raw_delta], dim=1)


class ImproveClassifierNet(_MobileNetFeatureBase):
    def __init__(self, *, num_classes: int, use_pretrained: bool = True):
        super().__init__(use_pretrained=use_pretrained, in_channels=3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.10)
        self.feature_dim = 576
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Hardswish(),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def head_parameters(self):
        return self.classifier.parameters()

    def extract_features(self, input_tensor: torch.Tensor) -> torch.Tensor:
        _, final_map = self.extract_feature_maps(input_tensor)
        feat = self.pool(final_map).flatten(1)
        return self.dropout(feat)

    def forward(self, input_tensor: torch.Tensor):
        feat = self.extract_features(input_tensor)
        return self.classifier(feat)


def load_state_dict_flexible(model: nn.Module, state: dict[str, Any]) -> bool:
    try:
        model.load_state_dict(state, strict=True)
        return True
    except RuntimeError:
        return False
