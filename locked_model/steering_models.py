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
            nn.Linear(32, 2),
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
            nn.Linear(24, 2),
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


class _SequenceFeatureMixin:
    def _init_sequence_input(self, num_frames: int) -> None:
        if num_frames < 2:
            raise ValueError(f"sequence model expects num_frames >= 2, got {num_frames}")
        self.num_frames = int(num_frames)
        self.temporal_input_channels = 3 * self.num_frames

    def _flatten_temporal_input(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.dim() != 4 or input_tensor.size(1) != self.temporal_input_channels:
            raise ValueError(
                f"expected input shape [B,{self.temporal_input_channels},H,W], got {tuple(input_tensor.shape)}"
            )
        batch_size, _, height, width = input_tensor.shape
        frames = input_tensor.reshape(batch_size, self.num_frames, 3, height, width)
        return frames.reshape(batch_size * self.num_frames, 3, height, width), batch_size

    def extract_feature_sequence(self, input_tensor: torch.Tensor) -> torch.Tensor:
        flat_frames, batch_size = self._flatten_temporal_input(input_tensor)
        frame_features = super().extract_features(flat_frames)
        return frame_features.reshape(batch_size, self.num_frames, self.feature_dim)


class RegressionSequenceGRUSteeringNet(_SequenceFeatureMixin, RegressionSteeringNet):
    """True temporal baseline: shared RGB backbone per frame followed by a compact GRU head."""

    def __init__(
        self,
        *,
        num_aux_classes: int = 11,
        use_pretrained: bool = True,
        num_frames: int = 3,
        hidden_dim: int = 48,
    ):
        super().__init__(num_aux_classes=num_aux_classes, use_pretrained=use_pretrained, in_channels=3)
        self._init_sequence_input(num_frames)
        self.hidden_dim = int(hidden_dim)
        self.temporal_rnn = nn.GRU(input_size=self.feature_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.reg_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.Hardswish(),
            nn.Dropout(0.10),
            nn.Linear(32, 2),
        )

    def head_parameters(self):
        return chain(
            self.mid_reduce.parameters(),
            self.final_reduce.parameters(),
            self.fuse.parameters(),
            self.spatial_pool.parameters(),
            self.temporal_rnn.parameters(),
            self.reg_head.parameters(),
            self.aux_head.parameters(),
        )

    def forward(self, input_tensor: torch.Tensor, return_aux: bool = False):
        feat_seq = self.extract_feature_sequence(input_tensor)
        out_seq, _ = self.temporal_rnn(feat_seq)
        angle = self.reg_head(out_seq[:, -1, :])
        if return_aux:
            return angle, self.aux_head(feat_seq[:, -1, :])
        return angle


def _build_sparse_recurrent_mask(hidden_dim: int, fanin: int, seed: int) -> torch.Tensor:
    hidden_dim = int(hidden_dim)
    fanin = max(1, min(int(fanin), hidden_dim))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    mask = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float32)
    for row in range(hidden_dim):
        indices = torch.randperm(hidden_dim, generator=generator)[:fanin]
        mask[row, indices] = 1.0
        mask[row, row] = 1.0
    return mask


class _CfCLiteCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, *, recurrent_fanin: int = 6, mask_seed: int = 20260626):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.z_input = nn.Linear(self.input_dim, self.hidden_dim)
        self.z_recurrent = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.g_input = nn.Linear(self.input_dim, self.hidden_dim)
        self.g_recurrent = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.register_buffer(
            "recurrent_mask",
            _build_sparse_recurrent_mask(self.hidden_dim, recurrent_fanin, mask_seed),
            persistent=True,
        )

    def _masked_recurrent(self, layer: nn.Linear, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, layer.weight * self.recurrent_mask, layer.bias)

    def forward(self, input_t: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        z_t = torch.tanh(self.z_input(input_t) + self._masked_recurrent(self.z_recurrent, hidden))
        g_t = torch.sigmoid(self.g_input(input_t) + self._masked_recurrent(self.g_recurrent, hidden))
        return g_t * hidden + (1.0 - g_t) * z_t


class RegressionSequenceCfCSteeringNet(_SequenceFeatureMixin, RegressionSteeringNet):
    """ONNX-friendly liquid-style controller with sparse recurrent dynamics."""

    def __init__(
        self,
        *,
        num_aux_classes: int = 11,
        use_pretrained: bool = True,
        num_frames: int = 3,
        projection_dim: int = 64,
        hidden_dim: int = 32,
        recurrent_fanin: int = 6,
        mask_seed: int = 20260626,
    ):
        super().__init__(num_aux_classes=num_aux_classes, use_pretrained=use_pretrained, in_channels=3)
        self._init_sequence_input(num_frames)
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)
        self.temporal_projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.projection_dim),
            nn.Dropout(0.10),
            nn.ReLU(inplace=True),
        )
        self.cfc_cell = _CfCLiteCell(
            self.projection_dim,
            self.hidden_dim,
            recurrent_fanin=recurrent_fanin,
            mask_seed=mask_seed,
        )
        self.reg_head = nn.Linear(self.hidden_dim, 2)

    def head_parameters(self):
        return chain(
            self.mid_reduce.parameters(),
            self.final_reduce.parameters(),
            self.fuse.parameters(),
            self.spatial_pool.parameters(),
            self.temporal_projection.parameters(),
            self.cfc_cell.parameters(),
            self.reg_head.parameters(),
            self.aux_head.parameters(),
        )

    def _run_cfc(self, projected_seq: torch.Tensor) -> torch.Tensor:
        hidden = projected_seq.new_zeros(projected_seq.size(0), self.hidden_dim)
        for frame_idx in range(self.num_frames):
            hidden = self.cfc_cell(projected_seq[:, frame_idx, :], hidden)
        return hidden

    def forward(self, input_tensor: torch.Tensor, return_aux: bool = False):
        feat_seq = self.extract_feature_sequence(input_tensor)
        projected_seq = self.temporal_projection(feat_seq)
        hidden = self._run_cfc(projected_seq)
        angle = self.reg_head(hidden)
        if return_aux:
            return angle, self.aux_head(feat_seq[:, -1, :])
        return angle


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
