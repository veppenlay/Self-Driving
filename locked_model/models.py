#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from steering_models import (
    RegressionSequenceCfCSteeringNet,
    RegressionSequenceGRUSteeringNet,
    RegressionSteeringNet,
    RegressionSteeringNetV1,
    RegressionTemporalSteeringNet,
    load_state_dict_flexible,
)


MODEL_OUTPUT_DIM = 2
MODEL_OUTPUT_NAMES = ("steering", "e_y")


class AutoDriveLegacyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_frames = 1
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.5),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 8 * 13, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.Linear(10, MODEL_OUTPUT_DIM),
        )

    def forward(self, input_tensor):
        x = input_tensor.view(input_tensor.size(0), 3, 120, 160)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.linear_layers(x)


class AutoDriveNet(RegressionSteeringNet):
    def __init__(self, *, num_aux_classes: int = 11, use_pretrained: bool = True):
        super().__init__(num_aux_classes=num_aux_classes, use_pretrained=use_pretrained)


class AutoDriveNetV1(RegressionSteeringNetV1):
    def __init__(self, *, num_aux_classes: int = 11, use_pretrained: bool = True):
        super().__init__(num_aux_classes=num_aux_classes, use_pretrained=use_pretrained)


class AutoDriveNetTemporal(RegressionTemporalSteeringNet):
    def __init__(self, *, num_aux_classes: int = 11, use_pretrained: bool = True, num_frames: int = 3):
        super().__init__(num_aux_classes=num_aux_classes, use_pretrained=use_pretrained, num_frames=num_frames)


class AutoDriveNetSeqGRU(RegressionSequenceGRUSteeringNet):
    def __init__(self, *, num_aux_classes: int = 11, use_pretrained: bool = True, num_frames: int = 3):
        super().__init__(num_aux_classes=num_aux_classes, use_pretrained=use_pretrained, num_frames=num_frames)


class AutoDriveNetSeqCfC(RegressionSequenceCfCSteeringNet):
    def __init__(self, *, num_aux_classes: int = 11, use_pretrained: bool = True, num_frames: int = 3):
        super().__init__(num_aux_classes=num_aux_classes, use_pretrained=use_pretrained, num_frames=num_frames)


def _infer_input_channels(state_dict) -> int:
    stem_weight = state_dict.get('backbone.0.0.weight')
    if stem_weight is not None and getattr(stem_weight, 'ndim', 0) == 4:
        return int(stem_weight.shape[1])
    return 3


def _is_temporal_state_dict(state_dict) -> bool:
    return any(str(key).startswith('temporal_adapter.') for key in state_dict.keys())


def _is_sequence_gru_state_dict(state_dict) -> bool:
    return any(str(key).startswith('temporal_rnn.') for key in state_dict.keys())


def _is_sequence_cfc_state_dict(state_dict) -> bool:
    return any(str(key).startswith(('temporal_projection.', 'cfc_cell.')) for key in state_dict.keys())


def _infer_num_frames_from_variant(variant: str, default: int = 3) -> int:
    import re

    match = re.search(r"temporal(\d+)", variant)
    if match:
        return max(2, int(match.group(1)))
    if "temporal" in variant or variant.startswith(("seq_gru", "seq_cfc")):
        return max(2, int(default))
    return max(1, int(default))


def _infer_num_aux_classes(state_dict, default: int = 11) -> int:
    aux_weight = state_dict.get('aux_head.3.weight')
    if aux_weight is not None and getattr(aux_weight, 'ndim', 0) == 2:
        return int(aux_weight.shape[0])
    return int(default)


def build_model_for_checkpoint(state_dict, model_variant: str | None = None):
    variant = (model_variant or '').strip().lower()
    inferred_in_channels = _infer_input_channels(state_dict)
    num_aux_classes = _infer_num_aux_classes(state_dict)
    if variant.startswith(("seq_gru", "sequence_gru")) or _is_sequence_gru_state_dict(state_dict):
        num_frames = _infer_num_frames_from_variant(variant, default=3)
        model = AutoDriveNetSeqGRU(num_aux_classes=num_aux_classes, num_frames=num_frames)
        model.load_state_dict(state_dict, strict=True)
        return model
    if variant.startswith(("seq_cfc", "sequence_cfc", "liquid_cfc")) or _is_sequence_cfc_state_dict(state_dict):
        num_frames = _infer_num_frames_from_variant(variant, default=3)
        model = AutoDriveNetSeqCfC(num_aux_classes=num_aux_classes, num_frames=num_frames)
        model.load_state_dict(state_dict, strict=True)
        return model
    if variant in {'temporal3', 'mobilenet_temporal3', 'mobilenet_v2_temporal3'} or inferred_in_channels > 3 or _is_temporal_state_dict(state_dict):
        num_frames = max(2, inferred_in_channels // 3)
        adapter_weight = state_dict.get('temporal_adapter.0.weight')
        if adapter_weight is not None and getattr(adapter_weight, 'ndim', 0) == 4:
            num_frames = max(2, int(adapter_weight.shape[1]))
        temporal = AutoDriveNetTemporal(num_aux_classes=num_aux_classes, num_frames=num_frames)
        temporal.load_state_dict(state_dict, strict=True)
        return temporal

    for cls in (AutoDriveNet, AutoDriveNetV1):
        model = cls(num_aux_classes=num_aux_classes)
        if load_state_dict_flexible(model, state_dict):
            return model
    legacy = AutoDriveLegacyNet()
    legacy.load_state_dict(state_dict, strict=True)
    return legacy
