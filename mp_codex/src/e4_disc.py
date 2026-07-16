from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_MODEL_DIR = REPO_ROOT / "locked_model"
if str(LOCKED_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCKED_MODEL_DIR))

from steering_models import RegressionSequenceCfCSteeringNet  # noqa: E402


class E4DiscCfC(RegressionSequenceCfCSteeringNet):
    """Locked Temporal3 CfC with frozen backbone BN statistics and projection LN."""

    def __init__(self, *, num_frames: int = 3, hidden_dim: int = 32, use_pretrained: bool = True):
        super().__init__(num_frames=num_frames, hidden_dim=hidden_dim, use_pretrained=use_pretrained)
        self.temporal_norm = nn.LayerNorm(self.projection_dim)

    def _freeze_backbone_bn_stats(self) -> None:
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._freeze_backbone_bn_stats()
        return self

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        feature_seq = self.extract_feature_sequence(input_tensor)
        projected_seq = self.temporal_norm(self.temporal_projection(feature_seq))
        hidden = self._run_cfc(projected_seq)
        return self.reg_head(hidden)

