from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_MODEL_DIR = REPO_ROOT / "locked_model"
if str(LOCKED_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCKED_MODEL_DIR))

from steering_models import RegressionSteeringNet, _CfCLiteCell  # noqa: E402


class TDiscCfC(RegressionSteeringNet):
    """Eight-frame training model with an explicit state/delta_t step interface."""

    def __init__(
        self,
        *,
        num_frames: int = 8,
        projection_dim: int = 64,
        hidden_dim: int = 32,
        dt_dim: int = 8,
        use_pretrained: bool = True,
    ):
        super().__init__(use_pretrained=use_pretrained, in_channels=3)
        self.num_frames = int(num_frames)
        self.temporal_input_channels = 3 * self.num_frames
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)
        self.dt_dim = int(dt_dim)
        self.temporal_projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.projection_dim),
            nn.Dropout(0.10),
            nn.ReLU(inplace=True),
        )
        self.dt_encoder = nn.Sequential(
            nn.Linear(1, self.dt_dim),
            nn.Tanh(),
        )
        cell_input_dim = self.projection_dim + self.dt_dim
        self.cfc_cell = _CfCLiteCell(cell_input_dim, self.hidden_dim, recurrent_fanin=6, mask_seed=20260626)
        self.time_rate = nn.Linear(cell_input_dim, self.hidden_dim)
        self.reg_head = nn.Linear(self.hidden_dim, 2)

    def _step_from_feature(self, feature: torch.Tensor, hidden_prev: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        projected = self.temporal_projection(feature)
        delta_t = delta_t.reshape(-1, 1).clamp(0.0, 0.5)
        dt_feature = self.dt_encoder(delta_t)
        cell_input = torch.cat([projected, dt_feature], dim=1)
        candidate = self.cfc_cell(cell_input, hidden_prev)
        rate = F.softplus(self.time_rate(cell_input)) + 1e-3
        alpha = 1.0 - torch.exp(-rate * delta_t * 30.0)
        return hidden_prev + alpha * (candidate - hidden_prev)

    def forward_step(
        self,
        image_t: torch.Tensor,
        hidden_prev: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.extract_features(image_t)
        hidden_next = self._step_from_feature(feature, hidden_prev, delta_t)
        return self.reg_head(hidden_next), hidden_next

    def forward(self, input_tensor: torch.Tensor, delta_t: torch.Tensor | None = None) -> torch.Tensor:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.dim() != 4 or input_tensor.size(1) != self.temporal_input_channels:
            raise ValueError(
                f"expected input shape [B,{self.temporal_input_channels},H,W], got {tuple(input_tensor.shape)}"
            )
        batch_size, _, height, width = input_tensor.shape
        frames = input_tensor.reshape(batch_size, self.num_frames, 3, height, width)
        flat_frames = frames.reshape(batch_size * self.num_frames, 3, height, width)
        feature_seq = self.extract_features(flat_frames).reshape(batch_size, self.num_frames, self.feature_dim)
        if delta_t is None:
            delta_t = feature_seq.new_full((batch_size, self.num_frames, 1), 1.0 / 30.0)
        elif delta_t.dim() == 2:
            delta_t = delta_t.unsqueeze(-1)
        if tuple(delta_t.shape) != (batch_size, self.num_frames, 1):
            raise ValueError(f"expected delta_t [B,{self.num_frames},1], got {tuple(delta_t.shape)}")
        hidden = feature_seq.new_zeros(batch_size, self.hidden_dim)
        for frame_index in range(self.num_frames):
            hidden = self._step_from_feature(feature_seq[:, frame_index], hidden, delta_t[:, frame_index])
        return self.reg_head(hidden)

