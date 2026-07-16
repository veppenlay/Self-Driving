#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1 affordance model: reuse the locked Seq-CfC backbone but regress geometry, not steering.

Output = [steering, e_y, theta]:
- `e_y`, `theta` are the PRIMARY control affordances consumed by the frozen controller.
- `steering` is kept as a weak auxiliary head purely for monitoring / compatibility with the
  existing 2-channel deploy interface; it is NOT used for control in P1.

The backbone / temporal CfC are identical to `locked_model/steering_models.py`, so the strong
augmentation + geometric target is the only change, keeping the model ONNX/TRT friendly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_DIR = REPO_ROOT / "locked_model"
if str(LOCKED_DIR) not in sys.path:
    sys.path.insert(0, str(LOCKED_DIR))

from steering_models import RegressionSequenceCfCSteeringNet  # type: ignore  # noqa: E402


AFFORDANCE_OUTPUT_NAMES = ("steering", "e_y", "theta")


class AffordanceCfCNet(RegressionSequenceCfCSteeringNet):
    """Seq-CfC net with a 3-dim regression head [steering, e_y, theta]."""

    def __init__(self, *, num_aux_classes: int = 11, use_pretrained: bool = True, num_frames: int = 3):
        super().__init__(num_aux_classes=num_aux_classes, use_pretrained=use_pretrained, num_frames=num_frames)
        self.reg_head = nn.Linear(self.hidden_dim, len(AFFORDANCE_OUTPUT_NAMES))


__all__ = ["AffordanceCfCNet", "AFFORDANCE_OUTPUT_NAMES"]
