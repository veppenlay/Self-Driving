#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Online classical-CV lane perception: BGR frame -> (e_y, theta) using the locked detector.

Reuses `locked_model/generate_2d_labels.py` detection primitives at TWO reference bands.
This is the P2 (diagnostic / safety-fallback) perception and also the temporal-hold fallback
branch for P1/P3 when the learned perception is unavailable.

IMPORTANT (plan discipline): the HSV thresholds live inside the locked detector and are NOT
to be re-tuned for the basement. The `width_prior` is a frozen source-domain constant passed
in from the labels summary; nothing here is scanned on the target scene.

Needs cv2 (runs on the GPU/dev environment). The temporal-hold logic is numpy-only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_DIR = REPO_ROOT / "locked_model"
if str(LOCKED_DIR) not in sys.path:
    sys.path.insert(0, str(LOCKED_DIR))

from .geometry import theta_from_centers


class CVLanePerception:
    def __init__(
        self,
        *,
        near_y_ratio: float = 0.54,
        far_y_ratio: float = 0.42,
        width_prior_px: float | None = None,
        hold_frames: int = 5,
    ):
        self.near_y_ratio = float(near_y_ratio)
        self.far_y_ratio = float(far_y_ratio)
        self.width_prior_px = width_prior_px
        self.hold_frames = int(hold_frames)
        self._g2d = None
        self._last: dict[str, Any] | None = None
        self._hold_count = 0

    def _primitives(self):
        if self._g2d is None:
            import generate_2d_labels as g2d  # type: ignore

            self._g2d = g2d
        return self._g2d

    def reset(self) -> None:
        self._last = None
        self._hold_count = 0

    def process(self, bgr) -> dict[str, Any]:
        g2d = self._primitives()
        near_raw = g2d._raw_detection(bgr, self.near_y_ratio)
        near = g2d._finalize_detection(near_raw, width_prior=self.width_prior_px)
        far_raw = g2d._raw_detection(bgr, self.far_y_ratio)
        far = g2d._finalize_detection(far_raw, width_prior=self.width_prior_px)

        e_y = near.get("eY")
        ey_quality = float(near.get("eyQuality") or 0.0)
        theta = 0.0
        theta_quality = 0.0
        if (
            near.get("laneCenterX") is not None
            and far.get("laneCenterX") is not None
            and ey_quality > 0
            and float(far.get("eyQuality") or 0.0) > 0
        ):
            theta = theta_from_centers(
                float(near["laneCenterX"]), float(near_raw["referenceY"]),
                float(far["laneCenterX"]), float(far_raw["referenceY"]),
            )
            theta_quality = float(min(ey_quality, float(far.get("eyQuality") or 0.0)))

        detected = e_y is not None and ey_quality > 0
        if detected:
            self._last = {"e_y": float(e_y), "theta": float(theta),
                          "ey_quality": ey_quality, "theta_quality": theta_quality}
            self._hold_count = 0
            return {**self._last, "status": str(near.get("status")), "source": "cv", "detected": True}

        # temporal hold fallback
        if self._last is not None and self._hold_count < self.hold_frames:
            self._hold_count += 1
            decay = 1.0 - self._hold_count / (self.hold_frames + 1)
            return {
                "e_y": self._last["e_y"], "theta": self._last["theta"],
                "ey_quality": self._last["ey_quality"] * decay,
                "theta_quality": self._last["theta_quality"] * decay,
                "status": str(near.get("status")), "source": "hold", "detected": False,
            }
        return {
            "e_y": 0.0, "theta": 0.0, "ey_quality": 0.0, "theta_quality": 0.0,
            "status": str(near.get("status")), "source": "blind", "detected": False,
        }


__all__ = ["CVLanePerception"]
