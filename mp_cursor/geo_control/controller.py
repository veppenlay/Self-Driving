#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen lateral controllers mapping geometric affordances to a steering/angular command.

Design rules (from the exploration plan):
- Gains/params are calibrated ONCE on the source domain (`dataset/26合并`) and then
  frozen. They are read-only at target/eval time; nothing here is meant to be scanned
  on `dataset/25地下室`.
- Pure numpy, no torch/cv2, so the harness and the on-car deploy can share the math.

Affordance conventions (matching `locked_model/generate_2d_labels.py`):
- `e_y`  : normalized lateral offset of the lane center w.r.t. image center,
           eY = (lane_center_x - image_center_x) / (width / 2). Range ~[-1, 1].
           Positive e_y => lane center is to the RIGHT of the car.
- `theta`: normalized lane heading (see `generate_geo_labels.py`). Positive theta
           => lane bends/points to the RIGHT going away from the car.

Sign relationship to `angular` (ROS: +angular_z turns LEFT) is NOT hardcoded; it is
absorbed by the calibrated gains, which may be negative.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _finite(value: float, fallback: float = 0.0) -> float:
    return float(value) if value is not None and math.isfinite(float(value)) else float(fallback)


@dataclass
class ControllerLimits:
    max_abs: float = 3.0          # matches deploy --max-angular default
    deadzone: float = 0.0
    slew_rate: float = 0.0        # max |Δangular| per step; 0 disables

    def clamp(self, value: float, prev: float | None) -> float:
        value = _finite(value, 0.0)
        if self.deadzone > 0.0 and abs(value) < self.deadzone:
            value = 0.0
        if self.slew_rate > 0.0 and prev is not None:
            lo, hi = prev - self.slew_rate, prev + self.slew_rate
            value = max(lo, min(hi, value))
        if self.max_abs > 0.0:
            value = max(-self.max_abs, min(self.max_abs, value))
        return value


class BaseController:
    kind: str = "base"

    def __init__(self, *, limits: ControllerLimits | None = None):
        self.limits = limits or ControllerLimits()
        self._prev_out: float | None = None
        self._prev_ey: float | None = None
        self._integral: float = 0.0

    def reset(self) -> None:
        self._prev_out = None
        self._prev_ey = None
        self._integral = 0.0

    def _raw(self, e_y: float, theta: float, v: float, dt: float) -> float:
        raise NotImplementedError

    def step(self, e_y: float, theta: float = 0.0, *, v: float = 0.5, dt: float = 1.0 / 30.0) -> float:
        raw = self._raw(_finite(e_y), _finite(theta), max(1e-3, _finite(v, 0.5)), max(1e-4, _finite(dt, 1.0 / 30.0)))
        out = self.limits.clamp(raw, self._prev_out)
        self._prev_out = out
        self._prev_ey = _finite(e_y)
        return out

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "BaseController":
        kind = str(data.get("kind", "pid")).lower()
        limits = ControllerLimits(**data.get("limits", {})) if data.get("limits") else ControllerLimits()
        if kind == "pid":
            return PIDLateralController(limits=limits, **data.get("gains", {}))
        if kind == "stanley":
            return StanleyController(limits=limits, **data.get("gains", {}))
        if kind == "pure_pursuit":
            return PurePursuitController(limits=limits, **data.get("gains", {}))
        raise ValueError(f"unknown controller kind: {kind}")


@dataclass
class _PIDGains:
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    ktheta: float = 0.0
    bias: float = 0.0
    integral_clamp: float = 1.0


class PIDLateralController(BaseController):
    """Linear feedback/feedforward controller: angular = kp*e_y + ki*∫e_y + kd*d(e_y) + ktheta*theta + bias.

    This is the natural controller for source-domain least-squares calibration: fitting
    kp/kd/ktheta/bias to human steering reduces to linear regression, after which the
    gains are frozen. It stays a fixed deterministic controller.
    """

    kind = "pid"

    def __init__(self, *, limits: ControllerLimits | None = None, **gains: float):
        super().__init__(limits=limits)
        self.g = _PIDGains(**gains)

    def _raw(self, e_y: float, theta: float, v: float, dt: float) -> float:
        d_ey = 0.0 if self._prev_ey is None else (e_y - self._prev_ey) / dt
        self._integral += e_y * dt
        clamp = abs(self.g.integral_clamp)
        if clamp > 0.0:
            self._integral = max(-clamp, min(clamp, self._integral))
        return (
            self.g.kp * e_y
            + self.g.ki * self._integral
            + self.g.kd * d_ey
            + self.g.ktheta * theta
            + self.g.bias
        )

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "gains": asdict(self.g), "limits": asdict(self.limits)}


@dataclass
class _StanleyGains:
    k: float = 1.0            # cross-track gain
    ktheta: float = 1.0       # heading gain
    scale: float = 1.0        # overall output scale (calibrated to command units)
    bias: float = 0.0
    soft: float = 0.2         # softening speed to avoid blow-up at low v


class StanleyController(BaseController):
    """Stanley-style law: out = scale * (ktheta*theta + atan2(k*e_y, v + soft)) + bias."""

    kind = "stanley"

    def __init__(self, *, limits: ControllerLimits | None = None, **gains: float):
        super().__init__(limits=limits)
        self.g = _StanleyGains(**gains)

    def _raw(self, e_y: float, theta: float, v: float, dt: float) -> float:
        cross = math.atan2(self.g.k * e_y, v + self.g.soft)
        return self.g.scale * (self.g.ktheta * theta + cross) + self.g.bias

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "gains": asdict(self.g), "limits": asdict(self.limits)}


@dataclass
class _PurePursuitGains:
    lookahead: float = 1.0    # normalized lookahead distance
    scale: float = 1.0        # maps curvature to command units (absorbs wheelbase/gearing)
    ktheta: float = 0.0       # optional heading feed-forward
    bias: float = 0.0


class PurePursuitController(BaseController):
    """Pure-pursuit curvature law: curvature = 2*e_y / Ld^2; out = scale*curvature + ktheta*theta + bias.

    `e_y` is used as the normalized lateral offset to a lookahead point; `lookahead` is
    the (frozen) normalized lookahead distance Ld.
    """

    kind = "pure_pursuit"

    def __init__(self, *, limits: ControllerLimits | None = None, **gains: float):
        super().__init__(limits=limits)
        self.g = _PurePursuitGains(**gains)

    def _raw(self, e_y: float, theta: float, v: float, dt: float) -> float:
        ld = max(1e-3, abs(self.g.lookahead))
        curvature = 2.0 * e_y / (ld * ld)
        return self.g.scale * curvature + self.g.ktheta * theta + self.g.bias

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "gains": asdict(self.g), "limits": asdict(self.limits)}


def save_controller(controller: BaseController, path: str | Path, *, meta: dict[str, Any] | None = None) -> None:
    payload = controller.to_dict()
    if meta:
        payload["meta"] = meta
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_controller(path: str | Path) -> BaseController:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return BaseController.from_dict(data)


__all__ = [
    "ControllerLimits",
    "BaseController",
    "PIDLateralController",
    "StanleyController",
    "PurePursuitController",
    "save_controller",
    "load_controller",
]
