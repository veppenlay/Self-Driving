#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""On-car geometry controller: affordances (e_y, theta) -> raw angular command.

Self-contained mirror of `mp_cursor/geo_control/controller.py` so the deploy tree has no
cross-package dependency. It loads the SAME frozen controller JSON produced by
`calibrate_controller.py` (calibrated once on the source domain, read-only here).

The server clamps/limits the output via its existing PostProcessor, so this module only
implements the raw control law. numpy-free, py3.
"""

from __future__ import annotations

import json
import math


class GeometryController(object):
    def __init__(self, kind, gains):
        self.kind = str(kind).lower()
        self.gains = dict(gains or {})
        self._prev_ey = None
        self._integral = 0.0

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as fh:
            data = json.load(fh)
        return cls(data.get("kind", "pid"), data.get("gains", {}))

    def reset(self):
        self._prev_ey = None
        self._integral = 0.0

    def _g(self, key, default=0.0):
        try:
            return float(self.gains.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def step(self, e_y, theta=0.0, v=0.5, dt=1.0 / 30.0):
        e_y = float(e_y) if _finite(e_y) else 0.0
        theta = float(theta) if _finite(theta) else 0.0
        v = max(1e-3, float(v) if _finite(v) else 0.5)
        dt = max(1e-4, float(dt) if _finite(dt) else 1.0 / 30.0)

        if self.kind == "pid":
            d_ey = 0.0 if self._prev_ey is None else (e_y - self._prev_ey) / dt
            self._integral += e_y * dt
            clamp = abs(self._g("integral_clamp", 1.0))
            if clamp > 0.0:
                self._integral = max(-clamp, min(clamp, self._integral))
            out = (self._g("kp") * e_y + self._g("ki") * self._integral
                   + self._g("kd") * d_ey + self._g("ktheta") * theta + self._g("bias"))
        elif self.kind == "stanley":
            cross = math.atan2(self._g("k", 1.0) * e_y, v + self._g("soft", 0.2))
            out = self._g("scale", 1.0) * (self._g("ktheta") * theta + cross) + self._g("bias")
        elif self.kind == "pure_pursuit":
            ld = max(1e-3, abs(self._g("lookahead", 1.0)))
            curvature = 2.0 * e_y / (ld * ld)
            out = self._g("scale", 1.0) * curvature + self._g("ktheta") * theta + self._g("bias")
        else:
            out = e_y

        self._prev_ey = e_y
        if not _finite(out):
            out = 0.0
        return float(out)


def _finite(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(v) or math.isinf(v))


__all__ = ["GeometryController"]
