#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared geometric helpers (numpy-only) reused by labeling, CV perception and seg geometry.

Keeps the `e_y` / `theta` conventions consistent everywhere:
- e_y   = (lane_center_x - image_center_x) / (width / 2), positive => lane center to the RIGHT.
- theta = normalized lane heading from a NEAR band to a FAR band (far band is higher up in the
          image). Positive theta => the lane points/bends to the RIGHT going away from the car.
          Normalized by pi/2 so it lives in roughly [-1, 1].
"""

from __future__ import annotations

import math


def e_y_from_center(lane_center_x: float, image_width: float) -> float:
    return (float(lane_center_x) - float(image_width) / 2.0) / (float(image_width) / 2.0)


def theta_from_centers(
    near_center_x: float,
    near_y: float,
    far_center_x: float,
    far_y: float,
) -> float:
    """Heading from near->far lane centers, normalized to ~[-1, 1].

    near_y is lower in the image (larger row index) than far_y.
    """
    dx = float(far_center_x) - float(near_center_x)
    dy = float(near_y) - float(far_y)  # positive when far band is above near band
    if abs(dy) < 1e-6:
        dy = 1e-6
    angle = math.atan2(dx, dy)  # radians; 0 => straight ahead
    return max(-1.0, min(1.0, angle / (math.pi / 2.0)))


__all__ = ["e_y_from_center", "theta_from_centers"]
