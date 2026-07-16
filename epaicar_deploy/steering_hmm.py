"""Causal steering-only HMM filter used by the locked deployment pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


class SteeringOnlyHMMOnlineFilter:
    """Stateful online filter compatible with ``apply_current_hmm_postprocess.py``."""

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("scheme") != "steering_only_hmm_online_filter" or payload.get("usesEy") is not False:
            raise ValueError(f"unsupported HMM model: {path}")

        self.path = path
        self.states = np.asarray(payload["states"], dtype=np.float64)
        self.transition = np.asarray(payload["transition"], dtype=np.float64)
        self.start_prob = np.asarray(payload["startProb"], dtype=np.float64)
        self.means = np.asarray(payload["emissionMeans"], dtype=np.float64)
        self.sigmas = np.asarray(payload["emissionSigmas"], dtype=np.float64)
        params = payload["params"]
        self.sigma_scale = float(params["sigma_scale"])
        self.switch_penalty = float(params["switch_penalty"])
        self._validate()
        self.transition = self._with_switch_penalty(self.transition, self.switch_penalty)
        self.sigma = np.clip(self.sigmas * self.sigma_scale, 1e-6, None)
        self.reset()

    def _validate(self) -> None:
        count = len(self.states)
        if count == 0 or self.transition.shape != (count, count):
            raise ValueError("invalid HMM transition/state dimensions")
        if self.start_prob.shape != (count,) or self.means.shape != (count,) or self.sigmas.shape != (count,):
            raise ValueError("invalid HMM emission/state dimensions")
        if not np.isfinite(self.transition).all() or not np.isfinite(self.start_prob).all():
            raise ValueError("HMM contains non-finite probabilities")

    @staticmethod
    def _with_switch_penalty(transition: np.ndarray, penalty: float) -> np.ndarray:
        adjusted = transition.astype(np.float64, copy=True)
        if penalty > 0:
            adjusted[~np.eye(adjusted.shape[0], dtype=bool)] *= math.exp(-penalty)
        return adjusted / np.clip(adjusted.sum(axis=1, keepdims=True), 1e-300, None)

    def reset(self) -> None:
        self.posterior = self.start_prob.astype(np.float64, copy=True)
        self.posterior /= np.clip(self.posterior.sum(), 1e-300, None)

    def apply(self, raw_steering: float) -> float:
        observation = float(raw_steering)
        if not math.isfinite(observation):
            observation = 0.0
        predicted = self.posterior @ self.transition
        emission = np.exp(-0.5 * ((observation - self.means) / self.sigma) ** 2) / self.sigma
        self.posterior = predicted * emission
        self.posterior /= np.clip(self.posterior.sum(), 1e-300, None)
        return float(self.states[int(np.argmax(self.posterior))])
