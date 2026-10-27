"""Split conformal prediction intervals, conditional on the hour of the day.

Idea: on a calibration window the model has never been trained on, we measure
how wrong it is. The (1 - alpha) quantile of the absolute errors gives a margin
that covers the truth (1 - alpha) of the time, without any assumption on the
error distribution. Errors differ between night and evening peak, so we compute
one margin per hour of the day.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def conformal_quantile(abs_residuals: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected quantile: ceil((n + 1)(1 - alpha)) / n."""
    r = np.sort(np.asarray(abs_residuals, dtype=float))
    n = len(r)
    if n == 0:
        return float("nan")
    k = math.ceil((n + 1) * (1 - alpha))
    return float(r[min(k, n) - 1])


class HourlyConformal:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.margins: dict[int, float] = {}

    def fit(self, hours: pd.Series, y_true: np.ndarray, y_pred: np.ndarray) -> "HourlyConformal":
        res = pd.DataFrame({"hour": np.asarray(hours), "r": np.abs(np.asarray(y_true) - np.asarray(y_pred))})
        global_q = conformal_quantile(res["r"].to_numpy(), self.alpha)
        self.margins = {
            h: conformal_quantile(g["r"].to_numpy(), self.alpha) if len(g) >= 20 else global_q
            for h, g in res.groupby("hour")
        }
        self._fallback = global_q
        return self

    def interval(self, hours: pd.Series, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = np.array([self.margins.get(int(h), self._fallback) for h in np.asarray(hours)])
        return y_pred - m, y_pred + m
