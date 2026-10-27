"""Production monitoring: is the model still trustworthy today?

Three independent checks, each with a clear alert rule:
1. Accuracy   : rolling MAPE of published forecasts vs the backtest reference
2. Calibration: empirical coverage of the 90 % intervals on recent days
3. Input drift: are recent temperatures outside what the model was trained on?
                (out-of-distribution share) and has the error distribution shifted?
                (Kolmogorov-Smirnov test between reference and recent residuals)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class Check:
    name: str
    value: float
    threshold: float
    alert: bool
    message: str

    def __post_init__(self):
        # numpy types are not JSON serialisable
        self.value, self.threshold, self.alert = float(self.value), float(self.threshold), bool(self.alert)


def accuracy_check(recent: pd.DataFrame, reference_mape: float, factor: float = 1.5) -> Check:
    mape = float(100 * np.mean(np.abs(recent["y"] - recent["yhat"]) / recent["y"]))
    limit = factor * reference_mape
    return Check("rolling_mape_pct", mape, limit, mape > limit,
                 f"MAPE {mape:.2f} % on recent days (alert above {limit:.2f} %)")


def coverage_check(recent: pd.DataFrame, target: float = 90.0, tolerance: float = 10.0) -> Check:
    cov = float(100 * np.mean((recent["y"] >= recent["lower"]) & (recent["y"] <= recent["upper"])))
    limit = target - tolerance
    return Check("interval_coverage_pct", cov, limit, cov < limit,
                 f"{cov:.1f} % of real values inside the {target:.0f} % intervals (alert below {limit:.0f} %)")


def temperature_ood_check(train_temp: pd.Series, recent_temp: pd.Series, max_share: float = 0.05) -> Check:
    lo, hi = np.nanpercentile(train_temp, [0.5, 99.5])
    share = float(np.mean((recent_temp < lo) | (recent_temp > hi)))
    return Check("temperature_out_of_range_share", share, max_share, share > max_share,
                 f"{100 * share:.1f} % of recent hours outside the training range [{lo:.1f}, {hi:.1f}] °C")


def residual_drift_check(ref_residuals: np.ndarray, recent_residuals: np.ndarray, p_value: float = 0.01) -> Check:
    stat = ks_2samp(ref_residuals, recent_residuals)
    return Check("residual_ks_pvalue", float(stat.pvalue), p_value, stat.pvalue < p_value,
                 f"KS test p-value {stat.pvalue:.3g} between reference and recent errors (alert below {p_value})")


def run_checks(recent: pd.DataFrame, reference: pd.DataFrame, train_temp: pd.Series,
               reference_mape: float, target_coverage: float = 90.0) -> list[dict]:
    """``recent`` and ``reference``: frames with y, yhat, lower, upper, temp_c."""
    checks = [
        accuracy_check(recent, reference_mape),
        coverage_check(recent, target_coverage),
        temperature_ood_check(train_temp, recent["temp_c"]),
        residual_drift_check((reference["y"] - reference["yhat"]).to_numpy(),
                             (recent["y"] - recent["yhat"]).to_numpy()),
    ]
    return [asdict(c) for c in checks]
