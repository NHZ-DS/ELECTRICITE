import numpy as np
import pandas as pd

from gridcast.conformal import HourlyConformal, conformal_quantile


def test_finite_sample_quantile():
    r = np.arange(1, 101)  # 1..100
    assert conformal_quantile(r, alpha=0.1) == 91  # ceil(101 * 0.9) = 91


def test_coverage_is_close_to_target():
    rng = np.random.default_rng(0)
    hours = pd.Series(np.tile(np.arange(24), 400))
    noise_scale = 1 + hours.to_numpy() / 10           # errors depend on the hour
    y_pred = np.zeros(len(hours))
    y_cal = rng.normal(0, noise_scale)
    conf = HourlyConformal(alpha=0.1).fit(hours, y_cal, y_pred)
    y_new = rng.normal(0, noise_scale)
    lo, hi = conf.interval(hours, y_pred)
    coverage = np.mean((y_new >= lo) & (y_new <= hi))
    assert 0.87 < coverage < 0.93
