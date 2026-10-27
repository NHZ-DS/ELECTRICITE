"""Synthetic but realistic data, used by the tests and for an offline demo.

The generator mimics the main structure of French demand: a daily and weekly
profile, strong heating sensitivity below 15 °C, light cooling above 22 °C,
holidays, and noise. It lets the whole pipeline run without internet access.
It is never used to report results.
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from gridcast.config import TIMEZONE


def make_synthetic(start: str = "2022-01-01", days: int = 730, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=days * 24, freq="1h", tz=TIMEZONE)
    n = len(ts)
    doy = ts.dayofyear.to_numpy()
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()

    # Temperature: seasonal cycle + daily cycle + persistent weather anomalies (AR(1))
    anomaly = np.zeros(n)
    eps = rng.normal(0, 0.35, n)
    for i in range(1, n):
        anomaly[i] = 0.995 * anomaly[i - 1] + eps[i]
    temp = 12.5 - 8.5 * np.cos(2 * np.pi * (doy - 20) / 365.25) - 3.0 * np.cos(2 * np.pi * (hour - 15) / 24) + anomaly

    fr_holidays = holidays.France(years=range(ts.year.min(), ts.year.max() + 1))
    is_holiday = np.array([d in fr_holidays for d in ts.date])

    daily = 1.0 + 0.10 * np.sin(2 * np.pi * (hour - 7) / 24) + 0.06 * np.exp(-((hour - 19) ** 2) / 6)
    weekly = np.where(dow >= 5, 0.88, 1.0) * np.where(is_holiday, 0.85, 1.0)
    # Buildings have thermal inertia: demand follows a smoothed temperature
    temp_smooth = pd.Series(temp).ewm(halflife=24, min_periods=1).mean().to_numpy()
    felt = 0.7 * temp_smooth + 0.3 * temp
    heating = 2400 * np.clip(15 - felt, 0, None)          # MW per °C below 15 °C
    cooling = 600 * np.clip(felt - 22, 0, None)
    load = 47000 * daily * weekly + heating + cooling + rng.normal(0, 900, n)

    return pd.DataFrame({"ts": ts, "load_mw": load, "temp_c": temp})
