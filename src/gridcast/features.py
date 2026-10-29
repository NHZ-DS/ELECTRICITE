"""Feature engineering for day-ahead forecasting.

Rule: every feature must be known when the forecast is issued (midnight before
the target day). Consumption lags are therefore >= 24 h, and temperature comes
from the weather forecast for future hours.
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from gridcast.config import CONFIG, ModelConfig


def _french_holidays(ts: pd.Series) -> np.ndarray:
    years = range(ts.dt.year.min() - 1, ts.dt.year.max() + 2)
    fr = holidays.France(years=years)
    return ts.dt.date.map(lambda d: d in fr).to_numpy()


def calendar_features(ts: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=ts.index)
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    doy = ts.dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    hol = _french_holidays(ts)
    out["is_holiday"] = hol.astype(int)
    # Bridge days ("ponts"): a working day squeezed between a holiday and a weekend
    day = ts.dt.normalize()
    hol_days = set(day[hol])
    prev_hol = (day - pd.Timedelta(days=1)).isin(hol_days)
    next_hol = (day + pd.Timedelta(days=1)).isin(hol_days)
    out["is_bridge"] = (((out["dow"] == 0) & next_hol) | ((out["dow"] == 4) & prev_hol)).astype(int)
    # Day type used by the physical model: 0 weekday, 1 Saturday, 2 Sunday or holiday
    out["day_type"] = np.select([out["is_holiday"].eq(1) | out["dow"].eq(6), out["dow"].eq(5)], [2, 1], 0)
    return out


def thermal_features(temp: pd.Series, cfg: ModelConfig = CONFIG) -> pd.DataFrame:
    """Heating and cooling degrees, plus a smoothed temperature for building inertia."""
    out = pd.DataFrame(index=temp.index)
    out["temp_c"] = temp
    # Buildings react with a delay: exponential smoothing with a 24 h half-life
    out["temp_smooth"] = temp.ewm(halflife=24, min_periods=1).mean()
    out["hdd"] = np.clip(cfg.heating_base - out["temp_smooth"], 0, None)
    out["hdd_now"] = np.clip(cfg.heating_base - temp, 0, None)
    out["cdd"] = np.clip(out["temp_smooth"] - cfg.cooling_base, 0, None)
    return out


def lag_features(load: pd.Series, cfg: ModelConfig = CONFIG) -> pd.DataFrame:
    out = pd.DataFrame(index=load.index)
    for lag in cfg.lags:
        out[f"load_lag_{lag}"] = load.shift(lag)
    out["load_mean_24_lag24"] = load.shift(24).rolling(24, min_periods=12).mean()
    return out


def build_features(df: pd.DataFrame, cfg: ModelConfig = CONFIG) -> pd.DataFrame:
    """Expects an hourly frame with columns ts, load_mw (may be NaN on future hours), temp_c."""
    df = df.sort_values("ts").reset_index(drop=True)
    feats = pd.concat(
        [df[["ts", "load_mw"] + (["rte_forecast_mw"] if "rte_forecast_mw" in df else [])],
         calendar_features(df["ts"]),
         thermal_features(df["temp_c"], cfg),
         lag_features(df["load_mw"], cfg)],
        axis=1,
    )
    return feats


FEATURES_ML = [
    "hour", "dow", "month", "doy_sin", "doy_cos", "is_holiday", "is_bridge", "day_type",
    "temp_c", "temp_smooth", "hdd", "hdd_now", "cdd",
    "load_lag_24", "load_lag_48", "load_lag_168", "load_mean_24_lag24",
]
