"""Rolling-origin backtest: the honest way to evaluate a forecasting model.

For each week of the test period:
  1. train on everything before the calibration window
  2. calibrate conformal intervals on the next ``calibration_days`` (unseen by the model)
  3. forecast the following week, and compare with what really happened
  4. compare with RTE's official day-ahead forecast on the same hours, when available

The model never sees the future, exactly as in production.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from gridcast.config import CONFIG, ModelConfig
from gridcast.conformal import HourlyConformal
from gridcast.features import FEATURES_ML, build_features
from gridcast.models import all_models

log = logging.getLogger(__name__)
RTE_BENCHMARK = "rte_official_d-1"

def point_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    err = y - yhat
    return {
        "mae_mw": float(np.mean(np.abs(err))),
        "rmse_mw": float(np.sqrt(np.mean(err ** 2))),
        "mape_pct": float(100 * np.mean(np.abs(err) / np.abs(y))),
    }


def interval_metrics(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    if np.isnan(lower).all():  # benchmark without interval
        return {"coverage_pct": np.nan, "mean_width_mw": np.nan}
    return {
        "coverage_pct": float(100 * np.mean((y >= lower) & (y <= upper))),
        "mean_width_mw": float(np.mean(upper - lower)),
    }
def with_forecast_temperature(df: pd.DataFrame, forecast_temp: pd.DataFrame) -> pd.DataFrame:
    """Replace the observed temperature by the one forecast the day before, where it exists."""
    fc = forecast_temp.set_index("ts")["temp_c"]
    out = df.copy()
    out["temp_c"] = out["ts"].map(fc).fillna(out["temp_c"])
    return out
def backtest(df: pd.DataFrame, test_weeks: int = 26, cfg: ModelConfig = CONFIG,
             models_factory=all_models,
             forecast_temp: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (predictions, summary).

    predictions: one row per (hour, model) with y, yhat, lower, upper
    summary: one row per model with point and interval metrics

    forecast_temp: temperature forecast one day before. When given, models still learn
    with the observed temperature, but calibrate and forecast with the forecast one,
    exactly like in production.
    """
    feats = build_features(df, cfg).dropna(subset=FEATURES_ML + ["load_mw"]).reset_index(drop=True)
    observed_temp = feats.set_index("ts")["temp_c"]
    if forecast_temp is None:
        feats_eval = feats
    else:
        feats_eval = (build_features(with_forecast_temperature(df, forecast_temp), cfg)
                      .assign(temp_forecast=lambda f: f["ts"].isin(forecast_temp["ts"]))
                      .dropna(subset=FEATURES_ML + ["load_mw"]).reset_index(drop=True))
    last_day = feats["ts"].dt.normalize().max()
    first_origin = last_day - pd.Timedelta(weeks=test_weeks) + pd.Timedelta(days=1)
    rows = []
    for w in range(test_weeks):
        origin = first_origin + pd.Timedelta(weeks=w)
        cal_start = origin - pd.Timedelta(days=cfg.calibration_days)
        train = feats[feats["ts"] < cal_start]
        cal = feats_eval[(feats_eval["ts"] >= cal_start) & (feats_eval["ts"] < origin)]
        test = feats_eval[(feats_eval["ts"] >= origin) & (feats_eval["ts"] < origin + pd.Timedelta(weeks=1))]
        if len(train) < 24 * 180 or test.empty:
            continue
        if forecast_temp is not None and not test["temp_forecast"].all():
            log.warning("Week of %s: forecast temperature missing on %d hours, observed used instead",
                        origin.date(), int((~test["temp_forecast"]).sum()))
        test_temp = test["ts"].map(observed_temp).to_numpy()  # cold-day metric uses the real temperature
        for model in models_factory(cfg):
            model.fit(train, train["load_mw"])
            conf = HourlyConformal(cfg.alpha).fit(cal["hour"], cal["load_mw"].to_numpy(), model.predict(cal))
            yhat = model.predict(test)
            lower, upper = conf.interval(test["hour"], yhat)
            rows.append(pd.DataFrame({
                "ts": test["ts"].to_numpy(), "model": model.name, "y": test["load_mw"].to_numpy(),
                "yhat": yhat, "lower": lower, "upper": upper, "temp_c": test["temp_c"].to_numpy(),
            }))
        if "rte_forecast_mw" in test and test["rte_forecast_mw"].notna().all():
            rows.append(pd.DataFrame({
                "ts": test["ts"].to_numpy(), "model": RTE_BENCHMARK, "y": test["load_mw"].to_numpy(),
                "yhat": test["rte_forecast_mw"].to_numpy(), "lower": np.nan, "upper": np.nan,
                "temp_c": test_temp,
            }))
        log.info("Backtest week %d/%d done (origin %s)", w + 1, test_weeks, origin.date())
    preds = pd.concat(rows, ignore_index=True)
    summary = summarize(preds)
    return preds, summary


def summarize(preds: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, g in preds.groupby("model", sort=False):
        y, yhat = g["y"].to_numpy(), g["yhat"].to_numpy()
        cold = g["temp_c"] < 5  # robustness check: how does the model behave on cold days?
        row = {"model": name} | point_metrics(y, yhat) | interval_metrics(y, g["lower"].to_numpy(), g["upper"].to_numpy())
        row["mape_cold_pct"] = point_metrics(y[cold], yhat[cold])["mape_pct"] if cold.any() else np.nan
        out.append(row)
    return pd.DataFrame(out).sort_values("mape_pct").reset_index(drop=True)
