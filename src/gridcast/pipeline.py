"""High-level steps used by the CLI, the API and the scheduled job."""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta

import pandas as pd
from gridcast.registry import save_model
from gridcast.config import CONFIG, REPORTS_DIR, TIMEZONE, ModelConfig
from gridcast.conformal import HourlyConformal
from gridcast.data import rte, weather
from gridcast.data.store import Store
from gridcast.data.synthetic import make_synthetic
from gridcast.evaluate import backtest
from gridcast.features import FEATURES_ML, build_features
from gridcast.models import ThermalModel, all_models
from gridcast.monitoring import run_checks

log = logging.getLogger(__name__)
MAIN_MODEL = "hybrid_physics_lgbm"


# ---------------------------------------------------------------- ingestion
def ingest(store: Store, start: str, end: str | None = None) -> dict:
    """Download consumption and temperature, then upsert them in the database."""
    today = date.today()
    end = end or (today + timedelta(days=1)).isoformat()
    n_load = store.save_consumption(rte.fetch_consumption(start, end))
    # The reanalysis archive lags a few days behind: stop it 6 days ago,
    # the forecast API covers the recent days and the next ones.
    hist_end = min(date.fromisoformat(end) - timedelta(days=1), today - timedelta(days=6))
    n_obs = 0
    if date.fromisoformat(start) <= hist_end:
        n_obs = store.save_temperature(weather.fetch_history(start, hist_end.isoformat()), "observed")
    n_fc = store.save_temperature(weather.fetch_forecast(past_days=10, forecast_days=3), "forecast")
    return {"consumption_rows": n_load, "temperature_observed_rows": n_obs, "temperature_forecast_rows": n_fc}


def load_demo(store: Store, days: int = 760) -> dict:
    """Fill the database with synthetic data ending today, to try the project offline."""
    today = pd.Timestamp.now(tz=TIMEZONE).normalize()
    start = (today - pd.Timedelta(days=days - 3)).strftime("%Y-%m-%d")
    df = make_synthetic(start=start, days=days)
    past = df[df["ts"] < today]
    store.save_consumption(past)
    store.save_temperature(past, "observed")
    future = df[df["ts"] >= today].copy()
    future["temp_c"] += 0.8  # a weather forecast is never perfect
    store.save_temperature(future, "forecast")
    return {"hours_with_consumption": len(past), "forecast_hours": len(future)}


# ---------------------------------------------------------------- backtest
def run_backtest(store: Store, test_weeks: int = 26, cfg: ModelConfig = CONFIG) -> pd.DataFrame:
    df = store.load_dataset()
    preds, summary = backtest(df.dropna(subset=["load_mw"]), test_weeks=test_weeks, cfg=cfg)
    REPORTS_DIR.mkdir(exist_ok=True)
    preds.to_parquet(REPORTS_DIR / "backtest_predictions.parquet", index=False)
    summary.to_csv(REPORTS_DIR / "backtest_summary.csv", index=False)

    feats = build_features(df.dropna(subset=["load_mw"]), cfg).dropna(subset=FEATURES_ML)
    sensitivity = ThermalModel().fit(feats, feats["load_mw"]).heating_sensitivity_mw_per_degree()
    report = {"summary": summary.to_dict(orient="records"),
              "heating_sensitivity_mw_per_degC": sensitivity,
              "test_weeks": test_weeks}
    (REPORTS_DIR / "backtest_report.json").write_text(json.dumps(report, indent=2))
    return summary


# ---------------------------------------------------------------- forecasting
def forecast_day(df: pd.DataFrame, target_day: pd.Timestamp, cfg: ModelConfig = CONFIG,
                 save: bool = False) -> pd.DataFrame:
    """Forecast the 24 hours of ``target_day`` using only information available at its midnight."""
    end = target_day + pd.Timedelta(days=1)
    visible = df[df["ts"] < end].copy()
    visible.loc[visible["ts"] >= target_day, "load_mw"] = float("nan")  # the future is unknown
    feats = build_features(visible, cfg)
    target = feats[feats["ts"] >= target_day].dropna(subset=FEATURES_ML)
    history = feats[feats["ts"] < target_day].dropna(subset=FEATURES_ML + ["load_mw"])
    cal_start = target_day - pd.Timedelta(days=cfg.calibration_days)
    train, cal = history[history["ts"] < cal_start], history[history["ts"] >= cal_start]
    if target.empty or len(train) < 24 * 180:
        raise ValueError(f"Not enough data to forecast {target_day.date()}")
    out = []
    for model in all_models(cfg):
        model.fit(train, train["load_mw"])
        conf = HourlyConformal(cfg.alpha).fit(cal["hour"], cal["load_mw"].to_numpy(), model.predict(cal))
        yhat = model.predict(target)
        lower, upper = conf.interval(target["hour"], yhat)
        if save and model.name == MAIN_MODEL:
            save_model(model, conf, {
                "trained_at": pd.Timestamp.now(tz=TIMEZONE),
                "train_start": train["ts"].min(),
                "train_end": train["ts"].max(),
                "train_rows": len(train),
                "target_day": target_day.date(),
                "backtest_mape_pct": backtest_mape(model.name),
            })
        out.append(pd.DataFrame({"issued_at": target_day, "ts": target["ts"].to_numpy(), "model": model.name,
                                 "yhat": yhat, "lower": lower, "upper": upper}))
    return pd.concat(out, ignore_index=True)


def next_target_day(df: pd.DataFrame) -> pd.Timestamp:
    """First day with no consumption data yet."""
    last_known = df.dropna(subset=["load_mw"])["ts"].max()
    return (last_known + pd.Timedelta(hours=1)).normalize()


def forecast(store: Store, target_day: str | None = None) -> pd.DataFrame:
    df = store.load_dataset()
    day = pd.Timestamp(target_day, tz=TIMEZONE) if target_day else next_target_day(df)
    fc = forecast_day(df, day, save=True)
    store.save_forecast(fc)
    return fc
def backtest_mape(model_name: str) -> float | None:
    """Backtest error of this model, if a backtest has been run."""
    path = REPORTS_DIR / "backtest_summary.csv"
    if not path.exists():
        return None
    summary = pd.read_csv(path)
    row = summary[summary["model"] == model_name]
    return float(row["mape_pct"].iloc[0]) if not row.empty else None

def replay(store: Store, days: int = 30) -> int:
    """Re-issue the forecasts of the last ``days`` days, as they would have been published.

    Useful to start monitoring with a history. Each day only sees the data available at its midnight.
    """
    df = store.load_dataset()
    last = next_target_day(df)
    n = 0
    for k in range(days, 0, -1):
        n += store.save_forecast(forecast_day(df, last - pd.Timedelta(days=k)))
    return n


# ---------------------------------------------------------------- monitoring
def monitor(store: Store, model: str = MAIN_MODEL, recent_days: int = 14) -> dict:
    fc = store.load_forecasts(model)
    data = store.load_dataset()
    joined = fc.merge(data[["ts", "load_mw", "temp_c"]], on="ts").dropna(subset=["load_mw"])
    joined = joined.sort_values("issued_at").drop_duplicates("ts", keep="last").rename(columns={"load_mw": "y"})
    cutoff = joined["ts"].max().normalize() - pd.Timedelta(days=recent_days)
    recent = joined[joined["ts"] >= cutoff]
    bt_path = REPORTS_DIR / "backtest_predictions.parquet"
    if recent.empty or not bt_path.exists():
        raise RuntimeError("Run `backtest` and publish forecasts (`forecast` or `replay`) before monitoring.")
    reference = pd.read_parquet(bt_path)
    reference = reference[reference["model"] == model]
    ref_mape = float(100 * (abs(reference["y"] - reference["yhat"]) / reference["y"]).mean())
    checks = run_checks(recent, reference, data["temp_c"], ref_mape, target_coverage=100 * (1 - CONFIG.alpha))
    report = {"model": model, "recent_days": recent_days, "from": str(cutoff.date()),
              "status": "ALERT" if any(c["alert"] for c in checks) else "OK", "checks": checks}
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "monitoring.json").write_text(json.dumps(report, indent=2))
    return report
