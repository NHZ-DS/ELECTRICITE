"""REST API exposing the latest forecast, the backtest results and the monitoring status.

    uvicorn api.main:app --reload
"""
from __future__ import annotations

import json

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from gridcast.config import REPORTS_DIR
from gridcast.data.store import Store
from gridcast.pipeline import MAIN_MODEL

app = FastAPI(
    title="GridCast FR",
    description="Day-ahead forecast of French electricity consumption, with 90 % prediction intervals.",
    version="0.1.0",
)


def _read_json(name: str) -> dict:
    path = REPORTS_DIR / name
    if not path.exists():
        raise HTTPException(404, f"{name} not found. Run the corresponding CLI command first.")
    return json.loads(path.read_text())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/forecast/latest")
def latest_forecast(model: str = Query(MAIN_MODEL)) -> dict:
    store = Store(read_only=True)
    try:
        fc = store.load_forecasts(model)
    finally:
        store.close()
    if fc.empty:
        raise HTTPException(404, f"No forecast published for model {model}.")
    last = fc[fc["issued_at"] == fc["issued_at"].max()]
    return {
        "model": model,
        "issued_at": last["issued_at"].iloc[0].isoformat(),
        "unit": "MW",
        "interval": "90 % conformal prediction interval",
        "hours": [
            {"ts": r.ts.isoformat(), "yhat": round(r.yhat), "lower": round(r.lower), "upper": round(r.upper)}
            for r in last.itertuples()
        ],
    }


@app.get("/metrics/backtest")
def backtest_metrics() -> dict:
    return _read_json("backtest_report.json")


@app.get("/monitoring")
def monitoring() -> dict:
    return _read_json("monitoring.json")


@app.get("/forecast/history")
def forecast_history(days: int = Query(14, ge=1, le=90), model: str = Query(MAIN_MODEL)) -> list[dict]:
    """Published forecasts next to what really happened."""
    store = Store(read_only=True)
    try:
        fc = store.load_forecasts(model)
        data = store.load_dataset()
    finally:
        store.close()
    joined = fc.merge(data[["ts", "load_mw"]], on="ts", how="left")
    cutoff = joined["ts"].max() - pd.Timedelta(days=days)
    joined = joined[joined["ts"] >= cutoff]
    return [
        {"ts": r.ts.isoformat(), "yhat": round(r.yhat), "lower": round(r.lower), "upper": round(r.upper),
         "actual": None if pd.isna(r.load_mw) else round(r.load_mw)}
        for r in joined.itertuples()
    ]
