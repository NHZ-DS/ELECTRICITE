"""Hourly temperature from Open-Meteo, aggregated into one national temperature.

- Historical data: archive API (reanalysis, available with a few days of delay)
- Recent days and next days: forecast API

The national temperature is a population-weighted mean over large French urban areas.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import requests

from gridcast.config import CITIES, TIMEZONE

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

def _common_params() -> dict:
    return {
        "latitude": ",".join(str(c.lat) for c in CITIES),
        "longitude": ",".join(str(c.lon) for c in CITIES),
        "hourly": "temperature_2m",
        "timezone": "GMT",
    }


def _weighted_national(payload, variable: str = "temperature_2m") -> pd.DataFrame:
    """Turn the per-city Open-Meteo answer into one weighted national series."""
    locations = payload if isinstance(payload, list) else [payload]
    if len(locations) != len(CITIES):
        raise ValueError(f"Expected {len(CITIES)} locations, got {len(locations)}")
    series = []
    for city, loc in zip(CITIES, locations):
        hourly = loc["hourly"]
        s = pd.Series(hourly[variable], index=pd.to_datetime(hourly["time"]), dtype=float)
        series.append(s.rename(city.name))
    temps = pd.concat(series, axis=1)
    weights = np.array([c.weight for c in CITIES])
    # Weighted mean that ignores missing cities hour by hour
    mask = temps.notna().to_numpy()
    values = np.nan_to_num(temps.to_numpy())
    national = (values * weights).sum(axis=1) / (mask * weights).sum(axis=1)
    out = pd.DataFrame({"ts": temps.index, "temp_c": national})
    out["ts"] = out["ts"].dt.tz_localize("UTC").dt.tz_convert(TIMEZONE)
    return out.dropna().reset_index(drop=True)
def fetch_history(start: str, end: str, timeout: int = 120) -> pd.DataFrame:
    """Observed national temperature, dates as YYYY-MM-DD (both included)."""
    params = _common_params() | {"start_date": start, "end_date": end}
    resp = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return _weighted_national(resp.json())


def fetch_forecast(past_days: int = 7, forecast_days: int = 3, timeout: int = 60) -> pd.DataFrame:
    """Recent days and short-term forecast of the national temperature."""
    params = _common_params() | {"past_days": past_days, "forecast_days": forecast_days}
    resp = requests.get(FORECAST_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return _weighted_national(resp.json())



def fetch_previous_day_forecast(start: str, end: str, timeout: int = 120) -> pd.DataFrame:
    """National temperature as it was forecast one day before (24 to 47 h ahead).

    Used to evaluate the models in realistic conditions: at forecast time,
    nobody knows the real temperature of the next day, only its forecast.
    Open-Meteo archives these past forecasts since 2024.
    """
    variable = "temperature_2m_previous_day1"
    params = _common_params() | {"hourly": variable, "start_date": start, "end_date": end}
    resp = requests.get(PREVIOUS_RUNS_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return _weighted_national(resp.json(), variable)