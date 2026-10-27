"""French national electricity consumption from RTE (eCO2mix), via the ODRE open-data portal.

Two datasets are combined:
- ``eco2mix-national-cons-def``: consolidated and final data (older periods)
- ``eco2mix-national-tr``: real-time data (recent period)

Raw data is published every 15 or 30 minutes, in MW. We resample it to hourly means.
"""
from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from gridcast.config import TIMEZONE

log = logging.getLogger(__name__)

BASE_URL = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/{dataset}/exports/csv"
DATASETS = ("eco2mix-national-cons-def", "eco2mix-national-tr")


def _fetch_dataset(dataset: str, start: str, end: str, timeout: int = 120) -> pd.DataFrame:
    params = {
        "select": "date_heure,consommation",
        "where": f"date_heure >= date'{start}' AND date_heure < date'{end}'",
        "timezone": TIMEZONE,
        "delimiter": ";",
    }
    resp = requests.get(BASE_URL.format(dataset=dataset), params=params, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), sep=";")
    if df.empty:
        return pd.DataFrame(columns=["ts", "load_mw"])
    missing = {"date_heure", "consommation"} - set(df.columns)
    if missing:
        raise ValueError(f"{dataset}: unexpected columns, missing {missing}. Got {list(df.columns)}")
    df = df.rename(columns={"date_heure": "ts", "consommation": "load_mw"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(TIMEZONE)
    df["source"] = dataset
    return df


def to_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Hourly mean consumption. Quarter-hours without a value are ignored."""
    if raw.empty:
        return pd.DataFrame(columns=["ts", "load_mw"])
    s = (raw.dropna(subset=["load_mw"])
            .set_index("ts")["load_mw"]
            .sort_index())
    hourly = s.resample("1h").mean().dropna()
    return hourly.rename("load_mw").reset_index()


def fetch_consumption(start: str, end: str) -> pd.DataFrame:
    """Hourly national consumption between ``start`` (included) and ``end`` (excluded), dates as YYYY-MM-DD.

    Consolidated data has priority. Real-time data only fills the hours it does not cover.
    """
    frames = []
    for ds in DATASETS:
        try:
            frames.append(_fetch_dataset(ds, start, end))
            log.info("%s: %d raw rows", ds, len(frames[-1]))
        except requests.RequestException as exc:
            log.warning("Could not download %s: %s", ds, exc)
    if not frames:
        raise RuntimeError("No consumption data could be downloaded from ODRE.")
    raw = pd.concat(frames, ignore_index=True)
    raw["priority"] = (raw["source"] == DATASETS[0]).map({True: 0, False: 1})
    raw = raw.sort_values(["ts", "priority"]).drop_duplicates("ts", keep="first")
    return to_hourly(raw)
