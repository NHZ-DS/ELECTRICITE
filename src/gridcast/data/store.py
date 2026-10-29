"""Small DuckDB storage layer.

Tables:
- consumption(ts TIMESTAMPTZ, load_mw DOUBLE, rte_forecast_mw DOUBLE)  RTE's own day-ahead forecast
- temperature(ts TIMESTAMPTZ, temp_c DOUBLE, kind VARCHAR)  kind = 'observed', 'forecast' or 'forecast_d1'
- forecasts(issued_at, ts, model, yhat, lower, upper)        every forecast ever published
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from gridcast.config import DB_PATH, TIMEZONE

SCHEMA = """
CREATE TABLE IF NOT EXISTS consumption (ts TIMESTAMPTZ PRIMARY KEY, load_mw DOUBLE, rte_forecast_mw DOUBLE);
CREATE TABLE IF NOT EXISTS temperature (ts TIMESTAMPTZ, kind VARCHAR, temp_c DOUBLE, PRIMARY KEY (ts, kind));
CREATE TABLE IF NOT EXISTS forecasts (
    issued_at TIMESTAMPTZ, ts TIMESTAMPTZ, model VARCHAR,
    yhat DOUBLE, lower DOUBLE, upper DOUBLE,
    PRIMARY KEY (issued_at, ts, model)
);
"""


class Store:
    def __init__(self, path: Path | str = DB_PATH, read_only: bool = False):
        """The API and the dashboard open the database read-only, so they never block the daily job."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path), read_only=read_only)
        self.con.execute(f"SET TimeZone = '{TIMEZONE}'")
        if not read_only:
            self.con.execute(SCHEMA)

    def _upsert(self, table: str, df: pd.DataFrame, keys: list[str]) -> int:
        if df.empty:
            return 0
        self.con.register("incoming", df)
        on = " AND ".join(f"t.{k} = incoming.{k}" for k in keys)
        self.con.execute(f"DELETE FROM {table} t USING incoming WHERE {on}")
        cols = ", ".join(df.columns)
        self.con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM incoming")
        self.con.unregister("incoming")
        return len(df)

    def save_consumption(self, df: pd.DataFrame) -> int:
        out = df[["ts", "load_mw"]].copy()
        out["rte_forecast_mw"] = df["rte_forecast_mw"] if "rte_forecast_mw" in df else float("nan")
        return self._upsert("consumption", out, ["ts"])

    def save_temperature(self, df: pd.DataFrame, kind: str) -> int:
        out = df[["ts", "temp_c"]].assign(kind=kind)
        return self._upsert("temperature", out, ["ts", "kind"])

    def save_forecast(self, df: pd.DataFrame) -> int:
        return self._upsert("forecasts", df[["issued_at", "ts", "model", "yhat", "lower", "upper"]],
                            ["issued_at", "ts", "model"])

    def load_dataset(self) -> pd.DataFrame:
        """Hourly table with consumption and the best available temperature.

        Observed temperature has priority over forecast temperature.
        """
        df = self.con.execute("""
            WITH temp AS (
                SELECT ts, arg_min(temp_c, CASE kind WHEN 'observed' THEN 0 ELSE 1 END) AS temp_c
                FROM temperature WHERE kind IN ('observed', 'forecast') GROUP BY ts
            )
            SELECT temp.ts, c.load_mw, temp.temp_c, c.rte_forecast_mw
            FROM temp LEFT JOIN consumption c USING (ts)
            ORDER BY temp.ts
        """).df()
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(TIMEZONE)
        return df
    def load_temperature(self, kind: str) -> pd.DataFrame:
        """One kind of temperature only, e.g. 'forecast_d1' (forecast issued one day before)."""
        df = self.con.execute("SELECT ts, temp_c FROM temperature WHERE kind = ? ORDER BY ts", [kind]).df()
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(TIMEZONE)
        return df

    def load_forecasts(self, model: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM forecasts" + (" WHERE model = ?" if model else "") + " ORDER BY issued_at, ts"
        df = self.con.execute(q, [model] if model else []).df()
        for col in ("issued_at", "ts"):
            df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert(TIMEZONE)
        return df

    def close(self) -> None:
        self.con.close()
