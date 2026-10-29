"""Command line entry point.

    python -m gridcast.cli demo                 # synthetic data, no internet needed
    python -m gridcast.cli ingest-forecast-weather --start 2025-01-01
    python -m gridcast.cli backtest --weeks 26 --weather forecast
    python -m gridcast.cli replay --days 30
    python -m gridcast.cli forecast
    python -m gridcast.cli monitor
    python -m gridcast.cli daily                # ingest last days + forecast + monitor
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta

import pandas as pd

from gridcast import pipeline
from gridcast.data.store import Store


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="gridcast")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="load synthetic data (offline)")
    pi = sub.add_parser("ingest", help="download RTE consumption and Open-Meteo temperature")
    pi.add_argument("--start", default="2021-01-01")
    pi.add_argument("--end", default=None)
    pw = sub.add_parser("ingest-forecast-weather", help="download the temperature forecast issued one day before")
    pw.add_argument("--start", default="2025-01-01")
    pw.add_argument("--end", default=None)
    pb = sub.add_parser("backtest", help="rolling-origin evaluation of all models")
    pb.add_argument("--weeks", type=int, default=26)
    pb.add_argument("--weather", choices=["observed", "forecast"], default="observed",
                    help="temperature used to forecast: observed (optimistic) or forecast issued the day before")
    pr = sub.add_parser("replay", help="re-issue past forecasts to seed monitoring")
    pr.add_argument("--days", type=int, default=30)
    pf = sub.add_parser("forecast", help="forecast the next day")
    pf.add_argument("--day", default=None, help="YYYY-MM-DD, default: first day without data")
    sub.add_parser("monitor", help="accuracy, calibration and drift checks")
    sub.add_parser("daily", help="scheduled job: ingest + forecast + monitor")
    args = p.parse_args(argv)

    store = Store()
    try:
        if args.cmd == "demo":
            print(pipeline.load_demo(store))
        elif args.cmd == "ingest":
            print(pipeline.ingest(store, args.start, args.end))
        elif args.cmd == "ingest-forecast-weather":
            print(pipeline.ingest_forecast_weather(store, args.start, args.end))
        elif args.cmd == "backtest":
            with pd.option_context("display.width", 160, "display.max_columns", None, "display.float_format", "{:,.2f}".format):
                print(pipeline.run_backtest(store, test_weeks=args.weeks, weather_mode=args.weather))
        elif args.cmd == "replay":
            print(f"{pipeline.replay(store, args.days)} forecast rows published")
        elif args.cmd == "forecast":
            fc = pipeline.forecast(store, args.day)
            out = fc[fc["model"] == pipeline.MAIN_MODEL]
            print(out.round({"yhat": 0, "lower": 0, "upper": 0}).to_string(index=False))
        elif args.cmd == "monitor":
            print(json.dumps(pipeline.monitor(store), indent=2))
        elif args.cmd == "daily":
            start = (date.today() - timedelta(days=10)).isoformat()
            print(pipeline.ingest(store, start))
            pipeline.forecast(store)
            print(json.dumps(pipeline.monitor(store), indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
