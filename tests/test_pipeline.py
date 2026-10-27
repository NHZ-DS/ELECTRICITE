import pandas as pd

from gridcast import pipeline
from gridcast.data.store import Store
from gridcast.monitoring import coverage_check


def test_forecast_day_end_to_end(tmp_path, synthetic):
    store = Store(tmp_path / "test.duckdb")
    store.save_consumption(synthetic)
    store.save_temperature(synthetic, "observed")
    df = store.load_dataset()
    day = pd.Timestamp("2024-05-01", tz="Europe/Paris")
    fc = pipeline.forecast_day(df, day)
    hybrid = fc[fc["model"] == pipeline.MAIN_MODEL]
    assert len(hybrid) == 24
    assert (hybrid["lower"] < hybrid["yhat"]).all() and (hybrid["yhat"] < hybrid["upper"]).all()
    store.save_forecast(fc)
    assert len(store.load_forecasts(pipeline.MAIN_MODEL)) == 24
    store.close()


def test_coverage_alert_triggers():
    recent = pd.DataFrame({"y": [10, 10, 10, 10], "lower": [0, 11, 11, 11], "upper": [20, 12, 12, 12]})
    check = coverage_check(recent, target=90)
    assert check.alert is True
