import numpy as np

from gridcast.evaluate import backtest


def test_forecast_weather_makes_the_backtest_harder(synthetic):
    """With a noisy temperature forecast, errors must grow and the cold metric must use real temperature."""
    rng = np.random.default_rng(0)
    forecast_temp = synthetic[["ts", "temp_c"]].assign(temp_c=lambda d: d["temp_c"] + rng.normal(0, 2.0, len(d)))
    _, observed = backtest(synthetic, test_weeks=2)
    preds, forecast = backtest(synthetic, test_weeks=2, forecast_temp=forecast_temp)
    mape = lambda s: s.set_index("model").loc["hybrid_physics_lgbm", "mape_pct"]
    assert mape(forecast) > mape(observed)
    real = synthetic.set_index("ts")["temp_c"]
    hybrid = preds[preds["model"] == "hybrid_physics_lgbm"]
    np.testing.assert_allclose(hybrid["temp_c"].to_numpy(), hybrid["ts"].map(real).to_numpy())