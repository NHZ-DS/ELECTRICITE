import numpy as np

from gridcast.features import FEATURES_ML, build_features
from gridcast.models import HybridModel, SeasonalNaive, ThermalModel


def _split(synthetic):
    f = build_features(synthetic).dropna(subset=FEATURES_ML + ["load_mw"])
    cut = f["ts"].max() - np.timedelta64(30, "D")
    return f[f["ts"] < cut], f[f["ts"] >= cut]


def test_thermal_model_recovers_the_true_heating_sensitivity(synthetic):
    """The synthetic data uses 2400 MW per °C: the physics model must find it back."""
    train, _ = _split(synthetic)
    m = ThermalModel().fit(train, train["load_mw"])
    assert abs(m.heating_sensitivity_mw_per_degree() - 2400) < 300


def test_hybrid_beats_the_naive_baseline(synthetic):
    train, test = _split(synthetic)
    y = test["load_mw"].to_numpy()
    naive = np.mean(np.abs(y - SeasonalNaive().predict(test)))
    hybrid = np.mean(np.abs(y - HybridModel().fit(train, train["load_mw"]).predict(test)))
    assert hybrid < 0.6 * naive
