import numpy as np

from gridcast import registry
from gridcast.conformal import HourlyConformal
from gridcast.features import FEATURES_ML, build_features
from gridcast.models import ThermalModel


def test_saved_model_gives_the_same_forecast(tmp_path, monkeypatch, synthetic):
    monkeypatch.setattr(registry, "ARTIFACTS_DIR", tmp_path)
    f = build_features(synthetic).dropna(subset=FEATURES_ML + ["load_mw"])
    model = ThermalModel().fit(f, f["load_mw"])
    conf = HourlyConformal(0.1).fit(f["hour"], f["load_mw"].to_numpy(), model.predict(f))

    version = registry.save_model(model, conf, {"note": "test"})
    loaded = registry.load_latest(model.name)

    assert (tmp_path / f"{model.name}_{version}.json").exists()
    np.testing.assert_allclose(loaded["model"].predict(f), model.predict(f))