"""Forecasting models, from the simplest to the hybrid one.

1. SeasonalNaive  : same hour last week. The baseline any model must beat.
2. ThermalModel   : physics-informed linear model. Consumption = calendar profile
                    + heating sensitivity (MW per °C below 15 °C) + cooling sensitivity.
                    Every coefficient has a physical meaning.
3. GBMModel       : LightGBM on calendar, weather and lag features. Purely data-driven.
4. HybridModel    : ThermalModel + LightGBM trained on its residuals. The physics gives
                    the structure, machine learning corrects what physics misses.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from gridcast.config import CONFIG, ModelConfig
from gridcast.features import FEATURES_ML


class SeasonalNaive:
    name = "seasonal_naive"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaive":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["load_lag_168"].to_numpy()


class ThermalModel:
    """Linear model with physically meaningful terms.

    Design matrix:
    - one intercept per (hour, day type): the daily load profile
    - heating degrees, with a slope that depends on the hour (heating is used more in the evening)
    - cooling degrees
    """
    name = "thermal_physics"

    def __init__(self, alpha: float = 1.0):
        self.reg = Ridge(alpha=alpha, fit_intercept=False)

    @staticmethod
    def _design(X: pd.DataFrame) -> np.ndarray:
        profile = pd.get_dummies(X["hour"].astype(str) + "_" + X["day_type"].astype(str))
        cols = [f"{h}_{d}" for h in range(24) for d in range(3)]
        profile = profile.reindex(columns=cols, fill_value=0).to_numpy(dtype=float)
        hour_block = pd.get_dummies(X["hour"] // 4).reindex(columns=range(6), fill_value=0).to_numpy(dtype=float)
        heating = hour_block * X["hdd"].to_numpy()[:, None]
        heating_now = X["hdd_now"].to_numpy()[:, None]
        cooling = X["cdd"].to_numpy()[:, None]
        return np.hstack([profile, heating, heating_now, cooling])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ThermalModel":
        self.reg.fit(self._design(X), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.reg.predict(self._design(X))

    def heating_sensitivity_mw_per_degree(self) -> float:
        """Average extra demand per °C of cold, in MW. The headline physical insight."""
        coef = self.reg.coef_
        # slow (building inertia) + immediate response to one degree of cold
        return float(np.mean(coef[72:78]) + coef[78])


class GBMModel:
    name = "lightgbm"

    def __init__(self, cfg: ModelConfig = CONFIG):
        self.model = lgb.LGBMRegressor(**cfg.lgbm_params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GBMModel":
        self.model.fit(X[FEATURES_ML], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[FEATURES_ML])


class HybridModel:
    name = "hybrid_physics_lgbm"

    def __init__(self, cfg: ModelConfig = CONFIG):
        self.physics = ThermalModel()
        params = dict(cfg.lgbm_params, n_estimators=400)
        self.correction = lgb.LGBMRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HybridModel":
        self.physics.fit(X, y)
        base = self.physics.predict(X)
        Xr = X[FEATURES_ML].assign(physics_pred=base)
        self.correction.fit(Xr, y - base)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base = self.physics.predict(X)
        Xr = X[FEATURES_ML].assign(physics_pred=base)
        return base + self.correction.predict(Xr)


def all_models(cfg: ModelConfig = CONFIG) -> list:
    return [SeasonalNaive(), ThermalModel(), GBMModel(cfg), HybridModel(cfg)]
