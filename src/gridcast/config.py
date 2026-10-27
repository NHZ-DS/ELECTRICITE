"""Project-wide settings.

Everything that a user may want to tune lives here, so the rest of the code
stays free of magic numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
ARTIFACTS_DIR = ROOT / "artifacts"
DB_PATH = DATA_DIR / "gridcast.duckdb"

TIMEZONE = "Europe/Paris"


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float
    weight: float  # approximate population of the urban area, in millions


# National temperature = population-weighted mean of large urban areas.
# Electricity demand follows where people live, not the geographic centre.
CITIES: tuple[City, ...] = (
    City("Paris", 48.857, 2.352, 13.0),
    City("Lyon", 45.764, 4.836, 2.3),
    City("Marseille", 43.296, 5.370, 1.9),
    City("Toulouse", 43.605, 1.444, 1.5),
    City("Lille", 50.629, 3.057, 1.5),
    City("Bordeaux", 44.838, -0.579, 1.4),
    City("Nantes", 47.218, -1.554, 1.0),
    City("Nice", 43.710, 7.262, 1.0),
    City("Strasbourg", 48.573, 7.752, 0.8),
    City("Rennes", 48.117, -1.678, 0.8),
)


@dataclass(frozen=True)
class ModelConfig:
    # Thermal physics: below this temperature (°C) France heats, above the
    # cooling threshold it starts to cool. Classic degree-day thresholds.
    heating_base: float = 15.0
    cooling_base: float = 22.0
    # Lags in hours. Day-ahead forecasting: we only use lags >= 24 h, so every
    # feature is known at forecast time.
    lags: tuple[int, ...] = (24, 48, 168)
    # Conformal prediction
    alpha: float = 0.10            # 90 % prediction intervals
    calibration_days: int = 56     # rolling calibration window
    lgbm_params: dict = field(default_factory=lambda: {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "verbose": -1,
    })


CONFIG = ModelConfig()
