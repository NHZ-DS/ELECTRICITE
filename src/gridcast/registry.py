"""Model registry: every trained model is saved with its metadata.

Each forecast can then be traced back to the exact model that produced it.
"""
from __future__ import annotations

import json
from datetime import datetime

import joblib

from gridcast.config import ARTIFACTS_DIR


def save_model(model, conformal, metadata: dict) -> str:
    """Save the model and its conformal margins, plus a JSON file describing them. Returns the version."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{model.name}_{version}"
    joblib.dump({"model": model, "conformal": conformal}, ARTIFACTS_DIR / f"{name}.joblib")
    info = {"version": version, "model": model.name, **metadata}
    (ARTIFACTS_DIR / f"{name}.json").write_text(json.dumps(info, indent=2, default=str))
    return version


def load_latest(model_name: str) -> dict:
    """Load the most recent saved version of a model."""
    files = sorted(ARTIFACTS_DIR.glob(f"{model_name}_*.joblib"))
    if not files:
        raise FileNotFoundError(f"No saved model for {model_name} in {ARTIFACTS_DIR}")
    return joblib.load(files[-1])