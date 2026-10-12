"""
The production model: everything needed to turn build_features output into MW.

A Champion holds the fitted LightGBM, the fitted target form (trend line), the
fitted bundle (thresholds, holiday_freq, trend origin) and the feature list.
It saves to a directory and wraps as an MLflow pyfunc, so P4 loads it with
mlflow.pyfunc.load_model and calls predict(features_df).

Gate: a row with no forecast temperature is refused (ForecastIncomplete).
A model trained on ERA5 has never seen missing temperature; in the archive gap
of Jan 2024 it produced 30-60% errors. The serving pipeline catches this and
falls back to the baseline with an alert.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from shef.features import bundle as bundle_mod
from shef.models import data, gbdt
from shef.models.targets import Target

REQUIRED_COMPLETE = ["temp_c"]


class ForecastIncomplete(ValueError):
    pass


@dataclass
class Champion:
    model: object                 # fitted LGBMRegressor
    target: Target
    bundle: bundle_mod.Bundle
    features: list[str]

    @classmethod
    def fit(cls, train: pd.DataFrame, features: list[str], target_kind: str = "trend", params: dict | None = None):
        """train: replayed feature rows with `y` (bundle not yet applied)."""
        b = bundle_mod.fit(train)
        tr = bundle_mod.apply(train, b)
        t = Target(target_kind).fit(tr, tr["y"])
        z = t.transform(tr, tr["y"])
        ok = np.isfinite(z)
        model = gbdt.lgbm(**(params or {}))
        model.fit(tr.loc[ok, features], z[ok])
        return cls(model, t, b, features)

    def check(self, rows: pd.DataFrame) -> None:
        missing = [c for c in self.features if c not in rows.columns and c not in data.BUNDLE_FEATURES]
        if missing:
            raise KeyError(f"feature rows lack {missing}")
        bad = rows[REQUIRED_COMPLETE].isna().any(axis=1)
        if bad.any():
            ts = rows.loc[bad, "ts_utc"].tolist() if "ts_utc" in rows else bad.sum()
            raise ForecastIncomplete(f"no forecast temperature for target hour(s): {ts}")

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        """rows: build_features output (bundle features are added here)."""
        self.check(rows)
        x = bundle_mod.apply(rows, self.bundle)
        return self.target.inverse(x, self.model.predict(x[self.features]))

    # --- persistence ----------------------------------------------------------
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(path / "lgbm.txt"))
        (path / "target.json").write_text(json.dumps(self.target.to_dict(), indent=2))
        (path / "bundle.json").write_text(self.bundle.to_json())
        (path / "features.json").write_text(json.dumps(self.features, indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> "Champion":
        import lightgbm as lgb

        path = Path(path)
        booster = lgb.Booster(model_file=str(path / "lgbm.txt"))
        return cls(_BoosterModel(booster), Target.from_dict(json.loads((path / "target.json").read_text())),
                   bundle_mod.Bundle.from_json((path / "bundle.json").read_text()),
                   json.loads((path / "features.json").read_text()))


class _BoosterModel:
    """A loaded booster behaves like the fitted regressor for predict()."""

    def __init__(self, booster):
        self.booster_ = booster

    def predict(self, X):
        return self.booster_.predict(X)


# --- MLflow pyfunc wrapper -------------------------------------------------------
try:
    import mlflow.pyfunc

    class ChampionPyfunc(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.champion = Champion.load(Path(context.artifacts["champion"]))

        def predict(self, context, model_input, params=None):
            return self.champion.predict(model_input)
except ImportError:  # serving image without mlflow can still use Champion.load
    pass
