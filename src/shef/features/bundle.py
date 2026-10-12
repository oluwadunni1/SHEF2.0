"""
Features fitted on training data (ported from v1 src/experiments/features.py).
fit() runs on the training rows only (per fold in P3); apply() is the same at
train and serve time, and the fitted values ship with the model.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import pandas as pd

TREND_ORIGIN = "2016-01-01"


@dataclass(frozen=True)
class Bundle:
    heat_tmax_c: float              # 95th percentile of tmax
    cold_tmin_c: float              # 5th percentile of tmin
    holiday_freq: dict[str, float] = field(default_factory=dict)  # share of holiday rows per holiday
    trend_origin: str = TREND_ORIGIN

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Bundle":
        return cls(**json.loads(text))


def fit(train: pd.DataFrame) -> Bundle:
    return Bundle(
        heat_tmax_c=float(train["tmax"].quantile(0.95)),
        cold_tmin_c=float(train["tmin"].quantile(0.05)),
        holiday_freq={str(k): float(v) for k, v in train["holiday_name"].value_counts(normalize=True).items()},
    )


def apply(df: pd.DataFrame, b: Bundle) -> pd.DataFrame:
    out = df.copy()
    out["is_extreme_heat_event"] = (out["tmax"] > b.heat_tmax_c).astype(int)
    out["is_extreme_cold_event"] = (out["tmin"] < b.cold_tmin_c).astype(int)
    out["holiday_freq"] = out["holiday_name"].map(b.holiday_freq).fillna(0.0)
    origin = pd.Timestamp(b.trend_origin, tz="UTC")
    out["trend_idx"] = (out["ts_utc"] - origin) / pd.Timedelta(days=365.25)
    return out
