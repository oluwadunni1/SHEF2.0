"""
Fold data: which rows train, which validate, with what weather, and the
bundle refitted on the training rows only.

Validation rows always come from the proxy table (forecast weather) and never
include Uri hours. Training rows come from:
    obs     ERA5-as-forecast, 2016+ (default: the long history)
    proxy   archived forecasts, 2021-03+ (what serving sees)
    hybrid  obs before the proxy start date, proxy after
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from shef import store
from shef.config import FC_PROXY_START
from shef.features import BUNDLE_FEATURES, bundle, contract_features
from shef.models import folds

META = ["issue_time_utc", "ts_utc", "holiday_name", "y", "is_uri", "is_dst_day", "simulated_delay_h"]
# Forecast humidity/precip only exist from 2024-01-19, so folds 1-2 would score
# them as all-missing against an ERA5-filled training set. Opt in with E5.
LATE_FORECAST_VARS = ["humidity_pct", "precip_mm"]


def default_features() -> list[str]:
    return [f for f in contract_features("day_ahead") if f not in LATE_FORECAST_VARS]


def load_tables(features: list[str]) -> dict[str, pd.DataFrame]:
    built = [c for c in features if c not in BUNDLE_FEATURES]
    cols = list(dict.fromkeys(META + built + ["tmax", "tmin"]))
    return {mode: store.read("features", mode, columns=cols) for mode in ("obs", "proxy")}


@dataclass
class FoldData:
    fold: int
    train: pd.DataFrame
    val: pd.DataFrame
    weight: np.ndarray            # training sample weights
    bundle: bundle.Bundle


def _training_rows(tables: dict, source: str, mask_fn) -> pd.DataFrame:
    if source in ("obs", "proxy"):
        t = tables[source]
        return t[mask_fn(t)]
    if source == "hybrid":
        start = pd.Timestamp(FC_PROXY_START["temp_c"], tz="UTC")
        o, p = tables["obs"], tables["proxy"]
        return pd.concat([o[mask_fn(o) & (o["ts_utc"] < start)], p[mask_fn(p) & (p["ts_utc"] >= start)]],
                         ignore_index=True)
    raise ValueError(source)


def _finish(train: pd.DataFrame, val: pd.DataFrame, fold_id, uri: str, uri_weight: float) -> FoldData:
    train = train.dropna(subset=["y"])
    if uri == "exclude":
        train = train[~train["is_uri"]]
    weight = np.where(train["is_uri"], uri_weight if uri == "weight" else 1.0, 1.0)
    val = val[~val["is_uri"]].dropna(subset=["y"])
    b = bundle.fit(train)
    return FoldData(fold_id, bundle.apply(train, b).reset_index(drop=True), bundle.apply(val, b).reset_index(drop=True),
                    weight, b)


def fold_data(tables: dict, fold: folds.Fold, train_source: str = "obs", uri: str = "exclude",
              uri_weight: float = 0.1) -> FoldData:
    train = _training_rows(tables, train_source, lambda t: folds.train_mask(t, fold))
    val = tables["proxy"][folds.val_mask(tables["proxy"], fold)]
    return _finish(train, val, fold.id, uri, uri_weight)


def holdout_data(tables: dict, train_source: str = "obs", uri: str = "exclude", uri_weight: float = 0.1,
                 final: bool = False) -> FoldData:
    train = _training_rows(tables, train_source, folds.holdout_train_mask)
    val = tables["proxy"][folds.holdout_mask(tables["proxy"], final=final)]
    return _finish(train, val, "holdout", uri, uri_weight)
