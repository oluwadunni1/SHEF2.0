"""
E0 baselines: the floor every model must beat. All of them only use features
the builder already produced, so they are as leak-free as the models.
"""
from __future__ import annotations

import pandas as pd

BASELINES = {
    "same_hour_last_week": lambda df: df["load_lag_168"],
    "same_hour_latest": lambda df: df["load_same_hour_latest"],
    "blend_latest_lastweek": lambda df: (df["load_same_hour_latest"] + df["load_lag_168"]) / 2,
}


def predict(name: str, df: pd.DataFrame) -> pd.Series:
    return BASELINES[name](df)


def score_on_folds(tables: dict) -> pd.DataFrame:
    """Validation rows of every fold (proxy table, Uri excluded) with one column per baseline."""
    from shef.models import folds
    from shef.models.cv import KEEP

    parts = []
    proxy = tables["proxy"]
    for f in folds.FOLDS:
        v = proxy[folds.val_mask(proxy, f)]
        v = v[~v["is_uri"]].dropna(subset=["y"])
        part = v[KEEP].copy()
        part["fold"] = f.id
        for name in BASELINES:
            part[name] = predict(name, v).to_numpy()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
