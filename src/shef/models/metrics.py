"""Scoring. MAPE is the headline; peak-MAPE (top 5% of actual load) is from the paper."""
from __future__ import annotations

import numpy as np
import pandas as pd

PEAK_PCT = 0.05


def mape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(p - y) / np.abs(y)) * 100)


def peak_mape(y, p, pct: float = PEAK_PCT) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    peak = y >= np.quantile(y, 1 - pct)
    return mape(y[peak], p[peak])


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def bias_pct(y, p) -> float:
    """Mean signed error as % of mean load; negative = under-forecasting."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(p - y) / np.mean(y) * 100)


def score(y, p) -> dict[str, float]:
    return {"mape": mape(y, p), "peak_mape": peak_mape(y, p), "rmse": rmse(y, p), "bias_pct": bias_pct(y, p)}


def score_folds(scored: pd.DataFrame, pred: str = "pred") -> dict[str, float]:
    """scored: rows with y, pred and fold. Peak hours are chosen within each fold."""
    out = {}
    for fold, g in scored.groupby("fold"):
        for k, v in score(g["y"], g[pred]).items():
            out[f"fold{fold}_{k}"] = v
    pooled = score(scored["y"], scored[pred])
    out.update(pooled)
    out["peak_mape"] = float(np.mean([out[f"fold{f}_peak_mape"] for f in scored["fold"].unique()]))
    return out


def by(scored: pd.DataFrame, key: str, pred: str = "pred") -> pd.DataFrame:
    """MAPE and bias grouped by e.g. lead_hours or month."""
    g = scored.assign(ape=(scored[pred] - scored["y"]).abs() / scored["y"] * 100,
                      err=scored[pred] - scored["y"])
    return g.groupby(key).agg(mape=("ape", "mean"), bias_mw=("err", "mean"), n=("y", "size"))
