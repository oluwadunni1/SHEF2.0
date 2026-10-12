"""Standard diagnostic plots, shown in notebooks and logged to MLflow."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from shef.models.metrics import by


def error_by(scored: pd.DataFrame, key: str, pred: str = "pred", title: str = ""):
    t = by(scored, key, pred)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(t.index, t["mape"], marker="o", lw=1.5)
    ax.set_xlabel(key)
    ax.set_ylabel("MAPE %")
    ax.set_title(title or f"MAPE by {key}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def compare_by(scored: pd.DataFrame, key: str, preds: list[str], title: str = ""):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for p in preds:
        t = by(scored, key, p)
        ax.plot(t.index, t["mape"], marker="o", lw=1.2, label=p)
    ax.set_xlabel(key)
    ax.set_ylabel("MAPE %")
    ax.set_title(title or f"MAPE by {key}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def peak_week(scored: pd.DataFrame, preds: list[str], title: str = ""):
    """The 7 days around the highest actual load in the scored rows."""
    s = scored.sort_values("ts_utc")
    peak = s.loc[s["y"].idxmax(), "ts_utc"]
    w = s[(s["ts_utc"] >= peak - pd.Timedelta(days=3)) & (s["ts_utc"] <= peak + pd.Timedelta(days=4))]
    local = w["ts_utc"].dt.tz_convert("America/Chicago")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(local, w["y"], color="black", lw=2, label="actual")
    for p in preds:
        ax.plot(local, w[p], lw=1.2, label=p)
    ax.set_ylabel("MW")
    ax.set_title(title or f"Peak week (max {w['y'].max():,.0f} MW)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def importance(names: list[str], values, top: int = 25, title: str = "Feature importance"):
    s = pd.Series(values, index=names).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(s) + 1))
    ax.barh(s.index, s.values)
    ax.set_title(title)
    fig.tight_layout()
    return fig
