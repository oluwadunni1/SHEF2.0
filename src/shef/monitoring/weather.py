"""
How good was the weather the model was actually served?

For every archived gfs_seamless snapshot whose target day is old enough for
ERA5 to be final (ERA5_LAG_DAYS), compare the served temperature on the
target-day hours with:

    served_vs_era5             what was observed (reanalysis)
    served_vs_previous_day1    the training proxy for that same hour
                               (Previous Runs, gfs_seamless, forecast 24 h earlier)

Shadow models archived in the same snapshot (SHADOW_MODELS) get the first
comparison too, as `<model>_vs_era5`, so a candidate serving model builds a
live record before anything switches.

The second comparison is why the snapshot archive exists: training used the
proxy because live forecasts were never archived. If the two differ, the
backtest overstated (or understated) live accuracy.

All series go through the same alignment (ts_utc = time - 1h) and population
weighting as training, so hours line up exactly.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from shef.align import align_weather
from shef.config import CENTRAL_TZ, CITIES
from shef.serving.schema import SERVED_WEATHER_MODEL
from shef.sources import openmeteo
from shef.timestamps import central_day_bounds
from shef.weather_weights import weight_cities

ERA5_LAG_DAYS = 7
SHADOW_MODELS = ["ecmwf_ifs025"]


def is_served(snapshot: pd.DataFrame) -> pd.Series:
    """Rows of the served model. Snapshots from before the model column existed were best_match."""
    model = snapshot["model"] if "model" in snapshot else pd.Series("best_match", index=snapshot.index)
    return model == SERVED_WEATHER_MODEL


def served_series(snapshot: pd.DataFrame, issue_date: date, model: str | None = None) -> pd.Series:
    """Weighted forecast temperature on the target day of one issue date, indexed by ts_utc:
    the served model's, or another archived model's."""
    rows = is_served(snapshot) if model is None else (snapshot["model"] == model if "model" in snapshot else False)
    snap = snapshot[(snapshot["issue_date_central"] == str(issue_date)) & rows]
    if snap.empty:
        return pd.Series(dtype=float)
    w = weight_cities(align_weather(snap, "forecast"), ["temp_c"]).set_index("ts_utc")["temp_c"]
    start, end = central_day_bounds(issue_date + timedelta(days=1))
    return w[(w.index >= start) & (w.index < end)]


def reference_series(source: str, day: date) -> pd.Series:
    """ERA5 ("archive") or the d1 proxy ("previous_runs", served model) for one Central day."""
    start, end = central_day_bounds(day)
    model = SERVED_WEATHER_MODEL if source == "previous_runs" else None
    raw = pd.concat([openmeteo.fetch(source, c, start.date(), (end + pd.Timedelta(hours=1)).date(), model=model)
                     for c in CITIES], ignore_index=True)
    if raw.empty:
        return pd.Series(dtype=float)
    col = "temp_c_d1" if source == "previous_runs" else "temp_c"
    w = weight_cities(align_weather(raw, source), [col]).set_index("ts_utc")[col]
    return w[(w.index >= start) & (w.index < end)]


def compare(served: pd.Series, ref: pd.Series) -> dict | None:
    both = pd.concat({"s": served, "r": ref}, axis=1).dropna()
    if len(both) < len(served) or both.empty:
        return None   # reference not complete for every served hour yet
    d = both["s"] - both["r"]
    return {"n_hours": int(len(d)), "mae": float(d.abs().mean()), "bias": float(d.mean()),
            "p95_abs": float(np.quantile(d.abs(), 0.95))}


def due_issue_dates(snapshot: pd.DataFrame, done: set[tuple[date, str]], today: date) -> list[date]:
    """Issue dates with a served snapshot, whose target day is ERA5-final, not yet scored for both comparisons."""
    has = is_served(snapshot)
    dates = sorted({date.fromisoformat(d) for d in snapshot.loc[has, "issue_date_central"].unique()})
    need = {"served_vs_era5", "served_vs_previous_day1"}
    return [d for d in dates
            if d + timedelta(days=1) <= today - timedelta(days=ERA5_LAG_DAYS)
            and not need <= {c for (dd, c) in done if dd == d}]


def due_shadow_dates(snapshot: pd.DataFrame, model: str, done: set[tuple[date, str]], today: date) -> list[date]:
    if "model" not in snapshot:
        return []
    dates = sorted({date.fromisoformat(d) for d in snapshot.loc[snapshot["model"] == model, "issue_date_central"].unique()})
    return [d for d in dates if d + timedelta(days=1) <= today - timedelta(days=ERA5_LAG_DAYS)
            and (d, f"{model}_vs_era5") not in done]


def check(snapshot: pd.DataFrame, done: set[tuple[date, str]], today: date | None = None) -> list[dict]:
    today = today or pd.Timestamp.now(tz=CENTRAL_TZ).date()
    out = []
    for d in due_issue_dates(snapshot, done, today):
        served = served_series(snapshot, d)
        for comparison, source in (("served_vs_era5", "archive"), ("served_vs_previous_day1", "previous_runs")):
            if (d, comparison) in done:
                continue
            if (r := compare(served, reference_series(source, d + timedelta(days=1)))) is not None:
                out.append({"issue_date": d, "comparison": comparison, "variable": "temp_c", **r})
    for model in SHADOW_MODELS:
        for d in due_shadow_dates(snapshot, model, done, today):
            fc = served_series(snapshot, d, model)
            if not fc.empty and (r := compare(fc, reference_series("archive", d + timedelta(days=1)))) is not None:
                out.append({"issue_date": d, "comparison": f"{model}_vs_era5", "variable": "temp_c", **r})
    return out
