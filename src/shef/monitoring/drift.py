"""
Input drift: PSI of recent live model inputs against a seasonal reference.

Reference: training rows of the serving model from the last REFERENCE_YEARS
years, split by calendar month, so a cool October is compared with past
Octobers (not with a whole year) and load growth older than a few years does
not register as drift. For each month and feature we keep decile edges and
the reference share of rows in each bin.

Current: the features JSONB the API logged for the last CURRENT_DAYS live days,
compared with the month most of those rows fall in.

PSI = sum((cur - ref) * ln(cur / ref)) over bins. The textbook cut-offs (0.1 /
0.25) are useless here: a 14-day window holds only 14 distinct daily highs, so
one heat wave fills two bins and every window "drifts" (checked: in-sample
windows score PSI 1-7 on tmax). Thresholds are therefore calibrated: PSI is
computed for every historical 14-day window of archived-forecast rows (what
live serving sees) and a live window is "watch" above that feature's 95th
percentile and "drift" above its 99th, i.e. rarer than 1 window in 100.

Drift is a signal for a human, not a retraining trigger: promotion is decided
by measured accuracy (monitoring.promote).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from shef.config import CENTRAL_TZ

# Hourly-varying inputs only. Daily-level features (tmax, tmin, temp_spike_vs_mean,
# last_known_roll_mean_168) take ~14 distinct values in a 14-day window, so PSI
# saturates near its 10-bin ceiling (~8.4) in ordinary history: their calibrated
# drift thresholds came out at 7.2-8.4, i.e. they could never alert.
FEATURES = ["temp_c", "cdd_x_hour", "load_same_hour_latest", "load_lag_168"]
REFERENCE_YEARS = 3
CURRENT_DAYS = 14
MIN_CURRENT_HOURS = 24 * 7
BINS = 10
EPS = 1e-4
WATCH_Q, DRIFT_Q = 0.95, 0.99


def psi(ref_share: np.ndarray, cur_share: np.ndarray) -> float:
    r = np.clip(np.asarray(ref_share, float), EPS, None)
    c = np.clip(np.asarray(cur_share, float), EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def status(value: float, thresholds: dict) -> str:
    return "drift" if value > thresholds["drift"] else "watch" if value > thresholds["watch"] else "ok"


def shares(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Share of values in each of the len(edges)+1 bins (-inf, e1], (e1, e2], ..., (ek, inf)."""
    idx = np.searchsorted(np.asarray(edges, float), values[~np.isnan(values)], side="left")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return counts / max(counts.sum(), 1)


def build_reference(train: pd.DataFrame, train_end: pd.Timestamp) -> dict:
    """train: feature rows the serving model was fit on (ts_utc + FEATURES)."""
    recent = train[train["ts_utc"] > train_end - pd.DateOffset(years=REFERENCE_YEARS)]
    month = recent["ts_utc"].dt.tz_convert(CENTRAL_TZ).dt.month
    ref = {"reference_years": REFERENCE_YEARS, "train_end": str(train_end.date()), "features": {}}
    for f in FEATURES:
        ref["features"][f] = {}
        for m, g in recent.groupby(month):
            v = g[f].to_numpy(float)
            v = v[~np.isnan(v)]
            edges = np.unique(np.quantile(v, np.linspace(0, 1, BINS + 1)[1:-1])).tolist()
            ref["features"][f][str(m)] = {"edges": edges, "share": shares(v, edges).tolist(), "n": int(len(v))}
    return ref


def current_month(ts: pd.Series) -> int:
    return int(ts.dt.tz_convert(CENTRAL_TZ).dt.month.mode().iloc[0])


def window_psi(reference: dict, current: pd.DataFrame) -> dict[str, float]:
    m = str(current_month(current["ts_utc"]))
    out = {}
    for f in FEATURES:
        r = reference["features"][f].get(m)
        if r is not None and f in current:
            out[f] = psi(r["share"], shares(current[f].to_numpy(float), r["edges"]))
    return out


def calibrate(reference: dict, rows: pd.DataFrame, step_days: int = 1) -> dict:
    """PSI thresholds per feature from every CURRENT_DAYS window of `rows` (archived-forecast features)."""
    rows = rows.sort_values("ts_utc")
    ends = pd.date_range(rows["ts_utc"].min() + pd.Timedelta(days=CURRENT_DAYS), rows["ts_utc"].max(),
                         freq=f"{step_days}D")
    values: dict[str, list[float]] = {f: [] for f in FEATURES}
    ts = rows["ts_utc"]
    for e in ends:
        w = rows[(ts > e - pd.Timedelta(days=CURRENT_DAYS)) & (ts <= e)]
        if len(w) >= MIN_CURRENT_HOURS:
            for f, v in window_psi(reference, w).items():
                values[f].append(v)
    return {f: {"watch": float(np.quantile(v, WATCH_Q)), "drift": float(np.quantile(v, DRIFT_Q)), "n_windows": len(v)}
            for f, v in values.items() if v}


def evaluate(reference: dict, current: pd.DataFrame) -> list[dict]:
    """current: ts_utc + FEATURES of recent live hours. Empty if there is too little data."""
    if len(current) < MIN_CURRENT_HOURS:
        return []
    m = str(current_month(current["ts_utc"]))
    return [{"feature": f, "psi": v, "status": status(v, reference["thresholds"][f]), "ref_months": m,
             "n_current": int(current[f].notna().sum())}
            for f, v in window_psi(reference, current).items()]


def load_reference(path: Path) -> dict:
    return json.loads(Path(path).read_text())
