"""
Client side of the API: build_features rows -> request JSON, and the HTTP calls.
Used by scripts/forecast_day.py and by the tests (same payload either way).
"""
from __future__ import annotations

import math

import pandas as pd

from shef.serving.schema import SERVED_WEATHER_MODEL, client_features


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def payload(rows: pd.DataFrame, features: list[str], weather_model: str = SERVED_WEATHER_MODEL) -> dict:
    """rows: build_features / build_live output for one target day (one issue time)."""
    issue = rows["issue_time_utc"].iloc[0]
    hol = rows["holiday_name"] if "holiday_name" in rows else pd.Series(None, index=rows.index)
    out = []
    for (_, r), h in zip(rows.iterrows(), hol):
        rec = {"ts_utc": r["ts_utc"].isoformat(), "holiday_name": h if isinstance(h, str) else None}
        rec.update({f: _num(r[f]) for f in client_features(features)})
        out.append(rec)
    return {"issue_time_utc": issue.isoformat(), "weather_model": weather_model, "rows": out}


def actuals_payload(load: pd.DataFrame) -> dict:
    """load: aligned load (ts_utc, load_mw)."""
    ok = load.dropna(subset=["load_mw"])
    return {"actuals": [{"ts_utc": t.isoformat(), "load_mw": float(v)} for t, v in zip(ok["ts_utc"], ok["load_mw"])]}


def get(url: str, path: str, timeout: float = 60) -> dict:
    import requests

    r = requests.get(url.rstrip("/") + path, timeout=timeout)
    r.raise_for_status()
    return r.json()


def post(url: str, path: str, body: dict, api_key: str, timeout: float = 60) -> dict:
    import requests

    r = requests.post(url.rstrip("/") + path, json=body, headers={"x-api-key": api_key}, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"{path} -> HTTP {r.status_code}: {r.text[:2000]}")
    return r.json()
