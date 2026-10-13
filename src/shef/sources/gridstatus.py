"""GridStatus fetcher for South Central load, shared by the audit, backfill and serving."""
from __future__ import annotations

import os
import sys

import pandas as pd
from dotenv import load_dotenv

from shef.config import ROOT

DATASET = "ercot_load_by_weather_zone"
LOAD_COLUMN = "south_central"
KEEP = ["interval_start_utc", "interval_end_utc", LOAD_COLUMN]


def client():
    from gridstatusio import GridStatusClient

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("GRIDSTATUS_API_KEY")
    if not api_key:
        sys.exit("GRIDSTATUS_API_KEY is not set. Add it to SHEF2.0/.env and rerun.")
    return GridStatusClient(api_key=api_key)


def fetch_load(start: pd.Timestamp, end: pd.Timestamp, gs=None) -> pd.DataFrame:
    """Hourly intervals whose start is in [start, end). Raw GridStatus labels, UTC."""
    gs = gs or client()
    df = gs.get_dataset(dataset=DATASET, start=start.isoformat(), end=end.isoformat(), verbose=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=KEEP)
    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        raise RuntimeError(f"GridStatus {DATASET} is missing {missing}; columns: {list(df.columns)}")
    out = df[KEEP].copy()
    for col in ("interval_start_utc", "interval_end_utc"):
        out[col] = pd.to_datetime(out[col], utc=True)
    out[LOAD_COLUMN] = pd.to_numeric(out[LOAD_COLUMN], errors="coerce").astype("float64")
    return out.sort_values("interval_start_utc").reset_index(drop=True)


# --- ERCOT's own forecast (benchmark) and the plan quota -------------------------
FORECAST_DATASET = "ercot_load_forecast_by_weather_zone"
FORECAST_LOOKBACK = pd.Timedelta(hours=3)   # ERCOT publishes hourly at :30; tolerate a missed publish


def fetch_ercot_forecast(target_day, issue_time: pd.Timestamp, gs=None) -> pd.DataFrame:
    """ERCOT's South Central forecast for one Central day as known at `issue_time`:
    per target hour, the latest publish at or before issue_time. One API request.
    Columns: ts_utc (hour start), publish_time_utc, forecast_mw."""
    from shef.timestamps import central_day_bounds

    gs = gs or client()
    start, end = central_day_bounds(target_day)
    df = gs.get_dataset(dataset=FORECAST_DATASET, start=start, end=end, columns=None,
                        publish_time_start=issue_time - FORECAST_LOOKBACK, publish_time_end=issue_time, verbose=False)
    cols = ["ts_utc", "publish_time_utc", "forecast_mw"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    df = df.assign(ts_utc=pd.to_datetime(df["interval_start_utc"], utc=True),
                   publish_time_utc=pd.to_datetime(df["publish_time_utc"], utc=True),
                   forecast_mw=pd.to_numeric(df[LOAD_COLUMN], errors="coerce"))
    df = df[(df["publish_time_utc"] <= issue_time) & df["forecast_mw"].notna()]
    latest = df.sort_values("publish_time_utc").groupby("ts_utc", as_index=False).last()
    return latest[cols].sort_values("ts_utc").reset_index(drop=True)


def usage(gs=None) -> dict:
    """Requests and rows used this period, with the plan's limits."""
    u = (gs or client()).get_api_usage()
    return {"period_start": u["current_usage_period_start"],
            "requests_used": u["current_period_usage"]["total_requests"],
            "requests_limit": u["limits"]["api_requests_limit"],
            "rows_used": u["current_period_usage"]["total_api_rows_returned"],
            "rows_limit": u["limits"]["api_rows_returned_limit"]}
