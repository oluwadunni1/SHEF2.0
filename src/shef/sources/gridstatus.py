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
