"""
GridStatus publishing latency, measured from fetches the daily job makes anyway.

A fetch that returns at T with the newest hourly interval ending at E says:
the hour ending E was published within T - E, and the hour ending E + 1h was
not yet published T - (E + 1h) after it ended. Many such brackets, at
different times of day, estimate the latency distribution the feature cutoff
(features.yaml: 1 h) depends on, at no extra API cost.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from shef.sources.gridstatus import LOAD_COLUMN

DEFAULT_PATH = Path("build/load_latency.json")


def observation(load_raw: pd.DataFrame, observed_at: pd.Timestamp, run_mode: str) -> dict | None:
    have = load_raw[load_raw[LOAD_COLUMN].notna()]
    if have.empty:
        return None
    newest = pd.Timestamp(have["interval_end_utc"].max())
    return {"observed_at": observed_at.isoformat(), "newest_interval_end": newest.isoformat(), "run_mode": run_mode}


def bounds(obs: dict) -> tuple[float, float]:
    """(next hour late by, newest hour published within), minutes."""
    gap = (pd.Timestamp(obs["observed_at"]) - pd.Timestamp(obs["newest_interval_end"])) / pd.Timedelta(minutes=1)
    return gap - 60, gap


def save(obs: dict | None, path: Path = DEFAULT_PATH) -> None:
    if obs is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obs))


def load(path: Path = DEFAULT_PATH) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None
