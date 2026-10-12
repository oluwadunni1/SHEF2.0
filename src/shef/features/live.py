"""
Serve-time path: the same build_features, fed from what the APIs return at
issue_time instead of from the clean store.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from shef.align import align_load, align_weather
from shef.features.asof import OBS, load_as_of, target_hours, weather_as_of
from shef.features.builder import build_features
from shef.weather_weights import weight_cities


def live_weather(snapshot_raw: pd.DataFrame) -> pd.DataFrame:
    """One forecast snapshot (raw Open-Meteo labels, 4 cities) -> weighted ts_utc series."""
    return weight_cities(align_weather(snapshot_raw, "forecast"), OBS)


def build_live(issue_time: pd.Timestamp, target_day: date,
               gridstatus_raw: pd.DataFrame, snapshot_raw: pd.DataFrame) -> pd.DataFrame:
    """gridstatus_raw: fetch_load(...) output covering at least the last 10 days.
    It may contain hours newer than the cutoff (a late run sees fresher data);
    load_as_of drops them, exactly as replay does."""
    known, cutoff = load_as_of(align_load(gridstatus_raw), issue_time)
    weather = weather_as_of(issue_time, target_day, "live", live=live_weather(snapshot_raw))
    out = build_features(issue_time, target_hours(issue_time, target_day), known, weather)
    out["cutoff_utc"] = cutoff
    return out
