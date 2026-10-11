"""
The only place that turns UTC instants into Central-time concepts.

Storage is UTC everywhere (`ts_utc` = UTC start of the hour a value describes).
Local hour, weekday and calendar day are Central-clock ideas, so they are
derived here and nowhere else — v1 once took them from a UTC column and
silently fed the model the wrong local hour.
"""
from __future__ import annotations

from datetime import date, time

import pandas as pd

from shef.config import CENTRAL_TZ, contract


def require_utc(s: pd.Series | pd.DatetimeIndex, name: str = "timestamp") -> None:
    tz = getattr(s.dtype, "tz", None)
    if tz is None or str(tz) != "UTC":
        raise ValueError(f"'{name}' must be tz-aware UTC, got {s.dtype}.")


def to_central(s: pd.Series) -> pd.Series:
    require_utc(s)
    return s.dt.tz_convert(CENTRAL_TZ)


def central_date(s: pd.Series) -> pd.Series:
    """Central calendar day each UTC instant falls on (as datetime.date)."""
    return to_central(s).dt.date


def central_day_bounds(day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) of a Central calendar day, in UTC. 23 or 25 hours on DST days."""
    start = pd.Timestamp(day, tz=CENTRAL_TZ)
    end = pd.Timestamp(day + pd.Timedelta(days=1), tz=CENTRAL_TZ)
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def issue_time_utc(target_day: date) -> pd.Timestamp:
    """Day-ahead issue time for a target Central day: issue_time_local on D-1, in UTC."""
    issue = contract()["issue"]
    local = time.fromisoformat(issue["issue_time_local"])
    d_minus = target_day - pd.Timedelta(days=issue["target_day_offset"])
    return pd.Timestamp.combine(d_minus, local).tz_localize(CENTRAL_TZ).tz_convert("UTC")


def add_calendar(df: pd.DataFrame, ts_col: str = "ts_utc") -> pd.DataFrame:
    """hour / dayofweek / month / is_weekend from the Central clock."""
    out = df.copy()
    local = to_central(out[ts_col])
    out["hour"] = local.dt.hour
    out["dayofweek"] = local.dt.dayofweek  # 0 = Monday
    out["month"] = local.dt.month
    out["is_weekend"] = out["dayofweek"].isin([5, 6])
    return out


def is_dst_transition_day(s: pd.Series) -> pd.Series:
    """True for every hour on a Central day that has 23 or 25 hours."""
    day = central_date(s)
    length = {d: central_day_hours(d) for d in day.unique()}
    return day.map(length) != 24


def central_day_hours(day: date) -> int:
    start, end = central_day_bounds(day)
    return int((end - start) / pd.Timedelta(hours=1))
