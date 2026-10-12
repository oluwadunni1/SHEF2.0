"""
What the world looked like at issue_time. The feature builder only ever sees
the frames these functions return, so it cannot read the future.

Load:    intervals with interval_end_utc <= cutoff, cutoff = issue_time - latency.
         The cutoff comes from the fixed latency in features.yaml, never from
         whatever rows happen to exist, so serving never sees fresher data than
         training replay did. Rows that are missing inside the cutoff (GridStatus
         late) are simply absent, and the builder falls back.

Weather: one hourly series per issue_time, weighted across the four cities.
         Hours whose valid time is <= issue_time are the "past" side; later hours
         are the "future" side. Modes:
           obs    ERA5 on both sides. A perfect forecast: for training on the
                  2016+ history only, never for evaluation.
           proxy  ERA5 past + Previous Runs future (previous_day1 when the valid
                  time is <= 24 h after issue, else previous_day2; day 2 also
                  fills a missing day 1). The honest backtest.
           live   The forecast snapshot fetched at issue_time, both sides.
         Known approximation: in replay the past side is ERA5, which does not
         exist yet for the last ~5 days at serve time; live uses the forecast
         API's past_days instead. The snapshot archive measures that gap (P3).
"""
from __future__ import annotations

import pandas as pd

from shef.config import contract
from shef.timestamps import central_day_bounds, require_utc

HOUR = pd.Timedelta(hours=1)
OBS = ["temp_c", "humidity_pct", "precip_mm"]
LOAD_LOOKBACK = pd.Timedelta(days=10)     # covers lag 168 + D-8 comparisons
WEATHER_LOOKBACK = pd.Timedelta(days=8)   # covers 72 h windows + D-1 daily stats
D1_MAX_LEAD = pd.Timedelta(hours=24)


def latency() -> pd.Timedelta:
    return pd.Timedelta(hours=contract()["latency"]["gridstatus_load_hours"])


def cutoff_for(issue_time: pd.Timestamp, extra_delay: pd.Timedelta = pd.Timedelta(0)) -> pd.Timestamp:
    return issue_time - latency() - extra_delay


def load_as_of(load: pd.DataFrame, issue_time: pd.Timestamp,
               extra_delay: pd.Timedelta = pd.Timedelta(0)) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Load rows known at issue_time. `extra_delay` simulates a late GridStatus
    publish (training augmentation); serving always passes 0."""
    require_utc(load["ts_utc"], "ts_utc")
    cutoff = cutoff_for(issue_time, extra_delay)
    window = load[(load["interval_end_utc"] <= cutoff) & (load["ts_utc"] >= issue_time - LOAD_LOOKBACK)]
    return window.dropna(subset=["load_mw"]).set_index("ts_utc")[["load_mw"]].sort_index(), cutoff


def target_hours(issue_time: pd.Timestamp, target_day) -> pd.DatetimeIndex:
    start, end = central_day_bounds(target_day)
    if start <= issue_time:
        raise ValueError(f"target day {target_day} starts before issue_time {issue_time}")
    return pd.date_range(start, end - HOUR, freq="h").as_unit("us")


def _span(issue_time: pd.Timestamp, target_day) -> pd.DatetimeIndex:
    _, end = central_day_bounds(target_day)
    return pd.date_range(issue_time - WEATHER_LOOKBACK, end - HOUR, freq="h").as_unit("us")


def _is_past(ts: pd.Index, issue_time: pd.Timestamp) -> pd.Series:
    # ts_utc is the hour start; the values describe the hour ending at ts + 1h.
    return pd.Series(ts + HOUR <= issue_time, index=ts)


def weather_as_of(issue_time: pd.Timestamp, target_day, mode: str, joined: pd.DataFrame | None = None,
                  live: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Hourly weighted weather from issue_time - 8 days to the end of the target day,
    indexed by ts_utc, with columns temp_c, humidity_pct, precip_mm, is_forecast.

    joined: the clean joined table (ERA5 columns + *_d1 / *_d2 proxy columns).
    live:   weighted snapshot (ts_utc + OBS columns), for mode="live".
    """
    span = _span(issue_time, target_day)
    past = _is_past(span, issue_time)

    if mode == "live":
        out = live.set_index("ts_utc").reindex(span)[OBS]
    else:
        src = joined.set_index("ts_utc").reindex(span)
        out = src[OBS].copy()
        if mode == "proxy":
            lead = (span + HOUR) - issue_time
            use_d1 = pd.Series(lead <= D1_MAX_LEAD, index=span)
            for col in OBS:
                fc = src[f"{col}_d1"].where(use_d1, src[f"{col}_d2"])
                fc = fc.fillna(src[f"{col}_d2"])  # day 2 is older, so always legal
                out.loc[~past, col] = fc[~past]
        elif mode != "obs":
            raise ValueError(f"unknown weather mode {mode!r}")

    out["is_forecast"] = ~past
    out.index.name = "ts_utc"
    return out
