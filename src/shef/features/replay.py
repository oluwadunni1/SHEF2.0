"""
Training set = replay every past day's issue time through the serving builder.

For each target Central day D: issue_time = 10:00 CT on D-1, the as-of views
are cut at that instant, build_features runs, and only then is the actual load
at each target hour attached as the label `y`.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from shef.features.asof import LOAD_LOOKBACK, WEATHER_LOOKBACK, load_as_of, target_hours, weather_as_of
from shef.features.builder import build_features
from shef.timestamps import issue_time_utc

LOAD_COLS = ["ts_utc", "interval_end_utc", "load_mw"]


def replay(joined: pd.DataFrame, start: date, end: date, mode: str,
           augment_frac: float = 0.0, max_extra_delay_h: int = 12, seed: int = 0) -> pd.DataFrame:
    """
    joined:       clean joined table (data/clean/joined)
    start, end:   target Central days, inclusive
    mode:         weather mode, see asof.weather_as_of
    augment_frac: share of issue days replayed as if GridStatus were 1..max_extra_delay_h
                  hours late, so the model has seen the serve-time fallback.
    """
    joined = joined.sort_values("ts_utc").reset_index(drop=True)
    ts = pd.DatetimeIndex(joined["ts_utc"])
    label = joined.set_index("ts_utc")["load_mw"]
    rng = np.random.default_rng(seed)

    rows = []
    day = start
    while day <= end:
        issue = issue_time_utc(day)
        lo = ts.searchsorted(issue - max(LOAD_LOOKBACK, WEATHER_LOOKBACK))
        hi = ts.searchsorted(issue + pd.Timedelta(days=3))
        window = joined.iloc[lo:hi]

        extra = int(rng.integers(1, max_extra_delay_h + 1)) if rng.random() < augment_frac else 0
        known, cutoff = load_as_of(window[LOAD_COLS], issue, pd.Timedelta(hours=extra))
        targets = target_hours(issue, day)
        f = build_features(issue, targets, known, weather_as_of(issue, day, mode, joined=window))
        f["cutoff_utc"] = cutoff
        f["simulated_delay_h"] = extra
        f["y"] = label.reindex(targets).to_numpy()
        rows.append(f)
        day += timedelta(days=1)

    out = pd.concat(rows, ignore_index=True)
    flags = joined.set_index("ts_utc")[["is_uri", "is_dst_day"]]
    out = out.join(flags, on="ts_utc")
    out["weather_mode"] = mode
    return out
