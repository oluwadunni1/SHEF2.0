"""P9 shadow challengers: frozen rules, point in time."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from shef.monitoring import challengers as ch
from shef.timestamps import issue_time_utc

DAY = date(2026, 10, 30)
ISSUE = issue_time_utc(DAY)                     # 2026-10-29 15:00 UTC (10:00 CDT)


def live_frame(err_by_ts=lambda ts: 100.0) -> pd.DataFrame:
    """Served-first live rows: 10 scored days before the issue, then the target day (no actuals)."""
    ts = pd.date_range(ISSUE - pd.Timedelta(days=10), ISSUE + pd.Timedelta(hours=37), freq="h", tz="UTC")
    p = np.full(len(ts), 10_000.0)
    y = np.array([10_000.0 - err_by_ts(t) for t in ts])
    df = pd.DataFrame({"ts_utc": ts, "p": p, "y": y, "served_as": "model", "issue_time_utc": ISSUE})
    df["target_day"] = df["ts_utc"].dt.tz_convert("America/Chicago").dt.date
    df.loc[df["target_day"] == DAY, "y"] = np.nan
    df = df[(df["ts_utc"] < ISSUE + pd.Timedelta(hours=14)) | (df["target_day"] == DAY)]
    return df.reset_index(drop=True)


def rows(out, name):
    return [r for r in out if r["challenger"] == name]


def test_recent_median_subtracts_half_the_median_known_error():
    out = rows(ch.build(live_frame(), pd.Series(dtype=float), DAY), "recent_median")
    assert len(out) == 24
    assert all(r["predicted_load_mw"] == pytest.approx(10_000 - 0.5 * 100) for r in out)
    assert out[0]["inputs"]["known_hours"] == 7 * 24


def test_errors_after_the_load_cutoff_are_not_used():
    # Poison every hour that ends after issue - 1 h: the correction must not move.
    cutoff = ISSUE - pd.Timedelta(hours=1)
    clean = rows(ch.build(live_frame(), pd.Series(dtype=float), DAY), "recent_median")
    poisoned = live_frame(lambda t: 1e6 if t + pd.Timedelta(hours=1) > cutoff else 100.0)
    assert poisoned["y"].notna().sum() > 7 * 24          # the poisoned actuals are present...
    out = rows(ch.build(poisoned, pd.Series(dtype=float), DAY), "recent_median")
    assert [r["predicted_load_mw"] for r in out] == [r["predicted_load_mw"] for r in clean]   # ...and ignored


def test_median_resists_one_bad_day():
    bad = ISSUE - pd.Timedelta(days=2)
    df = live_frame(lambda t: 3000.0 if bad <= t < bad + pd.Timedelta(days=1) else 100.0)
    out = rows(ch.build(df, pd.Series(dtype=float), DAY), "recent_median")
    assert out[0]["inputs"]["median_err_mw"] == 100.0


def test_too_little_history_logs_no_correction():
    df = live_frame()
    df = df[(df["ts_utc"] > ISSUE - pd.Timedelta(hours=60)) | (df["target_day"] == DAY)]
    assert rows(ch.build(df, pd.Series(dtype=float), DAY), "recent_median") == []


def test_stack_weights_and_fallback_where_ercot_is_missing():
    df = live_frame()
    target = df.loc[df["target_day"] == DAY, "ts_utc"]
    ercot = pd.Series(12_000.0, index=target.iloc[:20])
    out = rows(ch.build(df, ercot, DAY), "stack_ercot")
    assert len(out) == 24
    assert out[0]["predicted_load_mw"] == pytest.approx(0.31 * 10_000 + 0.70 * 12_000)
    assert out[-1]["predicted_load_mw"] == 10_000 and "fallback" in out[-1]["inputs"]


def test_no_rows_when_the_champion_fell_back():
    df = live_frame()
    df.loc[df["target_day"] == DAY, "served_as"] = "fallback"
    assert ch.build(df, pd.Series(dtype=float), DAY) == []
