"""
Pandera schemas: the hard gates at each stage boundary (raw -> clean -> joined).

Timestamps are checked for "tz-aware and UTC" rather than one exact dtype,
because pandas 3 picks the resolution (us/ns) per source. A tz-naive or
non-UTC column fails immediately.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from shef.config import CITIES

HOUR = pd.Timedelta(hours=1)

# South Central hourly load. Bounds are deliberately loose: they catch unit
# errors and garbage, not unusual-but-real days. The 2016-2026 backfill spans
# 3,772-16,113 MW (audit/FINDINGS.md §3).
LOAD_MIN_MW = 2_000.0
LOAD_MAX_MW = 30_000.0

is_utc = pa.Check(lambda s: isinstance(s.dtype, pd.DatetimeTZDtype) and str(s.dtype.tz) == "UTC",
                  name="tz_aware_utc", element_wise=False)
on_the_hour = pa.Check(lambda s: (s.dt.minute == 0) & (s.dt.second == 0) & (s.dt.microsecond == 0),
                       name="on_the_hour")


def _ts(unique: bool = False, nullable: bool = False) -> pa.Column:
    return pa.Column(checks=[is_utc, on_the_hour], unique=unique, nullable=nullable)


def _weather_value_checks(prefix: str) -> list[pa.Check]:
    if prefix.startswith("temp_c"):
        return [pa.Check.in_range(-25, 50)]
    if prefix.startswith("humidity_pct"):
        return [pa.Check.in_range(0, 100)]
    return [pa.Check.in_range(0, 150)]  # precip_mm per hour


def _weather_cols(names: list[str], nullable: bool = True) -> dict[str, pa.Column]:
    return {n: pa.Column(float, checks=_weather_value_checks(n), nullable=nullable) for n in names}


OBS_COLS = ["temp_c", "humidity_pct", "precip_mm"]
PROXY_COLS = [f"{v}_{d}" for v in OBS_COLS for d in ("d1", "d2")]

# --- raw (as fetched, before alignment) -----------------------------------
RawLoad = pa.DataFrameSchema(
    {
        "interval_start_utc": _ts(unique=True),
        "interval_end_utc": _ts(),
        "south_central": pa.Column(float, nullable=True),
    },
    checks=pa.Check(lambda df: (df["interval_end_utc"] - df["interval_start_utc"]) == HOUR, name="one_hour_interval"),
    strict=False,
    name="raw_load",
)

RawWeather = pa.DataFrameSchema(
    {
        "time": _ts(),
        "city": pa.Column(str, pa.Check.isin(CITIES)),
    },
    unique=["time", "city"],
    strict=False,
    name="raw_weather",
)

# --- clean (aligned to ts_utc) ----------------------------------------------
CleanLoad = pa.DataFrameSchema(
    {
        "ts_utc": _ts(unique=True),
        "interval_end_utc": _ts(),
        "load_mw": pa.Column(float, pa.Check.in_range(LOAD_MIN_MW, LOAD_MAX_MW), nullable=False),
    },
    checks=pa.Check(lambda df: df["interval_end_utc"] == df["ts_utc"] + HOUR, name="end_is_start_plus_1h"),
    strict=True,
    name="clean_load",
)

CleanWeatherObs = pa.DataFrameSchema(
    {"ts_utc": _ts(), "valid_time_utc": _ts(), "city": pa.Column(str, pa.Check.isin(CITIES)), **_weather_cols(OBS_COLS)},
    checks=pa.Check(lambda df: df["valid_time_utc"] == df["ts_utc"] + HOUR, name="valid_time_is_end_of_hour"),
    unique=["ts_utc", "city"],
    strict=True,
    name="clean_weather_obs",
)

CleanWeatherProxy = pa.DataFrameSchema(
    {"ts_utc": _ts(), "valid_time_utc": _ts(), "city": pa.Column(str, pa.Check.isin(CITIES)), **_weather_cols(PROXY_COLS)},
    checks=pa.Check(lambda df: df["valid_time_utc"] == df["ts_utc"] + HOUR, name="valid_time_is_end_of_hour"),
    unique=["ts_utc", "city"],
    strict=True,
    name="clean_weather_proxy",
)

# --- joined (one row per hour, population-weighted weather) -----------------
# The joined table spans every hour from the first to the last load hour, so
# a missing load hour is an explicit row with is_load_gap=True, never a jump.
Joined = pa.DataFrameSchema(
    {
        "ts_utc": _ts(unique=True),
        "interval_end_utc": _ts(),
        "load_mw": pa.Column(float, pa.Check.in_range(LOAD_MIN_MW, LOAD_MAX_MW), nullable=True),
        **_weather_cols(OBS_COLS),
        **_weather_cols(PROXY_COLS),
        "is_load_gap": pa.Column(bool),
        "is_uri": pa.Column(bool),
        "is_dst_day": pa.Column(bool),
    },
    checks=[
        pa.Check(lambda df: df["ts_utc"].diff().iloc[1:].eq(HOUR).all(), name="strictly_hourly_contiguous"),
        pa.Check(lambda df: df["load_mw"].isna() == df["is_load_gap"], name="gap_flag_matches_nulls"),
        pa.Check(lambda df: df["interval_end_utc"] == df["ts_utc"] + HOUR, name="end_is_start_plus_1h"),
    ],
    strict=True,
    name="joined",
)
