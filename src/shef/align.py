"""
Put every source on one key: `ts_utc` = UTC start of the hour a value describes.

    GridStatus load       average MW over [interval_start, interval_end)
                          -> ts_utc = interval_start_utc
    Open-Meteo temp, RH   instantaneous at `time`
                          -> the reading at the END of the hour: ts_utc = time - 1h
    Open-Meteo precip     sum over the hour ending at `time` ("preceding hour sum")
                          -> ts_utc = time - 1h, exactly the load interval

All weather variables take the same shift, so a row's temperature, humidity and
precipitation all belong to the load interval on that row. v1 joined `time`
straight onto interval_start, which put precipitation an hour early.
"""
from __future__ import annotations

import pandas as pd

from shef.config import FORECAST_LEADS, WEATHER_VARS
from shef.sources.gridstatus import LOAD_COLUMN
from shef.timestamps import require_utc

HOUR = pd.Timedelta(hours=1)
UNIT = "us"  # one resolution for every timestamp, so joins never compare us with ns


def align_load(raw: pd.DataFrame) -> pd.DataFrame:
    require_utc(raw["interval_start_utc"], "interval_start_utc")
    require_utc(raw["interval_end_utc"], "interval_end_utc")
    width = raw["interval_end_utc"] - raw["interval_start_utc"]
    if not (width == HOUR).all():
        bad = raw.loc[width != HOUR, ["interval_start_utc", "interval_end_utc"]].head()
        raise ValueError(f"GridStatus intervals are not 1 hour wide:\n{bad}")
    return pd.DataFrame({
        "ts_utc": raw["interval_start_utc"].dt.as_unit(UNIT),
        "interval_end_utc": raw["interval_end_utc"].dt.as_unit(UNIT),
        "load_mw": raw[LOAD_COLUMN].astype("float64"),
    }).reset_index(drop=True)


def _weather_columns(source: str) -> dict[str, str]:
    if source == "previous_runs":
        return {f"{v}_{lead}": f"{name}_{suffix}"
                for v, name in WEATHER_VARS.items() for lead, suffix in FORECAST_LEADS.items()}
    return dict(WEATHER_VARS)


def align_weather(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    """Long frame, one row per (ts_utc, city). Keeps `valid_time_utc` (the API's `time`)."""
    require_utc(raw["time"], "time")
    cols = _weather_columns(source)
    present = [c for c in cols if c in raw.columns]
    out = raw[["time", "city", *present]].rename(columns={"time": "valid_time_utc", **cols})
    out["valid_time_utc"] = out["valid_time_utc"].dt.as_unit(UNIT)
    out.insert(0, "ts_utc", out["valid_time_utc"] - HOUR)
    for c in cols.values():
        if c not in out.columns:
            out[c] = float("nan")
        out[c] = out[c].astype("float64")
    return out.reset_index(drop=True)
