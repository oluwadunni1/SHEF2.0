"""
raw -> clean -> joined. Every step goes through a Pandera gate.

The joined table has one row per UTC hour from the first to the last load
hour. Weather is joined on `ts_utc` after align.py has shifted it onto the
hour it describes, so this is a plain key join with nothing to fudge.
"""
from __future__ import annotations

import pandas as pd

from shef import schemas
from shef.align import align_load, align_weather
from shef.config import FC_PROXY_START, URI_DAYS_CENTRAL
from shef.timestamps import central_date, is_dst_transition_day
from shef.validate import GateFailed, completeness_by_year, gate, gaps
from shef.weather_weights import weight_cities

HOUR = pd.Timedelta(hours=1)
MIN_YEAR_COMPLETENESS = 0.98   # hard: below this a year is not usable
WARN_YEAR_COMPLETENESS = 0.995  # soft: listed in the report
ERA5_LAG_ALLOWANCE = pd.Timedelta(days=7)
# Above this a single-point hourly value is treated as a source glitch, not weather.
# Example: previous_day1 precip of 171 mm at San Marcos, 2026-05-20 07:00 UTC,
# where ERA5 had 8.6 mm and previous_day2 had 0.0.
QUARANTINE_ABOVE = {"precip_mm": 100.0}


def clean_load(raw: pd.DataFrame) -> pd.DataFrame:
    gate(raw, schemas.RawLoad, "raw_load")
    df = align_load(raw)
    nulls = int(df["load_mw"].isna().sum())
    df = df.dropna(subset=["load_mw"])  # a null reading is a gap; the joined table marks it
    return gate(df, schemas.CleanLoad, "clean_load", stats={"null_readings_dropped": nulls})


def quarantine(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """Null out values that are physically possible but not credible for this region,
    and return them so the report shows exactly what was removed. Anything beyond
    the schema's hard bounds still fails the gate."""
    removed = []
    for col in cols:
        limit = next((v for k, v in QUARANTINE_ABOVE.items() if col.startswith(k)), None)
        if limit is None:
            continue
        bad = df[col] > limit
        for _, r in df.loc[bad, ["ts_utc", "city", col]].iterrows():
            removed.append({"column": col, "ts_utc": str(r["ts_utc"]), "city": r["city"], "value": float(r[col])})
        df.loc[bad, col] = float("nan")
    return removed


def clean_weather(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    gate(raw, schemas.RawWeather, f"raw_{source}")
    df = align_weather(raw, source)
    if source == "previous_runs":
        for col in schemas.PROXY_COLS:
            start = pd.Timestamp(FC_PROXY_START[col.rsplit("_", 1)[0]], tz="UTC")
            df.loc[df["ts_utc"] < start, col] = float("nan")
        removed = quarantine(df, schemas.PROXY_COLS)
        return gate(df, schemas.CleanWeatherProxy, "clean_weather_proxy",
                    warnings=[f"quarantined {len(removed)} value(s)"] if removed else [], stats={"quarantined": removed})
    removed = quarantine(df, schemas.OBS_COLS)
    return gate(df, schemas.CleanWeatherObs, "clean_weather_obs",
                warnings=[f"quarantined {len(removed)} value(s)"] if removed else [], stats={"quarantined": removed})


def join(load: pd.DataFrame, obs: pd.DataFrame, proxy: pd.DataFrame) -> pd.DataFrame:
    hours = pd.date_range(load["ts_utc"].min(), load["ts_utc"].max(), freq="h").as_unit("us")
    df = pd.DataFrame({"ts_utc": hours})
    df = df.merge(load.drop(columns="interval_end_utc"), on="ts_utc", how="left")
    df = df.merge(weight_cities(obs, schemas.OBS_COLS), on="ts_utc", how="left")
    df = df.merge(weight_cities(proxy, schemas.PROXY_COLS), on="ts_utc", how="left")
    df.insert(1, "interval_end_utc", df["ts_utc"] + HOUR)
    df["is_load_gap"] = df["load_mw"].isna()
    day = central_date(df["ts_utc"])
    df["is_uri"] = day.between(*URI_DAYS_CENTRAL)
    df["is_dst_day"] = is_dst_transition_day(df["ts_utc"])
    return df


def build_joined(load: pd.DataFrame, obs: pd.DataFrame, proxy: pd.DataFrame) -> pd.DataFrame:
    df = join(load, obs, proxy)

    load_by_year = completeness_by_year(df, "load_mw")
    recent = df["ts_utc"] > df["ts_utc"].max() - ERA5_LAG_ALLOWANCE
    temp_missing_old = df.loc[~recent, "temp_c"].isna()
    proxy_from = pd.Timestamp(FC_PROXY_START["temp_c"], tz="UTC")
    proxy_rows = df[df["ts_utc"] >= proxy_from]

    stats = {
        "first_ts_utc": str(df["ts_utc"].min()),
        "last_ts_utc": str(df["ts_utc"].max()),
        "load_gaps": gaps(df.loc[~df["is_load_gap"], "ts_utc"]),
        "load_completeness_by_year": load_by_year,
        "temp_obs_missing_hours_older_than_7d": int(temp_missing_old.sum()),
        "temp_obs_last_valid_utc": str(df.loc[df["temp_c"].notna(), "ts_utc"].max()),
        "temp_proxy_d1_completeness_since_start": round(float(proxy_rows["temp_c_d1"].notna().mean()), 5),
        "temp_proxy_d2_completeness_since_start": round(float(proxy_rows["temp_c_d2"].notna().mean()), 5),
        "uri_hours": int(df["is_uri"].sum()),
        "dst_day_hours": int(df["is_dst_day"].sum()),
    }
    warnings = [f"load completeness {y}: {v:.3%}" for y, v in load_by_year.items() if v < WARN_YEAR_COMPLETENESS]
    if stats["temp_obs_missing_hours_older_than_7d"]:
        warnings.append(f"{stats['temp_obs_missing_hours_older_than_7d']} hours of ERA5 temperature missing outside the lag window")

    df = gate(df, schemas.Joined, "joined", warnings=warnings, stats=stats)

    hard = [f"load completeness {y}: {v:.3%} < {MIN_YEAR_COMPLETENESS:.0%}"
            for y, v in load_by_year.items() if v < MIN_YEAR_COMPLETENESS and y != df["ts_utc"].dt.year.max()]
    if temp_missing_old.mean() > 1 - MIN_YEAR_COMPLETENESS:
        hard.append(f"ERA5 temperature missing for {temp_missing_old.mean():.2%} of hours")
    if hard:
        raise GateFailed(f"[joined] {hard}")
    return df
