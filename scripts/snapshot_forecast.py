"""
snapshot_forecast.py — archive the Open-Meteo forecast exactly as we would
serve it at issue time. The Previous Runs API is only a proxy for this; over
time this archive is what proves (or disproves) that the proxy matches serving.

Three models per snapshot:
    ecmwf_ifs025  what serving requests (SERVED_WEATHER_MODEL, from issue date 2026-10-29)
    best_match    Open-Meteo's default blend, kept for comparison
    gfs_seamless  served until 2026-10-28; kept as a shadow so the two keep being compared.
                  Live it is HRRR at our 25-38 h leads, which the GFS-global backtest never saw.

Stores raw API labels in data/raw/weather_fc_live, keyed by Central issue date,
model, valid time and city, plus past_days of recent hours (the "observed" side
of windows that straddle issue_time is not in ERA5 yet at serve time).

    uv run python scripts/snapshot_forecast.py                 # snapshot now
    uv run python scripts/snapshot_forecast.py --only-at-issue # no-op unless it is 10:xx Central
    uv run python scripts/snapshot_forecast.py --dry-run       # fetch + validate, store nothing

Run by .github/workflows/daily.yml at 10:xx Central, just before the forecast,
which then reads the served model's rows back (forecast_day.py --from-snapshot).
"""
from __future__ import annotations

import argparse
from datetime import time

import pandas as pd

from shef import schemas, store
from shef.config import CENTRAL_TZ, CITIES, contract
from shef.serving.schema import SERVED_WEATHER_MODEL
from shef.sources import openmeteo
from shef.validate import gate

MODELS = [SERVED_WEATHER_MODEL, None]   # None = best_match
SHADOW_MODELS = ["gfs_seamless"]   # best effort: a failure here never blocks the forecast
KEYS = ["issue_date_central", "model", "time", "city"]


def _migrate() -> None:
    """Snapshots taken before the model column existed were best_match."""
    old = store.read("raw", "weather_fc_live")
    if not old.empty and "model" not in old.columns:
        old["model"] = "best_match"
        store.write(old, "raw", "weather_fc_live", ts_col="time")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only-at-issue", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    now = pd.Timestamp.now(tz="UTC")
    local = now.tz_convert(CENTRAL_TZ)
    issue_hour = time.fromisoformat(contract()["issue"]["issue_time_local"]).hour
    if args.only_at_issue and local.hour != issue_hour:
        print(f"{local:%Y-%m-%d %H:%M %Z}: not the issue hour ({issue_hour}:xx Central); skipping.")
        return

    df = pd.concat([openmeteo.fetch_live_forecast(c, model=m) for m in MODELS for c in CITIES], ignore_index=True)
    for m in SHADOW_MODELS:
        try:
            shadow = pd.concat([openmeteo.fetch_live_forecast(c, model=m) for c in CITIES], ignore_index=True)
            gate(shadow, _schema(), f"raw_weather_fc_live_{m}")
            df = pd.concat([df, shadow], ignore_index=True)
        except Exception as e:
            print(f"::warning::shadow model {m} not archived: {type(e).__name__}: {str(e)[:300]}")
    gate(df, _schema(), "raw_weather_fc_live")
    df["run_fetched_at_utc"] = now
    df["issue_date_central"] = str(local.date())
    summary = df.groupby("model")["temperature_2m"].agg(lambda s: f"{s.notna().mean():.1%} temp present").to_dict()
    if args.dry_run:
        print(f"{local:%Y-%m-%d %H:%M %Z}: dry run, {len(df)} rows fetched {summary}; nothing stored")
        return
    _migrate()
    n = store.upsert(df, "raw", "weather_fc_live", keys=KEYS, ts_col="time")
    print(f"{local:%Y-%m-%d %H:%M %Z}: stored {len(df)} rows {summary} "
          f"({df['time'].min()} -> {df['time'].max()}), table rows touched {n}")


def _schema():
    """RawWeather, but unique per (time, city, model): several models per snapshot."""
    import pandera.pandas as pa

    return pa.DataFrameSchema({**schemas.RawWeather.columns, "model": pa.Column(str)},
                              unique=["time", "city", "model"], strict=False, name="raw_weather_fc_live")


if __name__ == "__main__":
    main()
