"""
snapshot_forecast.py — archive the Open-Meteo forecast exactly as we would
serve it at issue time. The Previous Runs API is only a proxy for this; over
time this archive is what proves (or disproves) that the proxy matches serving.

Two models per snapshot:
    gfs_seamless  what serving requests (matches the training proxy, P3 E10)
    best_match    Open-Meteo's default blend, kept for comparison

Stores raw API labels in data/raw/weather_fc_live, keyed by Central issue date,
model, valid time and city, plus past_days of recent hours (the "observed" side
of windows that straddle issue_time is not in ERA5 yet at serve time).

    uv run python scripts/snapshot_forecast.py                 # snapshot now
    uv run python scripts/snapshot_forecast.py --only-at-issue # no-op unless it is 10:xx Central
    uv run python scripts/snapshot_forecast.py --dry-run       # fetch + validate, store nothing

Run by .github/workflows/daily.yml at 10:xx Central, just before the forecast,
which then reads the gfs_seamless rows back (forecast_day.py --from-snapshot).
"""
from __future__ import annotations

import argparse
from datetime import time

import pandas as pd

from shef import schemas, store
from shef.config import CENTRAL_TZ, CITIES, contract
from shef.sources import openmeteo
from shef.validate import gate

MODELS = ["gfs_seamless", None]   # None = best_match
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
    """RawWeather, but unique per (time, city, model): two models per snapshot."""
    import pandera.pandas as pa

    return pa.DataFrameSchema({**schemas.RawWeather.columns, "model": pa.Column(str)},
                              unique=["time", "city", "model"], strict=False, name="raw_weather_fc_live")


if __name__ == "__main__":
    main()
