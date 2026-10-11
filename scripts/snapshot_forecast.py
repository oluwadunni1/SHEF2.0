"""
snapshot_forecast.py — archive the Open-Meteo forecast exactly as we would
serve it at issue time. The Previous Runs API is only a proxy for this; over
time this archive is what proves (or disproves) that the proxy matches serving.

Stores raw API labels in data/raw/weather_fc_live, keyed by the Central issue
date, plus past_days of recent hours (the "observed" side of windows that
straddle issue_time is not in ERA5 yet at serve time).

    uv run python scripts/snapshot_forecast.py                 # snapshot now
    uv run python scripts/snapshot_forecast.py --only-at-issue # no-op unless it is 10:xx Central

Scheduled twice a day (15:00 and 16:00 UTC) so one run lands at 10:xx Central
in both CST and CDT; --only-at-issue drops the other one.
"""
from __future__ import annotations

import argparse
from datetime import time

import pandas as pd

from shef import schemas, store
from shef.config import CENTRAL_TZ, CITIES, contract
from shef.sources import openmeteo
from shef.validate import gate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only-at-issue", action="store_true")
    args = p.parse_args()

    now = pd.Timestamp.now(tz="UTC")
    local = now.tz_convert(CENTRAL_TZ)
    issue_hour = time.fromisoformat(contract()["issue"]["issue_time_local"]).hour
    if args.only_at_issue and local.hour != issue_hour:
        print(f"{local:%Y-%m-%d %H:%M %Z}: not the issue hour ({issue_hour}:xx Central); skipping.")
        return

    df = pd.concat([openmeteo.fetch_live_forecast(c) for c in CITIES], ignore_index=True)
    gate(df, schemas.RawWeather, "raw_weather_fc_live")
    df["run_fetched_at_utc"] = now
    df["issue_date_central"] = str(local.date())
    n = store.upsert(df, "raw", "weather_fc_live", keys=["issue_date_central", "time", "city"], ts_col="time")
    print(f"{local:%Y-%m-%d %H:%M %Z}: stored {len(df)} rows "
          f"({df['time'].min()} -> {df['time'].max()}), table rows touched {n}")


if __name__ == "__main__":
    main()
