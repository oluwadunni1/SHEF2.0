"""
backfill.py — pull history into data/raw/, one year at a time.

Raw tables keep the source's own labels (no shifting); each chunk passes the
Raw schema before it is stored. Reruns are safe: the store upserts.

    uv run python scripts/backfill.py --source all
    uv run python scripts/backfill.py --source load --start 2016-01-01
    uv run python scripts/backfill.py --source weather_fc_proxy --start 2021-01-01
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone

import pandas as pd

from shef import schemas, store
from shef.config import AUDIT_WEATHER_CACHE, CACHE, CITIES, TRAIN_START
from shef.sources import gridstatus, openmeteo
from shef.validate import GateFailed, gate

WEATHER_SOURCES = {"weather_obs": "archive", "weather_fc_proxy": "previous_runs"}
DEFAULT_START = {"load": TRAIN_START, "weather_obs": TRAIN_START, "weather_fc_proxy": date(2021, 1, 1)}


def years(start: date, end: date):
    for y in range(start.year, end.year + 1):
        yield max(start, date(y, 1, 1)), min(end, date(y, 12, 31))


def backfill_load(start: date, end: date) -> None:
    gs = gridstatus.client()
    for a, b in years(start, end):
        df = gridstatus.fetch_load(pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1), gs)
        gate(df, schemas.RawLoad, f"raw_load_{a.year}")
        df["fetched_at_utc"] = datetime.now(timezone.utc)
        n = store.upsert(df, "raw", "load", keys=["interval_start_utc"], ts_col="interval_start_utc")
        print(f"load {a}..{b}: fetched {len(df)}, stored {n}")


def backfill_weather(name: str, start: date, end: date) -> None:
    source = WEATHER_SOURCES[name]
    cache_dirs = [CACHE / "openmeteo", AUDIT_WEATHER_CACHE]
    for a, _ in years(start, end):
        frames = []
        for city in CITIES:
            # Whole-year ranges match the audit's cache keys, so past years are not refetched.
            fa, fb = date(a.year, 1, 1), min(date(a.year, 12, 31), end)
            frames.append(openmeteo.fetch(source, city, fa, fb, cache_dirs=cache_dirs))
        df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
        gate(df, schemas.RawWeather, f"raw_{name}_{a.year}")
        n = store.upsert(df, "raw", name, keys=["time", "city"], ts_col="time")
        print(f"{name} {a.year}: {len(df)} rows ({df['city'].nunique()} cities), stored {n}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["all", "load", *WEATHER_SOURCES], default="all")
    p.add_argument("--start", type=date.fromisoformat)
    p.add_argument("--end", type=date.fromisoformat, default=date.today())
    args = p.parse_args()

    names = ["load", *WEATHER_SOURCES] if args.source == "all" else [args.source]
    try:
        for name in names:
            start = args.start or DEFAULT_START[name]
            if name == "load":
                backfill_load(start, args.end)
            else:
                backfill_weather(name, start, args.end)
    except GateFailed as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
