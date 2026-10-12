"""
forecast_day.py — the day-ahead client: fetch what is known at issue time,
build the feature rows, ask the API for the forecast, report realised load.

    1. GridStatus load for the last 10 days + Open-Meteo gfs_seamless (4 cities)
    2. features.live.build_live(issue_time, target_day, ...)   (parity-tested vs replay)
    3. POST /forecast                                          (API predicts + logs)
    4. POST /actuals with the fetched load                     (fills actual_load_mw)

    uv run --env-file .env python scripts/forecast_day.py --dry-run   # in-process API, nothing logged
    uv run --env-file .env python scripts/forecast_day.py             # SHEF_API_URL + SHEF_API_KEY

Exit code 2 when the API served the baseline fallback, so a scheduled run that
fell back shows as failed (and GitHub's failure email is the alert).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from shef.align import align_load
from shef.config import CENTRAL_TZ, CITIES
from shef.features.live import build_live
from shef.serving import client
from shef.serving.schema import SERVED_WEATHER_MODEL
from shef.sources import gridstatus, openmeteo
from shef.timestamps import issue_time_utc

LATE_WARNING = pd.Timedelta(hours=3)


def fetch(issue: pd.Timestamp, now: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    load = gridstatus.fetch_load(issue - pd.Timedelta(days=10), now)
    weather = pd.concat([openmeteo.fetch_live_forecast(c, model=SERVED_WEATHER_MODEL) for c in CITIES],
                        ignore_index=True)
    return load, weather


def show(out: dict) -> None:
    print(f"\n{out['target_day']}  status={out['status']}  model={out['model_name']} v{out['model_version']}"
          f"  logged={out['logged']}" + (f"  WARNING: {out['warning']}" if out.get("warning") else ""))
    if out.get("reason"):
        print("reason:", out["reason"])
    for p in out["predictions"]:
        local = pd.Timestamp(p["ts_utc"]).tz_convert(CENTRAL_TZ)
        print(f"  {local:%Y-%m-%d %H:%M %Z}  lead {p['lead_hours']:>2}h  {p['load_mw']:>9,.1f} MW")


def local_api(model_dir: Path):
    """The real app in-process, with logging stubbed out: validates the payload exactly as Lambda would."""
    from fastapi.testclient import TestClient

    from shef.serving.app import create_app, load_model_dir

    champion, meta = load_model_dir(model_dir)
    app = create_app(champion, meta, "dry-run", log_predictions=lambda r: 0, log_actuals=lambda a: 0)
    return TestClient(app), champion.features


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target-day", type=pd.Timestamp, help="Central date; default tomorrow")
    p.add_argument("--dry-run", action="store_true", help="validate + predict in-process; log nothing")
    p.add_argument("--model-dir", type=Path, default=Path("build/model"), help="for --dry-run")
    args = p.parse_args()

    now = pd.Timestamp.now(tz="UTC")
    target_day = (args.target_day or now.tz_convert(CENTRAL_TZ) + pd.Timedelta(days=1)).date()
    issue = issue_time_utc(target_day)
    if now < issue:
        print(f"issue time {issue} has not arrived; a forecast for {target_day} cannot be made yet")
        return 1
    if now - issue > LATE_WARNING:
        print(f"note: running {now - issue} after issue time; load is still cut at issue time, "
              f"but the weather forecast is a later run than 10:00 Central would have seen")

    load_raw, weather_raw = fetch(issue, now)
    rows = build_live(issue, target_day, load_raw, weather_raw)
    print(f"built {len(rows)} rows for {target_day} as of {issue} (load cutoff {rows['cutoff_utc'].iloc[0]})")

    if args.dry_run:
        api, features = local_api(args.model_dir)
        r = api.post("/forecast", json=client.payload(rows, features), headers={"x-api-key": "dry-run"})
        if r.status_code != 200:
            print(f"payload rejected: HTTP {r.status_code} {r.text[:2000]}")
            return 1
        out = r.json()
    else:
        url, key = os.environ["SHEF_API_URL"], os.environ["SHEF_API_KEY"]
        health = client.get(url, "/health")
        print(f"API serving {health['model_name']} v{health['model_version']}")
        out = client.post(url, "/forecast", client.payload(rows, health["features"]), key)
        act = client.post(url, "/actuals", client.actuals_payload(align_load(load_raw)), key)
        print(f"actuals: {act['updated']} of {act['received']} hours filled in")
    show(out)
    return 2 if out["status"] == "fallback" else 0


if __name__ == "__main__":
    sys.exit(main())
