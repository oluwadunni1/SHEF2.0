"""
Open-Meteo fetcher, shared by the P0 audit, the backfill and live serving.

All requests use timezone=UTC, so `time` in every response is a UTC instant.
Values are returned exactly as the API labels them; shifting them onto the
hour they describe is align.py's job.
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from shef.config import CITY_COORDS, FORECAST_LEADS, WEATHER_VARS

URLS = {
    "archive": "https://archive-api.open-meteo.com/v1/archive",
    "historical_fc": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "previous_runs": "https://previous-runs-api.open-meteo.com/v1/forecast",
    "forecast": "https://api.open-meteo.com/v1/forecast",
}

# ERA5 lands ~5 days late and recent forecasts get backfilled, so a range that
# ends this close to today is fetched fresh every time and never cached.
RECENT_DAYS = 10


def hourly_vars(source: str) -> list[str]:
    if source == "previous_runs":
        return [f"{v}_{lead}" for v in WEATHER_VARS for lead in FORECAST_LEADS]
    return list(WEATHER_VARS)


def _get(url: str, params: dict) -> requests.Response:
    for attempt in range(4):
        resp = requests.get(url, params=params, timeout=120)
        if resp.status_code != 429:
            return resp
        time.sleep(30 * (attempt + 1))
    return resp


def _to_frame(payload: dict, city: str, source: str) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.insert(1, "city", city)
    df["source"] = source
    return df


def fetch(source: str, city: str, start: date, end: date, cache_dirs: list[Path] | None = None,
          model: str | None = None) -> pd.DataFrame:
    """
    One city, one source, one date range (inclusive, UTC dates).

    cache_dirs: looked up in order; a fresh pull is written to the first one.
    Returns an empty frame if the API does not serve that range (HTTP 400).
    """
    cache_dirs = cache_dirs or []
    name = f"{source}_{city}_{start}_{end}.json" if model is None else f"{source}_{model}_{city}_{start}_{end}.json"
    cacheable = end < date.today() - timedelta(days=RECENT_DAYS)

    if cacheable:
        for d in cache_dirs:
            if (d / name).exists():
                return _to_frame(json.loads((d / name).read_text()), city, source)

    coords = CITY_COORDS[city]
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(hourly_vars(source)),
        "timezone": "UTC",
    }
    if model:
        params["models"] = model
    resp = _get(URLS[source], params)
    if resp.status_code != 200:
        print(f"  {source} {city} {start}..{end}: HTTP {resp.status_code} {resp.text[:160]}")
        return pd.DataFrame()
    payload = resp.json()
    if cacheable and cache_dirs:
        cache_dirs[0].mkdir(parents=True, exist_ok=True)
        (cache_dirs[0] / name).write_text(json.dumps(payload))
    return _to_frame(payload, city, source)


def fetch_live_forecast(city: str, past_days: int = 3, forecast_days: int = 3) -> pd.DataFrame:
    """The forecast we would serve right now: latest run, best_match model."""
    coords = CITY_COORDS[city]
    resp = _get(URLS["forecast"], {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": ",".join(hourly_vars("forecast")),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    })
    resp.raise_for_status()
    return _to_frame(resp.json(), city, "forecast")
