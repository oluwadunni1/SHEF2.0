"""Paths, constants and the features.yaml contract. One place, imported everywhere."""
from __future__ import annotations

from datetime import date
from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
REPORTS = DATA / "reports"
CACHE = DATA / "cache"
AUDIT_WEATHER_CACHE = ROOT / "audit" / "cache" / "weather"

CENTRAL_TZ = "America/Chicago"

# Same four cities v1 used; weighted by population in weather_weights.py.
CITY_COORDS = {
    "san_antonio": {"lat": 29.4241, "lon": -98.4936},
    "austin":      {"lat": 30.2672, "lon": -97.7431},
    "round_rock":  {"lat": 30.5083, "lon": -97.6789},
    "san_marcos":  {"lat": 29.8833, "lon": -97.9414},
}
CITIES = list(CITY_COORDS)

# Open-Meteo variable -> our column name.
WEATHER_VARS = {
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "humidity_pct",
    "precipitation": "precip_mm",
}
FORECAST_LEADS = {"previous_day1": "d1", "previous_day2": "d2"}

TRAIN_START = date(2016, 1, 1)
# First day Previous Runs serves each variable (audit/FINDINGS.md §1).
FC_PROXY_START = {"temp_c": date(2021, 3, 24), "humidity_pct": date(2024, 1, 19), "precip_mm": date(2024, 1, 19)}

# Winter Storm Uri rotating outages: measured load is below true demand.
URI_DAYS_CENTRAL = (date(2021, 2, 10), date(2021, 2, 20))


@cache
def contract() -> dict:
    """features.yaml, parsed."""
    return yaml.safe_load((ROOT / "features.yaml").read_text())
