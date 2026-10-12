"""
E10: the same forecast proxy, but from one fixed weather model instead of
Open-Meteo's `best_match` blend, so the two can be compared on identical rows.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from shef import schemas
from shef.align import align_weather
from shef.config import CACHE, CITIES, FC_PROXY_START
from shef.sources import openmeteo
from shef.weather_weights import weight_cities


def proxy_variant(model: str, start: date, end: date) -> pd.DataFrame:
    """Population-weighted *_d1/*_d2 columns from Previous Runs for one model."""
    frames = []
    for year in range(start.year, end.year + 1):
        a, b = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        for city in CITIES:
            frames.append(openmeteo.fetch("previous_runs", city, a, b, cache_dirs=[CACHE / "openmeteo"], model=model))
    raw = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    aligned = align_weather(raw, "previous_runs")
    for col in schemas.PROXY_COLS:
        first = pd.Timestamp(FC_PROXY_START[col.rsplit("_", 1)[0]], tz="UTC")
        aligned.loc[aligned["ts_utc"] < first, col] = float("nan")
    return weight_cities(aligned, schemas.PROXY_COLS)


def joined_with(joined: pd.DataFrame, proxy: pd.DataFrame) -> pd.DataFrame:
    """The clean joined table with its proxy columns swapped for another model's."""
    base = joined.drop(columns=schemas.PROXY_COLS)
    return base.merge(proxy, on="ts_utc", how="left")
