"""
Parquet store: data/<layer>/<name>/year=YYYY.parquet, partitioned by the UTC
year of `ts_col`. Writes are upserts on `keys` (the newest copy of a row wins),
so rerunning a backfill never duplicates rows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from shef.config import DATA


def _dir(layer: str, name: str) -> Path:
    return DATA / layer / name


def upsert(df: pd.DataFrame, layer: str, name: str, keys: list[str], ts_col: str = "ts_utc") -> int:
    """Merge df into the store. Returns the total row count across touched partitions."""
    if df.empty:
        return 0
    d = _dir(layer, name)
    d.mkdir(parents=True, exist_ok=True)
    total = 0
    for year, part in df.groupby(df[ts_col].dt.year):
        path = d / f"year={year}.parquet"
        if path.exists():
            part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
        part = part.drop_duplicates(subset=keys, keep="last").sort_values(keys).reset_index(drop=True)
        part.to_parquet(path, index=False)
        total += len(part)
    return total


def read(layer: str, name: str) -> pd.DataFrame:
    d = _dir(layer, name)
    files = sorted(d.glob("year=*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def write(df: pd.DataFrame, layer: str, name: str, ts_col: str = "ts_utc") -> None:
    """Replace the whole table (used for derived tables like clean/joined)."""
    d = _dir(layer, name)
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("year=*.parquet"):
        f.unlink()
    for year, part in df.groupby(df[ts_col].dt.year):
        part.reset_index(drop=True).to_parquet(d / f"year={year}.parquet", index=False)
