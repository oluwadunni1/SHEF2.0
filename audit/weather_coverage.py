"""
weather_coverage.py — P0 audit: which weather data exists for training, and
how far off a day-ahead forecast is from what actually happened.

Three Open-Meteo sources, same four cities v1 used:

  archive           archive-api.open-meteo.com      ERA5 reanalysis ("observed").
                                                    What v1 trained on.
  historical_fc     historical-forecast-api...      Stitched archive of the
                                                    first hours of each model run.
                                                    Short lead time only.
  previous_runs     previous-runs-api...            The same hour as forecast
                                                    1 and 2 days earlier
                                                    (`*_previous_day1/2`). This
                                                    is the closest match to a
                                                    day-ahead serving path.

For each source and year it reports the share of non-null hourly values, then
for every hour where previous_runs and archive both exist it reports the
forecast error by lead day and month. Results go to audit/reports/.

USAGE
    uv run python audit/weather_coverage.py [--start-year 2016] [--end-year 2026]
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from shef.config import CITY_COORDS, FORECAST_LEADS, WEATHER_VARS
from shef.sources import openmeteo

VARS = list(WEATHER_VARS)
LEADS = list(FORECAST_LEADS)
SOURCES = ["archive", "historical_fc", "previous_runs"]

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "weather"
REPORTS = ROOT / "reports"


def fetch(source: str, city: str, start: date, end: date) -> pd.DataFrame:
    return openmeteo.fetch(source, city, start, end, cache_dirs=[CACHE])


def coverage_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Share of non-null values per source, year and variable (all cities pooled)."""
    rows = []
    for source, df in frames.items():
        if df.empty:
            continue
        value_cols = [c for c in df.columns if c not in ("time", "city", "source")]
        df = df.assign(year=df["time"].dt.year)
        for year, grp in df.groupby("year"):
            row = {"source": source, "year": int(year), "hours": len(grp) // len(CITY_COORDS)}
            for col in value_cols:
                row[col] = round(float(grp[col].notna().mean()), 3)
            rows.append(row)
    return pd.DataFrame(rows)


def skill_table(archive: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """Forecast minus reanalysis, per lead day and calendar month, for hours where both exist."""
    if archive.empty or prev.empty:
        return pd.DataFrame()
    merged = prev.merge(archive, on=["time", "city"], suffixes=("_fc", ""))
    merged["month"] = merged["time"].dt.month
    rows = []
    for lead in LEADS:
        for var in VARS:
            fc_col = f"{var}_{lead}"
            pair = merged[[fc_col, var, "month"]].dropna()
            if pair.empty:
                continue
            err = pair[fc_col] - pair[var]
            for month, e in err.groupby(pair["month"]):
                rows.append({
                    "lead": lead, "variable": var, "month": int(month), "n": len(e),
                    "mae": round(float(e.abs().mean()), 3),
                    "bias": round(float(e.mean()), 3),
                    "p95_abs": round(float(e.abs().quantile(0.95)), 3),
                })
            rows.append({
                "lead": lead, "variable": var, "month": "all", "n": len(err),
                "mae": round(float(err.abs().mean()), 3),
                "bias": round(float(err.mean()), 3),
                "p95_abs": round(float(err.abs().quantile(0.95)), 3),
            })
    return pd.DataFrame(rows)


def first_full_year(cov: pd.DataFrame, source: str, col: str, threshold: float = 0.99) -> int | None:
    sub = cov[(cov["source"] == source) & (cov[col] >= threshold)].sort_values("year")
    return int(sub["year"].iloc[0]) if not sub.empty else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    args = parser.parse_args()

    today = date.today()
    frames: dict[str, list[pd.DataFrame]] = {s: [] for s in SOURCES}
    for year in range(args.start_year, args.end_year + 1):
        start = date(year, 1, 1)
        end = min(date(year, 12, 31), today)
        for source in SOURCES:
            for city in CITY_COORDS:
                print(f"{year} {source:14s} {city}")
                frames[source].append(fetch(source, city, start, end))

    combined = {s: pd.concat([f for f in fs if not f.empty], ignore_index=True) if any(not f.empty for f in fs) else pd.DataFrame()
                for s, fs in frames.items()}

    REPORTS.mkdir(parents=True, exist_ok=True)
    cov = coverage_table(combined)
    cov.to_csv(REPORTS / "weather_coverage.csv", index=False)

    archive = combined["archive"].drop(columns=["source"]) if not combined["archive"].empty else pd.DataFrame()
    prev = combined["previous_runs"].drop(columns=["source"]) if not combined["previous_runs"].empty else pd.DataFrame()
    skill = skill_table(archive, prev)
    skill.to_csv(REPORTS / "weather_forecast_skill.csv", index=False)

    summary = {
        "archive_temperature_first_full_year": first_full_year(cov, "archive", "temperature_2m"),
        "historical_fc_temperature_first_full_year": first_full_year(cov, "historical_fc", "temperature_2m"),
        "previous_runs_day1_first_full_year": first_full_year(cov, "previous_runs", "temperature_2m_previous_day1"),
        "previous_runs_day2_first_full_year": first_full_year(cov, "previous_runs", "temperature_2m_previous_day2"),
    }
    (REPORTS / "weather_summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    print("\n=== Coverage (share of non-null hourly values, 4 cities pooled) ===")
    print(cov.to_string(index=False))
    print("\n=== Day-ahead forecast error vs ERA5 reanalysis (all months) ===")
    if not skill.empty:
        print(skill[skill["month"] == "all"].to_string(index=False))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
