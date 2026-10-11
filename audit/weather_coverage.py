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
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

CITY_COORDS = {
    "san_antonio": {"lat": 29.4241, "lon": -98.4936},
    "austin":      {"lat": 30.2672, "lon": -97.7431},
    "round_rock":  {"lat": 30.5083, "lon": -97.6789},
    "san_marcos":  {"lat": 29.8833, "lon": -97.9414},
}
VARS = ["temperature_2m", "relative_humidity_2m", "precipitation"]
LEADS = ["previous_day1", "previous_day2"]

URLS = {
    "archive": "https://archive-api.open-meteo.com/v1/archive",
    "historical_fc": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "previous_runs": "https://previous-runs-api.open-meteo.com/v1/forecast",
}

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "weather"
REPORTS = ROOT / "reports"


def _hourly_vars(source: str) -> list[str]:
    if source == "previous_runs":
        return [f"{v}_{lead}" for v in VARS for lead in LEADS]
    return VARS


def fetch(source: str, city: str, start: date, end: date) -> pd.DataFrame:
    """One city, one source, one date range. Cached on disk as JSON."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"{source}_{city}_{start}_{end}.json"
    if cache_file.exists():
        payload = json.loads(cache_file.read_text())
    else:
        coords = CITY_COORDS[city]
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(_hourly_vars(source)),
            "timezone": "UTC",
        }
        for attempt in range(4):
            resp = requests.get(URLS[source], params=params, timeout=120)
            if resp.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            break
        if resp.status_code != 200:
            # A range the API does not serve at all comes back as 400 — record it
            # as empty coverage rather than failing the whole audit.
            print(f"  {source} {city} {start}..{end}: HTTP {resp.status_code} {resp.text[:160]}")
            return pd.DataFrame()
        payload = resp.json()
        cache_file.write_text(json.dumps(payload))

    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["city"] = city
    df["source"] = source
    return df


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
    frames: dict[str, list[pd.DataFrame]] = {s: [] for s in URLS}
    for year in range(args.start_year, args.end_year + 1):
        start = date(year, 1, 1)
        end = min(date(year, 12, 31), today)
        for source in URLS:
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
