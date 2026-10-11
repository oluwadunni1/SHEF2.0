"""
gridstatus_latency.py — P0 audit: how long after an hour ends does its South
Central load appear in GridStatus?

That number decides which load lags are legal at forecast time:
  - day-ahead: anything older than (issue_time - latency) is usable
  - hour-ahead: only possible if latency is well under an hour

Two modes:

  snapshot   One call. Prints the dataset's columns, the newest hour available
             right now, and how old it is. Good for a first look.

  probe      Appends one observation per call to audit/data/gridstatus_probe.csv:
             when we checked, and every interval visible in the last 6 hours.
             `--loop SECONDS` repeats until stopped. Run it for a few days.

  report     Reads the probe log. For each interval, latency = first time we saw
             it minus interval_end. Resolution is the probe spacing, so probe
             every 5–10 minutes, not hourly.

Needs GRIDSTATUS_API_KEY in SHEF2.0/.env (same key v1 used).

USAGE
    uv run python audit/gridstatus_latency.py snapshot
    uv run python audit/gridstatus_latency.py probe --loop 600
    uv run python audit/gridstatus_latency.py report
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from shef.sources.gridstatus import DATASET, LOAD_COLUMN
from shef.sources.gridstatus import client as _client

LOOKBACK_HOURS = 6

ROOT = Path(__file__).resolve().parent
PROBE_LOG = ROOT / "data" / "gridstatus_probe.csv"
REPORTS = ROOT / "reports"


def _recent(client, hours: int) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    df = client.get_dataset(
        dataset=DATASET,
        start=(now - pd.Timedelta(hours=hours)).isoformat(),
        end=(now + pd.Timedelta(hours=1)).isoformat(),
        verbose=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    for col in ("interval_start_utc", "interval_end_utc"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def snapshot() -> None:
    client = _client()
    now = pd.Timestamp.now(tz="UTC")
    df = _recent(client, 48)
    if df.empty:
        print(f"No rows returned for the last 48h (checked {now}).")
        return
    print("Columns:", list(df.columns))
    newest = df.sort_values("interval_start_utc").iloc[-1]
    end = newest.get("interval_end_utc", newest["interval_start_utc"] + pd.Timedelta(hours=1))
    print(f"Checked at        {now}")
    print(f"Newest interval   {newest['interval_start_utc']} -> {end}")
    print(f"Age of newest end {(now - end)}")
    print(f"{LOAD_COLUMN} value     {newest.get(LOAD_COLUMN)}")
    print(f"Rows in last 48h  {len(df)} (expect ~48 for hourly data)")
    extra = [c for c in df.columns if "publish" in c.lower() or "updated" in c.lower()]
    if extra:
        print("Publish-time columns found:", extra)
        print(df[["interval_start_utc", *extra]].tail(6).to_string(index=False))


def probe_once(client) -> int:
    now = pd.Timestamp.now(tz="UTC")
    df = _recent(client, LOOKBACK_HOURS)
    PROBE_LOG.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        rows = pd.DataFrame([{"checked_at_utc": now, "interval_start_utc": pd.NaT, "interval_end_utc": pd.NaT, LOAD_COLUMN: None}])
    else:
        rows = df[["interval_start_utc", "interval_end_utc", LOAD_COLUMN]].copy()
        rows.insert(0, "checked_at_utc", now)
    rows.to_csv(PROBE_LOG, mode="a", header=not PROBE_LOG.exists(), index=False)
    newest = df["interval_end_utc"].max() if not df.empty else None
    print(f"{now:%Y-%m-%d %H:%M:%S}Z  newest interval end {newest}  age {now - newest if newest is not None else 'n/a'}")
    return len(rows)


def probe(loop_seconds: int | None) -> None:
    client = _client()
    while True:
        try:
            probe_once(client)
        except Exception as exc:  # noqa: BLE001 — keep a long-running probe alive through a transient failure
            print(f"probe failed: {exc}")
        if not loop_seconds:
            return
        time.sleep(loop_seconds)


def report() -> None:
    if not PROBE_LOG.exists():
        sys.exit(f"No probe log at {PROBE_LOG}. Run `probe` first.")
    log = pd.read_csv(PROBE_LOG, parse_dates=["checked_at_utc", "interval_start_utc", "interval_end_utc"]).dropna(subset=["interval_end_utc"])
    checks = log["checked_at_utc"].drop_duplicates().sort_values()
    spacing = checks.diff().median()

    first_seen = log.groupby("interval_end_utc")["checked_at_utc"].min().rename("first_seen").reset_index()
    # The first interval visible in the very first probe was already there
    # before we started looking — its latency is unknown, so drop it.
    first_seen = first_seen[first_seen["first_seen"] > checks.iloc[0]]
    first_seen["latency_min"] = (first_seen["first_seen"] - first_seen["interval_end_utc"]).dt.total_seconds() / 60
    first_seen["hour_central"] = first_seen["interval_end_utc"].dt.tz_convert("America/Chicago").dt.hour

    REPORTS.mkdir(parents=True, exist_ok=True)
    first_seen.to_csv(REPORTS / "gridstatus_latency.csv", index=False)

    lat = first_seen["latency_min"]
    print(f"Probes: {len(checks)}  median spacing: {spacing}  intervals measured: {len(lat)}")
    if lat.empty:
        print("Not enough probes yet.")
        return
    print("Latency after interval end (minutes, upper bound within one probe spacing):")
    print(lat.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(1).to_string())
    print(f"Share published within 60 min: {(lat <= 60).mean():.1%}")
    print(f"Share published within 120 min: {(lat <= 120).mean():.1%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot")
    p = sub.add_parser("probe")
    p.add_argument("--loop", type=int, default=None, help="seconds between probes; omit for a single probe")
    sub.add_parser("report")
    args = parser.parse_args()

    if args.cmd == "snapshot":
        snapshot()
    elif args.cmd == "probe":
        probe(args.loop)
    else:
        report()


if __name__ == "__main__":
    main()
