"""
monitor.py — daily monitoring after the forecast: weather-forecast error of
what was served, input drift, GridStatus publishing latency (from the
forecast's own load fetch), ERCOT's own forecast for tomorrow (benchmark;
only while the GridStatus quota has room), and the quota itself. Writes to
the monitoring schema with the monitor_writer login (MONITOR_DATABASE_URL), which cannot touch predictions.

    uv run --env-file .env python scripts/monitor.py [--dry-run]

Best effort by design: the daily workflow runs it with continue-on-error, so a
monitoring hiccup never blocks or fails the forecast.
"""
from __future__ import annotations

import argparse
import os
from datetime import timedelta

import pandas as pd
import psycopg

from shef import store
from shef.config import CENTRAL_TZ, REPORTS
from shef.monitoring import benchmark, drift, latency, weather
from shef.sources import gridstatus
from shef.timestamps import issue_time_utc

UPSERT_WEATHER = """
INSERT INTO monitoring.weather_checks (issue_date, comparison, variable, n_hours, mae, bias, p95_abs)
VALUES (%(issue_date)s, %(comparison)s, %(variable)s, %(n_hours)s, %(mae)s, %(bias)s, %(p95_abs)s)
ON CONFLICT (issue_date, comparison, variable) DO UPDATE
SET n_hours = EXCLUDED.n_hours, mae = EXCLUDED.mae, bias = EXCLUDED.bias, p95_abs = EXCLUDED.p95_abs, computed_at = now()
"""

INSERT_LATENCY = """
INSERT INTO monitoring.load_latency (observed_at, newest_interval_end, run_mode)
VALUES (%(observed_at)s, %(newest_interval_end)s, %(run_mode)s)
ON CONFLICT (observed_at) DO NOTHING
"""

UPSERT_DRIFT = """
INSERT INTO monitoring.drift (computed_on, feature, psi, status, ref_months, n_current)
VALUES (%(computed_on)s, %(feature)s, %(psi)s, %(status)s, %(ref_months)s, %(n_current)s)
ON CONFLICT (computed_on, feature) DO UPDATE
SET psi = EXCLUDED.psi, status = EXCLUDED.status, ref_months = EXCLUDED.ref_months,
    n_current = EXCLUDED.n_current, computed_at = now()
"""


def live_inputs(conn, days: int) -> pd.DataFrame:
    rows = conn.execute("""SELECT ts_utc, features FROM monitoring.v_live_inputs
                           WHERE ts_utc >= now() - make_interval(days => %s) ORDER BY ts_utc, model_version""",
                        (days + 2,)).fetchall()
    if not rows:
        return pd.DataFrame(columns=["ts_utc", *drift.FEATURES])
    df = pd.DataFrame([{"ts_utc": ts, **f} for ts, f in rows]).drop_duplicates("ts_utc")
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    cutoff = df["ts_utc"].max() - pd.Timedelta(days=days)
    return df[df["ts_utc"] > cutoff]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    today = pd.Timestamp.now(tz=CENTRAL_TZ).date()

    with psycopg.connect(os.environ["MONITOR_DATABASE_URL"], prepare_threshold=None, connect_timeout=15) as conn:
        done = {(d, c) for d, c in conn.execute("SELECT issue_date, comparison FROM monitoring.weather_checks")}
        snap = store.read("raw", "weather_fc_live")
        due = weather.due_issue_dates(snap, done, today)
        checks = weather.check(snap, done, today)
        print(f"weather: {len(due)} issue date(s) due (target day >= {weather.ERA5_LAG_DAYS} days old), "
              f"{len(checks)} comparison(s) scored")
        for c in checks:
            print(f"  {c['issue_date']} {c['comparison']:<26} MAE {c['mae']:.2f} C  bias {c['bias']:+.2f}  p95 {c['p95_abs']:.2f}")

        current = live_inputs(conn, drift.CURRENT_DAYS)
        results = drift.evaluate(drift.load_reference(REPORTS / "drift_reference.json"), current)
        if not results:
            print(f"drift: {len(current)} live hours, need {drift.MIN_CURRENT_HOURS}; skipped")
        for r in results:
            print(f"  drift {r['feature']:<26} PSI {r['psi']:.3f}  {r['status']}")

        gs = gridstatus.client()
        usage = gridstatus.usage(gs)
        tomorrow = today + timedelta(days=1)
        issue = issue_time_utc(tomorrow)
        ercot = pd.DataFrame()
        if pd.Timestamp.now(tz="UTC") < issue:
            print(f"ercot: issue time {issue} not reached; skipped")
        elif benchmark.can_spend(usage):
            ercot = gridstatus.fetch_ercot_forecast(tomorrow, issue, gs)
            usage = gridstatus.usage(gs)
            print(f"ercot: {len(ercot)} hours of ERCOT's forecast for {tomorrow} (published by {issue})")
        else:
            print(f"ercot: skipped to protect the quota ({usage['requests_used']}/{usage['requests_limit']} used)")
        print(f"gridstatus quota: {usage['requests_used']}/{usage['requests_limit']} requests, "
              f"{usage['rows_used']:,}/{usage['rows_limit']:,} rows this period")

        obs = latency.load()
        if obs:
            late, within = latency.bounds(obs)
            print(f"load latency: newest hour published within {within:.0f} min; next hour > {late:.0f} min late")

        if args.dry_run:
            print("dry run: nothing written")
            return
        with conn.transaction():
            for c in checks:
                conn.execute(UPSERT_WEATHER, c)
            for r in results:
                conn.execute(UPSERT_DRIFT, {**r, "computed_on": today})
            benchmark.store(conn, ercot, issue)
            benchmark.record_usage(conn, usage)
            if obs:
                conn.execute(INSERT_LATENCY, obs)


if __name__ == "__main__":
    main()
