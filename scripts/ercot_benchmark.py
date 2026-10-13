"""
ercot_benchmark.py — backfill ERCOT's point-in-time day-ahead forecast for a
range of target days, one GridStatus request per day, resumable.

Days already stored are skipped, and it stops before the plan's request
budget falls under the reserve the daily job needs, so it can be re-run each
month after the quota resets until the whole holdout is covered.

    uv run --env-file .env python scripts/ercot_benchmark.py --start 2026-06-12 --end 2026-10-09
"""
from __future__ import annotations

import argparse
import os
from datetime import date, timedelta

import psycopg

from shef.monitoring import benchmark
from shef.sources import gridstatus
from shef.timestamps import issue_time_utc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--reserve", type=int, default=benchmark.RESERVE)
    args = p.parse_args()

    gs = gridstatus.client()
    with psycopg.connect(os.environ["MONITOR_DATABASE_URL"], prepare_threshold=None, connect_timeout=15,
                         autocommit=True) as conn:
        have = benchmark.have_days(conn)
        todo = [args.start + timedelta(days=i) for i in range((args.end - args.start).days + 1)]
        todo = [d for d in todo if d not in have]
        usage = gridstatus.usage(gs)
        budget = usage["requests_limit"] - args.reserve - usage["requests_used"]
        print(f"{len(todo)} day(s) to fetch; quota {usage['requests_used']}/{usage['requests_limit']} used, "
              f"reserve {args.reserve} -> budget {max(budget, 0)}")
        done = 0
        for d in todo:
            if done >= budget:
                print(f"stopping at the reserve: {len(todo) - done} day(s) left for after the quota resets")
                break
            it = issue_time_utc(d)
            fc = gridstatus.fetch_ercot_forecast(d, it, gs)
            done += 1
            n = benchmark.store(conn, fc, it)
            if n < 23:
                print(f"  {d}: only {n} hours published before {it}")
        usage = gridstatus.usage(gs)
        benchmark.record_usage(conn, usage)
        print(f"fetched {done} day(s); quota now {usage['requests_used']}/{usage['requests_limit']}")


if __name__ == "__main__":
    main()
