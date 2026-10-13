"""
ERCOT benchmark: store ERCOT's own point-in-time day-ahead forecast next to
ours (monitoring.ercot_forecast; compared in monitoring.v_benchmark*).

Every fetch costs one GridStatus request and the Free plan allows 250 a month,
so optional calls go through `can_spend`, which keeps RESERVE requests back
for what the forecast itself needs (the daily load fetch, the monthly refresh).
"""
from __future__ import annotations

import pandas as pd

RESERVE = 60

UPSERT = """
INSERT INTO monitoring.ercot_forecast (target_timestamp, issue_time_utc, publish_time_utc, forecast_mw)
VALUES (%s, %s, %s, %s)
ON CONFLICT (target_timestamp) DO UPDATE
SET issue_time_utc = EXCLUDED.issue_time_utc, publish_time_utc = EXCLUDED.publish_time_utc,
    forecast_mw = EXCLUDED.forecast_mw, fetched_at = now()
"""

USAGE = """
INSERT INTO monitoring.api_usage (period_start, requests_used, requests_limit, rows_used, rows_limit)
VALUES (%(period_start)s, %(requests_used)s, %(requests_limit)s, %(rows_used)s, %(rows_limit)s)
"""


def can_spend(usage: dict, n: int = 1, reserve: int = RESERVE) -> bool:
    return usage["requests_used"] + n <= usage["requests_limit"] - reserve


def store(conn, forecast: pd.DataFrame, issue_time: pd.Timestamp) -> int:
    for r in forecast.itertuples(index=False):
        conn.execute(UPSERT, (r.ts_utc.to_pydatetime(), issue_time.to_pydatetime(),
                              r.publish_time_utc.to_pydatetime(), float(r.forecast_mw)))
    return len(forecast)


def record_usage(conn, usage: dict) -> None:
    conn.execute(USAGE, usage)


def have_days(conn) -> set:
    return {d for (d,) in conn.execute(
        "SELECT DISTINCT (target_timestamp AT TIME ZONE 'America/Chicago')::date FROM monitoring.ercot_forecast")}
