"""
render_dashboard.py — the public monitoring page (GitHub Pages).

Reads only the monitoring schema, as dashboard_reader (DASHBOARD_DATABASE_URL),
and writes one static site/index.html with the data embedded: no database
credentials or queries ever reach the browser.

    uv run --env-file .env python scripts/render_dashboard.py [--out site]
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

TEMPLATE = Path(__file__).with_name("dashboard_template.html")


def _jsonable(v):
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return v


def rows(conn, sql: str, *args) -> list[dict]:
    return [{k: _jsonable(v) for k, v in r.items()} for r in conn.execute(sql, args).fetchall()]


def gather(conn) -> dict:
    live = rows(conn, """SELECT target_timestamp, target_day, issue_time_utc, lead_hours, model_name, model_version,
                                mlflow_run_id, served_as, predicted_load_mw, actual_load_mw
                         FROM monitoring.v_live WHERE served_first ORDER BY target_timestamp""")
    latest_issue = max((r["issue_time_utc"] for r in live), default=None)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "summary": rows(conn, "SELECT * FROM monitoring.v_summary ORDER BY kind, model_version"),
        "window_30d": rows(conn, "SELECT * FROM monitoring.v_window_30d ORDER BY kind"),
        "daily": rows(conn, """SELECT kind, target_day, mape, bias_pct, n_hours FROM monitoring.v_daily
                               WHERE model_version = 'backtest-1' OR kind = 'live' ORDER BY target_day"""),
        "by_lead": rows(conn, "SELECT kind, lead_hours, mape, n_hours FROM monitoring.v_by_lead ORDER BY kind, lead_hours"),
        "latest": [r for r in live if r["issue_time_utc"] == latest_issue],
        "recent": [r for r in live if r["actual_load_mw"] is not None][-24 * 14:],
        "live_days": len({r["target_day"] for r in live if r["actual_load_mw"] is not None}),
        "fallback_hours": sum(r["served_as"] == "fallback" for r in live),
        "weather": rows(conn, "SELECT * FROM monitoring.weather_checks ORDER BY issue_date DESC, comparison LIMIT 60"),
        "drift": rows(conn, """SELECT * FROM monitoring.drift
                               WHERE computed_on = (SELECT max(computed_on) FROM monitoring.drift) ORDER BY feature"""),
        "promotions": rows(conn, "SELECT * FROM monitoring.promotions ORDER BY evaluated_at DESC LIMIT 12"),
        "model": live[-1] if live else None,
        "run_url": (f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
                    if os.environ.get("GITHUB_RUN_ID") else None),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("site"))
    args = p.parse_args()
    with psycopg.connect(os.environ["DASHBOARD_DATABASE_URL"], prepare_threshold=None, row_factory=dict_row,
                         connect_timeout=15) as conn:
        data = gather(conn)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)
    (args.out / "index.html").write_text(html, encoding="utf-8")
    (args.out / ".nojekyll").write_text("")
    print(f"{args.out / 'index.html'}: {data['live_days']} live day(s), {len(data['daily'])} daily rows, "
          f"{len(data['latest'])} hours in the latest forecast, {len(html) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
