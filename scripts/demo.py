"""
demo.py — one safe command per live-demo step. Everything here is read-only:
nothing is deployed, logged or written, and no file is edited.

    uv run --env-file .env python scripts/demo.py status     # registry @champion + the API's front door
    uv run --env-file .env python scripts/demo.py numbers    # headline numbers, read from the database views
    uv run python scripts/demo.py leak                       # the leakage test, clean then with v1's leak planted
    uv run --env-file .env python scripts/demo.py row        # one logged forecast hour, fully traceable

The database commands use DASHBOARD_DATABASE_URL (the read-only role), never
the postgres login.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date

import pandas as pd

LINE = "─" * 64


def head(title: str) -> None:
    print(f"\n── {title} {LINE[len(title) + 4:]}")


def _db():
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(os.environ["DASHBOARD_DATABASE_URL"], prepare_threshold=None, row_factory=dict_row,
                           connect_timeout=15)


# --- status ------------------------------------------------------------------
def status() -> None:
    import requests

    from shef.models import registry

    head(f"registry: {registry.NAME}  (DagsHub MLflow)")
    mv = registry.resolve(registry.ALIAS)
    run = registry.tracking.init().get_run(mv.run_id)
    print(f"  @{registry.ALIAS:<10} v{mv.version}   run {mv.run_id[:12]}   trained through {run.data.params.get('train_end')}")

    url = os.environ.get("SHEF_API_URL") or _repo_variable("SHEF_API_URL")
    head(f"API: {url}")
    r = requests.get(url.rstrip("/") + "/health", timeout=60)
    print(f"  unsigned GET /health  ->  HTTP {r.status_code}  {r.text[:60]}")
    print("  (AuthType AWS_IAM: Lambda refuses any request not SigV4-signed by an allowed role,")
    print("   before the app runs. The signed call is made by the GitHub OIDC role: see the daily run.)")


def _repo_variable(name: str) -> str:
    import shutil
    import subprocess

    gh = shutil.which("gh") or r"C:\Program Files\GitHub CLI\gh.exe"
    return subprocess.run([gh, "variable", "get", name], capture_output=True, text=True, check=True).stdout.strip()


# --- numbers -----------------------------------------------------------------
def numbers() -> None:
    with _db() as c:
        s = c.execute("SELECT * FROM monitoring.v_summary WHERE model_version = 'backtest-1'").fetchone()
        b = c.execute("SELECT * FROM monitoring.v_benchmark_summary WHERE kind = 'backtest'").fetchone()
        live = c.execute("SELECT * FROM monitoring.v_window_30d WHERE kind = 'live'").fetchone()
        nlive = c.execute("SELECT count(DISTINCT target_day) AS d, count(*) AS h FROM monitoring.v_live").fetchone()
    head("holdout backtest (archived weather forecasts, scored once)")
    print(f"  {s['first_day']} -> {s['last_day']}  {s['n_days']} days, {s['n_hours']:,} hours")
    print(f"  MAPE {s['mape']:.2f}%   Peak-MAPE {s['peak_mape']:.2f}%   RMSE {s['rmse']:,.0f} MW   bias {s['bias_pct']:+.2f}%")
    print("  baseline (freshest legal same-hour load): MAPE 7.62%   RMSE 1,013 MW")
    head("against ERCOT's own forecast (09:30 CT publish, same cut-off)")
    print(f"  {b['first_day']} -> {b['last_day']}  {b['n_days']} matched days")
    print(f"  ERCOT  MAPE {b['ercot_mape']:.2f}%   Peak {b['ercot_peak_mape']:.2f}%   RMSE {b['ercot_rmse']:,.0f}   bias {b['ercot_bias_pct']:+.2f}%")
    print(f"  ours   MAPE {b['ours_mape']:.2f}%   Peak {b['ours_peak_mape']:.2f}%   RMSE {b['ours_rmse']:,.0f}   bias {b['ours_bias_pct']:+.2f}%")
    print(f"  we are closer in {float(b['ours_wins_pct']):.0f}% of hours")
    head("live")
    print(f"  {nlive['d']} target day(s), {nlive['h']} hour(s) forecast so far")
    if live:
        print(f"  last 30 scored days: MAPE {live['mape']:.2f}%   Peak {live['peak_mape']:.2f}%   bias {live['bias_pct']:+.2f}%")
    else:
        print("  no scored live day yet: actuals arrive the morning after each target day")


# --- leak --------------------------------------------------------------------
def leak() -> int:
    """Run the real leakage check on five tricky days (DST, summer, autumn), first as
    the code stands, then with v1's mistake planted in memory: the load view reads
    24 hours past the cutoff, i.e. all of yesterday, which a 10:00 forecast cannot have."""
    sys.path.insert(0, "tests")
    from test_features import features_only, poison_future, synthetic_joined

    from shef.features import replay as replay_mod
    from shef.features.asof import load_as_of
    from shef.timestamps import issue_time_utc

    days = [date(2025, 3, 9), date(2025, 3, 10), date(2025, 7, 1), date(2025, 11, 2), date(2025, 11, 3)]
    joined = synthetic_joined()

    def check(label: str) -> bool:
        head(label)
        ok = True
        for d in days:
            issue = issue_time_utc(d)
            clean = features_only(replay_mod.replay(joined, d, d, "proxy"))
            poisoned = features_only(replay_mod.replay(poison_future(joined, issue), d, d, "proxy"))
            moved = [c for c in clean.columns
                     if not clean[c].equals(poisoned[c]) and not (clean[c].isna() & poisoned[c].isna()).all()]
            print(f"  {d}  " + ("PASS  no feature moved" if not moved else
                               f"FAIL  {len(moved)} feature(s) moved: {', '.join(moved[:4])}{' …' if len(moved) > 4 else ''}"))
            ok &= not moved
        return ok

    print("Every value after the 10:00 cut-off is poisoned (load x1000, weather +77 °C).")
    print("If a feature changes, it read the future.")
    clean_ok = check("the code as it is")

    def leaky_load_as_of(load, issue_time, extra_delay=pd.Timedelta(0)):
        return load_as_of(load, issue_time, extra_delay - pd.Timedelta(hours=24))   # v1: all of yesterday

    original = replay_mod.load_as_of
    replay_mod.load_as_of = leaky_load_as_of
    try:
        leak_caught = not check("with v1's leak planted: load read 24 h past the cut-off")
    finally:
        replay_mod.load_as_of = original
    head("result")
    print(f"  clean code passes: {clean_ok}   planted leak caught: {leak_caught}   (nothing on disk was changed)")
    return 0 if clean_ok and leak_caught else 1


# --- row ---------------------------------------------------------------------
def row() -> None:
    with _db() as c:
        r = c.execute("""SELECT l.*, i.features FROM monitoring.v_live l
                         JOIN monitoring.v_live_inputs i
                           ON i.ts_utc = l.target_timestamp AND i.model_version = l.model_version
                         WHERE l.served_first ORDER BY l.issue_time_utc DESC, l.target_timestamp
                         OFFSET 17 LIMIT 1""").fetchone()
    if r is None:
        print("no live forecast logged yet (the first scheduled run logs tomorrow's 24 hours)")
        return
    local = pd.Timestamp(r["target_timestamp"]).tz_convert("America/Chicago")
    head(f"one logged forecast hour: {local:%Y-%m-%d %H:%M %Z}")
    print(f"  forecast        {r['predicted_load_mw']:,.1f} MW" +
          (f"   actual {r['actual_load_mw']:,.1f} MW" if r["actual_load_mw"] is not None else "   actual: not in yet"))
    print(f"  served by       {r['model_name']} v{r['model_version']} ({r['served_as']})   MLflow run {r['mlflow_run_id'][:12]}")
    print(f"  issued          {pd.Timestamp(r['issue_time_utc']).tz_convert('America/Chicago'):%Y-%m-%d %H:%M %Z}"
          f"   lead {r['lead_hours']} h")
    f = r["features"]
    print(f"  {len(f)} model inputs, exactly as the model saw them, e.g.")
    for k in ("temp_c", "tmax", "load_same_hour_latest", "same_hour_age_h", "load_lag_168", "trend_idx", "load_staleness_h"):
        if k in f:
            print(f"    {k:<24} {f[k]}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["status", "numbers", "leak", "row"])
    args = p.parse_args()
    return {"status": status, "numbers": numbers, "leak": leak, "row": row}[args.cmd]() or 0


if __name__ == "__main__":
    sys.exit(main())
