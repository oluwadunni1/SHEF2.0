"""
promote.py — monthly champion/challenger check (monitoring.promote).

    uv run --env-file .env python scripts/promote.py                       # real: may register + move @champion
    uv run --env-file .env python scripts/promote.py --dry-run             # evaluate and log; change nothing
    uv run --env-file .env python scripts/promote.py --dry-run \\
        --as-of 2026-09-30 --simulate-champion-end 2026-07-31              # rehearse on a past window

--simulate-champion-end stands in a champion refit through that date (instead
of the registered one), so the whole path can be exercised before enough live
time has passed. It always implies --dry-run.

Every evaluation is logged to MLflow (tag experiment=promotion, dry_run=...);
real runs also go to monitoring.promotions (MONITOR_DATABASE_URL) for the dashboard.
On a promotion it writes promoted=true and version=N to $GITHUB_OUTPUT so the
workflow can dispatch deploy_api.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from shef.config import CENTRAL_TZ, REPORTS
from shef.models import data, registry, tracking
from shef.models.predict import Champion
from shef.monitoring import promote


def log_db(ev: promote.Evaluation, champion_version: str, challenger_version: str | None, run_id: str) -> None:
    url = os.environ.get("MONITOR_DATABASE_URL")
    if not url:
        return
    import psycopg

    c, h = ev.champion or {}, ev.challenger or {}
    with psycopg.connect(url, prepare_threshold=None, connect_timeout=15) as conn:
        conn.execute("""INSERT INTO monitoring.promotions
            (champion_version, challenger_version, window_start, window_end, n_days, champion_mape, challenger_mape,
             champion_peak_mape, challenger_peak_mape, decision, reason, mlflow_run_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                     (champion_version, challenger_version, ev.window_start, ev.window_end, ev.n_days,
                      c.get("mape"), h.get("mape"), c.get("peak_mape"), h.get("peak_mape"),
                      ev.decision, ev.reason, run_id))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--as-of", type=date.fromisoformat, default=None, help="last day the window may use (default: today, Central)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--simulate-champion-end", type=date.fromisoformat, default=None)
    args = p.parse_args()
    dry = args.dry_run or args.simulate_champion_end is not None
    as_of = args.as_of or pd.Timestamp.now(tz=CENTRAL_TZ).date()
    selection = json.loads((REPORTS / "p3_selection.json").read_text())
    params = selection["params"] or None

    mlflow = tracking.init()
    tables = data.load_tables(selection["features"])
    if args.simulate_champion_end:
        any_champ = Champion.load(Path(_fetch("champion")) / "champion")
        champion, _ = promote.refit_all(tables, any_champ, args.simulate_champion_end, params)
        champ_version, champ_end = f"simulated-{args.simulate_champion_end}", args.simulate_champion_end
    else:
        out = Path(_fetch("champion"))
        champion = Champion.load(out / "champion")
        meta = json.loads((out / "model_meta.json").read_text())
        champ_version, champ_end = meta["model_version"], date.fromisoformat(meta["trained_through"])

    ev, challenger = promote.evaluate(tables, champion, champ_end, as_of, params)
    print(f"champion {champ_version} (trained through {champ_end}) | window {ev.window_start}..{ev.window_end} "
          f"({ev.n_days} days) | challenger through {ev.challenger_train_end}")
    for who, s in (("champion", ev.champion), ("challenger", ev.challenger)):
        if s:
            print(f"  {who:<10} MAPE {s['mape']:.2f}%  Peak-MAPE {s['peak_mape']:.2f}%  RMSE {s['rmse']:.0f}  bias {s['bias_pct']:+.2f}%")
    print(f"decision: {ev.decision.upper()} - {ev.reason}" + ("  [dry run]" if dry else ""))

    metrics = {f"{who}_{k}": v for who, s in (("champion", ev.champion), ("challenger", ev.challenger)) if s for k, v in s.items()}
    with mlflow.start_run(run_name=f"promotion_{as_of}") as run:
        mlflow.set_tags({**tracking.provenance(), "experiment": "promotion", "decision": ev.decision, "dry_run": str(dry)})
        mlflow.log_params({"champion_version": champ_version, "champion_train_end": str(champ_end), "as_of": str(as_of),
                           "window_days": promote.WINDOW_DAYS, "margin_pts": promote.MARGIN})
        mlflow.log_metrics(metrics)
        mlflow.log_dict(ev.to_dict(), "evaluation.json")
        run_id = run.info.run_id

    new_version = None
    if ev.decision == "promote" and not dry:
        model, n = promote.refit_all(tables, champion, ev.window_end, params)
        new_version, _ = registry.register(
            model, f"promoted_{as_of}",
            params={"model": "lgbm", "target": model.target.kind, "train_source": "obs", "uri": "exclude",
                    "train_rows": n, "train_end": str(ev.window_end), "n_features": len(model.features)},
            metrics=metrics, tags={"experiment": "production", "promoted_over": champ_version},
            dicts={"evaluation.json": ev.to_dict()})
        registry.set_champion(new_version)
        print(f"registered {registry.NAME} v{new_version}; @champion -> v{new_version}")
        if gh := os.environ.get("GITHUB_OUTPUT"):
            with open(gh, "a") as f:
                f.write(f"promoted=true\nversion={new_version}\ntrained_through={ev.window_end}\n")
    if not dry:
        log_db(ev, champ_version, new_version, run_id)


def _fetch(ref: str) -> str:
    out = Path(tempfile.mkdtemp()) / "model"
    registry.fetch(ref, out)
    return str(out)


if __name__ == "__main__":
    main()
