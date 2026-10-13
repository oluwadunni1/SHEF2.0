"""
load_backtest.py — put the P3 holdout into Supabase as labelled backtest rows,
so monitoring has a reference series from day one.

The rows are recomputed, not copied: the holdout model logged by notebook 05
(MLflow run P3_holdout_lgbm_trend, artifact champion/champion_holdout) is run
on the holdout feature rows again, and must reproduce the logged out-of-fold
predictions before anything is written. They are bulk-loaded (COPY into a temp
table, then one INSERT ... ON CONFLICT DO NOTHING, so re-running is a no-op) with:

    model_version  'backtest-1'
    weather_model  'previous_runs_d1'   (archived proxy, not a live snapshot)
    actual_load_mw the realised load

    uv run --env-file .env python scripts/load_backtest.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from shef.config import REPORTS
from shef.features import bundle as bundle_mod
from shef.models import data, tracking
from shef.models.predict import Champion
from shef.serving import db

RUN_NAME = "P3_holdout_lgbm_trend"
VERSION = "backtest-1"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    mlflow = tracking.init()
    run = mlflow.search_runs(experiment_names=[tracking.EXPERIMENT],
                             filter_string=f"tags.mlflow.runName = '{RUN_NAME}'",
                             order_by=["start_time DESC"], max_results=1).iloc[0]
    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(run_id=run.run_id, artifact_path="champion/champion_holdout",
                                                   dst_path=tmp)
        champ = Champion.load(Path(path))
    oof = tracking.load_oof(RUN_NAME).sort_values("ts_utc").reset_index(drop=True)

    features = json.loads((REPORTS / "p3_selection.json").read_text())["features"]
    fd = data.holdout_data(data.load_tables(features), train_source="obs", uri="exclude", final=True)
    rows = fd.val.sort_values("ts_utc").reset_index(drop=True)
    pred = champ.predict(rows)
    assert len(rows) == len(oof) and (rows["ts_utc"].to_numpy() == oof["ts_utc"].to_numpy()).all()
    gap = float(np.max(np.abs(pred - oof["pred"].to_numpy())))
    assert gap < 1e-6, f"recomputed holdout differs from the logged OOF by {gap} MW"
    print(f"{len(rows):,} holdout hours recomputed; max difference vs logged OOF {gap:.1e} MW")

    inputs = bundle_mod.apply(rows, champ.bundle)[champ.features]
    records = [{"target_timestamp": ts.to_pydatetime(), "features": feats, "predicted_load_mw": float(y_hat),
                "model_name": "shef2-day-ahead", "model_version": VERSION,
                "issue_time_utc": it.to_pydatetime(), "lead_hours": int(lead),
                "weather_model": "previous_runs_d1", "mlflow_run_id": run.run_id}
               for ts, feats, y_hat, it, lead in zip(rows["ts_utc"], inputs.to_dict("records"), pred,
                                                     rows["issue_time_utc"], rows["lead_hours"])]
    if args.dry_run:
        print("dry run: nothing written")
        return
    n = bulk_insert(records, rows["y"].to_numpy(float))
    print(f"inserted {n:,} rows as {VERSION} (0 means already loaded)")


def bulk_insert(records: list[dict], actual: np.ndarray) -> int:
    from psycopg.types.json import Jsonb

    cols = ["target_timestamp", "features", "predicted_load_mw", "model_name", "model_version",
            "issue_time_utc", "lead_hours", "weather_model", "mlflow_run_id"]
    with db._connect() as conn, conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE bt (LIKE predictions INCLUDING DEFAULTS) ON COMMIT DROP")
        with cur.copy(f"COPY bt ({', '.join(cols)}, actual_load_mw) FROM STDIN") as cp:
            for r, a in zip(records, actual):
                cp.write_row([Jsonb(db.json_safe(r["features"])) if c == "features" else r[c] for c in cols] + [float(a)])
        cur.execute(f"""INSERT INTO predictions ({', '.join(cols)}, actual_load_mw)
                        SELECT {', '.join(cols)}, actual_load_mw FROM bt
                        ON CONFLICT (target_timestamp, model_version) DO NOTHING""")
        return cur.rowcount


if __name__ == "__main__":
    main()
