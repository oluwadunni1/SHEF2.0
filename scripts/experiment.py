"""
experiment.py — run P9 challenger experiments through the harness.

    uv run python scripts/experiment.py --list
    uv run python scripts/experiment.py --parity          # harness reproduces the P3 holdout (3.79%)
    uv run python scripts/experiment.py A3_linear_tree [A4_recency_2y ...] [--no-mlflow]

Each experiment runs the cv and rolling suites (shef.models.harness), is
compared with the baseline under the same protocol (computed once, cached in
build/harness keyed by recipe and feature-data hash), logged to MLflow
(experiment shef2-harness) and recorded in data/reports/harness.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from shef.config import REPORTS, ROOT
from shef.models import data, folds, harness, tracking
from shef.models.experiments import EXPERIMENTS
from shef.models.harness import Window

CACHE = ROOT / "build" / "harness"
LEDGER = REPORTS / "harness.json"
CHAMPION_FEATURES = ROOT / "build" / "model" / "champion" / "features.json"
P3_HOLDOUT_MAPE = 3.79


def base_features() -> list[str]:
    if CHAMPION_FEATURES.exists():
        return json.loads(CHAMPION_FEATURES.read_text())
    return data.default_features()


def data_hash() -> str:
    p = tracking.provenance()
    return "-".join(p[k][:8] for k in sorted(p) if k.startswith("dvc_"))


def scored(exp, feats, tables, suite) -> pd.DataFrame:
    path = CACHE / f"{exp.name}-{exp.signature()}-{data_hash()}-{suite}.parquet"
    if path.exists():
        print(f"  {exp.name:<22} {suite:<7} cached ({path.name})")
        return pd.read_parquet(path)
    s = harness.run_suite(exp, feats, tables, suite)
    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_parquet(path, index=False)
    return s


def record(entry: dict) -> None:
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    ledger = [e for e in ledger if not (e["name"] == entry["name"] and e["signature"] == entry["signature"])]
    LEDGER.write_text(json.dumps(ledger + [entry], indent=2) + "\n")


def brief(m: dict, keys=("mape", "peak_mape", "rmse", "bias_pct")) -> dict:
    return {k: round(float(m[k]), 3) for k in keys if k in m}


def parity(feats, tables) -> int:
    """One fit through the holdout's training cut-off, scored on the whole holdout: must give P3's number."""
    w = Window("holdout", folds.HOLDOUT[0] - folds.GAP - timedelta(days=1), *folds.HOLDOUT)
    s = harness.fit_predict(EXPERIMENTS["baseline"], feats, tables, w)
    m = harness.score(s["y"], s["pred"])
    ok = round(m["mape"], 2) == P3_HOLDOUT_MAPE
    print(f"parity: holdout MAPE {m['mape']:.3f}% peak {m['peak_mape']:.3f}% RMSE {m['rmse']:.0f} "
          f"-> {'matches' if ok else 'DOES NOT match'} P3's {P3_HOLDOUT_MAPE}%")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*")
    p.add_argument("--list", action="store_true")
    p.add_argument("--parity", action="store_true")
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args()

    if args.list or not (args.names or args.parity):
        for e in EXPERIMENTS.values():
            print(f"{e.name:<18} {e.hypothesis}")
        return 0
    unknown = [n for n in args.names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"unknown experiment(s) {unknown}; see --list")

    feats = base_features()
    tables = data.load_tables(list(dict.fromkeys(feats + ["load_lag_168"])))
    if args.parity:
        return parity(feats, tables)

    base_exp = EXPERIMENTS["baseline"]
    base_scored = {s: scored(base_exp, feats, tables, s) for s in harness.SUITES}
    base = {s: harness.suite_metrics(base_scored[s], s) for s in harness.SUITES}
    mlflow = None if args.no_mlflow else tracking.init("shef2-harness")

    for name in args.names:
        exp = EXPERIMENTS[name]
        print(f"\n== {name}: {exp.hypothesis}")
        sc = base_scored if name == "baseline" else {s: scored(exp, feats, tables, s) for s in harness.SUITES}
        res = {s: harness.suite_metrics(sc[s], s) for s in harness.SUITES}
        decision, reason = ("reference", "the champion recipe") if name == "baseline" else harness.verdict(base, res)
        months = harness.by_month(sc["rolling"])
        print(f"   {decision.upper()}: {reason}")
        print("   rolling by month (MAPE / bias %), baseline -> experiment:")
        bm = harness.by_month(base_scored["rolling"])
        for ym in months.index:
            print(f"     {ym}  {bm.loc[ym, 'mape']:5.2f} -> {months.loc[ym, 'mape']:5.2f}   "
                  f"{bm.loc[ym, 'bias_pct']:+5.2f} -> {months.loc[ym, 'bias_pct']:+5.2f}")

        run_id = None
        if mlflow:
            metrics = {**{f"cv_{k}": v for k, v in res["cv"].items()},
                       **{f"rolling_{k}": v for k, v in res["rolling"].items()},
                       "delta_rolling_mape": res["rolling"]["mape"] - base["rolling"]["mape"],
                       "delta_rolling_peak_mape": res["rolling"]["peak_mape"] - base["rolling"]["peak_mape"],
                       "delta_cv_mape": res["cv"]["mape"] - base["cv"]["mape"]}
            params = {"experiment": name, "signature": exp.signature(), "target": exp.target,
                      "lgbm_overrides": exp.params, "weight": exp.weight.__name__ if exp.weight else "none",
                      "derive": list(exp.derive), "drop": exp.drop, "n_features": len(exp.features(feats))}
            run_id = tracking.log_run(name, params, metrics,
                                      frames={f"oof_{s}": sc[s] for s in harness.SUITES},
                                      dicts={"rolling_by_month": months.reset_index().to_dict("list")},
                                      tags={"verdict": decision, "reason": reason, "hypothesis": exp.hypothesis})
        record({"name": name, "signature": exp.signature(), "hypothesis": exp.hypothesis,
                "run_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
                "git_commit": tracking.provenance()["git_commit"], "data": data_hash(),
                "cv": brief(res["cv"]), "rolling": brief(res["rolling"]),
                "rolling_2026_09_mape": round(float(res["rolling"].get("2026-09_mape", float("nan"))), 3),
                "verdict": decision, "reason": reason, "mlflow_run_id": run_id})
    return 0


if __name__ == "__main__":
    sys.exit(main())
