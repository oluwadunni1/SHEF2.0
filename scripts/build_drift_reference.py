"""
build_drift_reference.py — seasonal reference for drift checks, from the rows
the serving model was trained on (obs features, Uri excluded, through its
trained-through date), with PSI thresholds calibrated on the archived-forecast
(proxy) rows of the same years. Rebuild after every promotion.

    uv run python scripts/build_drift_reference.py --train-end 2026-10-09
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from shef import store
from shef.config import REPORTS
from shef.models import folds
from shef.monitoring import drift


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-end", required=True, help="the model's trained_through (Central date)")
    args = p.parse_args()

    end = pd.Timestamp(args.train_end).date()
    obs = store.read("features", "obs", columns=["ts_utc", "issue_time_utc", "is_uri", *drift.FEATURES])
    rows = obs[(folds.target_day(obs) <= end) & ~obs["is_uri"]]
    ref = drift.build_reference(rows, rows["ts_utc"].max())
    ref["train_end"] = str(end)
    proxy = store.read("features", "proxy", columns=["ts_utc", "issue_time_utc", "is_uri", *drift.FEATURES])
    proxy = proxy[(folds.target_day(proxy) <= end) & ~proxy["is_uri"]]
    proxy = proxy[proxy["ts_utc"] > rows["ts_utc"].max() - pd.DateOffset(years=drift.REFERENCE_YEARS)]
    ref["thresholds"] = drift.calibrate(ref, proxy)
    ref["calibrated_on"] = f"archived-forecast rows, {proxy['ts_utc'].min().date()}..{proxy['ts_utc'].max().date()}"
    out = REPORTS / "drift_reference.json"
    out.write_text(json.dumps(ref, indent=1))
    n = {m: v["n"] for m, v in ref["features"]["temp_c"].items()}
    for f, t in ref["thresholds"].items():
        print(f"  {f:<26} watch > {t['watch']:.2f}  drift > {t['drift']:.2f}  ({t['n_windows']} windows)")
    print(f"reference from {len(rows):,} rows (last {drift.REFERENCE_YEARS} y to {ref['train_end']}) -> {out}; rows per month {n}")


if __name__ == "__main__":
    main()
