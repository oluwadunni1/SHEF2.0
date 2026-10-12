"""
build_features.py — replay every past issue time into data/features/<mode>/.

    obs    ERA5 as a perfect forecast, 2016+, 5% of days replayed as if GridStatus
           were late. The long training history.
    proxy  Archived day-1/day-2 forecasts from their start date, no augmentation.
           The honest backtest; never trained on in the same fold it scores.

    uv run python scripts/build_features.py --mode obs
    uv run python scripts/build_features.py --mode proxy
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta

from shef import store
from shef.config import FC_PROXY_START, REPORTS, TRAIN_START
from shef.features.replay import replay

MODES = {
    # first target day whose 8-day weather lookback is fully covered
    "obs": {"start": TRAIN_START + timedelta(days=11), "augment_frac": 0.05},
    "proxy": {"start": FC_PROXY_START["temp_c"] + timedelta(days=2), "augment_frac": 0.0},
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=list(MODES), required=True)
    p.add_argument("--end", type=date.fromisoformat, help="last target day (default: last full day in the data)")
    args = p.parse_args()

    joined = store.read("clean", "joined")
    last_full = (joined["ts_utc"].max() - timedelta(days=2)).date()
    cfg = MODES[args.mode]
    end = args.end or last_full

    t0 = time.time()
    df = replay(joined, cfg["start"], end, args.mode, augment_frac=cfg["augment_frac"], seed=2016)
    store.write(df, "features", args.mode)

    summary = {
        "mode": args.mode, "rows": len(df), "issue_days": int(df["issue_time_utc"].nunique()),
        "first_target": str(df["ts_utc"].min()), "last_target": str(df["ts_utc"].max()),
        "label_missing": int(df["y"].isna().sum()),
        "simulated_delay_days": int(df.loc[df["simulated_delay_h"] > 0, "issue_time_utc"].nunique()),
        "null_share": {c: round(float(v), 5) for c, v in df.isna().mean().items() if v > 0},
        "seconds": round(time.time() - t0, 1),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"features_{args.mode}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
