"""
Shadow challengers: frozen post-processing of the champion's forecast, logged
daily next to it (monitoring.challengers) and never served. Each builds a live
record that the promotion gate can judge later.

Both rules were fixed on 2026-10-28 from backtest analysis (FINDINGS / P9) and
must not be tuned on the live record they are now accumulating:

  recent_median   p - K * median(known errors of the served model, last 7 days)
                  backtest 3.79 -> 3.56 MAPE, peak 3.35 -> 2.66
  stack_ercot     STACK_OURS * p + STACK_ERCOT * ERCOT's 09:30 forecast
                  walk-forward 2.73 vs ERCOT 2.78; falls back to p where ERCOT is missing

Point in time like the model itself: an error is "known" at issue time only if
its hour ended by the load cutoff (issue - 1 h); ERCOT's forecast is the latest
publish at or before issue time (monitoring.ercot_forecast).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

PREREGISTERED_FROM = date(2026, 10, 29)   # first target day issued after the rules were frozen
K = 0.5
WINDOW = pd.Timedelta(days=7)
MIN_KNOWN = 72
STACK_OURS, STACK_ERCOT = 0.31, 0.70
HOUR = pd.Timedelta(hours=1)


def known_errors(history: pd.DataFrame, issue: pd.Timestamp) -> pd.Series:
    """history: ts_utc, err (served model hours with an actual). Errors knowable at `issue`."""
    cutoff = issue - HOUR
    end = history["ts_utc"] + HOUR
    return history.loc[(end <= cutoff) & (end > cutoff - WINDOW), "err"]


def recent_median(history: pd.DataFrame, champion: pd.DataFrame, issue: pd.Timestamp) -> list[dict]:
    known = known_errors(history, issue)
    if len(known) < MIN_KNOWN:
        return []
    corr = float(known.median())
    return [{"target_timestamp": r.ts_utc, "challenger": "recent_median", "issue_time_utc": issue,
             "predicted_load_mw": r.p - K * corr, "champion_mw": r.p,
             "inputs": {"k": K, "median_err_mw": corr, "known_hours": int(len(known))}}
            for r in champion.itertuples(index=False)]


def stack_ercot(champion: pd.DataFrame, ercot: pd.Series, issue: pd.Timestamp) -> list[dict]:
    """ercot: forecast_mw indexed by ts_utc (may be empty or partial)."""
    out = []
    for r in champion.itertuples(index=False):
        e = ercot.get(r.ts_utc)
        if e is None or pd.isna(e):
            pred, inputs = r.p, {"fallback": "no ERCOT forecast at issue time"}
        else:
            pred, inputs = STACK_OURS * r.p + STACK_ERCOT * float(e), {"ercot_mw": float(e)}
        out.append({"target_timestamp": r.ts_utc, "challenger": "stack_ercot", "issue_time_utc": issue,
                    "predicted_load_mw": pred, "champion_mw": r.p,
                    "inputs": {"w_ours": STACK_OURS, "w_ercot": STACK_ERCOT, **inputs}})
    return out


def build(live: pd.DataFrame, ercot: pd.Series, target_day: date) -> list[dict]:
    """live: served-first live rows (ts_utc, target_day, issue_time_utc, served_as, p, y).
    Challenger rows for target_day, or [] if the champion's model forecast is not there."""
    champion = live[(live["target_day"] == target_day) & (live["served_as"] == "model")]
    if champion.empty:
        return []
    issue = champion["issue_time_utc"].iloc[0]
    scored = live[(live["served_as"] == "model") & live["y"].notna()]
    history = pd.DataFrame({"ts_utc": scored["ts_utc"], "err": scored["p"] - scored["y"]})
    champion = champion[["ts_utc", "p"]]
    return recent_median(history, champion, issue) + stack_ercot(champion, ercot, issue)
