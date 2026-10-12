"""Standard logging of a CV result: the same metrics, plots and artifacts for every run."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from shef.models import plots, tracking
from shef.models.metrics import by, score_folds


def summary(scored: pd.DataFrame, preds: list[str]) -> pd.DataFrame:
    rows = {p: score_folds(scored.dropna(subset=[p]), p) for p in preds}
    cols = ["mape", "peak_mape", "rmse", "bias_pct"] + [f"fold{i}_mape" for i in (1, 2, 3)]
    return pd.DataFrame(rows).T[[c for c in cols if c in next(iter(rows.values()))]].round(2)


def log_scored(name: str, scored: pd.DataFrame, pred: str, params: dict, *, figures: dict | None = None,
               dicts: dict | None = None, tags: dict | None = None) -> str:
    s = scored.dropna(subset=[pred])
    metrics = score_folds(s, pred)
    standard = {
        "mape_by_lead": plots.error_by(s, "lead_hours", pred, f"{name}: MAPE by lead hour"),
        "mape_by_month": plots.error_by(s, "month", pred, f"{name}: MAPE by month"),
        "peak_week": plots.peak_week(s, [pred], f"{name}: peak week"),
    }
    figs = {**standard, **(figures or {})}
    frames = {"oof_predictions": s[["ts_utc", "issue_time_utc", "fold", "lead_hours", "y", pred]]
              .rename(columns={pred: "pred"})}
    tables = {"by_lead": by(s, "lead_hours", pred).reset_index().to_dict("list"),
              "by_month": by(s, "month", pred).reset_index().to_dict("list")}
    run_id = tracking.log_run(name, params, metrics, figures=figs, frames=frames,
                              dicts={**tables, **(dicts or {})}, tags=tags)
    for f in standard.values():  # only our own; the caller may still want to show theirs
        plt.close(f)
    return run_id


def log_result(result, *, figures: dict | None = None, dicts: dict | None = None, tags: dict | None = None) -> str:
    """Log a cv.Result (scored rows carry `pred`)."""
    fitted = {f"fold{k}": {"target": t.to_dict(), "bundle": b.__dict__} for k, (_, t, b) in result.models.items()}
    return log_scored(result.spec.name, result.scored, "pred", {**result.spec.params(), "seconds": round(result.seconds, 1)},
                      figures=figures, dicts={"fitted_per_fold": fitted, **(dicts or {})}, tags=tags)
