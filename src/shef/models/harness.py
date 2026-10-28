"""
Experiment harness (P9): each challenger idea is ONE change to the champion's
recipe, scored the same two ways and judged by the same rule.

Suites
    cv       the three P3 yearly folds (validation 2022-08 .. 2025-08):
             does the idea hold across years?
    rolling  monthly origins over 2025-08-10 .. 2026-10-09: retrain through
             month start - 8 days, predict that month. This is production with
             the monthly promote.yml retrain, and it includes September 2026
             (record demand, bias -4.3%), the regime the challengers target.

Verdict, against the baseline (champion recipe) under the same protocol
    passes   rolling MAPE better by >= MARGIN, rolling Peak-MAPE no worse,
             and cv MAPE not worse by more than MARGIN
    fails    otherwise, with the reason

The rolling window is the old holdout, already scored once for the champion;
selecting on it makes its numbers optimistic. Passing earns a shadow-logged
live record (monitoring.challengers), not promotion: live data decides.

Validation rows are always proxy rows (archived forecast weather), Uri
excluded, hours without forecast temperature dropped, as in P3.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from shef.features import bundle as bundle_mod
from shef.models import folds
from shef.models.metrics import by, score, score_folds
from shef.models.targets import Target

MARGIN = 0.10
ROLLING_START, ROLLING_END = folds.HOLDOUT
KEEP = ["ts_utc", "issue_time_utc", "lead_hours", "hour", "month", "y"]


@dataclass
class Experiment:
    """The champion recipe plus one change. Leave everything else at its default."""
    name: str
    hypothesis: str
    target: str = "trend"                                     # trend | ratio | delta | raw
    params: dict = field(default_factory=dict)                # LightGBM overrides
    weight: Callable[[pd.DataFrame], np.ndarray] | None = None   # training rows -> sample weights
    derive: dict[str, Callable[[pd.DataFrame], pd.Series]] = field(default_factory=dict)
    # derive: new columns computed from columns of the SAME row only (each is legal at issue time,
    # so a function of them is too). Anything needing other rows belongs in the feature builder.
    drop: list[str] = field(default_factory=list)             # features removed

    def features(self, base: list[str]) -> list[str]:
        return [f for f in base if f not in self.drop] + list(self.derive)

    def signature(self) -> str:
        """Changes when the recipe changes, so a cached result is never reused for different code."""
        parts = [self.target, json.dumps(self.params, sort_keys=True), sorted(self.drop),
                 inspect.getsource(self.weight) if self.weight else "",
                 {k: inspect.getsource(f) for k, f in sorted(self.derive.items())}]
        return hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Window:
    id: str
    train_end: date
    val_start: date
    val_end: date


def cv_windows() -> list[Window]:
    return [Window(str(f.id), f.train_end, f.val_start, f.val_end) for f in folds.FOLDS]


def rolling_windows(start: date = ROLLING_START, end: date = ROLLING_END) -> list[Window]:
    out, s = [], start
    while s <= end:
        nxt = (pd.Timestamp(s) + pd.offsets.MonthBegin(1)).date()
        e = min(nxt - timedelta(days=1), end)
        out.append(Window(f"{s:%Y-%m}", s - folds.GAP - timedelta(days=1), s, e))
        s = nxt
    return out


SUITES = {"cv": cv_windows, "rolling": rolling_windows}


def _days(df: pd.DataFrame) -> pd.Series:
    return folds.target_day(df)


def fit_predict(exp: Experiment, base_features: list[str], tables: dict, w: Window) -> pd.DataFrame:
    """Train on obs rows through w.train_end, predict proxy rows in [val_start, val_end]."""
    from shef.models import gbdt

    obs, proxy = tables["obs"], tables["proxy"]
    od, pd_ = _days(obs), _days(proxy)
    train = obs[(od >= folds.TRAIN_START) & (od <= w.train_end) & ~obs["is_uri"]].dropna(subset=["y"])
    val = proxy[(pd_ >= w.val_start) & (pd_ <= w.val_end) & ~proxy["is_uri"]].dropna(subset=["y", "temp_c"])

    b = bundle_mod.fit(train)
    tr, va = bundle_mod.apply(train, b).reset_index(drop=True), bundle_mod.apply(val, b).reset_index(drop=True)
    for col, f in exp.derive.items():
        tr[col], va[col] = f(tr), f(va)
    feats = exp.features(base_features)

    t = Target(exp.target).fit(tr, tr["y"])
    z = t.transform(tr, tr["y"])
    ok = np.isfinite(z)
    weight = exp.weight(tr) if exp.weight else np.ones(len(tr))
    model = gbdt.lgbm(**exp.params)
    model.fit(tr.loc[ok, feats], z[ok], sample_weight=np.asarray(weight, float)[ok])

    out = va[KEEP].copy()
    out["fold"], out["pred"] = w.id, t.inverse(va, model.predict(va[feats]))
    return out


def run_suite(exp: Experiment, base_features: list[str], tables: dict, suite: str,
              log: Callable[[str], None] = print) -> pd.DataFrame:
    parts = []
    for w in SUITES[suite]():
        t0 = time.time()
        part = fit_predict(exp, base_features, tables, w)
        parts.append(part)
        m = score(part["y"], part["pred"])
        log(f"  {exp.name:<22} {suite:<7} {w.id:<8} MAPE {m['mape']:5.2f}%  bias {m['bias_pct']:+5.2f}%  "
            f"({len(part):,} h, {time.time() - t0:.0f}s)")
    return pd.concat(parts, ignore_index=True)


def suite_metrics(scored: pd.DataFrame, suite: str) -> dict[str, float]:
    """cv: P3's score_folds (peak hours chosen per fold). rolling: pooled over the whole window,
    as the holdout was scored, plus each month's MAPE and bias."""
    if suite == "cv":
        return score_folds(scored)
    out = score(scored["y"], scored["pred"])
    for wid, g in scored.groupby("fold"):
        s = score(g["y"], g["pred"])
        out[f"{wid}_mape"], out[f"{wid}_bias_pct"] = s["mape"], s["bias_pct"]
    return out


def verdict(base: dict, exp: dict, margin: float = MARGIN) -> tuple[str, str]:
    """base/exp: {'cv': metrics, 'rolling': metrics}."""
    r_gain = base["rolling"]["mape"] - exp["rolling"]["mape"]
    cv_loss = exp["cv"]["mape"] - base["cv"]["mape"]
    bp, ep = base["rolling"]["peak_mape"], exp["rolling"]["peak_mape"]
    summary = (f"rolling MAPE {base['rolling']['mape']:.2f} -> {exp['rolling']['mape']:.2f} ({r_gain:+.2f}), "
               f"peak {bp:.2f} -> {ep:.2f}; cv MAPE {base['cv']['mape']:.2f} -> {exp['cv']['mape']:.2f}")
    if ep > bp:
        return "fails", f"rolling Peak-MAPE worse. {summary}"
    if r_gain < margin:
        return "fails", f"rolling gain {r_gain:+.2f} below the {margin:.2f} margin. {summary}"
    if cv_loss > margin:
        return "fails", f"cv MAPE worse by {cv_loss:.2f}: does not hold across years. {summary}"
    return "passes", summary


def by_month(scored: pd.DataFrame) -> pd.DataFrame:
    s = scored.assign(ym=pd.to_datetime(_days(scored)).dt.to_period("M").astype(str))
    t = by(s, "ym")
    t["bias_pct"] = t["bias_mw"] / s.groupby("ym")["y"].mean() * 100
    return t
