"""
Hill-climbing ensemble (ported from v1 cv.py), scored on MAPE.

Weights picked on the same out-of-fold rows they are scored on look better
than they are, so `leave_one_fold_out` fits weights on the other folds and
applies them to each held-out fold: that is the honest ensemble score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shef.models.metrics import mape


def hill_climb(preds: dict[str, np.ndarray], y: np.ndarray, n_iter: int = 50, tol: float = 1e-4) -> pd.Series:
    """Greedy forward selection with replacement; returns weights (selection frequency)."""
    solo = {k: mape(y, p) for k, p in preds.items()}
    chosen = [min(solo, key=solo.get)]
    blend, best = preds[chosen[0]].copy(), solo[chosen[0]]
    for _ in range(n_iter):
        trial = {k: (blend * len(chosen) + p) / (len(chosen) + 1) for k, p in preds.items()}
        k = min(trial, key=lambda name: mape(y, trial[name]))
        score = mape(y, trial[k])
        if score >= best - tol:
            break
        chosen.append(k)
        blend, best = trial[k], score
    w = pd.Series(chosen).value_counts(normalize=True)
    return w.reindex(list(preds)).fillna(0.0)


def blend(preds: dict[str, np.ndarray], weights: pd.Series) -> np.ndarray:
    return sum(weights[k] * np.asarray(p) for k, p in preds.items() if weights.get(k, 0) > 0)


def leave_one_fold_out(scored: pd.DataFrame, members: list[str]) -> tuple[np.ndarray, dict[int, pd.Series]]:
    """scored: rows with y, fold and one column per member. Returns honest blended
    predictions and the weights used for each held-out fold."""
    out = np.full(len(scored), np.nan)
    used = {}
    for f in sorted(scored["fold"].unique()):
        tr, te = scored["fold"] != f, scored["fold"] == f
        w = hill_climb({m: scored.loc[tr, m].to_numpy() for m in members}, scored.loc[tr, "y"].to_numpy())
        out[te.to_numpy()] = blend({m: scored.loc[te, m].to_numpy() for m in members}, w)
        used[int(f)] = w
    return out, used
