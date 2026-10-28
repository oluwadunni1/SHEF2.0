"""P9 experiment harness: windows, verdict rule, recipe signatures, delta target."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from shef.models import folds, harness
from shef.models.experiments import EXPERIMENTS
from shef.models.targets import Target


def test_rolling_windows_tile_the_holdout_with_a_purge_gap():
    ws = harness.rolling_windows()
    assert ws[0].val_start == folds.HOLDOUT[0] and ws[-1].val_end == folds.HOLDOUT[1]
    assert ws[0].id == "2025-08" and ws[1].val_start == date(2025, 9, 1)
    for a, b in zip(ws, ws[1:]):
        assert (b.val_start - a.val_end).days == 1                     # no gaps, no overlap
    for w in ws:
        assert (w.val_start - w.train_end).days == folds.GAP.days + 1  # 8 clear days, as in P3


def test_cv_windows_are_the_p3_folds():
    assert [(w.train_end, w.val_start, w.val_end) for w in harness.cv_windows()] == \
        [(f.train_end, f.val_start, f.val_end) for f in folds.FOLDS]


def m(mape, peak):
    return {"mape": mape, "peak_mape": peak}


@pytest.mark.parametrize("cv, rolling, expected", [
    (m(4.18, 3.0), m(3.40, 3.0), "passes"),      # 0.20 better, peak equal, cv equal
    (m(4.18, 3.0), m(3.55, 3.0), "fails"),       # 0.05 better: inside the margin
    (m(4.18, 3.0), m(3.40, 3.1), "fails"),       # better MAPE, worse peak
    (m(4.40, 3.0), m(3.40, 3.0), "fails"),       # wins recent months, loses 0.22 across years
])
def test_verdict_rule(cv, rolling, expected):
    base = {"cv": m(4.18, 3.0), "rolling": m(3.60, 3.0)}
    assert harness.verdict(base, {"cv": cv, "rolling": rolling})[0] == expected


def test_signature_changes_with_the_recipe_only():
    a = EXPERIMENTS["A3_linear_tree"]
    assert a.signature() == harness.Experiment("other name", "x", params=dict(a.params)).signature()
    assert a.signature() != EXPERIMENTS["baseline"].signature()
    assert EXPERIMENTS["A4_recency_2y"].signature() != EXPERIMENTS["B5_inverse_load"].signature()


def test_delta_target_round_trips():
    df = pd.DataFrame({"load_same_hour_latest": [9_000.0, 11_000.0]})
    y = pd.Series([10_000.0, 10_500.0])
    t = Target("delta").fit(df, y)
    z = t.transform(df, y)
    np.testing.assert_allclose(z, [1_000.0, -500.0])
    np.testing.assert_allclose(t.inverse(df, z), y)


def test_recency_weights_halve_every_two_years():
    from shef.models.experiments import recency_2y
    ts = pd.to_datetime(["2022-01-01", "2024-01-01", "2026-01-01"], utc=True)
    w = recency_2y(pd.DataFrame({"ts_utc": ts}))
    np.testing.assert_allclose(w, [0.25, 0.5, 1.0], rtol=2e-3)
