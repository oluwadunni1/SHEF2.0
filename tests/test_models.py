from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from shef.models import folds, metrics
from shef.models.targets import Target


def days_frame(start, end):
    ts = pd.date_range(pd.Timestamp(start, tz="America/Chicago"), pd.Timestamp(end, tz="America/Chicago"),
                       freq="h").tz_convert("UTC")
    return pd.DataFrame({"ts_utc": ts})


def test_folds_do_not_overlap_and_keep_the_gap():
    df = days_frame("2016-01-12", "2026-10-09")
    for f in folds.FOLDS:
        tr, va = df[folds.train_mask(df, f)], df[folds.val_mask(df, f)]
        last_train = folds.target_day(tr).max()
        first_val = folds.target_day(va).min()
        assert first_val - last_train == timedelta(days=9)  # 8 full days between
        assert first_val == f.val_start


def test_validation_windows_are_disjoint_and_end_before_holdout():
    ends = [f.val_end for f in folds.FOLDS]
    starts = [f.val_start for f in folds.FOLDS]
    assert all(e < s for e, s in zip(ends, starts[1:]))
    assert ends[-1] < folds.HOLDOUT[0]


def test_holdout_requires_final_flag():
    df = days_frame("2025-08-01", "2025-08-20")
    with pytest.raises(PermissionError):
        folds.holdout_mask(df)
    assert folds.holdout_mask(df, final=True).any()


def test_metrics_by_hand():
    y = np.array([100.0, 200.0, 400.0, 1000.0])
    p = np.array([110.0, 180.0, 400.0, 900.0])
    assert metrics.mape(y, p) == pytest.approx((10 + 10 + 0 + 10) / 4)
    assert metrics.peak_mape(y, p, pct=0.25) == pytest.approx(10.0)  # top 25% = the 1000
    assert metrics.rmse(y, p) == pytest.approx(np.sqrt((100 + 400 + 0 + 10000) / 4))
    assert metrics.bias_pct(y, p) == pytest.approx((10 - 20 + 0 - 100) / 4 / 425 * 100)


@pytest.mark.parametrize("kind", ["raw", "trend", "ratio"])
def test_target_forms_invert_exactly(kind):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"trend_idx": np.linspace(0, 10, 200), "last_known_roll_mean_168": rng.uniform(8e3, 1e4, 200)})
    y = pd.Series(9000 + 150 * df["trend_idx"] + rng.normal(0, 300, 200))
    t = Target(kind).fit(df, y)
    np.testing.assert_allclose(t.inverse(df, t.transform(df, y)), y)
    assert Target.from_dict(t.to_dict()) == t


def test_trend_target_learns_the_line():
    df = pd.DataFrame({"trend_idx": np.arange(10.0)})
    y = pd.Series(1000 + 50 * df["trend_idx"])
    t = Target("trend").fit(df, y)
    assert (t.intercept, t.slope) == pytest.approx((1000, 50))


def test_hill_climb_prefers_the_better_member_and_lofo_is_honest():
    from shef.models import ensemble

    rng = np.random.default_rng(1)
    y = rng.uniform(8000, 12000, 600)
    good, bad = y * (1 + rng.normal(0, 0.03, 600)), y * (1 + rng.normal(0, 0.10, 600))
    w = ensemble.hill_climb({"good": good, "bad": bad}, y)
    assert w["good"] > w["bad"]
    scored = pd.DataFrame({"y": y, "good": good, "bad": bad, "fold": np.repeat([1, 2, 3], 200)})
    blended, used = ensemble.leave_one_fold_out(scored, ["good", "bad"])
    assert set(used) == {1, 2, 3} and not np.isnan(blended).any()
