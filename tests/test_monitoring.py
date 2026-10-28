"""P6: drift statistic, weather-error alignment, and SQL views vs metrics.py."""
import os
from datetime import date

import numpy as np
import pandas as pd
import pytest

from shef.monitoring import drift, weather


def test_psi_by_hand_and_zero_for_identical():
    ref, cur = np.array([0.5, 0.5]), np.array([0.8, 0.2])
    expected = (0.8 - 0.5) * np.log(0.8 / 0.5) + (0.2 - 0.5) * np.log(0.2 / 0.5)
    assert drift.psi(ref, cur) == pytest.approx(expected)
    assert drift.psi(ref, ref) == 0


def test_shares_put_edge_values_in_the_lower_bin():
    s = drift.shares(np.array([1.0, 2.0, 2.0, 3.0, np.nan]), edges=[2.0])
    np.testing.assert_allclose(s, [0.75, 0.25])


def test_thresholds_are_calibrated_from_history():
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", "2026-06-30 23:00", freq="h", tz="UTC")
    rows = pd.DataFrame({"ts_utc": ts, **{f: rng.normal(size=len(ts)) for f in drift.FEATURES}})
    ref = drift.build_reference(rows, ts.max())
    ref["thresholds"] = drift.calibrate(ref, rows, step_days=7)
    window = rows.tail(24 * 14)
    assert all(r["status"] != "drift" for r in drift.evaluate(ref, window))   # 'watch' is a 1-in-20 event
    shifted = window.assign(temp_c=window["temp_c"] + 3)
    assert {r["feature"]: r["status"] for r in drift.evaluate(ref, shifted)}["temp_c"] == "drift"
    assert drift.evaluate(ref, window.head(24)) == []          # too little data: no verdict


def snapshot_rows(issue_date, temp=20.0, model="gfs_seamless"):
    from shef.config import CITIES
    t = pd.date_range("2026-10-13 00:00", "2026-10-15 12:00", freq="h", tz="UTC")
    return pd.concat([pd.DataFrame({"time": t, "city": c, "temperature_2m": temp, "relative_humidity_2m": 50.0,
                                    "precipitation": 0.0, "model": model, "issue_date_central": issue_date})
                      for c in CITIES], ignore_index=True)


def test_served_series_is_the_target_day_in_utc_hour_starts():
    s = weather.served_series(snapshot_rows("2026-10-13"), date(2026, 10, 13))
    assert len(s) == 24
    assert s.index[0] == pd.Timestamp("2026-10-14 05:00", tz="UTC")     # 00:00 CDT, ts_utc = time - 1h
    assert weather.served_series(snapshot_rows("2026-10-13", model="best_match"), date(2026, 10, 13)).empty


def test_compare_needs_every_served_hour():
    idx = pd.date_range("2026-10-14 05:00", periods=24, freq="h", tz="UTC")
    served = pd.Series(21.0, index=idx)
    r = weather.compare(served, pd.Series(20.0, index=idx))
    assert r == {"n_hours": 24, "mae": 1.0, "bias": 1.0, "p95_abs": 1.0}
    assert weather.compare(served, pd.Series(20.0, index=idx[:20])) is None


def test_only_era5_final_days_are_due():
    snap = pd.concat([snapshot_rows("2026-10-13"), snapshot_rows("2026-10-20")])
    assert weather.due_issue_dates(snap, set(), today=date(2026, 10, 22)) == [date(2026, 10, 13)]
    done = {(date(2026, 10, 13), "served_vs_era5"), (date(2026, 10, 13), "served_vs_previous_day1")}
    assert weather.due_issue_dates(snap, done, today=date(2026, 10, 22)) == []


def test_shadow_model_is_scored_separately_and_never_served():
    snap = pd.concat([snapshot_rows("2026-10-13", temp=20.0), snapshot_rows("2026-10-13", temp=25.0, model="ecmwf_ifs025")])
    assert (weather.served_series(snap, date(2026, 10, 13)) == 20.0).all()            # serving still sees GFS
    assert (weather.served_series(snap, date(2026, 10, 13), "ecmwf_ifs025") == 25.0).all()
    assert weather.due_shadow_dates(snap, "ecmwf_ifs025", set(), today=date(2026, 10, 22)) == [date(2026, 10, 13)]
    assert weather.due_shadow_dates(snap, "ecmwf_ifs025", {(date(2026, 10, 13), "ecmwf_ifs025_vs_era5")},
                                    today=date(2026, 10, 22)) == []
    assert weather.due_shadow_dates(snapshot_rows("2026-10-13"), "ecmwf_ifs025", set(), date(2026, 10, 22)) == []


def test_served_model_switches_on_its_start_date():
    from shef.serving.schema import SERVED_SINCE, served_model_for
    before, after = SERVED_SINCE - pd.Timedelta(days=1).to_pytimedelta(), SERVED_SINCE
    assert served_model_for(before) == "gfs_seamless" and served_model_for(after) == "ecmwf_ifs025"
    snap = pd.concat([snapshot_rows(str(after), temp=20.0), snapshot_rows(str(after), temp=25.0, model="ecmwf_ifs025")])
    assert (weather.served_series(snap, after) == 25.0).all()                       # ECMWF is served from then on
    today = after + pd.Timedelta(days=10).to_pytimedelta()
    assert weather.due_shadow_dates(snap, "gfs_seamless", set(), today) == [after]  # and GFS becomes the shadow
    assert weather.due_shadow_dates(snap, "ecmwf_ifs025", set(), today) == []


@pytest.mark.skipif(not os.environ.get("DASHBOARD_DATABASE_URL"), reason="needs the monitoring database")
def test_sql_summary_matches_metrics_py():
    import psycopg

    from shef.models.metrics import score

    with psycopg.connect(os.environ["DASHBOARD_DATABASE_URL"], prepare_threshold=None) as c:
        rows = c.execute("SELECT actual_load_mw, predicted_load_mw FROM monitoring.v_scored "
                         "WHERE model_version = 'backtest-1'").fetchall()
        view = c.execute("SELECT mape, peak_mape, rmse, bias_pct FROM monitoring.v_summary "
                         "WHERE model_version = 'backtest-1'").fetchone()
    y, p = np.array(rows).T
    expected = score(y, p)
    np.testing.assert_allclose(view, [expected[k] for k in ("mape", "peak_mape", "rmse", "bias_pct")], rtol=1e-9)


@pytest.mark.parametrize("champ,chall,expected", [
    ({"mape": 4.00, "peak_mape": 3.0}, {"mape": 3.95, "peak_mape": 2.9}, "keep"),     # gain below the margin
    ({"mape": 4.00, "peak_mape": 3.0}, {"mape": 3.85, "peak_mape": 3.1}, "keep"),     # better MAPE, worse peaks
    ({"mape": 4.00, "peak_mape": 3.0}, {"mape": 3.85, "peak_mape": 3.0}, "promote"),
])
def test_promotion_rule(champ, chall, expected):
    from shef.monitoring.promote import decide
    assert decide(champ, chall)[0] == expected


def test_window_must_postdate_the_champion():
    from shef.monitoring import promote

    days = pd.date_range("2026-08-01", "2026-10-31", freq="D")
    proxy = pd.DataFrame({"ts_utc": days.tz_localize("UTC") + pd.Timedelta(hours=17), "y": 1.0})
    tables = {"proxy": proxy, "obs": proxy}
    ev, model = promote.evaluate(tables, champion=None, champion_train_end=date(2026, 10, 9), as_of=date(2026, 10, 31))
    assert ev.decision == "insufficient" and model is None


def test_latency_brackets_the_newest_published_hour():
    import pandas as pd

    from shef.monitoring import latency

    ends = pd.date_range("2026-10-14 13:00", periods=3, freq="h", tz="UTC")
    raw = pd.DataFrame({"interval_start_utc": ends - pd.Timedelta(hours=1), "interval_end_utc": ends,
                        "south_central": [1.0, 2.0, float("nan")]})   # the 15:00 hour is not in yet
    obs = latency.observation(raw, pd.Timestamp("2026-10-14 15:07", tz="UTC"), "issue")
    assert obs["newest_interval_end"] == "2026-10-14T14:00:00+00:00" and obs["run_mode"] == "issue"
    assert latency.bounds(obs) == (7.0, 67.0)   # next hour still missing 7 min after it ended; newest within 67
    assert latency.observation(raw.iloc[:0], pd.Timestamp.now(tz="UTC"), "issue") is None
