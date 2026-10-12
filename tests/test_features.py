"""
P2 tests for the point-in-time feature builder.

The leakage test is the structural fix for v1: it does not inspect the code,
it corrupts every value that lies in the future of issue_time and asserts the
features do not move. Any accidental future read makes it fail.
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from shef.config import CITIES
from shef.features import BUNDLE_FEATURES, bundle, contract_features
from shef.features.asof import cutoff_for, target_hours, weather_as_of
from shef.features.live import build_live
from shef.features.replay import replay
from shef.timestamps import issue_time_utc

HOUR = pd.Timedelta(hours=1)
META = {"issue_time_utc", "ts_utc", "holiday_name", "cutoff_utc", "simulated_delay_h", "y",
        "is_uri", "is_dst_day", "weather_mode"}


def synthetic_joined(start="2025-02-20", end="2025-11-10", seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, end, freq="h", tz="UTC").as_unit("us")
    n = len(ts)
    h = ts.hour.to_numpy()
    temp = np.round(20 + 8 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 1, n), 3)
    df = pd.DataFrame({
        "ts_utc": ts,
        "interval_end_utc": ts + HOUR,
        "load_mw": np.round(9000 + 2500 * np.sin(2 * np.pi * (h - 11) / 24) + rng.normal(0, 200, n), 3),
        "temp_c": temp,
        "humidity_pct": np.round(rng.uniform(30, 90, n), 3),
        "precip_mm": np.round(rng.exponential(0.2, n), 3),
    })
    for v in ("temp_c", "humidity_pct", "precip_mm"):
        df[f"{v}_d1"] = np.round(df[v] + rng.normal(0, 1, n), 3)
        df[f"{v}_d2"] = np.round(df[v] + rng.normal(0, 2, n), 3)
    df["humidity_pct_d1"] = df["humidity_pct_d1"].clip(0, 100)
    df["humidity_pct_d2"] = df["humidity_pct_d2"].clip(0, 100)
    df["is_load_gap"] = False
    df["is_uri"] = False
    df["is_dst_day"] = False
    return df


@pytest.fixture(scope="module")
def joined():
    return synthetic_joined()


def features_only(df):
    return df.drop(columns=[c for c in META if c in df.columns]).reset_index(drop=True)


def poison_future(j: pd.DataFrame, issue: pd.Timestamp) -> pd.DataFrame:
    """Corrupt everything not knowable at issue_time."""
    p = j.copy()
    after_cutoff = p["interval_end_utc"] > cutoff_for(issue)
    p.loc[after_cutoff, "load_mw"] = p.loc[after_cutoff, "load_mw"] * 1000
    future = p["ts_utc"] + HOUR > issue
    for v in ("temp_c", "humidity_pct", "precip_mm"):
        p.loc[future, v] = p.loc[future, v] + 77            # ERA5 in the future
        lead = p["ts_utc"] + HOUR - issue
        p.loc[lead > pd.Timedelta(hours=24), f"{v}_d1"] += 55  # day-1 forecasts made after issue
    return p


# --- leakage ----------------------------------------------------------------
@pytest.mark.parametrize("day", [date(2025, 3, 9), date(2025, 3, 10), date(2025, 7, 1),
                                 date(2025, 11, 2), date(2025, 11, 3)])
def test_future_data_cannot_move_features(joined, day):
    issue = issue_time_utc(day)
    clean = replay(joined, day, day, "proxy")
    poisoned = replay(poison_future(joined, issue), day, day, "proxy")
    pd.testing.assert_frame_equal(features_only(clean), features_only(poisoned))


def test_obs_mode_is_an_oracle_and_does_move(joined):
    """Sanity check on the poison itself: ERA5-as-forecast must react to it."""
    day = date(2025, 7, 1)
    clean = replay(joined, day, day, "obs")
    poisoned = replay(poison_future(joined, issue_time_utc(day)), day, day, "obs")
    assert not np.allclose(clean["temp_c"], poisoned["temp_c"])


def test_label_is_attached_after_build(joined):
    day = date(2025, 7, 1)
    f = replay(joined, day, day, "proxy")
    actual = joined.set_index("ts_utc")["load_mw"].reindex(f["ts_utc"]).to_numpy()
    np.testing.assert_allclose(f["y"], actual)


# --- contract ---------------------------------------------------------------
def test_builder_emits_exactly_the_contract(joined):
    f = replay(joined, date(2025, 7, 1), date(2025, 7, 1), "proxy")
    built = set(f.columns) - META
    assert built == set(contract_features("day_ahead")) - BUNDLE_FEATURES


def test_bundle_completes_the_contract_and_round_trips(joined):
    f = replay(joined, date(2025, 7, 1), date(2025, 7, 7), "proxy")
    b = bundle.fit(f)
    assert bundle.Bundle.from_json(b.to_json()) == b
    out = bundle.apply(f, b)
    assert set(contract_features("day_ahead")) <= set(out.columns)


# --- time handling ----------------------------------------------------------
@pytest.mark.parametrize("day, rows, first_lead", [(date(2025, 3, 9), 23, 14), (date(2025, 11, 2), 25, 14),
                                                    (date(2025, 7, 1), 24, 14)])
def test_target_day_rows_and_lead(joined, day, rows, first_lead):
    f = replay(joined, day, day, "proxy")
    assert len(f) == rows
    assert f["lead_hours"].iloc[0] == first_lead
    assert f["hour"].iloc[0] == 0


def test_same_hour_age_split_at_cutoff(joined):
    f = replay(joined, date(2025, 7, 1), date(2025, 7, 1), "proxy")
    assert (f.loc[f["hour"] <= 8, "same_hour_age_h"] == 24).all()
    assert (f.loc[f["hour"] >= 9, "same_hour_age_h"] == 48).all()
    assert (f["load_staleness_h"] == 1).all()


def test_proxy_uses_day1_then_day2_by_lead(joined):
    day = date(2025, 7, 1)
    issue = issue_time_utc(day)
    w = weather_as_of(issue, day, "proxy", joined=joined)
    src = joined.set_index("ts_utc")
    lead = (w.index + HOUR) - issue
    near = w.index[(lead > pd.Timedelta(0)) & (lead <= pd.Timedelta(hours=24))]
    far = w.index[lead > pd.Timedelta(hours=24)]
    np.testing.assert_allclose(w.loc[near, "temp_c"], src.loc[near, "temp_c_d1"])
    np.testing.assert_allclose(w.loc[far, "temp_c"], src.loc[far, "temp_c_d2"])
    past = w.index[lead <= pd.Timedelta(0)]
    np.testing.assert_allclose(w.loc[past, "temp_c"], src.loc[past, "temp_c"])


def test_missing_day1_falls_back_to_day2(joined):
    day = date(2025, 7, 1)
    issue = issue_time_utc(day)
    j = joined.copy()
    j["temp_c_d1"] = np.nan
    w = weather_as_of(issue, day, "proxy", joined=j)
    fut = w.index[w["is_forecast"]]
    np.testing.assert_allclose(w.loc[fut, "temp_c"], j.set_index("ts_utc").loc[fut, "temp_c_d2"])


# --- late GridStatus --------------------------------------------------------
def test_missing_age24_value_falls_back(joined):
    day = date(2025, 7, 1)
    issue = issue_time_utc(day)
    t8 = target_hours(issue, day)[8]  # 08:00 CDT on D
    j = joined[joined["ts_utc"] != t8 - pd.Timedelta(hours=24)]
    f = replay(j, day, day, "proxy")
    row = f.set_index("ts_utc").loc[t8]
    assert row["same_hour_age_h"] == 48
    assert row["load_staleness_h"] == 2  # newest known hour is now 07:00


def test_augmentation_simulates_late_publish(joined):
    f = replay(joined, date(2025, 7, 1), date(2025, 7, 5), "proxy", augment_frac=1.0, seed=3)
    assert (f["simulated_delay_h"] >= 1).all()
    np.testing.assert_allclose(f["load_staleness_h"], 1 + f["simulated_delay_h"])


# --- parity: live path == replay path ---------------------------------------
def test_live_path_matches_replay(joined):
    day = date(2025, 7, 1)
    issue = issue_time_utc(day)
    replayed = replay(joined, day, day, "proxy")

    # What GridStatus would return on a run 40 min late: includes an hour past the cutoff.
    recent = joined[(joined["ts_utc"] >= issue - pd.Timedelta(days=12)) & (joined["interval_end_utc"] <= issue + HOUR)]
    gridstatus_raw = pd.DataFrame({"interval_start_utc": recent["ts_utc"], "interval_end_utc": recent["interval_end_utc"],
                                   "south_central": recent["load_mw"]})

    # What the forecast snapshot would hold: the same as-of series, as raw per-city labels.
    w = weather_as_of(issue, day, "proxy", joined=joined)
    snap = []
    for city in CITIES:
        snap.append(pd.DataFrame({"time": w.index + HOUR, "city": city, "temperature_2m": w["temp_c"].to_numpy(),
                                  "relative_humidity_2m": w["humidity_pct"].to_numpy(),
                                  "precipitation": w["precip_mm"].to_numpy()}))
    snapshot_raw = pd.concat(snap, ignore_index=True)

    live = build_live(issue, day, gridstatus_raw, snapshot_raw)
    pd.testing.assert_frame_equal(features_only(live), features_only(replayed), check_exact=False, rtol=1e-9)


# --- LSTM sequences ---------------------------------------------------------
@pytest.mark.parametrize("day", [date(2025, 3, 9), date(2025, 7, 1), date(2025, 11, 2)])
def test_sequences_ignore_the_future(joined, day):
    from shef.features import sequences

    issue = issue_time_utc(day)
    clean = sequences.build(joined, day, day, "proxy")
    poisoned = sequences.build(poison_future(joined, issue), day, day, "proxy")
    np.testing.assert_array_equal(clean.enc, poisoned.enc)
    np.testing.assert_array_equal(clean.dec, poisoned.dec)
    np.testing.assert_array_equal(clean.level, poisoned.level)


def test_sequences_pad_dst_days(joined):
    from shef.features import sequences

    s = sequences.build(joined, date(2025, 3, 9), date(2025, 3, 10), "proxy")
    assert s.mask[0].sum() == 23 and s.mask[1].sum() == 24
    assert np.isnat(s.target_ts[0, 23:]).all()
    assert s.enc.shape[1:] == (sequences.ENC_HOURS, len(sequences.ENC_FEATURES))
