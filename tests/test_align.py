import pandas as pd
import pytest

from shef.align import align_load, align_weather
from shef.build import join
from shef.config import CITIES


def utc(s):
    return pd.Timestamp(s, tz="UTC")


def raw_load(starts, values=None):
    starts = pd.DatetimeIndex([utc(s) for s in starts])
    return pd.DataFrame({
        "interval_start_utc": starts,
        "interval_end_utc": starts + pd.Timedelta(hours=1),
        "south_central": values or [10_000.0] * len(starts),
    })


def raw_weather(times, temp, precip, rh=50.0):
    rows = []
    for city in CITIES:
        for t, tc, p in zip(times, temp, precip):
            rows.append({"time": utc(t), "city": city, "temperature_2m": tc,
                         "relative_humidity_2m": rh, "precipitation": p})
    return pd.DataFrame(rows)


def test_load_keyed_by_interval_start():
    df = align_load(raw_load(["2025-07-01 20:00"]))
    assert df.loc[0, "ts_utc"] == utc("2025-07-01 20:00")
    assert df.loc[0, "interval_end_utc"] == utc("2025-07-01 21:00")


def test_load_rejects_non_hourly_interval():
    raw = raw_load(["2025-07-01 20:00"])
    raw["interval_end_utc"] = utc("2025-07-01 20:15")
    with pytest.raises(ValueError):
        align_load(raw)


def test_weather_shifted_onto_the_hour_it_describes():
    df = align_weather(raw_weather(["2025-07-01 21:00"], [35.0], [2.0]), "archive")
    row = df.iloc[0]
    assert row["ts_utc"] == utc("2025-07-01 20:00")
    assert row["valid_time_utc"] == utc("2025-07-01 21:00")


def test_join_puts_end_of_hour_weather_on_the_load_interval():
    """Load for 20:00-21:00 UTC must get precip summed over 20:00-21:00 (Open-Meteo time 21:00)
    and the temperature read at 21:00 — not the 20:00 values v1 would have joined."""
    load = align_load(raw_load(["2025-07-01 19:00", "2025-07-01 20:00"]))
    obs = align_weather(raw_weather(["2025-07-01 20:00", "2025-07-01 21:00"], [30.0, 35.0], [0.0, 2.0]), "archive")
    proxy = align_weather(pd.DataFrame({"time": obs["valid_time_utc"], "city": obs["city"]}), "previous_runs")
    joined = join(load, obs, proxy).set_index("ts_utc")
    assert joined.loc[utc("2025-07-01 20:00"), "precip_mm"] == pytest.approx(2.0)
    assert joined.loc[utc("2025-07-01 20:00"), "temp_c"] == pytest.approx(35.0)
    assert joined.loc[utc("2025-07-01 19:00"), "temp_c"] == pytest.approx(30.0)


def test_previous_runs_columns_renamed_by_lead():
    raw = pd.DataFrame({"time": [utc("2024-06-01 12:00")], "city": ["austin"],
                        "temperature_2m_previous_day1": [30.0], "temperature_2m_previous_day2": [31.0]})
    df = align_weather(raw, "previous_runs")
    assert df.loc[0, "temp_c_d1"] == 30.0 and df.loc[0, "temp_c_d2"] == 31.0
    assert df["humidity_pct_d1"].isna().all()  # absent in the response -> explicit NaN


def test_missing_city_gives_nan_not_partial_average():
    raw = raw_weather(["2025-07-01 21:00"], [35.0], [0.0])
    raw.loc[raw["city"] == "austin", "temperature_2m"] = float("nan")
    load = align_load(raw_load(["2025-07-01 20:00"]))
    obs = align_weather(raw, "archive")
    proxy = align_weather(pd.DataFrame({"time": obs["valid_time_utc"], "city": obs["city"]}), "previous_runs")
    assert pd.isna(join(load, obs, proxy).loc[0, "temp_c"])


def test_weighting_saturated_humidity_stays_in_range():
    raw = raw_weather([f"2025-07-01 {h:02d}:00" for h in range(24)], [20.0] * 24, [0.0] * 24, rh=100.0)
    obs = align_weather(raw, "archive")
    from shef.weather_weights import weight_cities
    assert (weight_cities(obs, ["humidity_pct"])["humidity_pct"] <= 100.0).all()
