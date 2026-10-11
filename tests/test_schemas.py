import pandas as pd
import pandera.pandas as pa
import pytest

from shef import schemas, store
from shef.align import align_load, align_weather
from shef.build import join
from tests.test_align import raw_load, raw_weather, utc


def joined_frame(hours=4):
    starts = [str(utc("2025-07-01 00:00") + pd.Timedelta(hours=h)) for h in range(hours)]
    load = align_load(raw_load(starts))
    times = [str(utc(s) + pd.Timedelta(hours=1)) for s in starts]
    obs = align_weather(raw_weather(times, [30.0] * hours, [0.0] * hours), "archive")
    proxy = align_weather(pd.DataFrame({"time": obs["valid_time_utc"], "city": obs["city"]}), "previous_runs")
    return join(load, obs, proxy)


def fails(schema, df):
    with pytest.raises(pa.errors.SchemaErrors):
        schema.validate(df, lazy=True)


def test_valid_joined_passes():
    schemas.Joined.validate(joined_frame(), lazy=True)


def test_tz_naive_rejected():
    df = align_load(raw_load(["2025-07-01 00:00", "2025-07-01 01:00"]))
    df["ts_utc"] = df["ts_utc"].dt.tz_localize(None)
    fails(schemas.CleanLoad, df)


def test_non_utc_timezone_rejected():
    df = align_load(raw_load(["2025-07-01 00:00"]))
    df["ts_utc"] = df["ts_utc"].dt.tz_convert("America/Chicago")
    fails(schemas.CleanLoad, df)


def test_duplicate_hour_rejected():
    df = align_load(raw_load(["2025-07-01 00:00", "2025-07-01 00:00"]))
    fails(schemas.CleanLoad, df)


def test_missing_hour_rejected_in_joined():
    df = joined_frame().drop(index=2).reset_index(drop=True)
    fails(schemas.Joined, df)


def test_out_of_range_load_rejected():
    df = align_load(raw_load(["2025-07-01 00:00"], [999_999.0]))
    fails(schemas.CleanLoad, df)


def test_out_of_range_weather_rejected():
    df = joined_frame()
    df.loc[1, "humidity_pct"] = 140.0
    fails(schemas.Joined, df)


def test_half_hour_timestamp_rejected():
    df = align_load(raw_load(["2025-07-01 00:30"]))
    fails(schemas.CleanLoad, df)


def test_gap_flag_must_match_nulls():
    df = joined_frame()
    df.loc[1, "load_mw"] = float("nan")  # null load without is_load_gap
    fails(schemas.Joined, df)


def test_store_upsert_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA", tmp_path)
    df = align_load(raw_load(["2024-12-31 23:00", "2025-01-01 00:00", "2025-01-01 01:00"]))
    store.upsert(df, "raw", "load", keys=["ts_utc"])
    store.upsert(df, "raw", "load", keys=["ts_utc"])
    back = store.read("raw", "load")
    assert len(back) == 3
    assert sorted(p.name for p in (tmp_path / "raw" / "load").iterdir()) == ["year=2024.parquet", "year=2025.parquet"]
    assert str(back["ts_utc"].dtype.tz) == "UTC"
