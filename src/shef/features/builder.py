"""
build_features(issue_time, target_day, load_known, weather) -> one row per target hour.

The single implementation used by training replay and live serving. Its inputs
are the as-of views from asof.py, so everything here is a pure function of data
that existed at issue_time. Column names are the day-ahead features in
features.yaml (minus the bundle ones, added by bundle.apply after fitting).
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from shef.timestamps import central_date, to_central

HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)
BASE_TEMP_C = 18.3  # 65 °F, the usual CDD/HDD base
SAME_HOUR_AGES = (24, 48, 72)


def _holiday_names(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(pd.Series(dates.unique()))
    names = USFederalHolidayCalendar().holidays(start=d.min() - DAY, end=d.max() + DAY, return_name=True)
    lookup = {k.date(): v for k, v in names.items()}
    return dates.map(lookup)


def _calendar(targets: pd.DatetimeIndex, issue_time: pd.Timestamp) -> pd.DataFrame:
    ts = pd.Series(targets, index=targets)
    local = to_central(ts)
    df = pd.DataFrame(index=targets)
    df["hour"] = local.dt.hour
    df["dayofweek"] = local.dt.dayofweek
    df["month"] = local.dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_monday"] = (df["dayofweek"] == 0).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["holiday_name"] = _holiday_names(central_date(ts))
    df["is_holiday"] = df["holiday_name"].notna().astype(int)
    df["lead_hours"] = ((targets - issue_time) / HOUR).astype(int)
    return df


def _weather(targets: pd.DatetimeIndex, w: pd.DataFrame, hour: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=targets)
    for col in ("temp_c", "humidity_pct", "precip_mm"):
        df[col] = w[col].reindex(targets)

    # Daily stats over Central days (23/24/25 hours), from the as-of series.
    day = central_date(pd.Series(w.index, index=w.index))
    daily = w["temp_c"].groupby(day).agg(["max", "min", "mean"])
    tday = central_date(pd.Series(targets, index=targets))
    prev_day = tday.map(lambda d: d - timedelta(days=1))
    for name, stat in (("tmax", "max"), ("tmin", "min"), ("tavg", "mean")):
        df[name] = tday.map(daily[stat])
        df[f"{name}_lag_24"] = prev_day.map(daily[stat])

    df["cdd"] = (df["tavg"] - BASE_TEMP_C).clip(lower=0)
    df["hdd"] = (BASE_TEMP_C - df["tavg"]).clip(lower=0)
    df["cdd_x_hour"] = df["cdd"] * hour
    df["hour_x_temp"] = hour * df["temp_c"]
    spike = (df["tmax"] - df["tavg"]) / (df["tavg"] + 1)
    df["temp_spike_vs_mean"] = spike.replace([np.inf, -np.inf], np.nan)

    roll = w["temp_c"].rolling(72, min_periods=24)
    df["tmax_roll_max_72"] = roll.max().reindex(targets)
    df["temp_c_roll_std_72"] = roll.std().reindex(targets)
    return df


def _ratio(known: pd.Series, hours: pd.Index, shift: pd.Timedelta) -> float:
    """sum(load at `hours`) / sum(load at `hours - shift`), only if every hour exists."""
    if len(hours) == 0:
        return np.nan
    earlier = known.reindex(hours - shift)
    if earlier.isna().any():
        return np.nan
    return float(known.reindex(hours).sum() / earlier.sum())


def _load(targets: pd.DatetimeIndex, known: pd.Series, issue_time: pd.Timestamp, hour: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=targets)

    same, age = [], []
    for ts in targets:
        for a in SAME_HOUR_AGES:
            v = known.get(ts - pd.Timedelta(hours=a))
            if v is not None and not np.isnan(v):
                same.append(v), age.append(a)
                break
        else:
            same.append(np.nan), age.append(np.nan)
    df["load_same_hour_latest"] = same
    df["same_hour_age_h"] = age

    df["load_lag_48"] = known.reindex(targets - pd.Timedelta(hours=48)).to_numpy()
    df["lag_48_x_hour"] = df["load_lag_48"] * hour
    df["load_lag_168"] = known.reindex(targets - pd.Timedelta(hours=168)).to_numpy()

    # Issue-time anchors: identical for every target row.
    if known.empty:
        anchors = dict.fromkeys(["last_known_load", "load_staleness_h", "last_known_roll_mean_24",
                                 "last_known_roll_max_24", "last_known_roll_std_24",
                                 "last_known_roll_mean_168", "dminus1_vs_lastweek", "dminus1_vs_dminus2"], np.nan)
    else:
        last_ts = known.index[-1]
        last24 = known[known.index > last_ts - pd.Timedelta(hours=24)]
        last168 = known[known.index > last_ts - pd.Timedelta(hours=168)]
        dminus1 = central_date(pd.Series(targets[:1], index=targets[:1])).iloc[0] - timedelta(days=1)
        d1_hours = known.index[central_date(pd.Series(known.index, index=known.index)) == dminus1]
        anchors = {
            "last_known_load": float(known.iloc[-1]),
            "load_staleness_h": float((issue_time - (last_ts + HOUR)) / HOUR),
            "last_known_roll_mean_24": float(last24.mean()),
            "last_known_roll_max_24": float(last24.max()),
            "last_known_roll_std_24": float(last24.std()),
            "last_known_roll_mean_168": float(last168.mean()),
            "dminus1_vs_lastweek": _ratio(known, d1_hours, pd.Timedelta(days=7)),
            "dminus1_vs_dminus2": _ratio(known, d1_hours, DAY),
        }
    for k, v in anchors.items():
        df[k] = v
    return df


def build_features(issue_time: pd.Timestamp, targets: pd.DatetimeIndex,
                   load_known: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    issue_time:  UTC
    targets:     UTC hour starts of the target Central day (asof.target_hours)
    load_known:  asof.load_as_of(...)[0]  (index ts_utc, column load_mw)
    weather:     asof.weather_as_of(...)  (index ts_utc)
    """
    cal = _calendar(targets, issue_time)
    hour = cal["hour"]
    out = pd.concat([cal, _weather(targets, weather, hour), _load(targets, load_known["load_mw"], issue_time, hour)],
                    axis=1)
    out.index.name = "ts_utc"
    out.insert(0, "issue_time_utc", issue_time)
    return out.reset_index()
