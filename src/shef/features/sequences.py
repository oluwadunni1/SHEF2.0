"""
Sequence samples for the LSTM, one per issue time, built from the same as-of
views as build_features, so the same leakage guarantees hold.

    encoder  the 168 hours ending at the load cutoff:
             load / level, a 0/1 mask for missing load, temperature (past side),
             hour and weekday as sin/cos
    decoder  each target hour (padded to 25 for DST days):
             forecast temperature, hour and weekday sin/cos, is_holiday, lead / 38
    target   load / level per target hour, with a mask for padding

level = mean of the known load in the encoder window (≈ last_known_roll_mean_168),
so the network learns shapes and ratios rather than absolute MW, the same idea
as the tabular ratio target.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from shef.features.asof import load_as_of, target_hours, weather_as_of
from shef.features.builder import _holiday_names
from shef.timestamps import central_date, issue_time_utc, to_central

HOUR = pd.Timedelta(hours=1)
ENC_HOURS = 168
DEC_HOURS = 25
ENC_FEATURES = ["load_ratio", "load_known", "temp_c", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
DEC_FEATURES = ["temp_c", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_holiday", "lead"]


@dataclass
class Sequences:
    enc: np.ndarray          # (n, 168, len(ENC_FEATURES))
    dec: np.ndarray          # (n, 25, len(DEC_FEATURES))
    y: np.ndarray            # (n, 25) load / level, 0 where padded or missing
    mask: np.ndarray         # (n, 25) 1 where y is a real, scoreable target hour
    level: np.ndarray        # (n,)
    target_ts: np.ndarray    # (n, 25) UTC target hour starts, NaT where padded
    issue: np.ndarray        # (n,)

    def __len__(self):
        return len(self.level)


def _clock(ts: pd.DatetimeIndex) -> np.ndarray:
    local = to_central(pd.Series(ts, index=ts))
    h, d = local.dt.hour.to_numpy(), local.dt.dayofweek.to_numpy()
    return np.column_stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                            np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)])


def one(issue: pd.Timestamp, day: date, joined_window: pd.DataFrame, mode: str, label: pd.Series | None):
    known, cutoff = load_as_of(joined_window[["ts_utc", "interval_end_utc", "load_mw"]], issue)
    weather = weather_as_of(issue, day, mode, joined=joined_window)

    enc_ts = pd.date_range(cutoff - ENC_HOURS * HOUR, cutoff - HOUR, freq="h").as_unit("us")
    load = known["load_mw"].reindex(enc_ts)
    level = float(load.mean())
    enc = np.column_stack([(load / level).fillna(0).to_numpy(), load.notna().to_numpy(float),
                           weather["temp_c"].reindex(enc_ts).ffill().to_numpy(), _clock(enc_ts)])

    tgt = target_hours(issue, day)
    n = len(tgt)
    hol = _holiday_names(central_date(pd.Series(tgt, index=tgt))).notna().to_numpy(float)
    dec_real = np.column_stack([weather["temp_c"].reindex(tgt).to_numpy(), _clock(tgt), hol,
                                ((tgt - issue) / HOUR).to_numpy(float) / 38])
    dec = np.zeros((DEC_HOURS, len(DEC_FEATURES)))
    dec[:n] = dec_real

    y = np.zeros(DEC_HOURS)
    mask = np.zeros(DEC_HOURS)
    ts = np.full(DEC_HOURS, np.datetime64("NaT"), dtype="datetime64[us]")
    ts[:n] = tgt.tz_convert(None).to_numpy()
    if label is not None:
        yl = label.reindex(tgt).to_numpy(float)
        y[:n] = np.nan_to_num(yl / level)
        mask[:n] = ~np.isnan(yl)
    else:
        mask[:n] = 1
    return enc, dec, y, mask, level, ts


def build(joined: pd.DataFrame, start: date, end: date, mode: str) -> Sequences:
    joined = joined.sort_values("ts_utc").reset_index(drop=True)
    idx = pd.DatetimeIndex(joined["ts_utc"])
    label = joined.set_index("ts_utc")["load_mw"]
    out = {k: [] for k in ("enc", "dec", "y", "mask", "level", "target_ts", "issue")}
    day = start
    while day <= end:
        issue = issue_time_utc(day)
        lo, hi = idx.searchsorted(issue - pd.Timedelta(days=10)), idx.searchsorted(issue + pd.Timedelta(days=3))
        enc, dec, y, mask, level, ts = one(issue, day, joined.iloc[lo:hi], mode, label)
        for k, v in zip(out, (enc, dec, y, mask, level, ts, issue.tz_convert(None).to_datetime64())):
            out[k].append(v)
        day += timedelta(days=1)
    # Uri days are never scored and not trained on (P3 default).
    seq = Sequences(*(np.stack(out[k]) for k in out))
    uri = joined.set_index("ts_utc")["is_uri"]
    is_uri = np.array([bool(uri.reindex(pd.DatetimeIndex(t[~np.isnat(t)]).tz_localize("UTC")).any())
                       for t in seq.target_ts])
    seq.mask[is_uri] = 0
    return seq
