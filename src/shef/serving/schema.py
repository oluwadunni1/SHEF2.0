"""
Request and response models. The row model is built from the champion's own
feature list (v1 build_request_model), so a model with different features gets
a matching schema on redeploy without a hand edit.

Bundle features (trend_idx, holiday_freq, extreme-event flags) are NOT accepted
from the client: the API derives them from ts_utc / holiday_name / tmax / tmin
with the bundle that shipped with the model.

Values are nullable because the builder legitimately emits NaN (a load lag
that fell in a gap). temp_c is nullable too, but a null there sends the whole
day to the baseline fallback rather than to the model.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, create_model, model_validator

from shef.config import CENTRAL_TZ, contract
from shef.features import BUNDLE_FEATURES
from shef.timestamps import central_day_hours, issue_time_utc

# Open-Meteo model served at issue time. The model trains on ERA5, so this is a serving choice, not
# part of the fitted model. Until 2026-10-28 it was gfs_seamless, which at our 25-38 h leads is HRRR
# live but GFS global in the Previous Runs archive the backtest used (FINDINGS 13). ECMWF IFS 0.25 is
# one model at every lead: day-1 temp MAE 0.84 vs 1.37 C, backtest MAPE 3.54 -> 3.34 (P9 C9a).
SERVED_WEATHER_MODEL = "ecmwf_ifs025"
SERVED_SINCE = date(2026, 10, 29)          # first issue date served with it
PREVIOUS_WEATHER_MODEL = "gfs_seamless"


def served_model_for(issue_date: date) -> str:
    """The weather model that was served for a Central issue date."""
    return SERVED_WEATHER_MODEL if issue_date >= SERVED_SINCE else PREVIOUS_WEATHER_MODEL
LOAD_MIN_MW, LOAD_MAX_MW = 2000.0, 30000.0   # same bounds as the clean-layer gate (schemas.py)
TEMP = (-30.0, 55.0)
LOAD = (LOAD_MIN_MW, LOAD_MAX_MW)
LEAD = tuple(float(x) for x in contract()["issue"]["lead_hours"])

# Bounds are the bad-data guardrail: wrong units or a broken upstream join
# are rejected at the door instead of reaching the model.
BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "hour": (0, 23), "dayofweek": (0, 6), "month": (1, 12), "lead_hours": LEAD,
    "hour_sin": (-1, 1), "hour_cos": (-1, 1), "dow_sin": (-1, 1), "dow_cos": (-1, 1),
    **{f: TEMP for f in ["temp_c", "tmax", "tmin", "tavg", "tmax_roll_max_72",
                         "tavg_lag_24", "tmax_lag_24", "tmin_lag_24"]},
    "cdd": (0, None), "hdd": (0, None), "temp_c_roll_std_72": (0, None),
    "humidity_pct": (0, 100), "precip_mm": (0, 100),
    **{f: LOAD for f in ["load_same_hour_latest", "load_lag_48", "load_lag_168", "last_known_load",
                         "last_known_roll_mean_24", "last_known_roll_max_24", "last_known_roll_mean_168"]},
    "last_known_roll_std_24": (0, None),
    "same_hour_age_h": (0, 240), "load_staleness_h": (0, 240),
}


def _field(name: str):
    lo, hi = BOUNDS.get(name, (None, None))
    if name.startswith("is_"):
        lo, hi = 0, 1
    return (float | None, Field(None, ge=lo, le=hi, allow_inf_nan=False))


def client_features(features: list[str]) -> list[str]:
    return [f for f in features if f not in BUNDLE_FEATURES]


def build_request_model(features: list[str]) -> type[BaseModel]:
    row = create_model("ForecastRow", __config__=ConfigDict(extra="forbid"), ts_utc=(AwareDatetime, ...), holiday_name=(str | None, None),
                       **{f: _field(f) for f in client_features(features)})

    class ForecastRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")

        issue_time_utc: AwareDatetime
        weather_model: Literal[SERVED_WEATHER_MODEL]  # type: ignore[valid-type]
        rows: list[row] = Field(min_length=1)  # type: ignore[valid-type]

        @model_validator(mode="after")
        def one_central_day(self):
            ts = pd.DatetimeIndex([pd.Timestamp(r.ts_utc).tz_convert("UTC") for r in self.rows])
            if not ts.is_monotonic_increasing or ts.has_duplicates:
                raise ValueError("rows must be strictly increasing in ts_utc")
            days = set(ts.tz_convert(CENTRAL_TZ).date)
            if len(days) != 1:
                raise ValueError(f"rows span {len(days)} Central days; send exactly one")
            day = days.pop()
            if len(ts) != central_day_hours(day):
                raise ValueError(f"{day} has {central_day_hours(day)} hours; got {len(ts)} rows")
            issue = pd.Timestamp(self.issue_time_utc).tz_convert("UTC")
            if issue != issue_time_utc(day):
                raise ValueError(f"issue_time_utc must be {issue_time_utc(day)} (10:00 Central on D-1) for {day}")
            if "lead_hours" in row.model_fields:
                expect = list((ts - issue) / pd.Timedelta(hours=1))
                if expect != [r.lead_hours for r in self.rows]:
                    raise ValueError("lead_hours do not match ts_utc - issue_time_utc")
            return self

        def target_day(self) -> date:
            return pd.Timestamp(self.rows[0].ts_utc).tz_convert(CENTRAL_TZ).date()

        def frame(self) -> pd.DataFrame:
            df = pd.DataFrame([r.model_dump() for r in self.rows])
            df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True).dt.as_unit("us")
            num = [c for c in df.columns if c not in ("ts_utc", "holiday_name")]
            df[num] = df[num].astype(float)
            return df

    return ForecastRequest


class HourForecast(BaseModel):
    ts_utc: datetime
    lead_hours: float | None
    load_mw: float


class ForecastResponse(BaseModel):
    status: Literal["ok", "fallback"]
    reason: str | None = None
    target_day: date
    issue_time_utc: datetime
    model_name: str
    model_version: str
    mlflow_run_id: str | None
    weather_model: str
    predictions: list[HourForecast]
    logged: int
    warning: str | None = None


class Actual(BaseModel):
    ts_utc: AwareDatetime
    load_mw: float = Field(ge=LOAD_MIN_MW, le=LOAD_MAX_MW, allow_inf_nan=False)


class ActualsRequest(BaseModel):
    actuals: list[Actual] = Field(min_length=1, max_length=24 * 14)


class ActualsResponse(BaseModel):
    received: int
    updated: int


class LoggedResponse(BaseModel):
    target_day: date
    hours_logged: int
    hours_expected: int
    complete: bool


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    mlflow_run_id: str | None
    trained_through: str | None
    weather_model: str
    features: list[str]
