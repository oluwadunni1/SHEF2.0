"""P4 API: request validation, the forecast gate's fallback, logging, auth."""
from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from shef.features import contract_features
from shef.features.replay import replay
from shef.models.predict import Champion
from shef.serving import client, db, schema
from shef.serving.app import BASELINE_NAME, create_app
from tests.test_features import synthetic_joined

META = {"model_name": "shef2-day-ahead", "model_version": "7", "mlflow_run_id": "abc",
        "trained_through": "2025-07-31", "weather_model": "gfs_seamless"}


@pytest.fixture(scope="module")
def setup():
    j = synthetic_joined("2025-04-15", "2025-11-10")
    rows = replay(j, date(2025, 4, 26), date(2025, 7, 31), "obs")
    feats = [f for f in contract_features() if f not in ("humidity_pct", "precip_mm")]
    champ = Champion.fit(rows.dropna(subset=["y"]), feats, params={"n_estimators": 50})
    days = {d: replay(j, d, d, "proxy").reset_index(drop=True)
            for d in (date(2025, 8, 5), date(2025, 11, 2))}  # a normal day and the 25-hour fall-back day
    return champ, days


@pytest.fixture
def api(setup):
    champ, days = setup
    logged, actuals = [], []

    def fake_log(records):
        logged.extend(records)
        return len(records)

    app = create_app(champ, META, log_predictions=fake_log,
                     log_actuals=lambda a: actuals.extend(a) or len(a))
    return TestClient(app), champ, days, logged, actuals


def body_for(day_rows, champ):
    return client.payload(day_rows, champ.features)


def post(c, body):
    return c.post("/forecast", json=body)


def test_health_is_open_and_names_the_model(api):
    c, champ, *_ = api
    h = c.get("/health").json()
    assert h["model_version"] == "7" and h["features"] == champ.features


def test_function_url_region_is_parsed_for_signing(monkeypatch):
    seen = {}

    class Creds:
        def get_frozen_credentials(self):
            from botocore.credentials import ReadOnlyCredentials
            return ReadOnlyCredentials("AKIDEXAMPLE", "secret", None)

    import botocore.session
    import requests
    monkeypatch.setattr(botocore.session.Session, "get_credentials", lambda self: Creds())

    def fake_request(method, url, data=None, headers=None, timeout=None):
        seen.update(method=method, url=url, headers=headers)
        r = requests.Response(); r.status_code = 200; r._content = b"{}"
        return r

    monkeypatch.setattr(requests, "request", fake_request)
    client.post("https://abc.lambda-url.eu-west-1.on.aws/", "/forecast", {"x": 1})
    auth = seen["headers"]["Authorization"]
    assert seen["url"].endswith("/forecast") and "/eu-west-1/lambda/aws4_request" in auth


@pytest.mark.parametrize("day", [date(2025, 8, 5), date(2025, 11, 2)])
def test_forecast_equals_champion_and_logs_every_hour(api, day):
    c, champ, days, logged, _ = api
    rows = days[day]
    r = post(c, body_for(rows, champ))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["status"] == "ok" and out["model_name"] == "shef2-day-ahead" and out["logged"] == len(rows)
    np.testing.assert_allclose([p["load_mw"] for p in out["predictions"]], champ.predict(rows), atol=0.05)
    rec = logged[0]
    assert set(rec["features"]) == set(champ.features)          # the exact model inputs, bundle applied
    assert rec["lead_hours"] == 14 and rec["weather_model"] == "gfs_seamless" and rec["mlflow_run_id"] == "abc"
    assert pd.Timestamp(rec["issue_time_utc"]) == rows["issue_time_utc"].iloc[0]
    assert len(out["predictions"]) == (25 if day.month == 11 else 24)


def test_missing_temperature_falls_back_to_baseline(api):
    c, champ, days, logged, _ = api
    rows = days[date(2025, 8, 5)].copy()
    rows.loc[3, "temp_c"] = np.nan
    out = post(c, body_for(rows, champ)).json()
    assert out["status"] == "fallback" and out["model_name"] == BASELINE_NAME
    assert "temperature" in out["reason"]
    np.testing.assert_allclose([p["load_mw"] for p in out["predictions"]], rows["load_same_hour_latest"], atol=0.05)
    assert all(r["model_version"] == "baseline" for r in logged)


def mutate(body, f):
    body = {**body, "rows": [dict(r) for r in body["rows"]]}
    f(body)
    return body


@pytest.mark.parametrize("name,change", [
    ("missing hour", lambda b: b["rows"].pop(5)),
    ("not 10:00 Central", lambda b: b.update(issue_time_utc="2025-08-04T16:00:00+00:00")),
    ("wrong weather model", lambda b: b.update(weather_model="best_match")),
    ("temperature in Fahrenheit", lambda b: b["rows"][0].update(temp_c=95.0)),
    ("load in kW", lambda b: b["rows"][0].update(load_lag_168=9_000_000.0)),
    ("naive timestamp", lambda b: b["rows"][0].update(ts_utc="2025-08-05T05:00:00")),
    ("client-sent trend", lambda b: b["rows"][0].update(trend_idx=99.0)),
    ("lead hours off", lambda b: b["rows"][0].update(lead_hours=20.0)),
])
def test_bad_requests_are_rejected(api, name, change):
    c, champ, days, logged, _ = api
    r = post(c, mutate(body_for(days[date(2025, 8, 5)], champ), change))
    assert r.status_code == 422, name
    assert not logged


def test_logging_failure_still_returns_the_forecast(setup):
    champ, days = setup

    def broken(records):
        raise ConnectionError("db down")

    c = TestClient(create_app(champ, META, log_predictions=broken))
    out = post(c, body_for(days[date(2025, 8, 5)], champ)).json()
    assert out["status"] == "ok" and out["logged"] == 0 and "logging failed" in out["warning"]


def test_actuals(api):
    c, *_, actuals = api
    body = {"actuals": [{"ts_utc": "2025-08-05T05:00:00+00:00", "load_mw": 9000.0}]}
    r = c.post("/actuals", json=body)
    assert r.json() == {"received": 1, "updated": 1} and actuals[0]["load_mw"] == 9000.0
    assert c.post("/actuals", json={"actuals": [{"ts_utc": "2025-08-05T05:00:00+00:00", "load_mw": -1}]}).status_code == 422


def test_bounds_match_the_clean_gate():
    from shef import schemas
    assert (schema.LOAD_MIN_MW, schema.LOAD_MAX_MW) == (schemas.LOAD_MIN_MW, schemas.LOAD_MAX_MW)


def test_nan_features_are_stored_as_null():
    assert db.json_safe({"a": float("nan"), "b": 1.0}) == {"a": None, "b": 1.0}
