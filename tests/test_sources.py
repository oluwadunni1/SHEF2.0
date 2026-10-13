"""Source clients: transient failures are retried, persistent ones surface."""
import pytest
import requests

from shef.sources import openmeteo


class Resp:
    def __init__(self, status):
        self.status_code = status


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(openmeteo.time, "sleep", lambda s: None)


def test_timeout_then_server_error_then_success(monkeypatch, no_sleep):
    outcomes = iter([requests.Timeout("handshake timed out"), Resp(503), Resp(200)])

    def fake_get(*a, **k):
        o = next(outcomes)
        if isinstance(o, Exception):
            raise o
        return o

    monkeypatch.setattr(openmeteo.requests, "get", fake_get)
    assert openmeteo._get("https://x", {}).status_code == 200


def test_persistent_network_failure_is_raised(monkeypatch, no_sleep):
    calls = []

    def fake_get(*a, **k):
        calls.append(1)
        raise requests.ConnectionError("down")

    monkeypatch.setattr(openmeteo.requests, "get", fake_get)
    with pytest.raises(requests.ConnectionError):
        openmeteo._get("https://x", {})
    assert len(calls) == openmeteo.RETRIES


def test_client_errors_are_not_retried(monkeypatch, no_sleep):
    calls = []
    monkeypatch.setattr(openmeteo.requests, "get", lambda *a, **k: calls.append(1) or Resp(400))
    assert openmeteo._get("https://x", {}).status_code == 400 and len(calls) == 1


def test_ercot_forecast_is_point_in_time():
    import pandas as pd
    from datetime import date

    from shef.sources import gridstatus
    from shef.timestamps import issue_time_utc

    day = date(2026, 10, 1)
    issue = issue_time_utc(day)                                  # 2026-09-30 15:00 UTC
    hours = pd.date_range("2026-10-01 05:00", periods=24, freq="h", tz="UTC")
    publishes = {issue - pd.Timedelta(minutes=90): 100.0,        # older
                 issue - pd.Timedelta(minutes=30): 200.0,        # latest legal: 09:30 CT
                 issue + pd.Timedelta(minutes=30): 999.0}        # after issue: must never be used

    class FakeGS:
        def get_dataset(self, **kw):
            assert kw["publish_time_end"] == issue
            return pd.DataFrame([{"interval_start_utc": h, "publish_time_utc": p, "south_central": v}
                                 for h in hours for p, v in publishes.items()])

    out = gridstatus.fetch_ercot_forecast(day, issue, FakeGS())
    assert len(out) == 24 and (out["forecast_mw"] == 200.0).all()
    assert (out["publish_time_utc"] <= issue).all()


def test_quota_guard_keeps_the_reserve():
    from shef.monitoring.benchmark import can_spend

    assert can_spend({"requests_used": 189, "requests_limit": 250}, reserve=60)
    assert not can_spend({"requests_used": 190, "requests_limit": 250}, reserve=60)
