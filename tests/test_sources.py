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
