"""The EventBridge trigger's inline Lambda code (infra/scheduler.yaml), run with fakes:
dispatch posts the right workflow inputs; check alerts and catches up only when a day is incomplete."""
import json
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


class _CfnLoader(yaml.SafeLoader):
    pass


_CfnLoader.add_multi_constructor("!", lambda loader, tag, node: None)   # ignore !Ref, !Sub, ...


@pytest.fixture
def fn(monkeypatch):
    tpl = yaml.load((ROOT / "infra" / "scheduler.yaml").read_text(), Loader=_CfnLoader)
    code = tpl["Resources"]["TriggerFunction"]["Properties"]["Code"]["ZipFile"]
    mod = types.ModuleType("scheduler_fn")
    exec(compile(code, "scheduler_fn", "exec"), mod.__dict__)
    calls = {"posts": [], "alerts": [], "status": None}

    class Resp:
        def __init__(self, status, body=b""):
            self.status, self._body = status, body
        def read(self):
            return self._body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def urlopen(req, timeout):
        if req.get_method() == "POST":
            calls["posts"].append((req.full_url, json.loads(req.data), req.headers))
            return Resp(204)
        return Resp(200, json.dumps(calls["status"]).encode())

    monkeypatch.setenv("REPO", "owner/repo")
    monkeypatch.setenv("API_URL", "https://x.lambda-url.eu-west-1.on.aws/")
    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(mod, "_token", lambda: "tkn")
    monkeypatch.setattr(mod, "_signed_headers", lambda url: {"Authorization": "AWS4-HMAC-SHA256 ..."})
    monkeypatch.setattr(mod, "_alert", lambda subject, message: calls["alerts"].append(subject))
    return mod, calls


def test_dispatch_starts_daily_as_eventbridge(fn):
    mod, calls = fn
    assert mod.handler({"action": "dispatch"}, None) == {"dispatched": True}
    url, body, headers = calls["posts"][0]
    assert url.endswith("/repos/owner/repo/actions/workflows/daily.yml/dispatches")
    assert body == {"ref": "main", "inputs": {"dry_run": "false", "source": "eventbridge"}}
    assert headers["Authorization"] == "Bearer tkn"


def test_check_is_quiet_when_tomorrow_is_logged(fn):
    mod, calls = fn
    calls["status"] = {"target_day": "x", "hours_logged": 24, "hours_expected": 24, "complete": True}
    mod.handler({"action": "check"}, None)
    assert not calls["posts"] and not calls["alerts"]


def test_check_alerts_and_catches_up_when_missing(fn):
    mod, calls = fn
    calls["status"] = {"target_day": "x", "hours_logged": 0, "hours_expected": 24, "complete": False}
    mod.handler({"action": "check"}, None)
    assert len(calls["posts"]) == 1 and len(calls["alerts"]) == 1 and "missing" in calls["alerts"][0]


def test_unknown_action_fails_loudly(fn):
    mod, _ = fn
    with pytest.raises(ValueError):
        mod.handler({"action": "nope"}, None)
