"""
build_clean.py — data/raw -> data/clean/{load,weather_obs,weather_fc_proxy,joined}.

Every stage passes its Pandera gate; reports land in data/reports/. A failed
gate exits nonzero and nothing downstream of it is written.

    uv run python scripts/build_clean.py
"""
from __future__ import annotations

import json
import sys

from shef import build, store
from shef.config import REPORTS
from shef.validate import GateFailed


def main() -> None:
    try:
        load = build.clean_load(store.read("raw", "load"))
        obs = build.clean_weather(store.read("raw", "weather_obs"), "archive")
        proxy = build.clean_weather(store.read("raw", "weather_fc_proxy"), "previous_runs")
        store.write(load, "clean", "load")
        store.write(obs, "clean", "weather_obs")
        store.write(proxy, "clean", "weather_fc_proxy")
        joined = build.build_joined(load, obs, proxy)
    except GateFailed as exc:
        sys.exit(str(exc))
    store.write(joined, "clean", "joined")

    report = json.loads((REPORTS / "validation_joined.json").read_text())
    print(f"joined: {len(joined):,} rows, {report['stats']['first_ts_utc']} -> {report['stats']['last_ts_utc']}")
    print(json.dumps({k: v for k, v in report["stats"].items() if k != "load_gaps"}, indent=2))
    print(f"load gaps: {len(report['stats']['load_gaps'])}")
    for g in report["stats"]["load_gaps"][:20]:
        print("  ", g)
    for w in report["warnings"]:
        print("WARN", w)


if __name__ == "__main__":
    main()
