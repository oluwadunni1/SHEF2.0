"""Point-in-time feature builder: one code path for training replay and serving."""
from __future__ import annotations

from shef.config import contract

BUNDLE_FEATURES = {"holiday_freq", "trend_idx", "is_extreme_heat_event", "is_extreme_cold_event"}


def contract_features(mode: str = "day_ahead") -> list[str]:
    """Feature names features.yaml enables for `mode`, in file order."""
    return [f["name"] for f in contract()["features"] if mode in f.get("modes", [])]
