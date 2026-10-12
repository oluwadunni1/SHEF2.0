"""
Evaluation windows, fixed before any model is trained (P3 plan).

Folds are by target Central day: expanding training window from the first
feature day, an 8-day gap, then a 12-month validation window. The holdout is
scored once, at the very end, and holdout() refuses unless final=True.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from shef.timestamps import central_date

TRAIN_START = date(2016, 1, 12)
GAP = timedelta(days=8)


@dataclass(frozen=True)
class Fold:
    id: int
    val_start: date
    val_end: date

    @property
    def train_end(self) -> date:
        return self.val_start - GAP - timedelta(days=1)


FOLDS = (
    Fold(1, date(2022, 8, 10), date(2023, 8, 9)),
    Fold(2, date(2023, 8, 10), date(2024, 8, 9)),
    Fold(3, date(2024, 8, 10), date(2025, 8, 9)),
)
HOLDOUT = (date(2025, 8, 10), date(2026, 10, 9))


def target_day(df: pd.DataFrame) -> pd.Series:
    return central_date(df["ts_utc"])


def train_mask(df: pd.DataFrame, fold: Fold, start: date = TRAIN_START) -> pd.Series:
    d = target_day(df)
    return (d >= start) & (d <= fold.train_end)


def val_mask(df: pd.DataFrame, fold: Fold) -> pd.Series:
    d = target_day(df)
    return (d >= fold.val_start) & (d <= fold.val_end)


def holdout_train_mask(df: pd.DataFrame, start: date = TRAIN_START) -> pd.Series:
    d = target_day(df)
    return (d >= start) & (d <= HOLDOUT[0] - GAP - timedelta(days=1))


def holdout_mask(df: pd.DataFrame, final: bool = False) -> pd.Series:
    if not final:
        raise PermissionError("The holdout is scored once, for the chosen model, in notebook 05 (pass final=True).")
    d = target_day(df)
    return (d >= HOLDOUT[0]) & (d <= HOLDOUT[1])
