"""
Target forms. Trees can't extrapolate past the load levels they trained on,
and South Central load grows year on year, so the model can learn something
other than raw MW and the prediction is mapped back afterwards.

    raw     z = y
    trend   z = y - (a + b * trend_idx), with the line fitted on the training rows (v1)
    ratio   z = y / last_known_roll_mean_168, the mean of the last week known at issue time

Each form is fitted on training rows only and serialises to a small dict, so
the fitted form ships with the model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

LEVEL = "last_known_roll_mean_168"


@dataclass
class Target:
    kind: str                     # raw | trend | ratio
    intercept: float = 0.0
    slope: float = 0.0

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "Target":
        if self.kind == "trend":
            self.slope, self.intercept = np.polyfit(df["trend_idx"].to_numpy(float), y.to_numpy(float), 1)
        return self

    def transform(self, df: pd.DataFrame, y: pd.Series) -> np.ndarray:
        y = y.to_numpy(float)
        if self.kind == "raw":
            return y
        if self.kind == "trend":
            return y - self._line(df)
        if self.kind == "ratio":
            return y / df[LEVEL].to_numpy(float)
        raise ValueError(self.kind)

    def inverse(self, df: pd.DataFrame, z) -> np.ndarray:
        z = np.asarray(z, float)
        if self.kind == "raw":
            return z
        if self.kind == "trend":
            return z + self._line(df)
        if self.kind == "ratio":
            return z * df[LEVEL].to_numpy(float)
        raise ValueError(self.kind)

    def _line(self, df: pd.DataFrame) -> np.ndarray:
        return self.intercept + self.slope * df["trend_idx"].to_numpy(float)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Target":
        return cls(**d)
