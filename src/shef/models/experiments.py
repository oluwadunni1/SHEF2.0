"""
P9 experiment registry: one Experiment per Buildboard task, each a single
change to the champion recipe (shef.models.harness). Run with
scripts/experiment.py; results land in MLflow (shef2-harness) and
data/reports/harness.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shef.models.harness import Experiment


def recency_2y(tr: pd.DataFrame) -> np.ndarray:
    age_years = (tr["ts_utc"].max() - tr["ts_utc"]).dt.total_seconds() / (365.25 * 86400)
    return 0.5 ** (age_years / 2.0).to_numpy()


def inverse_load(tr: pd.DataFrame) -> np.ndarray:
    return (tr["y"].mean() / tr["y"]).to_numpy()


def peak_x2(tr: pd.DataFrame) -> np.ndarray:
    return np.where(tr["y"] >= tr["y"].quantile(0.90), 2.0, 1.0)


def wow_ratio(df: pd.DataFrame) -> pd.Series:
    return df["load_same_hour_latest"] / df["load_lag_168"]


EXPERIMENTS = {e.name: e for e in [
    Experiment("baseline", "The champion recipe: LightGBM, trend target, 42 features, obs training rows."),
    Experiment("A2_delta", "Predicting the change from the freshest legal same-hour load takes the level from "
               "data rather than a linear trend line, so record-demand months are not under-forecast.",
               target="delta"),
    # Unregularised linear leaves diverged (325,000 MW predictions, MAPE 1,227% on fold 1: collinear,
    # unscaled inputs such as trend_idx). linear_lambda=100 was chosen after seeing fold 1, which
    # leans toward the idea; it is still 0.5 pts worse than the baseline on that fold.
    Experiment("A3_linear_tree", "Linear leaves extrapolate the temperature response past training peaks.",
               params={"linear_tree": True, "linear_lambda": 100.0}),
    Experiment("A4_recency_2y", "Recent behaviour (AC stock, growth) should outweigh 2016: half-life 2 years.",
               weight=recency_2y),
    Experiment("B5_inverse_load", "L2 on MW optimises RMSE; weighting by 1/load moves the fit toward MAPE.",
               weight=inverse_load),
    Experiment("B6_peak_x2", "Peak-MAPE is the weak spot: double weight on the top-decile load hours.",
               weight=peak_x2),
    Experiment("D12_wow_ratio", "Week-over-week growth of the freshest load is a cheap level signal.",
               derive={"wow_ratio": wow_ratio}),
]}
