"""
Combine the four cities into one South Central weather signal, weighted by
population (anchors and interpolation ported from v1 src/weather_transform.py).
A weighted value is NaN if any city is missing, so a gap never silently turns
into a three-city average.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shef.config import CITIES

POPULATION_ANCHORS: dict[int, dict[str, int]] = {
    2016: {"san_antonio": 1_492_510, "austin": 938_200, "round_rock": 120_955, "san_marcos": 61_782},
    2018: {"san_antonio": 1_532_233, "austin": 962_800, "round_rock": 129_046, "san_marcos": 63_802},
    2020: {"san_antonio": 1_434_625, "austin": 962_163, "round_rock": 120_556, "san_marcos": 68_199},
    2022: {"san_antonio": 1_472_909, "austin": 978_358, "round_rock": 127_340, "san_marcos": 70_572},
    2024: {"san_antonio": 1_526_656, "austin": 993_588, "round_rock": 135_665, "san_marcos": 74_319},
}


def yearly_weights(years: list[int]) -> pd.DataFrame:
    """Interpolate populations between anchor years (flat beyond the ends); rows sum to 1."""
    anchors = sorted(POPULATION_ANCHORS)
    pops = {c: np.interp(years, anchors, [POPULATION_ANCHORS[y][c] for y in anchors]) for c in CITIES}
    df = pd.DataFrame(pops, index=pd.Index(years, name="year"))
    return df.div(df.sum(axis=1), axis=0)


def weight_cities(long: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """(ts_utc, city, values...) -> (ts_utc, values...) population-weighted."""
    wide = long.pivot(index="ts_utc", columns="city", values=value_cols)
    years = wide.index.year
    w = yearly_weights(sorted(set(years)))
    out = pd.DataFrame(index=wide.index)
    for col in value_cols:
        block = wide[col][CITIES]
        weights = w.loc[years, CITIES].to_numpy()
        # Plain numpy sum propagates NaN, so any missing city gives NaN. Rounded
        # because sources report 0.1 precision and four cities at 100% RH
        # otherwise sum to 100.00000000000003.
        out[col] = (block.to_numpy() * weights).sum(axis=1).round(3)
    return out.reset_index()
