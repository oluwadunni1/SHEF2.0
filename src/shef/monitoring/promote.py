"""
Champion / challenger. Should a model retrained on newer data replace the one
serving? Decided on evidence, never on a schedule alone.

Rolling origin (the same protocol as the P3 folds):

    window      the last WINDOW_DAYS target days that have labels (as of `as_of`)
    challenger  the champion's recipe (Champion.fit: same features, target form,
                parameters), trained on obs rows through window start - GAP
    champion    the serving model, trained through `champion_train_end`

Neither model has seen the window, so the only thing the challenger gains is
the data that arrived after the champion's cut-off: exactly what a retrain
would add. Both are scored on the proxy rows (archived day-ahead forecasts),
Uri excluded and hours without forecast temperature dropped, as in P3.

Rule (the P3 champion rule with a margin): promote only if the challenger's
MAPE is lower by at least MARGIN points AND its Peak-MAPE is no worse.
If the challenger would not train on any data the champion has not seen, or
the window is not entirely after the champion's cut-off, the verdict is
"insufficient" and nothing changes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from shef.models import folds
from shef.models.metrics import score
from shef.models.predict import Champion

WINDOW_DAYS = 28
MARGIN = 0.10


@dataclass
class Evaluation:
    decision: str                    # promote | keep | insufficient
    reason: str
    window_start: date | None = None
    window_end: date | None = None
    n_days: int = 0
    challenger_train_end: date | None = None
    champion: dict | None = None     # metrics.score on the window
    challenger: dict | None = None

    def to_dict(self) -> dict:
        return {k: (str(v) if isinstance(v, date) else v) for k, v in asdict(self).items()}


def decide(champion: dict, challenger: dict, margin: float = MARGIN) -> tuple[str, str]:
    gain = champion["mape"] - challenger["mape"]
    if challenger["peak_mape"] > champion["peak_mape"]:
        return "keep", (f"challenger Peak-MAPE {challenger['peak_mape']:.2f}% is worse than "
                        f"{champion['peak_mape']:.2f}%")
    if gain < margin:
        return "keep", f"MAPE gain {gain:+.2f} pts is below the {margin:.2f} pt margin"
    return "promote", (f"MAPE {champion['mape']:.2f}% -> {challenger['mape']:.2f}% ({gain:+.2f} pts), "
                       f"Peak-MAPE {champion['peak_mape']:.2f}% -> {challenger['peak_mape']:.2f}%")


def window(proxy: pd.DataFrame, as_of: date) -> tuple[date, date] | None:
    labelled = proxy.loc[proxy["y"].notna(), :]
    days = folds.target_day(labelled)
    end = min(as_of, days.max())
    return end - timedelta(days=WINDOW_DAYS - 1), end


def evaluate(tables: dict, champion: Champion, champion_train_end: date, as_of: date,
             params: dict | None = None) -> tuple[Evaluation, Champion | None]:
    w = window(tables["proxy"], as_of)
    start, end = w
    chall_end = start - folds.GAP
    if start <= champion_train_end:
        return Evaluation("insufficient", f"window {start}..{end} overlaps the champion's training "
                          f"(through {champion_train_end}); needs labels through "
                          f"{champion_train_end + timedelta(days=WINDOW_DAYS)}", start, end), None
    if chall_end <= champion_train_end:
        return Evaluation("insufficient", f"a challenger would train through {chall_end}, no later than "
                          f"the champion ({champion_train_end}): no new data to learn from", start, end), None

    proxy = tables["proxy"]
    pdays = folds.target_day(proxy)
    val = proxy[(pdays >= start) & (pdays <= end) & ~proxy["is_uri"]].dropna(subset=["y", "temp_c"])
    obs = tables["obs"]
    odays = folds.target_day(obs)
    train = obs[(odays >= folds.TRAIN_START) & (odays <= chall_end) & ~obs["is_uri"]].dropna(subset=["y"])
    challenger = Champion.fit(train, champion.features, target_kind=champion.target.kind, params=params)

    y = val["y"].to_numpy(float)
    s_champ = score(y, champion.predict(val))
    s_chall = score(y, challenger.predict(val))
    decision, reason = decide(s_champ, s_chall)
    n_days = int(folds.target_day(val).nunique())
    return Evaluation(decision, reason, start, end, n_days, chall_end, s_champ, s_chall), challenger


def refit_all(tables: dict, champion: Champion, through: date, params: dict | None = None) -> tuple[Champion, int]:
    """The promoted model: the same recipe on every labelled obs row through `through`."""
    obs = tables["obs"]
    d = folds.target_day(obs)
    rows = obs[(d >= folds.TRAIN_START) & (d <= through) & ~obs["is_uri"]].dropna(subset=["y"])
    return Champion.fit(rows, champion.features, target_kind=champion.target.kind, params=params), len(rows)


def finite(d: dict | None) -> dict | None:
    return None if d is None else {k: (float(v) if np.isfinite(v) else None) for k, v in d.items()}
