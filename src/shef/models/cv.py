"""
One CV runner for every tabular experiment: fold data -> target form ->
fit on training rows -> predict validation rows -> map back to MW.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from shef.models import data, folds
from shef.models.metrics import score_folds
from shef.models.targets import Target

KEEP = ["ts_utc", "issue_time_utc", "lead_hours", "hour", "month", "is_dst_day", "y"]


@dataclass
class Spec:
    name: str
    model: Callable[[], object]          # zero-arg builder, e.g. lambda: gbdt.lgbm()
    target: str = "raw"                  # raw | trend | ratio
    features: list[str] = field(default_factory=data.default_features)
    train_source: str = "obs"            # obs | proxy | hybrid
    uri: str = "exclude"                 # exclude | weight | keep
    fold_ids: tuple[int, ...] = (1, 2, 3)

    def params(self) -> dict:
        return {"model": self.name, "target": self.target, "train_source": self.train_source, "uri": self.uri,
                "n_features": len(self.features), "features": self.features, "folds": list(self.fold_ids)}


@dataclass
class Result:
    spec: Spec
    scored: pd.DataFrame                 # validation rows: KEEP + fold + pred
    metrics: dict[str, float]
    models: dict                         # fold -> (fitted model, Target, Bundle)
    seconds: float


def fit_predict(spec: Spec, fd: data.FoldData):
    target = Target(spec.target).fit(fd.train, fd.train["y"])
    z = target.transform(fd.train, fd.train["y"])
    ok = np.isfinite(z)
    model = spec.model()
    X = fd.train.loc[ok, spec.features]
    if isinstance(model, Pipeline):  # sklearn routes weights to a pipeline step by name
        model.fit(X, z[ok], **{f"{model.steps[-1][0]}__sample_weight": fd.weight[ok]})
    else:
        model.fit(X, z[ok], sample_weight=fd.weight[ok])
    pred = target.inverse(fd.val, model.predict(fd.val[spec.features]))
    return model, target, pred


def run_cv(spec: Spec, tables: dict, verbose: bool = True) -> Result:
    t0 = time.time()
    parts, models = [], {}
    for fold in folds.FOLDS:
        if fold.id not in spec.fold_ids:
            continue
        fd = data.fold_data(tables, fold, spec.train_source, spec.uri)
        model, target, pred = fit_predict(spec, fd)
        part = fd.val[KEEP].copy()
        part["fold"], part["pred"] = fold.id, pred
        parts.append(part)
        models[fold.id] = (model, target, fd.bundle)
        if verbose:
            m = score_folds(part)
            print(f"  {spec.name:<28} fold {fold.id}: MAPE {m['mape']:.2f}%  peak {m['peak_mape']:.2f}%  "
                  f"(train {len(fd.train):,} / val {len(fd.val):,})")
    scored = pd.concat(parts, ignore_index=True)
    return Result(spec, scored, score_folds(scored), models, time.time() - t0)
