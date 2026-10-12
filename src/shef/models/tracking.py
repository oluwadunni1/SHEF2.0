"""
MLflow on DagsHub (credentials from .env, as in v1). One log_run call per
experiment run, from every notebook, so runs are comparable.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from shef.config import ROOT

EXPERIMENT = "shef2-day-ahead"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def init(experiment: str = EXPERIMENT):
    import mlflow

    load_dotenv(ROOT / ".env")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment)
    return mlflow


def provenance() -> dict[str, str]:
    """Which code and which data produced a run."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    out = {"git_commit": git("rev-parse", "--short", "HEAD"), "git_dirty": str(bool(git("status", "--porcelain")))}
    lock = yaml.safe_load((ROOT / "dvc.lock").read_text())
    for stage, spec in lock.get("stages", {}).items():
        for o in spec.get("outs", []):
            if o["path"].startswith("data/features/"):
                out[f"dvc_{o['path'].rsplit('/', 1)[-1]}"] = o["md5"]
    return out


def log_run(name: str, params: dict, metrics: dict, *, figures: dict | None = None,
            frames: dict[str, pd.DataFrame] | None = None, dicts: dict[str, dict] | None = None,
            files: dict[str, Path] | None = None, tags: dict | None = None) -> str:
    """Log one experiment run. Returns the run id."""
    import mlflow

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({**provenance(), **(tags or {})})
        mlflow.log_params({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        for fname, fig in (figures or {}).items():
            mlflow.log_figure(fig, f"plots/{fname}.png")
        for fname, d in (dicts or {}).items():
            mlflow.log_dict(d, f"{fname}.json")
        with tempfile.TemporaryDirectory() as tmp:
            for fname, df in (frames or {}).items():
                path = Path(tmp) / f"{fname}.parquet"
                df.to_parquet(path, index=False)
                mlflow.log_artifact(str(path), "data")
        for sub, path in (files or {}).items():
            mlflow.log_artifact(str(path), sub)
        return run.info.run_id
