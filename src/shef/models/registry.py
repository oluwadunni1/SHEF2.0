"""
MLflow Model Registry helpers for shef2-day-ahead: fetch a version (or the
@champion alias) as plain files, register a new Champion, move the alias.
Used by scripts/fetch_model.py (deploy), scripts/promote.py, notebook 05's recipe.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from shef.config import ROOT
from shef.models import tracking
from shef.models.predict import Champion
from shef.serving.schema import SERVED_WEATHER_MODEL

NAME = "shef2-day-ahead"
ALIAS = "champion"


def resolve(ref: str):
    """ref: a version number ("3") or an alias ("champion") -> ModelVersion."""
    client = tracking.init().MlflowClient()
    return client.get_model_version(NAME, ref) if ref.isdigit() else client.get_model_version_by_alias(NAME, ref)


def fetch(ref: str, out: Path) -> dict:
    """Write <out>/champion/ (lgbm, target, bundle, features) and <out>/model_meta.json."""
    mlflow = tracking.init()
    mv = resolve(ref)
    run = mlflow.get_run(mv.run_id)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(f"models:/{NAME}/{mv.version}", dst_path=tmp))
        src = next(local.rglob("features.json")).parent
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(src, out / "champion")
    meta = {"model_name": NAME, "model_version": str(mv.version), "mlflow_run_id": mv.run_id,
            "trained_through": run.data.params.get("train_end"),
            "weather_model": SERVED_WEATHER_MODEL}   # a serving choice: the model trains on ERA5
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def register(champion: Champion, run_name: str, params: dict, metrics: dict, tags: dict,
             dicts: dict | None = None) -> tuple[str, str]:
    """Log `champion` as a pyfunc and register it. Returns (version, run_id). Does not move the alias."""
    from shef.models.predict import ChampionPyfunc

    mlflow = tracking.init()
    with tempfile.TemporaryDirectory() as tmp:
        art = champion.save(Path(tmp) / "champion")
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags({**tracking.provenance(), "weather_model_for_serving": SERVED_WEATHER_MODEL, **tags})
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            for name, d in (dicts or {}).items():
                mlflow.log_dict(d, name)
            info = mlflow.pyfunc.log_model(name="model", python_model=ChampionPyfunc(), artifacts={"champion": str(art)},
                                           code_paths=[str(ROOT / "src" / "shef")], registered_model_name=NAME)
    return str(info.registered_model_version), run.info.run_id


def set_champion(version: str) -> None:
    tracking.init().MlflowClient().set_registered_model_alias(NAME, ALIAS, version)
