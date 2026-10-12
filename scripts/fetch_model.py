"""
fetch_model.py — pull one registered model version from MLflow into a plain
directory the API can load without MLflow:

    <out>/champion/        lgbm.txt, target.json, bundle.json, features.json
    <out>/model_meta.json  name, version, run id, trained-through, weather model

The deploy workflow runs this before `docker build`, so each image carries
exactly one model version and the API never calls the registry.

    uv run --env-file .env python scripts/fetch_model.py --version 1 --out build/model
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from shef.models import tracking

NAME = "shef2-day-ahead"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="1")
    p.add_argument("--out", type=Path, default=Path("build/model"))
    args = p.parse_args()

    mlflow = tracking.init()
    mv = mlflow.MlflowClient().get_model_version(NAME, args.version)
    run = mlflow.get_run(mv.run_id)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(f"models:/{NAME}/{args.version}", dst_path=tmp))
        champion = next(local.rglob("features.json")).parent
        if args.out.exists():
            shutil.rmtree(args.out)
        shutil.copytree(champion, args.out / "champion")
    meta = {"model_name": NAME, "model_version": str(args.version), "mlflow_run_id": mv.run_id,
            "trained_through": run.data.params.get("train_end"),
            "weather_model": run.data.tags.get("weather_model_for_serving", "gfs_seamless")}
    (args.out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{NAME} v{args.version} -> {args.out}: {sorted(f.name for f in (args.out / 'champion').iterdir())} {meta}")


if __name__ == "__main__":
    main()
