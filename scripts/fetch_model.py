"""
fetch_model.py — pull one registered model (a version, or the @champion alias)
from MLflow into a plain directory the API can load without MLflow:

    <out>/champion/        lgbm.txt, target.json, bundle.json, features.json
    <out>/model_meta.json  name, version, run id, trained-through, weather model

The deploy workflow runs this before `docker build`, so each image carries
exactly one model version and the API never calls the registry.

    uv run --env-file .env python scripts/fetch_model.py                 # @champion
    uv run --env-file .env python scripts/fetch_model.py --version 1 --out build/model
"""
from __future__ import annotations

import argparse
from pathlib import Path

from shef.models import registry


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", default=registry.ALIAS, help="version number or alias (default: champion)")
    p.add_argument("--out", type=Path, default=Path("build/model"))
    args = p.parse_args()
    meta = registry.fetch(args.version, args.out)
    print(f"{registry.NAME} {args.version} -> {args.out}: {meta}")


if __name__ == "__main__":
    main()
