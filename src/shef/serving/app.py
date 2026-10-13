"""
The forecast API. Its whole job: validate one Central day of feature rows,
predict with the champion (or the baseline, if the forecast gate trips), log
every hour to Supabase, return the forecast. Feature engineering happens in
the client (scripts/forecast_day.py -> features.live.build_live).

    GET  /health     which model is serving
    POST /forecast   one day of rows -> 23-25 hourly forecasts
    POST /actuals    realised load -> fills actual_load_mw

Authentication is not in this code: the Function URL uses AWS_IAM auth, so
Lambda rejects any request not SigV4-signed by a principal allowed to invoke
it (the GitHub Actions OIDC role) before the app runs. No shared secret exists.

On Lambda, `handler` (Mangum) adapts Function URL events to ASGI. The model is
baked into the image at /opt/model by the deploy workflow and loaded once per
cold start. Locally:

    MODEL_DIR=<dir with champion/ and model_meta.json> uv run uvicorn shef.serving.app:app
"""

import json
import logging
import os
from pathlib import Path
from typing import Callable

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from shef.models.predict import Champion, ForecastIncomplete
from shef.serving import db
from shef.serving.schema import (ActualsRequest, ActualsResponse, ForecastResponse, HealthResponse,
                                 HourForecast, build_request_model)

log = logging.getLogger("shef.api")
log.setLevel(logging.INFO)

BASELINE_NAME = "baseline_same_hour_latest"
BASELINE_COLUMN = "load_same_hour_latest"


def create_app(champion: Champion, meta: dict,
               log_predictions: Callable[[list[dict]], int] = db.log_predictions,
               log_actuals: Callable[[list[dict]], int] = db.log_actuals) -> FastAPI:
    app = FastAPI(title="SHEF 2.0 day-ahead load forecast", version=str(meta["model_version"]))
    Request_ = build_request_model(champion.features)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError):
        # Pydantic error details can echo inputs as non-JSON types (dates); stringify.
        return JSONResponse(status_code=422, content={"detail": json.loads(json.dumps(exc.errors(), default=str))})

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(status="ok", model_name=meta["model_name"], model_version=str(meta["model_version"]),
                              mlflow_run_id=meta.get("mlflow_run_id"), trained_through=meta.get("trained_through"),
                              weather_model=meta["weather_model"], features=champion.features)

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(body: Request_):  # type: ignore[valid-type]
        rows = body.frame()
        status, reason = "ok", None
        name, version = meta["model_name"], str(meta["model_version"])
        try:
            mw = champion.predict(rows)
            logged_features = model_inputs(champion, rows)
        except ForecastIncomplete as exc:
            # P3: a model trained on ERA5 has never seen a missing temperature and is
            # badly wrong without one. Serve the freshest legal same-hour load instead.
            if rows[BASELINE_COLUMN].isna().any():
                raise HTTPException(503, f"forecast incomplete and baseline unavailable: {exc}") from exc
            status, reason = "fallback", str(exc)
            name, version = BASELINE_NAME, "baseline"
            mw = rows[BASELINE_COLUMN].to_numpy(float)
            logged_features = rows.drop(columns=["ts_utc"])
        log.info("forecast %s day=%s model=%s v%s", status, body.target_day(), name, version)

        issue = pd.Timestamp(body.issue_time_utc).tz_convert("UTC")
        lead = rows["lead_hours"] if "lead_hours" in rows else (rows["ts_utc"] - issue) / pd.Timedelta(hours=1)
        records = [{"target_timestamp": ts.to_pydatetime(), "features": feats,
                    "predicted_load_mw": float(y), "model_name": name, "model_version": version,
                    "issue_time_utc": issue.to_pydatetime(), "lead_hours": int(h),
                    "weather_model": body.weather_model, "mlflow_run_id": meta.get("mlflow_run_id")}
                   for ts, feats, y, h in zip(rows["ts_utc"], logged_features.to_dict("records"), mw, lead)]
        logged, warning = 0, None
        try:
            logged = log_predictions(records)
        except Exception as exc:  # noqa: BLE001
            # As in v1: a logging failure must not turn a good forecast into a 500,
            # but it has to be visible (CloudWatch + the response).
            warning = f"prediction logging failed: {type(exc).__name__}"
            log.exception("prediction logging failed")

        return ForecastResponse(
            status=status, reason=reason, target_day=body.target_day(), issue_time_utc=issue,
            model_name=name, model_version=version, mlflow_run_id=meta.get("mlflow_run_id"),
            weather_model=body.weather_model, logged=logged, warning=warning,
            predictions=[HourForecast(ts_utc=r["target_timestamp"], lead_hours=r["lead_hours"],
                                      load_mw=round(r["predicted_load_mw"], 1)) for r in records])

    @app.post("/actuals", response_model=ActualsResponse)
    def actuals(body: ActualsRequest):
        updated = log_actuals([{"ts_utc": a.ts_utc, "load_mw": a.load_mw} for a in body.actuals])
        return ActualsResponse(received=len(body.actuals), updated=updated)

    return app


def model_inputs(champion: Champion, rows: pd.DataFrame) -> pd.DataFrame:
    """The exact feature matrix the model saw (bundle applied), for the features JSONB column."""
    from shef.features import bundle as bundle_mod

    return bundle_mod.apply(rows, champion.bundle)[champion.features]


def load_model_dir(path: Path) -> tuple[Champion, dict]:
    path = Path(path)
    return Champion.load(path / "champion"), json.loads((path / "model_meta.json").read_text())


def _default_app() -> FastAPI | None:
    model_dir = Path(os.environ.get("MODEL_DIR", "/opt/model"))
    if not (model_dir / "model_meta.json").exists():
        return None  # imported by tests or tooling without a baked model
    champion, meta = load_model_dir(model_dir)
    return create_app(champion, meta)


app = _default_app()

if app is not None:
    from mangum import Mangum

    handler = Mangum(app, lifespan="off")
