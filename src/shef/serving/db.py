"""
Supabase logging (v1 src/api/db.py, extended with v2's columns).

One connection per call: the API is invoked about once a day, and DATABASE_URL
is Supabase's transaction pooler, which hands out short-lived connections.
prepare_threshold=None because a transaction pooler cannot keep prepared
statements across transactions.
"""
from __future__ import annotations

import math
import os

INSERT_SQL = """
INSERT INTO predictions
    (target_timestamp, features, predicted_load_mw, model_name, model_version,
     issue_time_utc, lead_hours, weather_model, mlflow_run_id)
VALUES
    (%(target_timestamp)s, %(features)s, %(predicted_load_mw)s, %(model_name)s, %(model_version)s,
     %(issue_time_utc)s, %(lead_hours)s, %(weather_model)s, %(mlflow_run_id)s)
ON CONFLICT (target_timestamp, model_version) DO NOTHING
RETURNING id
"""

ACTUALS_SQL = """
UPDATE predictions SET actual_load_mw = %(load_mw)s
WHERE target_timestamp = %(ts_utc)s AND actual_load_mw IS NULL
"""


def json_safe(features: dict) -> dict:
    """NaN is not valid JSON(B); store it as null."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in features.items()}


def _connect():
    import psycopg

    return psycopg.connect(os.environ["DATABASE_URL"], prepare_threshold=None, connect_timeout=10)


def log_predictions(records: list[dict]) -> int:
    """records: one dict per hour with the INSERT_SQL keys. All or nothing; returns rows inserted
    (0 on a re-run: the unique key makes logging idempotent)."""
    from psycopg.types.json import Jsonb

    with _connect() as conn, conn.cursor() as cur:
        n = 0
        for r in records:
            cur.execute(INSERT_SQL, {**r, "features": Jsonb(json_safe(r["features"]))})
            n += cur.fetchone() is not None
        return n


def log_actuals(actuals: list[dict]) -> int:
    """Fill actual_load_mw where still NULL; returns rows updated."""
    with _connect() as conn, conn.cursor() as cur:
        n = 0
        for a in actuals:
            cur.execute(ACTUALS_SQL, a)
            n += cur.rowcount
        return n
