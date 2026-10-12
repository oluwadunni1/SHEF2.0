"""
db_migrate.py — apply sql/*.sql to the Supabase database, each file once.

Applied files are recorded in schema_migrations, so re-running is a no-op.

    uv run --env-file .env --group serve python scripts/db_migrate.py
"""
from __future__ import annotations

import os
from pathlib import Path

import psycopg

SQL = Path(__file__).resolve().parents[1] / "sql"


def main() -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], prepare_threshold=None, connect_timeout=15) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations "
                     "(name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())")
        conn.execute("ALTER TABLE schema_migrations ENABLE ROW LEVEL SECURITY")
        done = {r[0] for r in conn.execute("SELECT name FROM schema_migrations")}
        for f in sorted(SQL.glob("*.sql")):
            if f.name in done:
                print(f"{f.name}: already applied")
                continue
            before = conn.execute("SELECT count(*) FROM predictions").fetchone()[0]
            with conn.transaction():
                conn.execute(f.read_text())
                conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (f.name,))
            after = conn.execute("SELECT count(*) FROM predictions").fetchone()[0]
            print(f"{f.name}: applied (predictions rows {before} -> {after})")


if __name__ == "__main__":
    main()
