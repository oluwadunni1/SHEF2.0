"""
set_role_login.py — give one of the least-privilege database roles a fresh
password and store its pooler URL as a GitHub secret and in .env. The
password is generated here and never printed.

    uv run --env-file .env python scripts/set_role_login.py api_writer API_DATABASE_URL
    uv run --env-file .env python scripts/set_role_login.py monitor_writer MONITOR_DATABASE_URL
    uv run --env-file .env python scripts/set_role_login.py dashboard_reader DASHBOARD_DATABASE_URL

Re-running rotates the password: the old URL stops working at once, so redeploy
anything that holds it (API_DATABASE_URL lives in the Lambda: run deploy-api).
"""
from __future__ import annotations

import argparse
import os
import secrets
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote, urlparse, urlunparse

import psycopg
from psycopg import sql

ROLES = {"api_writer", "monitor_writer", "dashboard_reader"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=sorted(ROLES))
    p.add_argument("secret", help="GitHub secret / .env variable name")
    p.add_argument("--no-github", action="store_true", help="only update .env")
    args = p.parse_args()

    admin = urlparse(os.environ["DATABASE_URL"])
    project_ref = admin.username.split(".", 1)[1]           # pooler users are <role>.<project_ref>
    password = secrets.token_urlsafe(32)
    with psycopg.connect(os.environ["DATABASE_URL"], prepare_threshold=None) as conn:
        conn.execute(sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {}").format(sql.Identifier(args.role), sql.Literal(password)))
    url = urlunparse(admin._replace(
        netloc=f"{args.role}.{project_ref}:{quote(password, safe='')}@{admin.hostname}:{admin.port}"))

    with psycopg.connect(url, prepare_threshold=None, connect_timeout=15) as conn:
        assert conn.execute("SELECT current_user").fetchone()[0] == args.role
    print(f"{args.role}: login works")

    if not args.no_github:
        gh = shutil.which("gh") or r"C:\Program Files\GitHub CLI\gh.exe"
        r = subprocess.run([gh, "secret", "set", args.secret], input=url, text=True, capture_output=True)
        print(f"GitHub secret {args.secret}:", "set" if r.returncode == 0 else r.stderr.strip())
    env = Path(".env")
    lines = [line for line in env.read_text().splitlines() if not line.startswith(f"{args.secret}=")]
    env.write_text("\n".join(lines) + f"\n{args.secret}={url}\n")
    print(f".env: {args.secret} updated")


if __name__ == "__main__":
    main()
