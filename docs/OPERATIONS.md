# Operations

How SHEF runs every day, how it fails, and how to deploy it. Back to the [README](../README.md).

## The daily loop

| When (America/Chicago) | What |
|---|---|
| **10:05** | EventBridge Scheduler dispatches the `daily` workflow (retries on delivery failure). |
| 10:06 | `daily` snapshots the Open-Meteo forecast to DVC, forecasts tomorrow through the API from that snapshot, fills in actual load for earlier days, runs monitoring, and rebuilds the dashboard. |
| **13:00** | EventBridge asks the API (`GET /logged`) whether all of tomorrow's 23–25 hours exist. If not: an email alert first, then a catch-up dispatch. |
| 11:37, 17:37 CDT | GitHub cron backups. They find tomorrow already logged and exit. |

Defined in [`.github/workflows/daily.yml`](../.github/workflows/daily.yml) and [`infra/scheduler.yaml`](../infra/scheduler.yaml).

**Run modes**, decided at run time on the Central clock:

| Mode | When | Does |
|---|---|---|
| `issue` | 10:xx | the normal forecast |
| `catchup` | 11:00–23:59 | still forecasts; load is still cut at 10:00, only the weather is later |
| `skip` | before 10:00, or tomorrow already logged | nothing; checked first via a read-only role, so duplicate triggers cost no API quota |
| `dry` | manual | snapshot without storing, signed `/health`, log nothing |

## Why EventBridge, not GitHub cron

GitHub cron is best-effort. In the first live days it dropped 2026-10-12 entirely and ran 2026-10-13's slots 5.5–6.5 hours late. Every run was green and nothing was forecast, because late runs correctly skipped. Two lessons:

1. **The clock moved** to EventBridge Scheduler: time-zone aware (no DST arithmetic), with retries and a dead-letter email.
2. **Check the output, not the job.** The 13:00 check asks the API whether the forecast exists, which catches every failure mode upstream of it.

Full write-up: [FINDINGS §11](../audit/FINDINGS.md).

## Alerts

| Failure | Signal |
|---|---|
| Trigger can't reach GitHub | Lambda retries twice, then SNS email |
| Forecast missing at 13:00 | SNS email, then an automatic catch-up run |
| Run fails, serves the baseline fallback, or monitoring fails quietly | GitHub issue labelled `ops-alert` (one per day; later failures comment on it) |
| Dashboard build fails | same `ops-alert` issue |

## Security

- **No long-lived cloud keys.** GitHub Actions assumes AWS roles through OIDC, with trust pinned to the repository's immutable owner and repo ids. Deploying and forecasting use separate roles. R2 keys (for DVC) are only given to DVC steps and never share an environment with AWS credentials.
- **Signed API.** The Lambda Function URL uses `AWS_IAM` auth: unsigned requests get `403`. The deploy workflow checks both cases after every deploy.
- **Least privilege in Postgres.** Each component logs in as its own role. Every role was probed, inside a rolled-back transaction, for what it can and cannot do:

  | Role | Used by | Can | Cannot |
  |---|---|---|---|
  | `api_writer` | Lambda | insert forecasts, fill `actual_load_mw` | read features, delete, change a forecast, touch backtest rows (row-level security) |
  | `monitor_writer` | daily monitoring | write monitoring tables, read monitoring views | read or write `predictions` |
  | `dashboard_reader` | dashboard build, run-mode check | read monitoring views | write anything, read `predictions` directly |

  The `postgres` login is used only for migrations, from a laptop.

## Serving

- **One model version per image.** `deploy-api` pulls the `@champion` version from MLflow, bakes it into the container, tests it under the Lambda runtime emulator, deploys, then smoke-tests `/health` and `/logged` with signed requests.
- **Validated boundary.** The API accepts only a full Central day (23–25 rows, the 10:00 issue time, the archived weather model, physical bounds, no unknown fields) and derives fitted features itself from the bundle shipped with the model.
- **Everything logged.** Each hour is stored with its exact model inputs, issue time, lead time, weather model and MLflow run id, so any forecast can be replayed.
- **Forecast gate.** If any target hour lacks forecast temperature, the baseline is served and labelled `status: fallback`, and the run fails loudly. (A model trained on observed weather has never seen a missing value; in a 2024 archive gap it was 30–60% wrong.)

## Deploying

1. Run `infra/bootstrap.yaml` once in AWS CloudShell (OIDC provider, roles, ECR).
2. Apply `sql/` with `scripts/db_migrate.py`, and create the three role logins with `scripts/set_role_login.py` (it stores each as a GitHub secret).
3. Run the `deploy-api` workflow (input: a model version, or `champion`).
4. In CloudShell: store a fine-grained GitHub token (this repo only, Actions read/write) as the SSM SecureString `/shef2/github-dispatch-token`, deploy `infra/scheduler.yaml` as stack `shef2-scheduler`, and confirm the SNS subscription email.
5. The `daily` workflow then runs at 10:05 Central.
