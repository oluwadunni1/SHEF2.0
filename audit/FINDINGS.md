# P0 availability audit — findings

Run date: 2026-10-11. Scripts and raw outputs are in this folder; rerun them to refresh.

## 1. Weather data for training

Source: `weather_coverage.py` → `reports/weather_coverage.csv`, `reports/weather_forecast_skill.csv`.
Four cities pooled (San Antonio, Austin, Round Rock, San Marcos).

| Source | What it is | Temperature | Humidity / precip |
|---|---|---|---|
| Archive (ERA5) | Reanalysis, closest to observed. What v1 trained on. | 2016 → now, complete | 2016 → now, complete |
| Historical Forecast API | First hours of each model run, stitched. Short lead only. | 2018 → now | 2018 → now |
| Previous Runs API, `previous_day1/2` | The same hour as forecast 1 or 2 days earlier. Closest match to day-ahead serving. | **2021-03-24** → now | **2024-01-19** → now |

Previous Runs has small gaps: about 0.5% of 2023 and early January 2024 are missing.

### How far off is a day-ahead forecast?

Forecast vs ERA5, 2021-03 → 2026-10, all four cities:

| Lead | Temp MAE | Temp bias | Temp p95 abs error | Humidity MAE | Precip MAE |
|---|---|---|---|---|---|
| day 1 | 1.32 °C | +0.18 °C | 3.7 °C | 7.7 pts (bias −1.8) | 0.20 mm |
| day 2 | 1.58 °C | +0.61 °C | 4.3 °C | 9.8 pts (bias −5.6) | 0.16 mm |

By month, day-1 temperature MAE ranges from 1.18 °C (November) to 1.46 °C (May). May and June have the largest misses (p95 4.1–4.4 °C).

**Reading:** your instinct holds for temperature. A day-ahead forecast is usually off by about 1.3 °C, and 1 hour in 20 is off by more than 3.7 °C. Humidity is noticeably worse and biased low. Precipitation error looks small only because most hours have no rain.

Open-Meteo's docs say: "`_previous_day1` is the value that was predicted 24 hours before valid time, `_previous_day2` 48 hours before." Our leads are 14–38 h. To stay on the pessimistic side, use **day 1 for target hours with lead ≤ 24 h and day 2 for lead 25–38 h**. The start dates also match the docs: GFS temperature is archived from March 2021, and most other variables and models from January 2024. The default "best match" model blends models, so record which one is used. A single fixed model such as `gfs_seamless` might be safer for training/serving parity. Test both in P3.

### Decision this enables

- Train on **ERA5 for 2016 → now** (long history, all variables).
- Backtest on **2022 → now with `previous_day1` temperature** in place of ERA5. That is the headline MAPE.
- Compare against a model trained only on forecast temperature from 2021-03 onward (P3).
- Humidity and precipitation forecasts only exist from 2024. Keep them only if they earn their place on the 2024+ backtest; otherwise drop them.

## 2. Load data (GridStatus `ercot_load_by_weather_zone`, `south_central`)

Source: `gridstatus_latency.py`.

- **History:** rows exist before 2016-01-01. Sampled days are complete (24 rows, no nulls).
- **Winter Storm Uri (Feb 2021):** load is present, but ERCOT shed load during rotating outages, so measured load is below true demand. Flag those days and exclude them from training or give them less weight.
- **Freshness snapshot:** at 18:56 UTC the newest available hour ended at 18:00 UTC (56 min old).
- **Publish latency (provisional):** the probe ran every 5 minutes from 18:56 to 19:52 UTC, then was stopped early when the machine ran low on memory. In that window, the hour ending 19:00 UTC first appeared between 19:22 and 19:27, a delay of **22–27 min**. The hour ending 18:00 was already visible at 18:56. That's one clean measurement, enough to plan with but not a worst case.
  - `features.yaml` uses **1 h** as a provisional delay, about twice the observed one. Day-ahead targets 00–08 get yesterday's same-hour load.
  - Re-run `gridstatus_latency.py probe --loop 300` over a few days to get a p99 before P2's leakage test treats the number as final.

## 3. P1 data layer (2026-10-11)

Built by `scripts/backfill.py` then `scripts/build_clean.py`. Every stage passes a Pandera gate. The reports are in `data/reports/validation_*.json`.

**One time key.** `ts_utc` is the UTC start of the hour a value describes.
- Load: `ts_utc = interval_start_utc`.
- Open-Meteo: `ts_utc = time − 1h`. The docs label temperature and humidity "Instant" and precipitation "Preceding hour sum". So every weather value on a row describes that row's load interval. v1 joined `time` directly to `interval_start`, which put precipitation one hour early.

| Table | Rows | Range (UTC) | Notes |
|---|---|---|---|
| load | 94,484 | 2016-01-01 00:00 → 2026-10-11 20:00 | 1 missing hour (2016-03-16 04:00 UTC); every other year 100% |
| weather_obs (ERA5, 4 cities) | 377k | 2016 → now | 0 missing hours older than 7 days |
| weather_fc_proxy (`previous_day1/2`) | 202k | temp from 2021-03-24 | day1 99.0%, day2 98.9% complete since start |
| joined | 94,485 | one row per hour | the gap is an explicit row with `is_load_gap` |

**Spot checks:**
- **Load range:** 3,772 to 16,113 MW. The maximum was 2026-08-27 15:00 CDT.
- **Summer daily cycle (Jul–Aug 2025, Central hours):** temperature peaks at 15:00 and load at 17:00. Load bottoms out at 04:00 and temperature at 06:00. Hourly correlation is 0.91. A timezone mix-up would shift these by 5–6 h.
- **Fall-back night:** the two 01:00 Central hours on 2025-11-02 are separate UTC rows, with no duplicate and no gap.
- **Uri (Feb 10–20, 2021; 264 flagged hours):**
  - Load rose with the cold, from about 5.8 GW to 11.7 GW on the 14th.
  - It then dropped to 8.4–8.9 GW on the 16th–17th even though the low was −13 °C. That's the load shedding.
  - The flag covers the whole event. Whether to exclude it or down-weight it is a P3 decision.

**Source glitch caught by the gate.** San Marcos `precipitation_previous_day1` showed 171 mm at 2026-05-20 07:00 UTC, while ERA5 had 8.6 mm and day 2 had 0.0. Precipitation above 100 mm/h is now quarantined: set to NaN and listed in the report.

**Live forecast archive** (`data/raw/weather_fc_live`):
- `scripts/snapshot_forecast.py` stores what we would serve: 3 past days plus 3 forecast days, all four cities.
- The first snapshot was taken 2026-10-11 16:59 CDT. That's a test, not issue-time.
- Two Windows tasks (16:00 and 17:00 WAT, which is 15:00 and 16:00 UTC) run `scripts/snapshot_forecast.cmd`. `--only-at-issue` keeps only the run at 10:xx Central. *(Retired 2026-10-15: `daily.yml` archived its first snapshot on 2026-10-14 and logged the first live forecast, for 2026-10-15. The tasks and the `.cmd` are deleted.)*
- Running the wrapper by hand works. Whether the task-launched run writes its log is **unverified**: check `data/reports/snapshot_forecast.log` after the first scheduled run.

**Raised for P2:**
- At serve time, ERA5 does not yet have the last ~5 days.
- The Open-Meteo archive still returned values up to the current hour. Those recent values are model output, not reanalysis, and they get revised. That's why ranges ending within 10 days are never cached.
- So the observed half of the windows that straddle issue time (`tmax_roll_max_72`, `*_lag_24`) must come from what is available at 10:00 CT, meaning the forecast API's `past_days`. Training must replay the same source, not ERA5.

## 4. P3 model selection (2026-10-12)

Notebooks `01`–`04`, all runs in MLflow (`shef2-day-ahead` on DagsHub). Validation: 3 yearly folds, Aug 2022 → Aug 2025, scored on archived forecast weather.

| | MAPE | Peak-MAPE |
|---|---|---|
| Floor: freshest legal same-hour load | 8.04% | 5.70% |
| **Champion: LightGBM + trend target, default parameters, ERA5 training** | **4.18%** | **2.99%** |

Scores cover hours that have an archived forecast; Uri is excluded.
- Tuning (E8) and ensembling (E7) gained ≤ 0.06 points, which is within noise. The LSTM (E6) scored 5.46% / 6.10% and didn't help the blend.

**The archived forecast has a hole: Dec 31 2023 → Jan 18 2024** (506 validation hours, no `temp_c` in Previous Runs).
- A model trained on ERA5 has never seen missing temperature. It output a flat ~10.8 GW: 30–60% too high on mild days, 23% too low in the Jan 15–17 cold snap.
- **P4/P5 requirement:** forecast temperature must be complete for all target hours, or no model prediction is made. In that case the pipeline falls back to the baseline and alerts.

**Weather model:** `gfs_seamless` and `best_match` give identical scores. In the archive, `best_match` temperature at these sites is GFS. **Serve with `models=gfs_seamless`** so live inputs match training, and archive it in the daily snapshot.

**Holdout (Aug 2025 → Oct 2026, scored once):** champion **3.79% MAPE / 3.38% Peak-MAPE**, vs the baseline's 7.62% / 3.79%; RMSE 482 vs 1,013 MW, bias −0.17%. The production model is retrained on 2016-01-12 → 2026-10-09 and registered as **`shef2-day-ahead` v1** (MLflow pyfunc; bundle, trend line and forecast gate included).

## 5. P4 serving (2026-10-13)

`shef2-day-ahead` v1 is served from AWS Lambda in eu-west-1, the same region as the Supabase database, as a container image with the model baked in.

**Access:**
- The Function URL uses **AWS_IAM** auth.
- GitHub Actions assumes `shef2-github-forecast` (invoke only) or `shef2-github-deploy` through OIDC.
- The trust is pinned to GitHub's immutable subject, `repo:oluwadunni1@171298512/SHEF2.0@1415648897:ref:refs/heads/main`. The first deploy failed against a name-only subject, which is why the ids are pinned.

**Logging:** v1's `predictions` table was cleared (75 v1 rows) and extended with `issue_time_utc`, `lead_hours`, `weather_model` and `mlflow_run_id`.

**Verified on 2026-10-13:**
- An unsigned call gets `403`; a signed `/health` reports v1, run `7b8094a8…`.
- The first cold start took 18.6 s, so the timeout was raised to 60 s.
- An end-to-end `forecast` workflow run for 2026-10-13 logged 24 rows with every column filled.
  - Re-running it logged 0 rows, so logging is idempotent.
  - Lambda vs a local `Champion` on the logged inputs: max difference 4.9e-11 MW.
  - The two hours already realised were 3.1% and 2.6% off.
- Those rows were a smoke test made 17 h after issue time, with a later weather run than 10:00 would have seen. They were deleted, so live MAPE starts with the first scheduled run.

**Deploy pitfalls hit:**
- LightGBM needs `libgomp`, which the Lambda base image lacks.
- Lambda needs an ECR repository policy to pull the image.

## 6. P5 orchestration (2026-10-13)

`.github/workflows/daily.yml` replaces `forecast_snapshot.yml` and the manual `forecast.yml`. One job runs at 10:xx Central:
1. Snapshot both weather models to DVC/R2.
2. Forecast tomorrow through the API **from the archived `gfs_seamless` rows**, so what was served is exactly what was stored.
3. Send the last 10 days of realised load to `/actuals`.

**Scheduling:**
- GitHub dropped the whole scheduled day on 2026-10-12. So the job has three slots at :07, off the top of the hour: 15:07 and 16:07 UTC, whichever is 10:xx Central, plus a 17:37 UTC catch-up.
- The catch-up snapshots only if today's snapshot is missing (with a warning that it is late), then re-forecasts. A re-forecast is idempotent: 0 rows if the forecast is already logged.

**Credentials:** the R2 keys (stored as `AWS_*`) reach only the DVC steps. The OIDC forecast role is assumed after the last DVC step.

**Alerting:** a fallback forecast exits 2, which fails the run, and GitHub sends a failure email.
- These emails go to whoever last edited the cron lines.
- A late snapshot shows as a warning annotation.

**Verified:** a dry run passed. It pulled from R2, snapshotted both models with 100% of temperatures present, and made a signed `/health` call with the forecast role. The first scheduled run is 2026-10-13 15:07 UTC, for 2026-10-14.

## 7. P6 monitoring and promotion (2026-10-13)

**Storage:**
- Everything lives in a `monitoring` schema, which Supabase's REST API does not expose.
- Two scoped logins:
  - `dashboard_reader` can read the views only;
  - `monitor_writer` can write the three monitoring tables.
- Neither can touch `predictions`. Both were tested with their own connections.

**Backtest as reference:**
- The holdout model from MLflow was re-run on the holdout feature rows and reproduced the logged OOF with a 0.0 MW difference.
- Loaded as `backtest-1` (10,224 rows).
- `monitoring.v_summary` returns 3.79% / 3.38% / 482 MW / −0.17%, identical to P3, which ties the SQL definitions to `metrics.py` (also a test).

**Drift: textbook PSI does not work here.**
- With 0.1 / 0.25 thresholds, every 14-day window "drifts", including in-sample ones (PSI 1–7 on `tmax`).
- Thresholds are therefore calibrated per feature on 1,082 historical windows of archived-forecast rows: watch above p95, drift above p99.
- `tmax`, `tmin`, `temp_spike_vs_mean` and `last_known_roll_mean_168` are dropped. With about 14 distinct values per window their PSI saturates near the 10-bin ceiling (calibrated drift thresholds of 7.2–8.4), so they could never alert.
- What's left (`temp_c`, `cdd_x_hour`, `load_same_hour_latest`, `load_lag_168`):
  - stays quiet on Jul and Feb 2026;
  - flags °F temperatures and a frozen load feed (PSI 8.3);
  - flags the two weeks to 2026-10-09 on load. That's genuine growth above previous Octobers, consistent with the early-September under-forecasts.

**Promotion:**
- Rolling origin: 28 labelled days, challenger trained through window start − 8 days.
- Rule: at least 0.10 points of MAPE gain and a Peak-MAPE no worse.
- The first real verdict needs labels through 2026-11-06. The 2026-11-01 run will report "insufficient"; December is the first real test.
- Rehearsal (champion refit through 2026-07-31, challenger through 08-26, window Sep 3–30):

  | | MAPE | Peak-MAPE |
  |---|---|---|
  | Champion | 3.27% | 2.93% |
  | Challenger | 3.23% | 3.00% |

  The verdict is **keep**. A month of extra data is worth about nothing, and the margin stops churn.
- v1 now carries the `@champion` alias, and `deploy_api` deploys whatever the alias points to.

## 8. P7 hour-ahead: no-go (2026-10-13)

**Decision: not built.** Measured, not assumed:
- GridStatus's Free plan allows **250 API requests a month** (`get_api_usage()`: 500k rows; 30 per minute).
- Hourly hour-ahead serving needs about 720 requests a month.
- The latency probe that should have gated it needs about 8,600 at 5-minute resolution.
- Day-ahead uses about 35 (one load fetch per daily run plus the monthly refresh), so it stays well inside the quota.

**What would reopen it:** a source without that cap. ERCOT's own public API (apiexplorer.ercot.com, free with registration) publishes the same weather-zone load. Even then, the latency p99 has to come first. A model that relies on the last hour's load is only as good as how reliably that hour is published.

**What was built instead:** the ERCOT benchmark (section 9). The quota is now watched as well:
- `monitoring.api_usage` is logged daily and shown on the dashboard;
- optional calls are skipped once fewer than 60 requests remain.

## 9. Benchmark against ERCOT's own forecast (2026-10-13)

**The comparison:**
- ERCOT's South Central day-ahead forecast, taken point in time: per target hour, the latest publish at or before our 10:00 CT issue time. That was the 09:30 CT publish for all 2,880 hours.
- Stored in `monitoring.ercot_forecast`, compared in `monitoring.v_benchmark*`.
- It uses one GridStatus request per day, so the quota decided the scope: 120 days now (Jun 12 → Oct 9 2026, the summer peak season). The rest of the holdout follows after the 1 Nov reset (`scripts/ercot_benchmark.py` resumes).
- Tomorrow's ERCOT forecast is also stored each day, so the comparison continues live.

| Jun 12 → Oct 9 2026, matched hours | MAPE | Peak-MAPE | RMSE | Bias |
|---|---|---|---|---|
| ERCOT | **2.79%** | **2.52%** | **437 MW** | −0.79% |
| Ours (holdout backtest) | 3.65% | 3.38% overall holdout; 3.89% on these hours | 551 MW | −1.97% |

**ERCOT is better overall.** Ours is better on 39% of hours, and better in July (2.85% vs 3.06%).

**The gap is one specific failure, not general weakness:**

| | Jun | Jul | Aug | Sep | Oct |
|---|---|---|---|---|---|
| Our bias | −0.9% | −1.1% | −1.7% | **−4.3%** | +0.9% |
| ERCOT's bias | −0.8% | −1.2% | −0.2% | −1.3% | +0.6% |

- Our under-forecasting grows as South Central sets demand records (16.1 GW on Aug 27).
- On the top 5% of load hours, our bias is −3.8% against ERCOT's −1.6%.
- Trees can't extrapolate beyond the levels they've seen, and the linear trend rises more slowly than 2026 demand did. ERCOT also has more weather inputs and real-time telemetry, and adapts within days.
- The same issue appeared in the RMSE analysis (early-September under-forecasts) and in the drift monitor (load above past Octobers).

**Next model work, in order of evidence:**
1. Make the model follow the current demand level: weight recent data more, or use the ratio-to-level target from P3 E2, evaluated on summer 2026.
2. Use ERCOT's 09:30 forecast as a feature. It's legal at 10:00, but the quota limits training history: about 3,650 requests for 10 years.
3. Prediction intervals.

None of this was tuned on the holdout. Any challenger goes through the monthly promotion rule.

## 10. Hardening before the write-up (2026-10-13)

- **The API no longer holds the postgres URL.** `api_writer` (`sql/006`) can insert forecasts and fill `actual_load_mw`, nothing else. A probe run as the role, then rolled back, gave:
  - allowed: the insert, an idempotent re-insert (0 rows), and the actuals update;
  - denied: inserting a `backtest-*` row (RLS), reading features, DELETE, changing a forecast, and reading the monitoring schema;
  - backtest rows are invisible to it (count 0).

  All four components now use their own role.
- **Latency is measured by every run.** `forecast_day.py` records the newest hour each load fetch returns, and `monitor.py` stores it in `monitoring.load_latency`. One fetch at T, with the newest hour ending at E, brackets the latency: the hour ending E took at most T − E, and the next hour took more than T − E − 1 h. This builds up the multi-day evidence the single P0 sample lacked, at zero extra GridStatus requests. The dashboard flags any observation that would break the 1 h cutoff.
- **Failures open an issue.** In a scheduled `daily` run, any of these opens an `ops-alert` GitHub issue (one per day; later slots comment on it):
  - a failed or fallback forecast;
  - a failed dashboard;
  - a monitoring step that failed under `continue-on-error`.
- **README:** model card, a limitations section, and a least-privilege role table.

## 11. GitHub cron lost the first live days, so the clock moved to EventBridge (2026-10-14)

- **2026-10-12:** no scheduled run at all.
- **2026-10-13:** all three slots ran, but late:

  | slot (UTC) | started (UTC) | Central time |
  |---|---|---|
  | 15:07 | 20:36 | 15:36 |
  | 16:07 | 21:08 | 16:08 |
  | 17:37 | 21:56 | 16:56 |

  The slot logic only accepted 10:00–12:59 Central, so each run chose `skip` and finished **green**. Nothing was forecast for 2026-10-14, and no alert fired, because nothing failed.

The lesson is twofold. The trigger is the unreliable part (GitHub runs `workflow_dispatch` immediately; the cron trigger is best-effort). And "the job passed" is the wrong thing to monitor.

Changes:
- **Clock: EventBridge Scheduler** (`infra/scheduler.yaml`).
  - Runs at 10:05 America/Chicago, so the time zone handles DST and the two UTC slots are gone.
  - Delivery is retried, and a Lambda dispatches `daily.yml`.
  - The Lambda's own retries end in an SNS topic.
  - Its only credential is a fine-grained GitHub token in an SSM SecureString, readable by that one function.
- **Output check: 13:00 Central.** The same Lambda calls the API's new `GET /logged?target_day=` (SigV4-signed; `api_writer` can count live hours, and RLS hides backtest rows from it). If tomorrow's day is incomplete, it emails and dispatches a catch-up.
- **The catch-up window is widened to midnight Central.** A late run is still a valid forecast: load is cut at the 10:00 issue time; only the weather is a later run, and the run warns about it.
- **Duplicate triggers are free.** A run first asks the read-only role whether tomorrow is already logged and exits before any GridStatus request. The monitor skips ERCOT's forecast when it is already stored. Without this, the extra triggers could have exhausted the 250-request plan.
- GitHub cron is kept as a backup (16:37 and 22:37 UTC).
- **Deployed 2026-10-15 and tested end to end from CloudShell.** Two first-run failures were fixed:
  - the API answered 403, because a Function URL with IAM auth needs `lambda:InvokeFunction` as well as `lambda:InvokeFunctionUrl`;
  - GitHub answered 401, because the token had been pasted twice into SSM (186 characters, not 93).

  The check also now alerts before it dispatches, so a failed catch-up can't be silent. A manual `{"action":"dispatch"}` started `daily` within seconds, and it correctly chose `skip` (06:50 Central, before the issue time).
- CI now runs `cfn-lint` on `infra/`. It caught an unquoted `: ` in `api.yaml` before it reached a deploy.

## 12. Still open

- [ ] GridStatus publish latency p99: accumulating daily in `monitoring.load_latency` (section 10). Needs a few weeks before it can be called a p99.
- [ ] 7 consecutive unattended `daily` runs (P5 definition of done). The first scheduled run is 2026-10-13 15:07 UTC.
- [x] ~~A day when GitHub drops every cron slot opens no issue.~~ The 13:00 output check alerts by email (section 11).
- [ ] Prediction intervals; a level-adaptive challenger for record-demand periods (README limitations).
- [x] Confirm Open-Meteo `previous_day1` lead definition: predicted 24 h before valid time
- [x] Weather model: `gfs_seamless` (identical to `best_match` in the archive; pinned for live parity)
