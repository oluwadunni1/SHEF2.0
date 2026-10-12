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
- Two Windows tasks (16:00 and 17:00 WAT, which is 15:00 and 16:00 UTC) run `scripts/snapshot_forecast.cmd`. `--only-at-issue` keeps only the run at 10:xx Central.
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

## 5. Still open

- [ ] GridStatus publish latency p99 over several days (provisional 1 h from one sample)
- [ ] Hour-ahead go / no-go: a ~25 min delay makes it technically possible, but one sample is not enough. Parked as optional P7.
- [x] Confirm Open-Meteo `previous_day1` lead definition: predicted 24 h before valid time
- [x] Weather model: `gfs_seamless` (identical to `best_match` in the archive; pinned for live parity)
