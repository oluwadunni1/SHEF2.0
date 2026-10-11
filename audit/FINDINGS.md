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

## 3. Still open

- [ ] GridStatus publish latency p99 over several days (provisional 1 h from one sample)
- [ ] Hour-ahead go / no-go: a ~25 min delay makes it technically possible, but one sample is not enough. Parked as optional P7.
- [x] Confirm Open-Meteo `previous_day1` lead definition: predicted 24 h before valid time
- [ ] Choose the weather model: `best_match` vs a fixed model such as `gfs_seamless` (P3)
