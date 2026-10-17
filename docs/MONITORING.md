# Monitoring and promotion

How SHEF knows whether it is still working, and how a better model gets in. Back to the [README](../README.md).

**Live dashboard: https://oluwadunni1.github.io/SHEF2.0/**: a static page rebuilt after every daily run, from a read-only database role. No credentials reach the browser.

## Accuracy

- Live 30-day MAPE, Peak-MAPE, RMSE and bias, next to the holdout backtest.
- The backtest rows sit in the same table as live rows (labelled `backtest-1`), so one set of SQL views scores both. The views reproduce the notebook results exactly (3.79% / 3.38% / 482 MW), which ties the SQL definitions to `metrics.py`; a test checks it.
- **ERCOT benchmark.** ERCOT's own forecast, taken as known at our issue time (its 09:30 publish), is stored daily and compared hour by hour.

## Served weather

Each archived `gfs_seamless` forecast is scored once ERA5 observations are final (about 6 days later):
- **against what was observed**: how wrong was the weather the model was given?
- **against the archived forecast used in training**: does training's stand-in match what serving actually sees?

Large forecast misses are usually weather misses; this separates the two.

## Input drift

PSI on hourly inputs against the same calendar month in the last 3 training years.
- **Textbook thresholds don't work here.** With 0.1 / 0.25, every 14-day window "drifts", including windows from the training data.
- Thresholds are therefore **calibrated** per feature on 1,082 historical windows: watch above the 95th percentile, drift above the 99th.
- Daily-level features are excluded because their PSI saturates.
- It catches °F temperatures and a frozen load feed (PSI 8.3), and it flagged real early-October 2026 load growth above past Octobers.

## Load publishing latency

The features assume load is published within 1 hour. Every daily fetch records the newest hour GridStatus returned, which brackets the true latency for free (no extra requests). It tests that assumption every day instead of trusting one early sample.

## Promotion

A monthly job (`promote.yml`):
1. refreshes the data through the DVC pipeline (all data gates apply);
2. retrains the champion's recipe as a challenger;
3. scores both on the latest 28 labelled days that neither has seen (rolling origin, 8-day gap);
4. promotes only if **MAPE improves by ≥ 0.10 points and Peak-MAPE is no worse**;
5. on promotion, registers vN, moves the MLflow `@champion` alias and redeploys the API.

Every evaluation, promoted or not, is logged to MLflow.

**Rehearsal (Sep 2026 window):** champion 3.27% / 2.93%, challenger 3.23% / 3.00%. Verdict: keep. A month of extra data was worth 0.04 points with a worse peak, and the margin exists to stop that kind of churn.

The first real verdict needs 28 labelled live days, so December 2026 is the first real test.
