-- ERCOT's own day-ahead forecast for South Central, point in time: for each
-- target hour, the latest ERCOT publish at or before our 10:00 Central issue
-- time on D-1 (in practice the 09:30 CT publish). Same rules as our model:
-- nothing published after issue time is used.
CREATE TABLE IF NOT EXISTS monitoring.ercot_forecast (
    target_timestamp  timestamptz PRIMARY KEY,   -- UTC start of the target hour
    issue_time_utc    timestamptz NOT NULL,      -- our 10:00 CT issue time for that day
    publish_time_utc  timestamptz NOT NULL,      -- ERCOT's publish used (<= issue_time_utc)
    forecast_mw       float8      NOT NULL,
    fetched_at        timestamptz NOT NULL DEFAULT now()
);

-- GridStatus Free plan: 250 requests / 500k rows a month. Logged daily so the
-- dashboard shows the headroom and jobs can refuse optional calls near the cap.
CREATE TABLE IF NOT EXISTS monitoring.api_usage (
    checked_at        timestamptz PRIMARY KEY DEFAULT now(),
    period_start      timestamptz NOT NULL,
    requests_used     int         NOT NULL,
    requests_limit    int         NOT NULL,
    rows_used         bigint      NOT NULL,
    rows_limit        bigint      NOT NULL
);

ALTER TABLE monitoring.ercot_forecast ENABLE ROW LEVEL SECURITY;
ALTER TABLE monitoring.api_usage ENABLE ROW LEVEL SECURITY;
GRANT SELECT ON monitoring.ercot_forecast, monitoring.api_usage TO dashboard_reader, monitor_writer;
GRANT INSERT, UPDATE ON monitoring.ercot_forecast, monitoring.api_usage TO monitor_writer;
CREATE POLICY dashboard_read ON monitoring.ercot_forecast FOR SELECT TO dashboard_reader USING (true);
CREATE POLICY dashboard_read ON monitoring.api_usage FOR SELECT TO dashboard_reader USING (true);
CREATE POLICY monitor_all ON monitoring.ercot_forecast FOR ALL TO monitor_writer USING (true) WITH CHECK (true);
CREATE POLICY monitor_all ON monitoring.api_usage FOR ALL TO monitor_writer USING (true) WITH CHECK (true);

-- Matched hours only: our served forecast, ERCOT's, and the actual.
CREATE OR REPLACE VIEW monitoring.v_benchmark AS
SELECT s.kind, s.target_timestamp, s.target_day, s.lead_hours,
       s.actual_load_mw,
       s.predicted_load_mw                                              AS ours_mw,
       e.forecast_mw                                                    AS ercot_mw,
       abs(s.predicted_load_mw - s.actual_load_mw) / s.actual_load_mw * 100 AS ours_ape,
       abs(e.forecast_mw - s.actual_load_mw) / s.actual_load_mw * 100       AS ercot_ape,
       s.predicted_load_mw - s.actual_load_mw                           AS ours_err,
       e.forecast_mw - s.actual_load_mw                                 AS ercot_err
FROM monitoring.v_scored s
JOIN monitoring.ercot_forecast e USING (target_timestamp)
WHERE s.served_first;

CREATE OR REPLACE VIEW monitoring.v_benchmark_summary AS
WITH q AS (
    SELECT kind, percentile_cont(0.95) WITHIN GROUP (ORDER BY actual_load_mw) AS peak_q
    FROM monitoring.v_benchmark GROUP BY kind
)
SELECT b.kind,
       count(*)                                                  AS n_hours,
       count(DISTINCT b.target_day)                              AS n_days,
       min(b.target_day)                                         AS first_day,
       max(b.target_day)                                         AS last_day,
       avg(b.ours_ape)                                           AS ours_mape,
       avg(b.ercot_ape)                                          AS ercot_mape,
       avg(b.ours_ape) FILTER (WHERE b.actual_load_mw >= q.peak_q)  AS ours_peak_mape,
       avg(b.ercot_ape) FILTER (WHERE b.actual_load_mw >= q.peak_q) AS ercot_peak_mape,
       sqrt(avg(b.ours_err * b.ours_err))                        AS ours_rmse,
       sqrt(avg(b.ercot_err * b.ercot_err))                      AS ercot_rmse,
       avg(b.ours_err) / avg(b.actual_load_mw) * 100             AS ours_bias_pct,
       avg(b.ercot_err) / avg(b.actual_load_mw) * 100            AS ercot_bias_pct,
       avg((b.ours_ape < b.ercot_ape)::int) * 100                AS ours_wins_pct
FROM monitoring.v_benchmark b JOIN q USING (kind)
GROUP BY b.kind;

CREATE OR REPLACE VIEW monitoring.v_benchmark_daily AS
SELECT kind, target_day, count(*) AS n_hours, avg(ours_ape) AS ours_mape, avg(ercot_ape) AS ercot_mape
FROM monitoring.v_benchmark GROUP BY kind, target_day;

GRANT SELECT ON monitoring.v_benchmark, monitoring.v_benchmark_summary, monitoring.v_benchmark_daily
    TO dashboard_reader, monitor_writer;
