-- Live forecasts with or without an actual yet (tomorrow's forecast, the last
-- 14 days chart). served_first as in v_scored.
CREATE OR REPLACE VIEW monitoring.v_live AS
SELECT p.target_timestamp,
       (p.target_timestamp AT TIME ZONE 'America/Chicago')::date AS target_day,
       p.issue_time_utc,
       p.lead_hours,
       p.model_name,
       p.model_version,
       p.mlflow_run_id,
       CASE WHEN p.model_name LIKE 'baseline%' THEN 'fallback' ELSE 'model' END AS served_as,
       row_number() OVER (PARTITION BY p.target_timestamp ORDER BY p.predicted_at) = 1 AS served_first,
       p.predicted_load_mw,
       p.actual_load_mw,
       p.predicted_at
FROM predictions p
WHERE p.model_version NOT LIKE 'backtest%';

GRANT SELECT ON monitoring.v_live TO dashboard_reader, monitor_writer;
