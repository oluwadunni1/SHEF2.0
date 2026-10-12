-- v2 takes over v1's predictions table: same columns and constraints, v1 rows
-- cleared, plus the four columns a day-ahead forecast needs.
TRUNCATE predictions RESTART IDENTITY;

ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS issue_time_utc timestamptz,   -- 10:00 Central on D-1
    ADD COLUMN IF NOT EXISTS lead_hours     smallint,      -- hours from issue to target start
    ADD COLUMN IF NOT EXISTS weather_model  text,          -- Open-Meteo model served (gfs_seamless)
    ADD COLUMN IF NOT EXISTS mlflow_run_id  text;          -- run that produced the model

CREATE INDEX IF NOT EXISTS idx_predictions_issue ON predictions (issue_time_utc);
