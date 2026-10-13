-- The forecast API (Lambda) logs with its own login instead of the postgres
-- connection string. It can do exactly what app.py does:
--   /forecast  INSERT ... ON CONFLICT DO NOTHING RETURNING id
--   /actuals   UPDATE actual_load_mw WHERE target_timestamp = .. AND actual_load_mw IS NULL
-- and nothing else: no DELETE, no other columns, no other tables, and the
-- backtest rows are invisible to it (RLS), so it cannot alter them.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'api_writer') THEN
        CREATE ROLE api_writer NOLOGIN;
    END IF;
END $$;

GRANT USAGE ON SCHEMA public TO api_writer;
GRANT INSERT (target_timestamp, features, predicted_load_mw, model_name, model_version,
              issue_time_utc, lead_hours, weather_model, mlflow_run_id)
    ON predictions TO api_writer;
GRANT SELECT (id, target_timestamp, model_version, actual_load_mw) ON predictions TO api_writer;
GRANT UPDATE (actual_load_mw) ON predictions TO api_writer;
GRANT USAGE ON SEQUENCE predictions_id_seq TO api_writer;

CREATE POLICY api_insert ON predictions FOR INSERT TO api_writer
    WITH CHECK (model_version NOT LIKE 'backtest%');
CREATE POLICY api_select ON predictions FOR SELECT TO api_writer
    USING (model_version NOT LIKE 'backtest%');
CREATE POLICY api_update ON predictions FOR UPDATE TO api_writer
    USING (model_version NOT LIKE 'backtest%') WITH CHECK (model_version NOT LIKE 'backtest%');
