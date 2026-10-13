-- The daily monitoring step writes weather checks, drift and promotion
-- results. It gets its own login that can do exactly that: read the
-- monitoring views (incl. live model inputs for drift) and write the three
-- monitoring tables. It cannot touch predictions.

CREATE OR REPLACE VIEW monitoring.v_live_inputs AS
SELECT target_timestamp AS ts_utc, model_name, model_version, features
FROM predictions
WHERE model_version NOT LIKE 'backtest%';

GRANT SELECT ON monitoring.v_live_inputs TO dashboard_reader;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'monitor_writer') THEN
        CREATE ROLE monitor_writer NOLOGIN;
    END IF;
END $$;
GRANT USAGE ON SCHEMA monitoring TO monitor_writer;
GRANT SELECT ON ALL TABLES IN SCHEMA monitoring TO monitor_writer;
GRANT INSERT, UPDATE ON monitoring.weather_checks, monitoring.drift, monitoring.promotions TO monitor_writer;
GRANT USAGE ON SEQUENCE monitoring.promotions_id_seq TO monitor_writer;
CREATE POLICY monitor_all ON monitoring.weather_checks FOR ALL TO monitor_writer USING (true) WITH CHECK (true);
CREATE POLICY monitor_all ON monitoring.drift FOR ALL TO monitor_writer USING (true) WITH CHECK (true);
CREATE POLICY monitor_all ON monitoring.promotions FOR ALL TO monitor_writer USING (true) WITH CHECK (true);
