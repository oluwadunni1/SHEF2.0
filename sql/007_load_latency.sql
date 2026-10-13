-- GridStatus publishing latency, measured for free: every daily load fetch
-- already returns the newest hour visible at that moment. One fetch at time T
-- whose newest hourly interval ends at E brackets the latency:
--   the hour ending E      was published within  T - E           (upper bound)
--   the hour ending E + 1h was NOT published in  T - (E + 1h)    (lower bound)
-- The feature contract assumes 1 h (features.yaml); this table tests that
-- assumption with every run instead of the single P0 sample.

CREATE TABLE IF NOT EXISTS monitoring.load_latency (
    observed_at         timestamptz PRIMARY KEY,   -- when the fetch returned
    newest_interval_end timestamptz NOT NULL,
    run_mode            text NOT NULL              -- issue | catchup | manual
);
ALTER TABLE monitoring.load_latency ENABLE ROW LEVEL SECURITY;

CREATE OR REPLACE VIEW monitoring.v_load_latency AS
SELECT observed_at, newest_interval_end, run_mode,
       extract(epoch FROM observed_at - newest_interval_end) / 60                         AS published_within_min,
       extract(epoch FROM observed_at - newest_interval_end - interval '1 hour') / 60     AS next_hour_late_by_min
FROM monitoring.load_latency;

GRANT SELECT, INSERT ON monitoring.load_latency TO monitor_writer;
CREATE POLICY monitor_all ON monitoring.load_latency FOR ALL TO monitor_writer USING (true) WITH CHECK (true);
GRANT SELECT ON monitoring.v_load_latency TO dashboard_reader, monitor_writer;
