-- P9 shadow challengers (shef.monitoring.challengers): frozen rules applied to
-- the champion's forecast at issue time, logged daily, never served. Rows are
-- written once and never rewritten, like predictions.
CREATE TABLE IF NOT EXISTS monitoring.challengers (
    target_timestamp   timestamptz NOT NULL,   -- UTC start of the target hour
    challenger         text        NOT NULL,   -- recent_median | stack_ercot
    issue_time_utc     timestamptz NOT NULL,   -- the champion forecast's issue time
    predicted_load_mw  float8      NOT NULL,
    champion_mw        float8      NOT NULL,   -- the served forecast it was derived from
    inputs             jsonb       NOT NULL,   -- correction applied / ERCOT value or fallback
    computed_at        timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (target_timestamp, challenger)
);

ALTER TABLE monitoring.challengers ENABLE ROW LEVEL SECURITY;
GRANT SELECT ON monitoring.challengers TO dashboard_reader, monitor_writer;
GRANT INSERT ON monitoring.challengers TO monitor_writer;
CREATE POLICY dashboard_read ON monitoring.challengers FOR SELECT TO dashboard_reader USING (true);
CREATE POLICY monitor_read ON monitoring.challengers FOR SELECT TO monitor_writer USING (true);
CREATE POLICY monitor_insert ON monitoring.challengers FOR INSERT TO monitor_writer WITH CHECK (true);

-- Scored hours: challenger vs the champion forecast it came from, same actual.
-- `shadow` = logged before the target day began (Central), i.e. a real live
-- record; false for rows replayed after the fact (the 2026-10-28 backfill).
-- `preregistered` = target day after the rules were frozen. The honest
-- verdict uses rows that are both.
CREATE OR REPLACE VIEW monitoring.v_challengers AS
SELECT c.challenger, c.target_timestamp, l.target_day, c.issue_time_utc,
       l.actual_load_mw,
       c.champion_mw,
       c.predicted_load_mw                                                  AS challenger_mw,
       abs(c.champion_mw - l.actual_load_mw) / l.actual_load_mw * 100       AS champion_ape,
       abs(c.predicted_load_mw - l.actual_load_mw) / l.actual_load_mw * 100 AS challenger_ape,
       c.champion_mw - l.actual_load_mw                                     AS champion_err,
       c.predicted_load_mw - l.actual_load_mw                               AS challenger_err,
       c.computed_at < (l.target_day::timestamp AT TIME ZONE 'America/Chicago') AS shadow,
       l.target_day >= DATE '2026-10-29'                                    AS preregistered
FROM monitoring.challengers c
JOIN monitoring.v_live l ON l.target_timestamp = c.target_timestamp AND l.served_first
WHERE l.actual_load_mw IS NOT NULL;

-- Per challenger, on its matched hours: the numbers the promotion rule needs.
-- scope 'record' = shadow and preregistered (the verdict); 'all' adds replayed days.
CREATE OR REPLACE VIEW monitoring.v_challenger_summary AS
WITH s AS (
    SELECT 'record'::text AS scope, * FROM monitoring.v_challengers WHERE shadow AND preregistered
    UNION ALL
    SELECT 'all', * FROM monitoring.v_challengers
), q AS (
    SELECT scope, challenger, percentile_cont(0.95) WITHIN GROUP (ORDER BY actual_load_mw) AS peak_q
    FROM s GROUP BY scope, challenger
)
SELECT s.scope, s.challenger,
       count(*)                                                          AS n_hours,
       count(DISTINCT s.target_day)                                      AS n_days,
       min(s.target_day)                                                 AS first_day,
       max(s.target_day)                                                 AS last_day,
       avg(s.champion_ape)                                               AS champion_mape,
       avg(s.challenger_ape)                                             AS challenger_mape,
       avg(s.champion_ape) FILTER (WHERE s.actual_load_mw >= q.peak_q)   AS champion_peak_mape,
       avg(s.challenger_ape) FILTER (WHERE s.actual_load_mw >= q.peak_q) AS challenger_peak_mape,
       avg(s.champion_err) / avg(s.actual_load_mw) * 100                 AS champion_bias_pct,
       avg(s.challenger_err) / avg(s.actual_load_mw) * 100               AS challenger_bias_pct
FROM s JOIN q USING (scope, challenger)
GROUP BY s.scope, s.challenger;

GRANT SELECT ON monitoring.v_challengers, monitoring.v_challenger_summary TO dashboard_reader, monitor_writer;
