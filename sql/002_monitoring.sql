-- P6 monitoring. Everything lives in its own schema: Supabase's REST API
-- exposes `public` and grants new objects there to anon by default, and these
-- views read predictions with their owner's rights.
CREATE SCHEMA IF NOT EXISTS monitoring;
REVOKE ALL ON SCHEMA monitoring FROM PUBLIC, anon, authenticated;

-- One row per hour that has an actual. `served_first` marks what the live system
-- served for that hour (a catch-up run can add a later row for the same hour).
CREATE OR REPLACE VIEW monitoring.v_scored AS
SELECT p.id,
       p.target_timestamp,
       (p.target_timestamp AT TIME ZONE 'America/Chicago')::date               AS target_day,
       p.issue_time_utc,
       p.lead_hours,
       p.model_name,
       p.model_version,
       CASE WHEN p.model_version LIKE 'backtest%' THEN 'backtest' ELSE 'live' END AS kind,
       CASE WHEN p.model_name LIKE 'baseline%' THEN 'fallback' ELSE 'model' END    AS served_as,
       row_number() OVER (PARTITION BY (p.model_version LIKE 'backtest%'), p.target_timestamp
                          ORDER BY p.predicted_at) = 1                          AS served_first,
       p.predicted_load_mw,
       p.actual_load_mw,
       p.predicted_load_mw - p.actual_load_mw                                   AS err,
       abs(p.predicted_load_mw - p.actual_load_mw) / p.actual_load_mw * 100     AS ape
FROM predictions p
WHERE p.actual_load_mw IS NOT NULL;

-- Same definitions as shef.models.metrics: MAPE, RMSE, bias % of mean load,
-- Peak-MAPE over hours with actual >= the 95th percentile (linear, = np.quantile).
CREATE OR REPLACE VIEW monitoring.v_daily AS
SELECT kind, model_name, model_version, target_day,
       count(*)                                                  AS n_hours,
       avg(ape)                                                  AS mape,
       sqrt(avg(err * err))                                      AS rmse,
       avg(err) / avg(actual_load_mw) * 100                      AS bias_pct,
       max(ape)                                                  AS max_ape
FROM monitoring.v_scored
GROUP BY kind, model_name, model_version, target_day;

-- Whole history per kind (served rows) and per model version.
CREATE OR REPLACE VIEW monitoring.v_summary AS
WITH s AS (
    SELECT kind, 'served'::text AS model_version, target_day, actual_load_mw, err, ape
    FROM monitoring.v_scored WHERE served_first
    UNION ALL
    SELECT kind, model_version, target_day, actual_load_mw, err, ape FROM monitoring.v_scored
), q AS (
    SELECT kind, model_version, percentile_cont(0.95) WITHIN GROUP (ORDER BY actual_load_mw) AS peak_q
    FROM s GROUP BY kind, model_version
)
SELECT s.kind, s.model_version,
       count(*)                                                   AS n_hours,
       count(DISTINCT s.target_day)                               AS n_days,
       min(s.target_day)                                          AS first_day,
       max(s.target_day)                                          AS last_day,
       avg(s.ape)                                                 AS mape,
       avg(s.ape) FILTER (WHERE s.actual_load_mw >= q.peak_q)     AS peak_mape,
       sqrt(avg(s.err * s.err))                                   AS rmse,
       avg(s.err) / avg(s.actual_load_mw) * 100                   AS bias_pct
FROM s JOIN q USING (kind, model_version)
GROUP BY s.kind, s.model_version;

-- The last 30 scored days of each kind (served rows): the definition-of-done number.
CREATE OR REPLACE VIEW monitoring.v_window_30d AS
WITH s AS (
    SELECT kind, target_day, actual_load_mw, err, ape,
           max(target_day) OVER (PARTITION BY kind) AS last_day
    FROM monitoring.v_scored WHERE served_first
), w AS (
    SELECT * FROM s WHERE target_day > last_day - 30
), q AS (
    SELECT kind, percentile_cont(0.95) WITHIN GROUP (ORDER BY actual_load_mw) AS peak_q FROM w GROUP BY kind
)
SELECT w.kind,
       count(*)                                                   AS n_hours,
       count(DISTINCT w.target_day)                               AS n_days,
       min(w.target_day)                                          AS first_day,
       max(w.target_day)                                          AS last_day,
       avg(w.ape)                                                 AS mape,
       avg(w.ape) FILTER (WHERE w.actual_load_mw >= q.peak_q)     AS peak_mape,
       sqrt(avg(w.err * w.err))                                   AS rmse,
       avg(w.err) / avg(w.actual_load_mw) * 100                   AS bias_pct
FROM w JOIN q USING (kind)
GROUP BY w.kind;

CREATE OR REPLACE VIEW monitoring.v_by_lead AS
SELECT kind, lead_hours, count(*) AS n_hours, avg(ape) AS mape,
       avg(err) / avg(actual_load_mw) * 100 AS bias_pct
FROM monitoring.v_scored WHERE served_first
GROUP BY kind, lead_hours;

-- Served weather vs what was observed later (ERA5) and vs the training proxy.
CREATE TABLE IF NOT EXISTS monitoring.weather_checks (
    issue_date   date        NOT NULL,
    comparison   text        NOT NULL,   -- served_vs_era5 | served_vs_previous_day1
    variable     text        NOT NULL DEFAULT 'temp_c',
    n_hours      int         NOT NULL,
    mae          float8      NOT NULL,
    bias         float8      NOT NULL,   -- served minus reference
    p95_abs      float8      NOT NULL,
    computed_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (issue_date, comparison, variable)
);

CREATE TABLE IF NOT EXISTS monitoring.drift (
    computed_on  date        NOT NULL,
    feature      text        NOT NULL,
    psi          float8      NOT NULL,
    status       text        NOT NULL,   -- ok | watch | drift
    ref_months   text        NOT NULL,
    n_current    int         NOT NULL,
    computed_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (computed_on, feature)
);

CREATE TABLE IF NOT EXISTS monitoring.promotions (
    id                  bigserial PRIMARY KEY,
    evaluated_at        timestamptz NOT NULL DEFAULT now(),
    champion_version    text        NOT NULL,
    challenger_version  text,                 -- set only when registered
    window_start        date,
    window_end          date,
    n_days              int,
    champion_mape       float8,
    challenger_mape     float8,
    champion_peak_mape  float8,
    challenger_peak_mape float8,
    decision            text        NOT NULL, -- promote | keep | insufficient
    reason              text        NOT NULL,
    mlflow_run_id       text
);

ALTER TABLE monitoring.weather_checks ENABLE ROW LEVEL SECURITY;
ALTER TABLE monitoring.drift ENABLE ROW LEVEL SECURITY;
ALTER TABLE monitoring.promotions ENABLE ROW LEVEL SECURITY;

-- The dashboard's read-only login (password set out of band, never in git).
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'dashboard_reader') THEN
        CREATE ROLE dashboard_reader NOLOGIN;
    END IF;
END $$;
GRANT USAGE ON SCHEMA monitoring TO dashboard_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA monitoring TO dashboard_reader;
CREATE POLICY dashboard_read ON monitoring.weather_checks FOR SELECT TO dashboard_reader USING (true);
CREATE POLICY dashboard_read ON monitoring.drift FOR SELECT TO dashboard_reader USING (true);
CREATE POLICY dashboard_read ON monitoring.promotions FOR SELECT TO dashboard_reader USING (true);
