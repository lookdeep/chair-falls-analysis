-- One-off session audit for talk-confirmed sitter analysis.
-- Retained sessions must pass the existing hour-level eligibility gates and
-- include at least one talk_clicked event during the local session window.
CREATE OR REPLACE TABLE `{{session_table}}` AS
WITH fall_events_source AS {{fall_events_compatible_subquery_sql}},
intervention_monitor_ids AS (
  SELECT DISTINCT CAST(monitor_id AS INT64) AS monitor_id
  FROM fall_events_source
  WHERE monitor_id IS NOT NULL
  UNION DISTINCT
  SELECT monitor_id
  FROM UNNEST([{{manual_intervention_monitor_ids_sql}}]) AS monitor_id
  WHERE monitor_id IS NOT NULL
),
hourly_base AS (
  SELECT
    h.hospital_id,
    h.division_id,
    CAST(h.monitor_id AS INT64) AS monitor_id,
    CAST(h.patient_id AS INT64) AS patient_id,
    DATETIME_TRUNC(CAST(h.hour_bin_local AS DATETIME), HOUR) AS hour_dt_local,
    (
      SAFE_CAST(h.in_chair AS FLOAT64) BETWEEN 0.0 AND 1.0
      AND SAFE_CAST(h.in_bed AS FLOAT64) BETWEEN 0.0 AND 1.0
      AND SAFE_CAST(h.in_room AS FLOAT64) BETWEEN 0.0 AND 1.0
      AND SAFE_CAST(h.not_located AS FLOAT64) BETWEEN 0.0 AND 1.0
      AND ABS(
        COALESCE(SAFE_CAST(h.in_chair AS FLOAT64), 0.0)
        + COALESCE(SAFE_CAST(h.in_bed AS FLOAT64), 0.0)
        + COALESCE(SAFE_CAST(h.in_room AS FLOAT64), 0.0)
        + COALESCE(SAFE_CAST(h.not_located AS FLOAT64), 0.0)
        - 1.0
      ) <= 0.02
    ) AS row_valid_pct
  FROM `{{project}}.{{dataset}}.hourly_graph_facts_alarms_cache` h
  JOIN intervention_monitor_ids monitors
    ON CAST(h.monitor_id AS INT64) = monitors.monitor_id
  WHERE h.hospital_id = {{study_hospital_id}}
    AND h.monitor_id IS NOT NULL
    AND h.patient_id IS NOT NULL
    AND DATE(CAST(h.hour_bin_local AS DATETIME))
      BETWEEN DATE "{{study_start_date}}" AND DATE "{{study_end_date}}"
),
eligibility AS (
  SELECT
    hospital_id,
    division_id,
    monitor_id,
    patient_id,
    MIN(hour_dt_local) AS session_start_dt_local,
    MAX(hour_dt_local) AS session_end_dt_local,
    COUNT(DISTINCT hour_dt_local) AS observed_hours,
    DATETIME_DIFF(MAX(hour_dt_local), MIN(hour_dt_local), HOUR) + 1 AS expected_hours,
    SAFE_DIVIDE(
      COUNT(DISTINCT hour_dt_local),
      DATETIME_DIFF(MAX(hour_dt_local), MIN(hour_dt_local), HOUR) + 1
    ) AS coverage_ratio,
    AVG(IF(row_valid_pct, 1.0, 0.0)) AS valid_pct_ratio
  FROM hourly_base
  GROUP BY hospital_id, division_id, monitor_id, patient_id
),
talk_events_raw AS (
  SELECT
    CAST(a.id AS STRING) AS talk_event_id,
    a.hospital_id,
    a.division_id,
    CAST(a.object_id AS INT64) AS monitor_id,
    CAST(a.created_at AS TIMESTAMP) AS event_ts_utc
  FROM `{{project}}.hospital_api.audit_logs` a
  WHERE a.hospital_id = {{study_hospital_id}}
    AND a.created_at IS NOT NULL
    AND a.object_id IS NOT NULL
    AND LOWER(TRIM(COALESCE(a.object_type, ""))) = "patientmonitor"
    AND LOWER(TRIM(COALESCE(a.category, ""))) LIKE "%talk_clicked%"
    AND DATE(CAST(a.created_at AS TIMESTAMP), "{{hospital_timezone}}")
      BETWEEN DATE "{{study_start_date}}" AND DATE "{{study_end_date}}"
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY hospital_id, division_id, monitor_id, event_ts_utc
    ORDER BY talk_event_id
  ) = 1
),
session_talks AS (
  SELECT
    e.hospital_id,
    e.division_id,
    e.monitor_id,
    e.patient_id,
    e.session_start_dt_local,
    DATETIME_ADD(e.session_end_dt_local, INTERVAL 1 HOUR) AS session_end_exclusive_dt_local,
    TIMESTAMP(e.session_start_dt_local, "{{hospital_timezone}}") AS session_start_ts_utc,
    TIMESTAMP(
      DATETIME_ADD(e.session_end_dt_local, INTERVAL 1 HOUR),
      "{{hospital_timezone}}"
    ) AS session_end_exclusive_ts_utc,
    e.observed_hours,
    e.expected_hours,
    e.coverage_ratio,
    e.valid_pct_ratio,
    COUNT(t.event_ts_utc) AS talk_clicked_events,
    MIN(t.event_ts_utc) AS first_talk_ts_utc,
    MAX(t.event_ts_utc) AS last_talk_ts_utc
  FROM eligibility e
  LEFT JOIN talk_events_raw t
    ON t.hospital_id = e.hospital_id
   AND t.monitor_id = e.monitor_id
   AND (t.division_id IS NULL OR t.division_id = e.division_id)
   AND DATETIME(t.event_ts_utc, "{{hospital_timezone}}") >= e.session_start_dt_local
   AND DATETIME(t.event_ts_utc, "{{hospital_timezone}}") < DATETIME_ADD(
     e.session_end_dt_local,
     INTERVAL 1 HOUR
   )
  GROUP BY
    hospital_id,
    division_id,
    monitor_id,
    patient_id,
    session_start_dt_local,
    session_end_exclusive_dt_local,
    session_start_ts_utc,
    session_end_exclusive_ts_utc,
    observed_hours,
    expected_hours,
    coverage_ratio,
    valid_pct_ratio
)
SELECT
  hospital_id,
  division_id,
  monitor_id,
  patient_id,
  session_start_dt_local,
  session_end_exclusive_dt_local,
  session_start_ts_utc,
  session_end_exclusive_ts_utc,
  observed_hours,
  expected_hours,
  coverage_ratio,
  valid_pct_ratio,
  talk_clicked_events,
  first_talk_ts_utc,
  last_talk_ts_utc,
  observed_hours >= {{min_observed_hours}} AS meets_observed_hours_gate,
  coverage_ratio >= {{min_coverage_ratio}} AS meets_coverage_gate,
  valid_pct_ratio >= {{min_valid_pct_ratio}} AS meets_valid_pct_gate,
  talk_clicked_events >= 1 AS meets_talk_gate,
  (
    observed_hours >= {{min_observed_hours}}
    AND coverage_ratio >= {{min_coverage_ratio}}
    AND valid_pct_ratio >= {{min_valid_pct_ratio}}
    AND talk_clicked_events >= 1
  ) AS retained_for_sitter_analysis,
  CASE
    WHEN session_start_dt_local >= DATETIME "{{talk_gate_pivot_date}} 00:00:00" THEN "post_2025_06_01"
    ELSE "pre_2025_06_01"
  END AS period_prepost_2025_06_01
FROM session_talks
ORDER BY hospital_id, division_id, monitor_id, patient_id, session_start_dt_local;
