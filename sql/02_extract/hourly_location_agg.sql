-- Hourly patient-location baseline extraction for chair-falls prep.
-- Source semantics:
--   - in_* columns are normalized shares on a 0-1 scale
--   - table is pre-aggregated at hourly grain
--
-- Scope for baseline:
--   - hospital_id = {{study_hospital_id}}
--   - years 2024-2025 (by hour_bin_local)
WITH baseline AS (
  SELECT
    hospital_id,
    division_id,
    monitor_id,
    patient_id,
    TIMESTAMP_TRUNC(hour_bin_local, HOUR) AS hour_ts,
    SAFE_CAST(in_chair AS FLOAT64) AS pct_chair,
    SAFE_CAST(in_bed AS FLOAT64) AS pct_bed,
    SAFE_CAST(in_room AS FLOAT64) AS pct_ambulatory,
    SAFE_CAST(not_located AS FLOAT64) AS pct_not_located,
    SAFE_CAST(num_alarms AS INT64) AS num_alarms,
    SAFE_CAST(num_nudges AS INT64) AS num_nudges,
    SAFE_CAST(num_announcements AS INT64) AS num_announcements
  FROM `{{project}}.{{dataset}}.hourly_graph_facts_alarms_cache`
  WHERE hospital_id = {{study_hospital_id}}
    AND DATE(hour_bin_local) >= DATE "2024-01-01"
    AND DATE(hour_bin_local) < DATE "2026-01-01"
)
SELECT
  hospital_id,
  division_id,
  monitor_id,
  patient_id,
  hour_ts,
  pct_chair,
  pct_bed,
  pct_ambulatory,
  pct_not_located,
  num_alarms,
  num_nudges,
  num_announcements
FROM baseline
WHERE patient_id IS NOT NULL
  AND monitor_id IS NOT NULL
  AND (
    COALESCE(pct_chair, 0.0)
    + COALESCE(pct_bed, 0.0)
    + COALESCE(pct_ambulatory, 0.0)
    + COALESCE(pct_not_located, 0.0)
  ) BETWEEN 0.98 AND 1.02;
