-- Chunk inventory for the non-fall derivatives cohort.
-- One row per contiguous five-minute non-fall window.
SELECT
  source_chunk_id,
  hospital_id,
  division_id,
  monitor_id,
  patient_id,
  chunk_start_ts_utc,
  chunk_end_ts_utc,
  anchor_ts_utc,
  frame_count,
  duration_seconds,
  anchor_hour_local,
  anchor_daypart_local,
  has_patient_id
FROM {{negative_control_source_chunk_subquery_sql}};
