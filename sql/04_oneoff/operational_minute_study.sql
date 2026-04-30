-- One-off minute exposure and nudge-duration table for talk-confirmed sitter sessions.
CREATE OR REPLACE TABLE `{{minute_table}}` AS
WITH retained_sessions AS (
  SELECT *
  FROM `{{session_table}}`
  WHERE retained_for_sitter_analysis
),
retained_hourly_base AS (
  SELECT
    rs.hospital_id,
    rs.division_id,
    rs.monitor_id,
    rs.patient_id,
    DATETIME_TRUNC(CAST(h.hour_bin_local AS DATETIME), HOUR) AS hour_dt_local
  FROM `{{project}}.{{dataset}}.hourly_graph_facts_alarms_cache` h
  JOIN retained_sessions rs
    ON rs.hospital_id = h.hospital_id
   AND rs.division_id = h.division_id
   AND rs.monitor_id = CAST(h.monitor_id AS INT64)
   AND rs.patient_id = CAST(h.patient_id AS INT64)
   AND DATETIME_TRUNC(CAST(h.hour_bin_local AS DATETIME), HOUR) >= rs.session_start_dt_local
   AND DATETIME_TRUNC(CAST(h.hour_bin_local AS DATETIME), HOUR) < rs.session_end_exclusive_dt_local
  WHERE h.hospital_id = {{study_hospital_id}}
    AND h.monitor_id IS NOT NULL
    AND h.patient_id IS NOT NULL
),
retained_hour_lookup AS (
  SELECT
    hospital_id,
    monitor_id,
    hour_dt_local,
    ANY_VALUE(division_id) AS division_id,
    ANY_VALUE(patient_id) AS patient_id,
    COUNT(
      DISTINCT CONCAT(CAST(division_id AS STRING), "|", CAST(patient_id AS STRING))
    ) AS unit_count
  FROM retained_hourly_base
  GROUP BY hospital_id, monitor_id, hour_dt_local
),
retained_hour_unique AS (
  SELECT
    hospital_id,
    division_id,
    monitor_id,
    patient_id,
    hour_dt_local,
    TIMESTAMP(hour_dt_local, "{{hospital_timezone}}") AS hour_start_utc
  FROM retained_hour_lookup
  WHERE unit_count = 1
),
retained_seconds AS (
  SELECT
    eh.hospital_id,
    eh.division_id,
    eh.monitor_id,
    eh.patient_id,
    second_ts_utc,
    TIMESTAMP_TRUNC(second_ts_utc, MINUTE) AS minute_ts_utc,
    DATETIME(TIMESTAMP_TRUNC(second_ts_utc, MINUTE), "{{hospital_timezone}}") AS minute_ts_local
  FROM retained_hour_unique eh,
  UNNEST(
    GENERATE_TIMESTAMP_ARRAY(
      eh.hour_start_utc,
      TIMESTAMP_ADD(eh.hour_start_utc, INTERVAL 3599 SECOND),
      INTERVAL 1 SECOND
    )
  ) AS second_ts_utc
),
posture_second_candidates AS (
  SELECT
    eh.hospital_id,
    eh.division_id,
    eh.monitor_id,
    eh.patient_id,
    TIMESTAMP_TRUNC(d.timestamp, SECOND) AS second_ts_utc,
    MIN(
      COALESCE(
        d.all_patient_chair_distances.min_obj1_obj2_distance,
        d.all_deduped_patient_chair_distances.min_obj1_obj2_distance
      )
    ) AS patient_chair_distance,
    MIN(
      COALESCE(
        d.all_patient_bed_distances.min_obj1_obj2_distance,
        d.all_deduped_patient_bed_distances.min_obj1_obj2_distance
      )
    ) AS patient_bed_distance,
    MIN(
      COALESCE(
        d.all_patient_other_distances.min_obj1_obj2_distance,
        d.all_deduped_patient_other_distances.min_obj1_obj2_distance
      )
    ) AS patient_room_distance
  FROM retained_hour_unique eh
  JOIN `{{project}}.{{dataset}}.{{live_stream_derivatives_table}}` d
    ON d.hospital_id = eh.hospital_id
   AND d.division_id = eh.division_id
   AND d.monitor_id = eh.monitor_id
   AND d.patient_id = eh.patient_id
   AND d.timestamp >= eh.hour_start_utc
   AND d.timestamp < TIMESTAMP_ADD(eh.hour_start_utc, INTERVAL 1 HOUR)
   AND DATETIME_TRUNC(DATETIME(d.timestamp, "{{hospital_timezone}}"), HOUR) = eh.hour_dt_local
  GROUP BY hospital_id, division_id, monitor_id, patient_id, second_ts_utc
),
retained_second_posture AS (
  SELECT
    rs.hospital_id,
    rs.division_id,
    rs.monitor_id,
    rs.patient_id,
    rs.second_ts_utc,
    rs.minute_ts_utc,
    rs.minute_ts_local,
    CASE
      WHEN p.second_ts_utc IS NULL THEN "missing_posture"
      WHEN p.patient_chair_distance IS NULL
           AND p.patient_bed_distance IS NULL
           AND p.patient_room_distance IS NULL THEN "no_patient"
      WHEN COALESCE(p.patient_chair_distance, 1e12) <= LEAST(
        COALESCE(p.patient_bed_distance, 1e12),
        COALESCE(p.patient_room_distance, 1e12)
      ) THEN "chair"
      WHEN COALESCE(p.patient_bed_distance, 1e12) <= COALESCE(p.patient_room_distance, 1e12) THEN "bed"
      ELSE "room"
    END AS dominant_location_label
  FROM retained_seconds rs
  LEFT JOIN posture_second_candidates p
    USING (hospital_id, division_id, monitor_id, patient_id, second_ts_utc)
),
minute_exposure AS (
  SELECT
    hospital_id,
    division_id,
    monitor_id,
    patient_id,
    minute_ts_utc,
    minute_ts_local,
    COUNTIF(dominant_location_label = "chair") AS seconds_chair,
    COUNTIF(dominant_location_label = "bed") AS seconds_bed,
    COUNTIF(dominant_location_label = "room") AS seconds_room,
    COUNTIF(dominant_location_label = "no_patient") AS seconds_no_patient,
    COUNTIF(dominant_location_label = "missing_posture") AS seconds_missing_posture
  FROM retained_second_posture
  GROUP BY hospital_id, division_id, monitor_id, patient_id, minute_ts_utc, minute_ts_local
),
nudge_intervals_raw AS (
  SELECT DISTINCT
    CAST(n.patient_monitor_id AS INT64) AS monitor_id,
    CAST(n.start_time AS TIMESTAMP) AS start_ts_utc,
    CAST(n.end_time AS TIMESTAMP) AS end_ts_utc,
    CAST(n.nudge_uuid AS STRING) AS nudge_uuid
  FROM `{{project}}.hospital_api.nudge_history` n
  WHERE n.patient_monitor_id IS NOT NULL
    AND n.start_time IS NOT NULL
    AND n.end_time IS NOT NULL
    AND CAST(n.end_time AS TIMESTAMP) > CAST(n.start_time AS TIMESTAMP)
    AND DATE(CAST(n.end_time AS TIMESTAMP), "{{hospital_timezone}}") >= DATE "{{study_start_date}}"
    AND DATE(CAST(n.start_time AS TIMESTAMP), "{{hospital_timezone}}") <= DATE "{{study_end_date}}"
),
nudge_interval_hour_overlap AS (
  SELECT
    eh.hospital_id,
    eh.division_id,
    eh.monitor_id,
    eh.patient_id,
    nr.nudge_uuid,
    TIMESTAMP_TRUNC(
      GREATEST(nr.start_ts_utc, eh.hour_start_utc),
      SECOND
    ) AS overlap_start_second_utc,
    TIMESTAMP_ADD(
      TIMESTAMP_TRUNC(
        LEAST(nr.end_ts_utc, TIMESTAMP_ADD(eh.hour_start_utc, INTERVAL 1 HOUR)),
        SECOND
      ),
      INTERVAL IF(
        UNIX_MICROS(LEAST(nr.end_ts_utc, TIMESTAMP_ADD(eh.hour_start_utc, INTERVAL 1 HOUR)))
          > UNIX_MICROS(
            TIMESTAMP_TRUNC(
              LEAST(nr.end_ts_utc, TIMESTAMP_ADD(eh.hour_start_utc, INTERVAL 1 HOUR)),
              SECOND
            )
          ),
        1,
        0
      ) SECOND
    ) AS overlap_end_second_exclusive_utc
  FROM nudge_intervals_raw nr
  JOIN retained_hour_unique eh
    ON eh.monitor_id = nr.monitor_id
   AND nr.end_ts_utc > eh.hour_start_utc
   AND nr.start_ts_utc < TIMESTAMP_ADD(eh.hour_start_utc, INTERVAL 1 HOUR)
  WHERE LEAST(nr.end_ts_utc, TIMESTAMP_ADD(eh.hour_start_utc, INTERVAL 1 HOUR))
      > GREATEST(nr.start_ts_utc, eh.hour_start_utc)
),
nudge_active_seconds_positioned AS (
  SELECT DISTINCT
    rsp.hospital_id,
    rsp.division_id,
    rsp.monitor_id,
    rsp.patient_id,
    rsp.second_ts_utc,
    rsp.minute_ts_utc,
    rsp.dominant_location_label
  FROM nudge_interval_hour_overlap n
  JOIN retained_second_posture rsp
    ON rsp.hospital_id = n.hospital_id
   AND rsp.division_id = n.division_id
   AND rsp.monitor_id = n.monitor_id
   AND rsp.patient_id = n.patient_id
   AND rsp.second_ts_utc >= n.overlap_start_second_utc
   AND rsp.second_ts_utc < n.overlap_end_second_exclusive_utc
),
nudge_rollup AS (
  SELECT
    hospital_id,
    division_id,
    monitor_id,
    patient_id,
    minute_ts_utc,
    COUNT(*) AS nudge_active_seconds_total_in_scope,
    COUNTIF(dominant_location_label = "chair") AS nudge_active_seconds_chair,
    COUNTIF(dominant_location_label = "bed") AS nudge_active_seconds_bed,
    COUNTIF(dominant_location_label = "room") AS nudge_active_seconds_room,
    COUNTIF(dominant_location_label = "no_patient") AS nudge_active_seconds_no_patient,
    COUNTIF(dominant_location_label = "missing_posture") AS nudge_active_seconds_missing_posture
  FROM nudge_active_seconds_positioned
  GROUP BY hospital_id, division_id, monitor_id, patient_id, minute_ts_utc
)
SELECT
  me.hospital_id,
  me.division_id,
  me.monitor_id,
  me.patient_id,
  me.minute_ts_utc,
  me.minute_ts_local,
  me.seconds_chair,
  me.seconds_bed,
  me.seconds_room,
  me.seconds_no_patient,
  me.seconds_missing_posture,
  COALESCE(nr.nudge_active_seconds_total_in_scope, 0) AS nudge_active_seconds_total_in_scope,
  COALESCE(nr.nudge_active_seconds_chair, 0) AS nudge_active_seconds_chair,
  COALESCE(nr.nudge_active_seconds_bed, 0) AS nudge_active_seconds_bed,
  COALESCE(nr.nudge_active_seconds_room, 0) AS nudge_active_seconds_room,
  COALESCE(nr.nudge_active_seconds_no_patient, 0) AS nudge_active_seconds_no_patient,
  COALESCE(nr.nudge_active_seconds_missing_posture, 0) AS nudge_active_seconds_missing_posture
FROM minute_exposure me
LEFT JOIN nudge_rollup nr
  USING (hospital_id, division_id, monitor_id, patient_id, minute_ts_utc)
ORDER BY hospital_id, division_id, monitor_id, patient_id, minute_ts_utc;
