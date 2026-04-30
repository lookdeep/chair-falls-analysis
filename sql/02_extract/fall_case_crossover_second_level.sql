-- Per-second event-local case-crossover panel aligned to the hazard 300s pre-anchor window.
-- Output schema intentionally mirrors fall_livestream_second_level.sql plus window_role and anchor_ts_utc.
WITH falls AS (
  SELECT
    ROW_NUMBER() OVER (
      ORDER BY timestamp, hospital_id, division_id, patient_id, monitor_id, summary
    ) AS fall_event_id,
    timestamp AS fall_ts_utc,
    timestamp_local AS fall_ts_local,
    hospital_id,
    division_id,
    patient_id,
    monitor_id
  FROM {{fall_events_compatible_subquery_sql}}
  WHERE timestamp IS NOT NULL
    AND hospital_id IS NOT NULL
    AND hospital_id = {{study_hospital_id}}
    AND division_id IS NOT NULL
    AND monitor_id IS NOT NULL
),
anchors AS (
  SELECT
    fall_event_id,
    "hazard" AS window_role,
    fall_ts_utc AS anchor_ts_utc,
    fall_ts_utc,
    fall_ts_local,
    hospital_id,
    division_id,
    patient_id,
    monitor_id
  FROM falls
{{fall_case_crossover_control_unions_second_level_sql}}
),
panel AS (
  SELECT
    a.fall_event_id,
    a.window_role,
    a.anchor_ts_utc,
    a.fall_ts_utc,
    a.fall_ts_local,
    a.hospital_id,
    a.division_id,
    a.patient_id,
    a.monitor_id,
    d.timestamp AS frame_ts_utc,
    TIMESTAMP_DIFF(d.timestamp, a.anchor_ts_utc, SECOND) AS second_offset,
    CASE
      WHEN d.timestamp < TIMESTAMP_SUB(a.anchor_ts_utc, INTERVAL 60 SECOND) THEN "pre_long"
      WHEN d.timestamp < a.anchor_ts_utc THEN "pre_immediate"
      ELSE "post"
    END AS window_phase,
    COALESCE(
      d.all_patient_chair_distances.min_obj1_obj2_distance,
      d.all_deduped_patient_chair_distances.min_obj1_obj2_distance
    ) AS patient_chair_distance,
    COALESCE(
      d.all_patient_bed_distances.min_obj1_obj2_distance,
      d.all_deduped_patient_bed_distances.min_obj1_obj2_distance
    ) AS patient_bed_distance,
    COALESCE(
      d.all_patient_other_distances.min_obj1_obj2_distance,
      d.all_deduped_patient_other_distances.min_obj1_obj2_distance
    ) AS patient_room_distance,
    GREATEST(
      COALESCE(d.all_patient_staff_distances.max_obj1_obj2_iou, 0.0),
      COALESCE(d.all_deduped_patient_staff_distances.max_obj1_obj2_iou, 0.0)
    ) AS patient_staff_iou,
    d.frame.nudge.state_to_show AS nudge_state,
    d.frame.nudge.score AS nudge_score,
    d.frame.bed.in_bed_score AS bed_in_bed_score,
    d.frame.motion.val_BAC AS motion_bac,
    GREATEST(IFNULL(ARRAY_LENGTH(d.all_patients), 0), IFNULL(ARRAY_LENGTH(d.all_deduped_patients), 0))
      AS patient_candidate_count,
    GREATEST(IFNULL(ARRAY_LENGTH(d.all_staff), 0), IFNULL(ARRAY_LENGTH(d.all_deduped_staff), 0))
      AS staff_candidate_count,
    GREATEST(IFNULL(ARRAY_LENGTH(d.all_other), 0), IFNULL(ARRAY_LENGTH(d.all_deduped_other), 0))
      AS other_candidate_count,
    IFNULL(ARRAY_LENGTH(d.all_beds), 0) AS bed_candidate_count,
    IFNULL(ARRAY_LENGTH(d.all_chairs), 0) AS chair_candidate_count,
    COALESCE(
      (
        SELECT AS STRUCT
          LOWER(NULLIF(TRIM(p.posture), "")) AS posture_label,
          p.score_sitting AS score_sitting,
          p.score_standing AS score_standing,
          p.score_lying AS score_lying,
          IFNULL(ARRAY_LENGTH(d.all_patients), 0) AS candidate_count,
          "all_patients" AS posture_source
        FROM UNNEST(d.all_patients) AS p
        ORDER BY
          COALESCE(p.score_patient, 0.0) DESC,
          COALESCE(p.score, 0.0) DESC,
          IF(COALESCE(p.has_tracked, FALSE), 1, 0) DESC
        LIMIT 1
      ),
      (
        SELECT AS STRUCT
          LOWER(NULLIF(TRIM(p.posture), "")) AS posture_label,
          p.score_sitting AS score_sitting,
          p.score_standing AS score_standing,
          p.score_lying AS score_lying,
          IFNULL(ARRAY_LENGTH(d.all_deduped_patients), 0) AS candidate_count,
          "all_deduped_patients" AS posture_source
        FROM UNNEST(d.all_deduped_patients) AS p
        ORDER BY
          COALESCE(p.score_patient, 0.0) DESC,
          COALESCE(p.score, 0.0) DESC,
          IF(COALESCE(p.has_tracked, FALSE), 1, 0) DESC
        LIMIT 1
      ),
      STRUCT(
        CAST(NULL AS STRING) AS posture_label,
        CAST(NULL AS FLOAT64) AS score_sitting,
        CAST(NULL AS FLOAT64) AS score_standing,
        CAST(NULL AS FLOAT64) AS score_lying,
        0 AS candidate_count,
        "none" AS posture_source
      )
    ) AS primary_patient_posture
  FROM anchors a
  LEFT JOIN `{{project}}.{{dataset}}.{{live_stream_derivatives_table}}` d
    ON d.hospital_id = a.hospital_id
   AND d.division_id = a.division_id
   AND (a.patient_id IS NULL OR d.patient_id = a.patient_id)
   AND d.monitor_id = a.monitor_id
   AND d.timestamp BETWEEN TIMESTAMP_SUB(a.anchor_ts_utc, INTERVAL 300 SECOND)
                       AND a.anchor_ts_utc
)
SELECT
  fall_event_id,
  window_role,
  anchor_ts_utc,
  fall_ts_utc,
  fall_ts_local,
  hospital_id,
  division_id,
  patient_id,
  monitor_id,
  frame_ts_utc,
  second_offset,
  window_phase,
  (
    patient_chair_distance IS NOT NULL
    OR patient_bed_distance IS NOT NULL
    OR patient_room_distance IS NOT NULL
  ) AS frame_has_location_signal,
  patient_chair_distance,
  patient_bed_distance,
  patient_room_distance,
  patient_staff_iou,
  CASE
    WHEN patient_chair_distance IS NULL
         AND patient_bed_distance IS NULL
         AND patient_room_distance IS NULL THEN "no_patient"
    WHEN COALESCE(patient_chair_distance, 1e12) <= LEAST(
      COALESCE(patient_bed_distance, 1e12),
      COALESCE(patient_room_distance, 1e12)
    ) THEN "chair"
    WHEN COALESCE(patient_bed_distance, 1e12) <= COALESCE(patient_room_distance, 1e12) THEN "bed"
    ELSE "room"
  END AS dominant_location_label,
  nudge_state,
  nudge_score,
  bed_in_bed_score,
  motion_bac,
  patient_candidate_count,
  staff_candidate_count,
  other_candidate_count,
  bed_candidate_count,
  chair_candidate_count,
  primary_patient_posture.posture_label AS primary_patient_posture_label,
  primary_patient_posture.score_sitting AS primary_patient_posture_score_sitting,
  primary_patient_posture.score_standing AS primary_patient_posture_score_standing,
  primary_patient_posture.score_lying AS primary_patient_posture_score_lying,
  primary_patient_posture.candidate_count AS patient_posture_candidate_count,
  primary_patient_posture.posture_source AS patient_posture_source
FROM panel
WHERE frame_ts_utc IS NOT NULL;
