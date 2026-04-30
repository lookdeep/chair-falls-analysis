-- Per-second matched non-fall negative-control panel aligned to the 120s pre-anchor control window.
-- Output schema mirrors fall_case_crossover_second_level.sql for non-fall matched controls.
WITH falls AS (
  SELECT
    ROW_NUMBER() OVER (
      ORDER BY timestamp, hospital_id, division_id, patient_id, monitor_id, summary
    ) AS fall_event_id,
    timestamp AS fall_ts_utc,
    hospital_id,
    division_id,
    patient_id,
    monitor_id,
    EXTRACT(HOUR FROM timestamp AT TIME ZONE "{{hospital_timezone}}") AS fall_hour_local,
    CASE
      WHEN EXTRACT(HOUR FROM timestamp AT TIME ZONE "{{hospital_timezone}}") BETWEEN 0 AND 5 THEN "window_00_05"
      WHEN EXTRACT(HOUR FROM timestamp AT TIME ZONE "{{hospital_timezone}}") BETWEEN 6 AND 8 THEN "window_06_08"
      WHEN EXTRACT(HOUR FROM timestamp AT TIME ZONE "{{hospital_timezone}}") BETWEEN 9 AND 11 THEN "window_09_11"
      WHEN EXTRACT(HOUR FROM timestamp AT TIME ZONE "{{hospital_timezone}}") BETWEEN 12 AND 14 THEN "window_12_14"
      WHEN EXTRACT(HOUR FROM timestamp AT TIME ZONE "{{hospital_timezone}}") BETWEEN 15 AND 17 THEN "window_15_17"
      WHEN EXTRACT(HOUR FROM timestamp AT TIME ZONE "{{hospital_timezone}}") BETWEEN 18 AND 20 THEN "window_18_20"
      ELSE "window_21_23"
    END AS fall_daypart_local
  FROM {{fall_events_compatible_subquery_sql}}
  WHERE timestamp IS NOT NULL
    AND hospital_id IS NOT NULL
    AND hospital_id = {{study_hospital_id}}
    AND division_id IS NOT NULL
    AND monitor_id IS NOT NULL
),
fall_monitors AS (
  SELECT DISTINCT monitor_id
  FROM falls
),
source_chunks AS (
  SELECT *
  FROM {{negative_control_source_chunk_subquery_sql}}
  WHERE monitor_id NOT IN (SELECT monitor_id FROM fall_monitors)
),
source_history AS (
  SELECT
    MIN(anchor_ts_utc) AS effective_source_start_ts_utc
  FROM source_chunks
),
eligible_falls AS (
  SELECT
    f.*,
    sh.effective_source_start_ts_utc
  FROM falls f
  CROSS JOIN source_history sh
  WHERE sh.effective_source_start_ts_utc IS NOT NULL
    AND f.fall_ts_utc >= sh.effective_source_start_ts_utc
),
candidates AS (
  SELECT
    f.fall_event_id,
    f.hospital_id,
    f.division_id,
    f.monitor_id AS source_monitor_id,
    s.monitor_id,
    s.patient_id,
    s.source_chunk_id,
    s.chunk_start_ts_utc AS source_chunk_start_ts_utc,
    s.chunk_end_ts_utc AS source_chunk_end_ts_utc,
    s.anchor_ts_utc,
    s.anchor_hour_local,
    s.anchor_daypart_local,
    ABS(TIMESTAMP_DIFF(s.anchor_ts_utc, f.fall_ts_utc, SECOND)) AS abs_anchor_gap_seconds,
    CASE
      WHEN s.anchor_hour_local = f.fall_hour_local THEN 1
      WHEN s.anchor_daypart_local = f.fall_daypart_local THEN 2
      ELSE 3
    END AS match_tier_rank,
    CASE
      WHEN s.anchor_hour_local = f.fall_hour_local THEN "same_local_hour"
      WHEN s.anchor_daypart_local = f.fall_daypart_local THEN "same_local_daypart"
      ELSE "same_division"
    END AS match_tier
  FROM eligible_falls f
  JOIN source_chunks s
    ON s.hospital_id = f.hospital_id
   AND s.division_id = f.division_id
),
best_tier AS (
  SELECT
    *,
    MIN(match_tier_rank) OVER (PARTITION BY fall_event_id) AS best_match_tier_rank
  FROM candidates
),
selected AS (
  SELECT
    fall_event_id,
    "nonfall_control" AS window_role,
    anchor_ts_utc,
    hospital_id,
    division_id,
    patient_id,
    monitor_id,
    source_chunk_id,
    source_chunk_start_ts_utc,
    source_chunk_end_ts_utc,
    source_monitor_id,
    ROW_NUMBER() OVER (
      PARTITION BY fall_event_id
      ORDER BY abs_anchor_gap_seconds, source_chunk_id
    ) AS control_rank,
    match_tier,
    COUNT(*) OVER (PARTITION BY fall_event_id) AS matching_candidates
  FROM best_tier
  WHERE match_tier_rank = best_match_tier_rank
  QUALIFY control_rank <= {{negative_control_match_limit}}
),
panel AS (
  SELECT
    s.fall_event_id,
    s.window_role,
    s.anchor_ts_utc,
    s.hospital_id,
    s.division_id,
    s.patient_id,
    s.monitor_id,
    s.source_chunk_id,
    s.source_chunk_start_ts_utc,
    s.source_chunk_end_ts_utc,
    s.source_monitor_id,
    s.control_rank,
    s.match_tier,
    s.matching_candidates,
    d.timestamp AS frame_ts_utc,
    TIMESTAMP_DIFF(d.timestamp, s.anchor_ts_utc, SECOND) AS second_offset,
    CASE
      WHEN d.timestamp < TIMESTAMP_SUB(s.anchor_ts_utc, INTERVAL 60 SECOND) THEN "pre_long"
      WHEN d.timestamp < s.anchor_ts_utc THEN "pre_immediate"
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
  FROM selected s
  LEFT JOIN `{{project}}.{{dataset}}.{{negative_control_source_table}}` d
    ON d.hospital_id = s.hospital_id
   AND d.division_id = s.division_id
   AND (s.patient_id IS NULL OR d.patient_id = s.patient_id)
   AND d.monitor_id = s.monitor_id
   AND d.timestamp BETWEEN s.source_chunk_start_ts_utc AND s.source_chunk_end_ts_utc
   AND d.timestamp >= TIMESTAMP_SUB(s.anchor_ts_utc, INTERVAL 120 SECOND)
   AND d.timestamp < s.anchor_ts_utc
)
SELECT
  fall_event_id,
  window_role,
  anchor_ts_utc,
  hospital_id,
  division_id,
  patient_id,
  monitor_id,
  source_chunk_id,
  source_chunk_start_ts_utc,
  source_chunk_end_ts_utc,
  source_monitor_id,
  control_rank,
  match_tier,
  matching_candidates,
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
