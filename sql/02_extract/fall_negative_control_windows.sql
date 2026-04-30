-- Matched non-fall negative-control windows for per-second stress testing.
-- Controls are drawn from the pre-windowed non-fall derivatives cohort.
-- Matching requires same division and prefers same local hour-of-day, then same local daypart,
-- then any remaining chunk in the same division. Up to {{negative_control_match_limit}} controls
-- are retained per fall and averaged downstream.
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
anchor_proximity AS (
  SELECT
    s.fall_event_id,
    COUNTIF(ABS(TIMESTAMP_DIFF(f.fall_ts_utc, s.anchor_ts_utc, SECOND)) <= 1800) AS nearby_fall_count_30m
  FROM selected s
  LEFT JOIN falls f
    ON f.hospital_id = s.hospital_id
   AND f.division_id = s.division_id
   AND f.monitor_id = s.monitor_id
  GROUP BY s.fall_event_id
),
frames AS (
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
    COALESCE(
      (
        SELECT AS STRUCT
          LOWER(NULLIF(TRIM(p.posture), "")) AS posture_label,
          p.score_sitting AS score_sitting,
          p.score_standing AS score_standing,
          p.score_lying AS score_lying
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
          p.score_lying AS score_lying
        FROM UNNEST(d.all_deduped_patients) AS p
        ORDER BY
          COALESCE(p.score_patient, 0.0) DESC,
          COALESCE(p.score, 0.0) DESC,
          IF(COALESCE(p.has_tracked, FALSE), 1, 0) DESC
        LIMIT 1
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
),
scored AS (
  SELECT
    *,
    (
      patient_chair_distance IS NOT NULL
      OR patient_bed_distance IS NOT NULL
      OR patient_room_distance IS NOT NULL
    ) AS frame_has_location_signal,
    (
      primary_patient_posture.posture_label IS NOT NULL
      OR primary_patient_posture.score_sitting IS NOT NULL
      OR primary_patient_posture.score_standing IS NOT NULL
      OR primary_patient_posture.score_lying IS NOT NULL
    ) AS frame_has_posture_signal,
    primary_patient_posture.posture_label AS primary_patient_posture_label,
    primary_patient_posture.score_sitting AS primary_patient_posture_score_sitting,
    primary_patient_posture.score_standing AS primary_patient_posture_score_standing,
    primary_patient_posture.score_lying AS primary_patient_posture_score_lying,
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
    END AS dominant_location_label
  FROM frames
),
with_switch AS (
  SELECT
    *,
    LAG(dominant_location_label) OVER (
      PARTITION BY fall_event_id, window_role, control_rank
      ORDER BY frame_ts_utc
    ) AS prev_dominant_location_label,
    LAST_VALUE(
      IF(frame_has_posture_signal, primary_patient_posture_label, NULL) IGNORE NULLS
    ) OVER (
      PARTITION BY fall_event_id, window_role, control_rank
      ORDER BY frame_ts_utc
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS prev_observed_posture_label
  FROM scored
),
aggregated AS (
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
    COUNT(frame_ts_utc) AS frame_count,
    COUNTIF(frame_has_location_signal) AS visible_frame_count,
    SAFE_DIVIDE(COUNTIF(frame_has_location_signal), NULLIF(COUNT(frame_ts_utc), 0)) AS visibility_ratio,
    SAFE_DIVIDE(COUNTIF(dominant_location_label = "chair"), NULLIF(COUNT(frame_ts_utc), 0)) AS chair_share,
    SAFE_DIVIDE(COUNTIF(dominant_location_label = "bed"), NULLIF(COUNT(frame_ts_utc), 0)) AS bed_share,
    SAFE_DIVIDE(COUNTIF(dominant_location_label = "room"), NULLIF(COUNT(frame_ts_utc), 0)) AS room_share,
    SAFE_DIVIDE(COUNTIF(dominant_location_label = "no_patient"), NULLIF(COUNT(frame_ts_utc), 0)) AS no_patient_share,
    COUNTIF(frame_has_posture_signal) AS posture_observed_frame_count,
    AVG(IF(frame_has_posture_signal, primary_patient_posture_score_sitting, NULL))
      AS posture_sitting_score_mean,
    AVG(IF(frame_has_posture_signal, primary_patient_posture_score_standing, NULL))
      AS posture_standing_score_mean,
    AVG(IF(frame_has_posture_signal, primary_patient_posture_score_lying, NULL))
      AS posture_lying_score_mean,
    SAFE_DIVIDE(
      COUNTIF(primary_patient_posture_label = "sitting"),
      NULLIF(COUNTIF(frame_has_posture_signal), 0)
    ) AS posture_sitting_share,
    SAFE_DIVIDE(
      COUNTIF(primary_patient_posture_label = "standing"),
      NULLIF(COUNTIF(frame_has_posture_signal), 0)
    ) AS posture_standing_share,
    SAFE_DIVIDE(
      COUNTIF(primary_patient_posture_label = "lying"),
      NULLIF(COUNTIF(frame_has_posture_signal), 0)
    ) AS posture_lying_share,
    AVG(patient_chair_distance) AS mean_patient_chair_distance,
    AVG(patient_bed_distance) AS mean_patient_bed_distance,
    AVG(patient_room_distance) AS mean_patient_room_distance,
    AVG(patient_staff_iou) AS mean_patient_staff_iou,
    MAX(patient_staff_iou) AS max_patient_staff_iou,
    COUNTIF(
      prev_dominant_location_label IS NOT NULL
      AND prev_dominant_location_label != dominant_location_label
    ) AS dominant_switch_count,
    COUNTIF(
      frame_has_posture_signal
      AND prev_observed_posture_label IS NOT NULL
      AND primary_patient_posture_label IS NOT NULL
      AND prev_observed_posture_label != primary_patient_posture_label
    ) AS posture_switch_count
  FROM with_switch
  GROUP BY
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
    matching_candidates
)
SELECT
  a.*,
  COALESCE(p.nearby_fall_count_30m, 0) AS nearby_fall_count_30m,
  (COALESCE(p.nearby_fall_count_30m, 0) = 0 AND a.frame_count > 0) AS eligible_window
FROM aggregated a
LEFT JOIN anchor_proximity p
  ON p.fall_event_id = a.fall_event_id;
