-- Event-local case-crossover windows using the hazard timestamp plus configured same-monitor day offsets.
-- Hazard/control windows use the 120 seconds before each anchor timestamp.
WITH falls AS (
  SELECT
    ROW_NUMBER() OVER (
      ORDER BY timestamp, hospital_id, division_id, patient_id, monitor_id, summary
    ) AS fall_event_id,
    timestamp AS fall_ts_utc,
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
    hospital_id,
    division_id,
    patient_id,
    monitor_id
  FROM falls
{{fall_case_crossover_control_unions_sql}}
),
anchor_proximity AS (
  SELECT
    a.fall_event_id,
    a.window_role,
    COUNTIF(
      ABS(TIMESTAMP_DIFF(f2.fall_ts_utc, a.anchor_ts_utc, SECOND)) <= 1800
      AND (a.window_role != "hazard" OR f2.fall_event_id != a.fall_event_id)
    ) AS nearby_fall_count_30m
  FROM anchors a
  LEFT JOIN falls f2
    ON f2.hospital_id = a.hospital_id
   AND f2.division_id = a.division_id
   AND f2.monitor_id = a.monitor_id
   AND ABS(TIMESTAMP_DIFF(f2.fall_ts_utc, a.anchor_ts_utc, SECOND)) <= 1800
  GROUP BY a.fall_event_id, a.window_role
),
frames AS (
  SELECT
    a.fall_event_id,
    a.window_role,
    a.anchor_ts_utc,
    a.hospital_id,
    a.division_id,
    a.patient_id,
    a.monitor_id,
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
  FROM anchors a
  LEFT JOIN `{{project}}.{{dataset}}.{{live_stream_derivatives_table}}` d
    ON d.hospital_id = a.hospital_id
   AND d.division_id = a.division_id
   AND (a.patient_id IS NULL OR d.patient_id = a.patient_id)
   AND d.monitor_id = a.monitor_id
   AND d.timestamp >= TIMESTAMP_SUB(a.anchor_ts_utc, INTERVAL 120 SECOND)
   AND d.timestamp < a.anchor_ts_utc
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
      PARTITION BY fall_event_id, window_role
      ORDER BY frame_ts_utc
    ) AS prev_dominant_location_label,
    LAST_VALUE(
      IF(frame_has_posture_signal, primary_patient_posture_label, NULL) IGNORE NULLS
    ) OVER (
      PARTITION BY fall_event_id, window_role
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
    MIN(patient_chair_distance) AS min_patient_chair_distance,
    MIN(patient_bed_distance) AS min_patient_bed_distance,
    MIN(patient_room_distance) AS min_patient_room_distance,
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
    monitor_id
)
SELECT
  a.*,
  p.nearby_fall_count_30m,
  (
    a.window_role = "hazard"
    OR (a.window_role IN ({{fall_case_crossover_control_roles_sql}}) AND p.nearby_fall_count_30m = 0)
  ) AS eligible_window
FROM aggregated a
LEFT JOIN anchor_proximity p
  ON p.fall_event_id = a.fall_event_id
 AND p.window_role = a.window_role;
