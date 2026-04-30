-- Falls-only livestream event-window aggregation.
-- Produces one row per fall event with:
--   - probabilistic pre-fall location (chair/bed/room/no_patient)
--   - 30-second focus-window location weighting immediately before the fall
--   - 180-second dropout/reliability features immediately before the fall
--   - first post-fall patient-staff overlap timestamp (IOU > 0)
--   - event-level support counts for QA/diagnostics
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
    monitor_id,
    summary
  FROM {{fall_events_compatible_subquery_sql}}
  WHERE timestamp IS NOT NULL
    AND hospital_id IS NOT NULL
    AND hospital_id = {{study_hospital_id}}
    AND division_id IS NOT NULL
    AND monitor_id IS NOT NULL
),
joined AS (
  SELECT
    f.*,
    d.timestamp AS frame_ts_utc,
    d.timestamp < f.fall_ts_utc AS is_prefall,
    d.timestamp >= f.fall_ts_utc AS is_postfall,
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
  FROM falls f
  LEFT JOIN `{{project}}.{{dataset}}.{{live_stream_derivatives_table}}` d
    ON d.hospital_id = f.hospital_id
   AND d.division_id = f.division_id
   AND (f.patient_id IS NULL OR d.patient_id = f.patient_id)
   AND d.monitor_id = f.monitor_id
   AND d.timestamp BETWEEN TIMESTAMP_SUB(f.fall_ts_utc, INTERVAL 15 MINUTE)
                       AND TIMESTAMP_ADD(f.fall_ts_utc, INTERVAL 15 MINUTE)
),
frame_features AS (
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
    IFNULL(1.0 / GREATEST(patient_chair_distance, 1e-6), 0.0) AS frame_weight_chair,
    IFNULL(1.0 / GREATEST(patient_bed_distance, 1e-6), 0.0) AS frame_weight_bed,
    IFNULL(1.0 / GREATEST(patient_room_distance, 1e-6), 0.0) AS frame_weight_room,
    (
      is_prefall
      AND frame_ts_utc IS NOT NULL
      AND frame_ts_utc >= TIMESTAMP_SUB(fall_ts_utc, INTERVAL 30 SECOND)
      AND frame_ts_utc < fall_ts_utc
    ) AS is_prefall_focus_frame,
    (
      is_prefall
      AND frame_ts_utc IS NOT NULL
      AND frame_ts_utc >= TIMESTAMP_SUB(fall_ts_utc, INTERVAL 180 SECOND)
      AND frame_ts_utc < fall_ts_utc
    ) AS is_prefall_dropout_frame,
    IF(
      frame_ts_utc IS NULL,
      NULL,
      TIMESTAMP_SECONDS(30 * DIV(UNIX_SECONDS(frame_ts_utc), 30))
    ) AS frame_ts_30s
  FROM joined
),
with_posture_switch AS (
  SELECT
    *,
    LAST_VALUE(
      IF(frame_has_posture_signal, primary_patient_posture_label, NULL) IGNORE NULLS
    ) OVER (
      PARTITION BY fall_event_id
      ORDER BY frame_ts_utc
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS prev_observed_posture_label
  FROM frame_features
),
agg AS (
  SELECT
    fall_event_id,
    fall_ts_utc,
    fall_ts_local,
    hospital_id,
    division_id,
    patient_id,
    monitor_id,
    summary,
    COUNTIF(frame_ts_utc IS NOT NULL AND is_prefall) AS pre_window_frames,
    COUNTIF(frame_ts_utc IS NOT NULL AND is_postfall) AS post_window_frames,
    COUNTIF(is_prefall_focus_frame) AS pre_focus_frames,
    COUNTIF(is_prefall_focus_frame AND frame_has_location_signal) AS pre_focus_visible_frames,
    COUNT(DISTINCT IF(is_prefall_focus_frame, frame_ts_30s, NULL)) AS pre_focus_bin_count,
    COUNT(DISTINCT IF(is_prefall_focus_frame AND frame_has_location_signal, frame_ts_30s, NULL))
      AS pre_focus_visible_bin_count,
    COUNTIF(is_prefall_dropout_frame) AS pre_dropout_frames,
    COUNTIF(is_prefall_dropout_frame AND frame_has_location_signal) AS pre_dropout_visible_frames,
    COUNT(DISTINCT IF(is_prefall_dropout_frame, frame_ts_30s, NULL)) AS pre_dropout_bin_count,
    COUNT(DISTINCT IF(is_prefall_dropout_frame AND frame_has_location_signal, frame_ts_30s, NULL))
      AS pre_dropout_visible_bin_count,
    MIN(IF(is_prefall_focus_frame, patient_chair_distance, NULL)) AS pre_focus_min_patient_chair_distance,
    MIN(IF(is_prefall_focus_frame, patient_bed_distance, NULL)) AS pre_focus_min_patient_bed_distance,
    MIN(IF(is_prefall_focus_frame, patient_room_distance, NULL)) AS pre_focus_min_patient_room_distance,
    MAX(IF(is_prefall_dropout_frame AND frame_has_location_signal, frame_ts_utc, NULL))
      AS pre_dropout_last_visible_ts_utc,
    SUM(IF(is_prefall_focus_frame, frame_weight_chair, 0.0)) AS pre_focus_weight_chair,
    SUM(IF(is_prefall_focus_frame, frame_weight_bed, 0.0)) AS pre_focus_weight_bed,
    SUM(IF(is_prefall_focus_frame, frame_weight_room, 0.0)) AS pre_focus_weight_room,
    COUNTIF(is_prefall_dropout_frame AND frame_has_posture_signal) AS pre_posture_observed_frame_count,
    AVG(IF(is_prefall_dropout_frame AND frame_has_posture_signal, primary_patient_posture_score_sitting, NULL))
      AS pre_posture_sitting_score_mean,
    AVG(IF(is_prefall_dropout_frame AND frame_has_posture_signal, primary_patient_posture_score_standing, NULL))
      AS pre_posture_standing_score_mean,
    AVG(IF(is_prefall_dropout_frame AND frame_has_posture_signal, primary_patient_posture_score_lying, NULL))
      AS pre_posture_lying_score_mean,
    SAFE_DIVIDE(
      COUNTIF(is_prefall_dropout_frame AND primary_patient_posture_label = "sitting"),
      NULLIF(COUNTIF(is_prefall_dropout_frame AND frame_has_posture_signal), 0)
    ) AS pre_posture_sitting_share,
    SAFE_DIVIDE(
      COUNTIF(is_prefall_dropout_frame AND primary_patient_posture_label = "standing"),
      NULLIF(COUNTIF(is_prefall_dropout_frame AND frame_has_posture_signal), 0)
    ) AS pre_posture_standing_share,
    SAFE_DIVIDE(
      COUNTIF(is_prefall_dropout_frame AND primary_patient_posture_label = "lying"),
      NULLIF(COUNTIF(is_prefall_dropout_frame AND frame_has_posture_signal), 0)
    ) AS pre_posture_lying_share,
    COUNTIF(
      is_prefall_dropout_frame
      AND frame_has_posture_signal
      AND prev_observed_posture_label IS NOT NULL
      AND primary_patient_posture_label IS NOT NULL
      AND prev_observed_posture_label != primary_patient_posture_label
    ) AS pre_posture_switch_count,
    MIN(IF(is_postfall AND patient_staff_iou > 0, frame_ts_utc, NULL)) AS first_staff_overlap_ts_utc,
    COUNTIF(is_postfall AND patient_staff_iou > 0) AS post_staff_overlap_frames
  FROM with_posture_switch
  GROUP BY
    fall_event_id,
    fall_ts_utc,
    fall_ts_local,
    hospital_id,
    division_id,
    patient_id,
    monitor_id,
    summary
),
scored AS (
  SELECT
    *,
    (
      IF(pre_focus_min_patient_chair_distance IS NULL, 0, 1)
      + IF(pre_focus_min_patient_bed_distance IS NULL, 0, 1)
      + IF(pre_focus_min_patient_room_distance IS NULL, 0, 1)
    ) AS pre_distance_signal_count,
    SAFE_DIVIDE(pre_dropout_visible_bin_count, NULLIF(pre_dropout_bin_count, 0)) AS pre_dropout_visibility_ratio,
    CASE
      WHEN pre_dropout_last_visible_ts_utc IS NULL THEN NULL
      ELSE TIMESTAMP_DIFF(fall_ts_utc, pre_dropout_last_visible_ts_utc, SECOND)
    END AS pre_dropout_last_visible_gap_seconds,
    CASE
      WHEN pre_dropout_last_visible_ts_utc IS NULL THEN 1.0
      ELSE LEAST(
        1.0,
        GREATEST(
          0.0,
          SAFE_DIVIDE(TIMESTAMP_DIFF(fall_ts_utc, pre_dropout_last_visible_ts_utc, SECOND), 180.0)
        )
      )
    END AS pre_dropout_gap_ratio,
    CASE
      WHEN pre_dropout_bin_count = 0 THEN 0.0
      ELSE 1.0 - SAFE_DIVIDE(pre_dropout_visible_bin_count, pre_dropout_bin_count)
    END AS pre_dropout_dropout_ratio,
    (pre_focus_weight_chair + pre_focus_weight_bed + pre_focus_weight_room) AS pre_focus_weight_sum,
    CASE
      WHEN pre_dropout_bin_count = 0 THEN 0.0
      WHEN pre_dropout_visible_bin_count = 0 THEN 0.0
      ELSE GREATEST(
        0.15,
        1.0 - LEAST(
          0.85,
          GREATEST(
            0.0,
            0.65 * (
              CASE
                WHEN pre_dropout_bin_count = 0 THEN 1.0
                ELSE 1.0 - SAFE_DIVIDE(pre_dropout_visible_bin_count, pre_dropout_bin_count)
              END
            )
            + 0.35 * (
              CASE
                WHEN pre_dropout_last_visible_ts_utc IS NULL THEN 1.0
                ELSE LEAST(
                  1.0,
                  GREATEST(
                    0.0,
                    SAFE_DIVIDE(TIMESTAMP_DIFF(fall_ts_utc, pre_dropout_last_visible_ts_utc, SECOND), 180.0)
                  )
                )
              END
            )
          )
        )
      )
    END AS pre_state_visible_prob,
    CASE
      WHEN pre_dropout_bin_count = 0 THEN 1.0
      ELSE 0.0
    END AS pre_state_unknown_prob
  FROM agg
),
probs AS (
  SELECT
    *,
    IF(
      pre_state_visible_prob > 0 AND pre_focus_weight_sum > 0,
      pre_focus_weight_chair / pre_focus_weight_sum,
      0.0
    ) AS pre_cond_prob_chair,
    IF(
      pre_state_visible_prob > 0 AND pre_focus_weight_sum > 0,
      pre_focus_weight_bed / pre_focus_weight_sum,
      0.0
    ) AS pre_cond_prob_bed,
    IF(
      pre_state_visible_prob > 0 AND pre_focus_weight_sum > 0,
      pre_focus_weight_room / pre_focus_weight_sum,
      0.0
    ) AS pre_cond_prob_room,
    CASE
      WHEN pre_dropout_bin_count = 0 THEN 0.0
      WHEN pre_dropout_visible_bin_count = 0 THEN 0.5
      ELSE LEAST(
        0.90,
        GREATEST(
          0.10,
          0.30
          + 0.55 * IF(pre_focus_weight_sum > 0, pre_focus_weight_room / pre_focus_weight_sum, 0.0)
          + 0.15 * pre_dropout_dropout_ratio
        )
      )
    END AS pre_out_of_room_share
  FROM scored
),
final_probs AS (
  SELECT
    *,
    CASE
      WHEN pre_dropout_bin_count = 0 THEN 0.0
      ELSE (1.0 - pre_state_visible_prob - pre_state_unknown_prob)
    END AS pre_missing_mass,
    CASE
      WHEN pre_dropout_bin_count = 0 THEN 0.0
      ELSE (1.0 - pre_state_visible_prob - pre_state_unknown_prob) * (1.0 - pre_out_of_room_share)
    END AS pre_state_not_visible_prob,
    CASE
      WHEN pre_dropout_bin_count = 0 THEN 0.0
      ELSE (1.0 - pre_state_visible_prob - pre_state_unknown_prob) * pre_out_of_room_share
    END AS pre_state_out_of_room_prob
  FROM probs
)
SELECT
  *,
  (pre_state_visible_prob * pre_cond_prob_chair) AS pre_prob_chair,
  (pre_state_visible_prob * pre_cond_prob_bed) AS pre_prob_bed,
  (pre_state_visible_prob * pre_cond_prob_room) AS pre_prob_room,
  (pre_state_not_visible_prob + pre_state_out_of_room_prob + pre_state_unknown_prob) AS pre_prob_no_patient,
  pre_state_not_visible_prob AS pre_prob_not_visible,
  pre_state_out_of_room_prob AS pre_prob_out_of_room,
  pre_state_unknown_prob AS pre_prob_unknown,
  CASE
    WHEN (
      pre_state_not_visible_prob + pre_state_out_of_room_prob + pre_state_unknown_prob
    ) >= GREATEST(
      pre_state_visible_prob * pre_cond_prob_chair,
      pre_state_visible_prob * pre_cond_prob_bed,
      pre_state_visible_prob * pre_cond_prob_room
    ) THEN "no_patient"
    WHEN (pre_state_visible_prob * pre_cond_prob_chair) >= GREATEST(
      pre_state_visible_prob * pre_cond_prob_bed,
      pre_state_visible_prob * pre_cond_prob_room,
      pre_state_not_visible_prob + pre_state_out_of_room_prob + pre_state_unknown_prob
    ) THEN "chair"
    WHEN (pre_state_visible_prob * pre_cond_prob_bed) >= GREATEST(
      pre_state_visible_prob * pre_cond_prob_room,
      pre_state_not_visible_prob + pre_state_out_of_room_prob + pre_state_unknown_prob
    ) THEN "bed"
    ELSE "room"
  END AS prefall_location_label,
  CASE
    WHEN (
      pre_state_not_visible_prob + pre_state_out_of_room_prob + pre_state_unknown_prob
    ) >= GREATEST(
      pre_state_visible_prob * pre_cond_prob_chair,
      pre_state_visible_prob * pre_cond_prob_bed,
      pre_state_visible_prob * pre_cond_prob_room
    ) THEN (
      pre_state_not_visible_prob + pre_state_out_of_room_prob + pre_state_unknown_prob
    ) - GREATEST(
      pre_state_visible_prob * pre_cond_prob_chair,
      pre_state_visible_prob * pre_cond_prob_bed,
      pre_state_visible_prob * pre_cond_prob_room
    )
    WHEN (pre_state_visible_prob * pre_cond_prob_chair) >= GREATEST(
      pre_state_visible_prob * pre_cond_prob_bed,
      pre_state_visible_prob * pre_cond_prob_room,
      pre_state_not_visible_prob + pre_state_out_of_room_prob + pre_state_unknown_prob
    ) THEN (pre_state_visible_prob * pre_cond_prob_chair) - GREATEST(
      pre_state_visible_prob * pre_cond_prob_bed,
      pre_state_visible_prob * pre_cond_prob_room,
      pre_state_not_visible_prob + pre_state_out_of_room_prob + pre_state_unknown_prob
    )
    WHEN (pre_state_visible_prob * pre_cond_prob_bed) >= GREATEST(
      pre_state_visible_prob * pre_cond_prob_room,
      pre_state_not_visible_prob + pre_state_out_of_room_prob + pre_state_unknown_prob
    ) THEN (pre_state_visible_prob * pre_cond_prob_bed) - GREATEST(
      pre_state_visible_prob * pre_cond_prob_chair,
      pre_state_visible_prob * pre_cond_prob_room,
      pre_state_not_visible_prob + pre_state_out_of_room_prob + pre_state_unknown_prob
    )
    ELSE (pre_state_visible_prob * pre_cond_prob_room) - GREATEST(
      pre_state_visible_prob * pre_cond_prob_chair,
      pre_state_visible_prob * pre_cond_prob_bed,
      pre_state_not_visible_prob + pre_state_out_of_room_prob + pre_state_unknown_prob
    )
  END AS prefall_location_margin,
  first_staff_overlap_ts_utc IS NOT NULL AS response_detected,
  IF(
    first_staff_overlap_ts_utc IS NULL,
    NULL,
    TIMESTAMP_DIFF(first_staff_overlap_ts_utc, fall_ts_utc, SECOND)
  ) AS response_latency_seconds
FROM final_probs;
