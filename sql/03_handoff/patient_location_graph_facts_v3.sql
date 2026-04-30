-- Upstream patient-location graph-facts v3 handoff query.
--
-- This file is intentionally not wired into the current chair-falls extraction
-- pipeline. The actual upstream graph-facts generator is not part of this repo.
-- Treat this as a concrete BigQuery implementation artifact for the SQL owner.
-- A readable Python companion and checked-in validation example live beside
-- this file at `patient_location_graph_facts_v3_reference.py` and
-- `patient_location_graph_facts_v3_reference_example.json`.
-- The current query assumes each scored row has prior history available inside
-- the 180-second lookback window; warmup-row handling should be added upstream
-- if those earliest rows must be retained.
--
-- Replace the `source` CTE below with the real upstream second-level table or
-- view before execution. Expected source columns:
--   Keys / time:
--     hospital_id, division_id, monitor_id, patient_id, ts_local
--   Base four-class scorer:
--     base_prob_chair, base_prob_bed, base_prob_room, base_prob_no_patient
--   Visibility regime inputs:
--     state_visible_prob, state_not_visible_prob, state_out_of_room_prob,
--     dropout_visibility_ratio, prob_not_visible, prob_out_of_room,
--     out_of_room_share, pre_distance_signal_count
--   Per-frame location evidence:
--     frame_has_location_signal, dominant_location_label,
--     patient_chair_distance, patient_bed_distance, patient_room_distance
--   Additional evidence:
--     patient_staff_iou, nudge_score, bed_in_bed_score, motion_bac,
--     patient_candidate_count, staff_candidate_count, other_candidate_count,
--     bed_candidate_count, chair_candidate_count
--   Posture:
--     primary_patient_posture_label,
--     primary_patient_posture_score_sitting,
--     primary_patient_posture_score_standing,
--     primary_patient_posture_score_lying
--
-- Public output contract:
--   hospital_id, division_id, monitor_id, patient_id, hour_ts,
--   pct_chair, pct_bed, pct_ambulatory, pct_not_located

CREATE TEMP FUNCTION clamp01(x FLOAT64)
AS (LEAST(1.0, GREATEST(0.0, COALESCE(x, 0.0))));

CREATE TEMP FUNCTION safe_ratio(numerator FLOAT64, denominator FLOAT64)
AS (COALESCE(SAFE_DIVIDE(numerator, NULLIF(denominator, 0.0)), 0.0));

CREATE TEMP FUNCTION scaled_support(value FLOAT64, scale FLOAT64)
AS (
  CASE
    WHEN scale <= 0.0 THEN 0.0
    ELSE clamp01(safe_ratio(value, scale))
  END
);

CREATE TEMP FUNCTION positive_margin_score(margin FLOAT64)
AS (
  CASE
    WHEN margin IS NULL OR margin <= 0.0 THEN 0.0
    ELSE clamp01(safe_ratio(margin, margin + 0.05))
  END
);

CREATE TEMP FUNCTION normalize_visible(
  chair FLOAT64,
  bed FLOAT64,
  room FLOAT64
)
RETURNS STRUCT<chair FLOAT64, bed FLOAT64, room FLOAT64>
AS ((
  WITH clean AS (
    SELECT
      GREATEST(0.0, COALESCE(chair, 0.0)) AS chair_value,
      GREATEST(0.0, COALESCE(bed, 0.0)) AS bed_value,
      GREATEST(0.0, COALESCE(room, 0.0)) AS room_value
  ),
  totals AS (
    SELECT
      chair_value,
      bed_value,
      room_value,
      chair_value + bed_value + room_value AS total_value
    FROM clean
  )
  SELECT AS STRUCT
    CASE WHEN total_value > 0.0 THEN chair_value / total_value ELSE 0.0 END AS chair,
    CASE WHEN total_value > 0.0 THEN bed_value / total_value ELSE 0.0 END AS bed,
    CASE WHEN total_value > 0.0 THEN room_value / total_value ELSE 0.0 END AS room
  FROM totals
));

CREATE TEMP FUNCTION normalize_posture(
  sitting FLOAT64,
  standing FLOAT64,
  lying FLOAT64
)
RETURNS STRUCT<sitting FLOAT64, standing FLOAT64, lying FLOAT64>
AS ((
  WITH clean AS (
    SELECT
      GREATEST(0.0, COALESCE(sitting, 0.0)) AS sitting_value,
      GREATEST(0.0, COALESCE(standing, 0.0)) AS standing_value,
      GREATEST(0.0, COALESCE(lying, 0.0)) AS lying_value
  ),
  totals AS (
    SELECT
      sitting_value,
      standing_value,
      lying_value,
      sitting_value + standing_value + lying_value AS total_value
    FROM clean
  )
  SELECT AS STRUCT
    CASE WHEN total_value > 0.0 THEN sitting_value / total_value ELSE 0.0 END AS sitting,
    CASE WHEN total_value > 0.0 THEN standing_value / total_value ELSE 0.0 END AS standing,
    CASE WHEN total_value > 0.0 THEN lying_value / total_value ELSE 0.0 END AS lying
  FROM totals
));

CREATE TEMP FUNCTION normalize_probs(
  chair FLOAT64,
  bed FLOAT64,
  room FLOAT64,
  no_patient FLOAT64
)
RETURNS STRUCT<chair FLOAT64, bed FLOAT64, room FLOAT64, no_patient FLOAT64>
AS ((
  WITH clean AS (
    SELECT
      GREATEST(0.0, COALESCE(chair, 0.0)) AS chair_value,
      GREATEST(0.0, COALESCE(bed, 0.0)) AS bed_value,
      GREATEST(0.0, COALESCE(room, 0.0)) AS room_value,
      GREATEST(0.0, COALESCE(no_patient, 0.0)) AS no_patient_value
  ),
  totals AS (
    SELECT
      chair_value,
      bed_value,
      room_value,
      no_patient_value,
      chair_value + bed_value + room_value + no_patient_value AS total_value
    FROM clean
  )
  SELECT AS STRUCT
    CASE WHEN total_value > 0.0 THEN chair_value / total_value ELSE 0.0 END AS chair,
    CASE WHEN total_value > 0.0 THEN bed_value / total_value ELSE 0.0 END AS bed,
    CASE WHEN total_value > 0.0 THEN room_value / total_value ELSE 0.0 END AS room,
    CASE WHEN total_value > 0.0 THEN no_patient_value / total_value ELSE 0.0 END AS no_patient
  FROM totals
));

CREATE TEMP FUNCTION dominant_visible_label(
  chair FLOAT64,
  bed FLOAT64,
  room FLOAT64,
  min_confidence FLOAT64
)
RETURNS STRING
AS ((
  WITH normalized AS (
    SELECT normalize_visible(chair, bed, room) AS probs
  ),
  winner AS (
    SELECT
      CASE
        WHEN probs.chair >= probs.bed AND probs.chair >= probs.room THEN 'chair'
        WHEN probs.bed >= probs.room THEN 'bed'
        ELSE 'room'
      END AS label,
      GREATEST(probs.chair, probs.bed, probs.room) AS confidence
    FROM normalized
  )
  SELECT
    CASE
      WHEN confidence < min_confidence THEN 'none'
      ELSE label
    END
  FROM winner
));

CREATE TEMP FUNCTION dominant_posture_label(
  score_sitting FLOAT64,
  score_standing FLOAT64,
  score_lying FLOAT64,
  share_sitting FLOAT64,
  share_standing FLOAT64,
  share_lying FLOAT64,
  min_confidence FLOAT64
)
RETURNS STRING
AS ((
  WITH score_norm AS (
    SELECT normalize_posture(score_sitting, score_standing, score_lying) AS score_probs
  ),
  share_norm AS (
    SELECT normalize_posture(share_sitting, share_standing, share_lying) AS share_probs
  ),
  chosen AS (
    SELECT
      CASE
        WHEN score_probs.sitting + score_probs.standing + score_probs.lying > 0.0
          THEN score_probs
        ELSE share_probs
      END AS probs
    FROM score_norm
    CROSS JOIN share_norm
  ),
  winner AS (
    SELECT
      CASE
        WHEN probs.sitting >= probs.standing AND probs.sitting >= probs.lying THEN 'sitting'
        WHEN probs.standing >= probs.lying THEN 'standing'
        ELSE 'lying'
      END AS label,
      GREATEST(probs.sitting, probs.standing, probs.lying) AS confidence
    FROM chosen
  )
  SELECT
    CASE
      WHEN confidence < min_confidence THEN 'none'
      ELSE label
    END
  FROM winner
));

CREATE TEMP FUNCTION shift_probability_to(
  chair FLOAT64,
  bed FLOAT64,
  room FLOAT64,
  no_patient FLOAT64,
  target_label STRING,
  desired_probability FLOAT64
)
RETURNS STRUCT<chair FLOAT64, bed FLOAT64, room FLOAT64, no_patient FLOAT64>
AS ((
  WITH normalized AS (
    SELECT normalize_probs(chair, bed, room, no_patient) AS probs
  ),
  target AS (
    SELECT
      probs,
      LEAST(0.95, GREATEST(0.0, desired_probability)) AS target_probability,
      CASE target_label
        WHEN 'chair' THEN probs.chair
        WHEN 'bed' THEN probs.bed
        WHEN 'room' THEN probs.room
        ELSE probs.no_patient
      END AS current_probability
    FROM normalized
  ),
  needed AS (
    SELECT
      probs,
      target_probability,
      CASE
        WHEN current_probability >= target_probability THEN 0.0
        ELSE target_probability - current_probability
      END AS needed_probability
    FROM target
  ),
  donor_rows AS (
    SELECT
      label,
      probability,
      CASE WHEN label = 'no_patient' THEN 0.55 ELSE 0.45 END AS donor_cap
    FROM needed,
    UNNEST([
      STRUCT('chair' AS label, probs.chair AS probability),
      STRUCT('bed' AS label, probs.bed AS probability),
      STRUCT('room' AS label, probs.room AS probability),
      STRUCT('no_patient' AS label, probs.no_patient AS probability)
    ])
    WHERE label != target_label
  ),
  ordered_donors AS (
    SELECT
      label,
      probability,
      donor_cap,
      probability * donor_cap AS capped_probability,
      SUM(probability * donor_cap) OVER (
        ORDER BY probability DESC, label
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
      ) AS capped_before
    FROM donor_rows
  ),
  donor_transfers AS (
    SELECT
      label,
      LEAST(
        capped_probability,
        GREATEST(0.0, (SELECT needed_probability FROM needed) - COALESCE(capped_before, 0.0))
      ) AS transferred_probability
    FROM ordered_donors
  ),
  rebalanced AS (
    SELECT
      CASE
        WHEN target_label = 'chair'
          THEN (SELECT probs.chair FROM needed) + COALESCE(SUM(transferred_probability), 0.0)
        ELSE (SELECT probs.chair FROM needed) - COALESCE(SUM(IF(label = 'chair', transferred_probability, 0.0)), 0.0)
      END AS chair_value,
      CASE
        WHEN target_label = 'bed'
          THEN (SELECT probs.bed FROM needed) + COALESCE(SUM(transferred_probability), 0.0)
        ELSE (SELECT probs.bed FROM needed) - COALESCE(SUM(IF(label = 'bed', transferred_probability, 0.0)), 0.0)
      END AS bed_value,
      CASE
        WHEN target_label = 'room'
          THEN (SELECT probs.room FROM needed) + COALESCE(SUM(transferred_probability), 0.0)
        ELSE (SELECT probs.room FROM needed) - COALESCE(SUM(IF(label = 'room', transferred_probability, 0.0)), 0.0)
      END AS room_value,
      CASE
        WHEN target_label = 'no_patient'
          THEN (SELECT probs.no_patient FROM needed) + COALESCE(SUM(transferred_probability), 0.0)
        ELSE (SELECT probs.no_patient FROM needed) - COALESCE(SUM(IF(label = 'no_patient', transferred_probability, 0.0)), 0.0)
      END AS no_patient_value
    FROM donor_transfers
  )
  SELECT
    CASE
      WHEN (SELECT needed_probability FROM needed) <= 0.0 THEN (SELECT probs FROM needed)
      ELSE normalize_probs(chair_value, bed_value, room_value, no_patient_value)
    END
  FROM rebalanced
));

CREATE TEMP FUNCTION set_no_patient_probability(
  chair FLOAT64,
  bed FLOAT64,
  room FLOAT64,
  no_patient FLOAT64,
  target_no_patient FLOAT64,
  visible_prior_chair FLOAT64,
  visible_prior_bed FLOAT64,
  visible_prior_room FLOAT64
)
RETURNS STRUCT<chair FLOAT64, bed FLOAT64, room FLOAT64, no_patient FLOAT64>
AS ((
  WITH normalized AS (
    SELECT normalize_probs(chair, bed, room, no_patient) AS probs
  ),
  target AS (
    SELECT
      probs,
      LEAST(0.95, GREATEST(0.0, target_no_patient)) AS desired_no_patient
    FROM normalized
  ),
  visible_prior AS (
    SELECT normalize_visible(visible_prior_chair, visible_prior_bed, visible_prior_room) AS visible_probs
  ),
  decreased AS (
    SELECT normalize_probs(
      probs.chair + (probs.no_patient - desired_no_patient) * visible_probs.chair,
      probs.bed + (probs.no_patient - desired_no_patient) * visible_probs.bed,
      probs.room + (probs.no_patient - desired_no_patient) * visible_probs.room,
      desired_no_patient
    ) AS adjusted_probs
    FROM target
    CROSS JOIN visible_prior
    WHERE desired_no_patient < probs.no_patient
  ),
  increased AS (
    SELECT normalize_probs(
      GREATEST(0.0, probs.chair - (desired_no_patient - probs.no_patient) * safe_ratio(probs.chair, probs.chair + probs.bed + probs.room)),
      GREATEST(0.0, probs.bed - (desired_no_patient - probs.no_patient) * safe_ratio(probs.bed, probs.chair + probs.bed + probs.room)),
      GREATEST(0.0, probs.room - (desired_no_patient - probs.no_patient) * safe_ratio(probs.room, probs.chair + probs.bed + probs.room)),
      desired_no_patient
    ) AS adjusted_probs
    FROM target
    WHERE desired_no_patient > probs.no_patient
  )
  SELECT
    COALESCE(
      (SELECT adjusted_probs FROM decreased),
      (SELECT adjusted_probs FROM increased),
      (SELECT probs FROM target)
    )
));

WITH
constants AS (
  SELECT
    0.45 AS anchor_weight_base,
    0.35 AS visible_no_patient_cap,
    0.50 AS transient_no_patient_cap,
    0.45 AS present_sparse_no_patient_cap,
    0.55 AS out_of_room_no_patient_floor,
    0.35 AS presence_min_support,
    0.35 AS departure_min_confidence,
    0.55 AS bed_in_bed_min_score,
    0.95 AS bed_in_bed_invisible_min_score,
    0.15 AS motion_presence_scale,
    0.50 AS room_object_mean_min,
    0.60 AS sparse_corroboration_min,
    6.0 AS posture_min_frames,
    2.0 AS posture_max_switch_count,
    0.60 AS posture_bed_min_lying_prob,
    0.20 AS posture_max_chair_bed_gap,
    0.10 AS velocity_tiebreak_margin,
    0.05 AS velocity_min_approach,
    45.0 AS recency_half_life_seconds,
    60.0 AS recent_visible_gap_seconds
),
source AS (
  SELECT
    hospital_id,
    division_id,
    monitor_id,
    patient_id,
    ts_local,
    TIMESTAMP_TRUNC(ts_local, HOUR) AS hour_ts,
    CAST(base_prob_chair AS FLOAT64) AS base_prob_chair,
    CAST(base_prob_bed AS FLOAT64) AS base_prob_bed,
    CAST(base_prob_room AS FLOAT64) AS base_prob_room,
    CAST(base_prob_no_patient AS FLOAT64) AS base_prob_no_patient,
    CAST(state_visible_prob AS FLOAT64) AS state_visible_prob,
    CAST(state_not_visible_prob AS FLOAT64) AS state_not_visible_prob,
    CAST(state_out_of_room_prob AS FLOAT64) AS state_out_of_room_prob,
    CAST(dropout_visibility_ratio AS FLOAT64) AS dropout_visibility_ratio,
    CAST(prob_not_visible AS FLOAT64) AS prob_not_visible,
    CAST(prob_out_of_room AS FLOAT64) AS prob_out_of_room,
    CAST(out_of_room_share AS FLOAT64) AS out_of_room_share,
    CAST(pre_distance_signal_count AS FLOAT64) AS pre_distance_signal_count,
    COALESCE(frame_has_location_signal, FALSE) AS frame_has_location_signal,
    LOWER(TRIM(COALESCE(dominant_location_label, 'no_patient'))) AS dominant_location_label,
    CAST(patient_chair_distance AS FLOAT64) AS patient_chair_distance,
    CAST(patient_bed_distance AS FLOAT64) AS patient_bed_distance,
    CAST(patient_room_distance AS FLOAT64) AS patient_room_distance,
    CAST(patient_staff_iou AS FLOAT64) AS patient_staff_iou,
    CASE
      WHEN nudge_score IS NULL THEN 0.0
      WHEN nudge_score > 1.0 AND nudge_score <= 100.0 THEN clamp01(nudge_score / 100.0)
      ELSE clamp01(CAST(nudge_score AS FLOAT64))
    END AS nudge_score_normalized,
    CAST(bed_in_bed_score AS FLOAT64) AS bed_in_bed_score,
    CAST(motion_bac AS FLOAT64) AS motion_bac,
    CAST(patient_candidate_count AS FLOAT64) AS patient_candidate_count,
    CAST(staff_candidate_count AS FLOAT64) AS staff_candidate_count,
    CAST(other_candidate_count AS FLOAT64) AS other_candidate_count,
    CAST(bed_candidate_count AS FLOAT64) AS bed_candidate_count,
    CAST(chair_candidate_count AS FLOAT64) AS chair_candidate_count,
    LOWER(TRIM(COALESCE(primary_patient_posture_label, ''))) AS primary_patient_posture_label,
    clamp01(CAST(primary_patient_posture_score_sitting AS FLOAT64)) AS posture_score_sitting_raw,
    clamp01(CAST(primary_patient_posture_score_standing AS FLOAT64)) AS posture_score_standing_raw,
    clamp01(CAST(primary_patient_posture_score_lying AS FLOAT64)) AS posture_score_lying_raw
  FROM `{{source_table}}`
),
source_normalized AS (
  SELECT
    *,
    normalize_probs(
      base_prob_chair,
      base_prob_bed,
      base_prob_room,
      base_prob_no_patient
    ) AS base_probs,
    normalize_posture(
      posture_score_sitting_raw,
      posture_score_standing_raw,
      posture_score_lying_raw
    ) AS posture_score_probs_frame,
    CASE
      WHEN base_prob_chair >= base_prob_bed
           AND base_prob_chair >= base_prob_room
           AND base_prob_chair >= base_prob_no_patient
        THEN 'chair'
      WHEN base_prob_bed >= base_prob_room
           AND base_prob_bed >= base_prob_no_patient
        THEN 'bed'
      WHEN base_prob_room >= base_prob_no_patient
        THEN 'room'
      ELSE 'no_patient'
    END AS base_label,
    (
      primary_patient_posture_label IN ('sitting', 'standing', 'lying')
      OR posture_score_sitting_raw > 0.0
      OR posture_score_standing_raw > 0.0
      OR posture_score_lying_raw > 0.0
    ) AS frame_has_posture_signal
  FROM source
),
panel_frames AS (
  SELECT
    score_row.hospital_id,
    score_row.division_id,
    score_row.monitor_id,
    score_row.patient_id,
    score_row.ts_local,
    score_row.hour_ts,
    history_row.ts_local AS history_ts_local,
    TIMESTAMP_DIFF(history_row.ts_local, score_row.ts_local, SECOND) AS relative_second,
    TIMESTAMP_DIFF(score_row.ts_local, history_row.ts_local, SECOND) AS lag_from_score_seconds,
    TIMESTAMP_DIFF(history_row.ts_local, score_row.ts_local, SECOND) BETWEEN -180 AND -61 AS in_anchor_window,
    TIMESTAMP_DIFF(history_row.ts_local, score_row.ts_local, SECOND) BETWEEN -60 AND -1 AS in_tail_window,
    POW(
      2.0,
      safe_ratio(TIMESTAMP_DIFF(history_row.ts_local, score_row.ts_local, SECOND), constants.recency_half_life_seconds)
    ) AS recency_weight,
    history_row.frame_has_location_signal,
    history_row.dominant_location_label,
    history_row.patient_chair_distance,
    history_row.patient_bed_distance,
    history_row.patient_room_distance,
    history_row.patient_staff_iou,
    history_row.nudge_score_normalized,
    history_row.bed_in_bed_score,
    history_row.motion_bac,
    history_row.patient_candidate_count,
    history_row.staff_candidate_count,
    history_row.other_candidate_count,
    history_row.bed_candidate_count,
    history_row.chair_candidate_count,
    history_row.frame_has_posture_signal,
    history_row.primary_patient_posture_label,
    history_row.posture_score_probs_frame.sitting AS posture_score_sitting,
    history_row.posture_score_probs_frame.standing AS posture_score_standing,
    history_row.posture_score_probs_frame.lying AS posture_score_lying
  FROM source_normalized AS score_row
  CROSS JOIN constants
  JOIN source_normalized AS history_row
    ON history_row.hospital_id = score_row.hospital_id
   AND history_row.division_id = score_row.division_id
   AND history_row.monitor_id = score_row.monitor_id
   AND history_row.patient_id = score_row.patient_id
   AND history_row.ts_local >= TIMESTAMP_SUB(score_row.ts_local, INTERVAL 180 SECOND)
   AND history_row.ts_local < score_row.ts_local
),
panel_with_velocity AS (
  SELECT
    *,
    CASE
      WHEN in_tail_window AND frame_has_location_signal THEN
        SAFE_DIVIDE(
          patient_chair_distance
            - LAG(patient_chair_distance) OVER (
              PARTITION BY hospital_id, division_id, monitor_id, patient_id, ts_local
              ORDER BY history_ts_local
            ),
          TIMESTAMP_DIFF(
            history_ts_local,
            LAG(history_ts_local) OVER (
              PARTITION BY hospital_id, division_id, monitor_id, patient_id, ts_local
              ORDER BY history_ts_local
            ),
            SECOND
          )
        )
      ELSE NULL
    END AS chair_distance_velocity,
    CASE
      WHEN in_tail_window AND frame_has_location_signal THEN
        SAFE_DIVIDE(
          patient_bed_distance
            - LAG(patient_bed_distance) OVER (
              PARTITION BY hospital_id, division_id, monitor_id, patient_id, ts_local
              ORDER BY history_ts_local
            ),
          TIMESTAMP_DIFF(
            history_ts_local,
            LAG(history_ts_local) OVER (
              PARTITION BY hospital_id, division_id, monitor_id, patient_id, ts_local
              ORDER BY history_ts_local
            ),
            SECOND
          )
        )
      ELSE NULL
    END AS bed_distance_velocity,
    CASE
      WHEN in_tail_window AND frame_has_location_signal THEN
        SAFE_DIVIDE(
          patient_room_distance
            - LAG(patient_room_distance) OVER (
              PARTITION BY hospital_id, division_id, monitor_id, patient_id, ts_local
              ORDER BY history_ts_local
            ),
          TIMESTAMP_DIFF(
            history_ts_local,
            LAG(history_ts_local) OVER (
              PARTITION BY hospital_id, division_id, monitor_id, patient_id, ts_local
              ORDER BY history_ts_local
            ),
            SECOND
          )
        )
      ELSE NULL
    END AS room_distance_velocity
  FROM panel_frames
),
panel_with_posture_switches AS (
  SELECT
    *,
    CASE
      WHEN frame_has_posture_signal
           AND primary_patient_posture_label IN ('sitting', 'standing', 'lying')
           AND LAG(primary_patient_posture_label) OVER (
             PARTITION BY hospital_id, division_id, monitor_id, patient_id, ts_local
             ORDER BY history_ts_local
           ) IN ('sitting', 'standing', 'lying')
           AND primary_patient_posture_label != LAG(primary_patient_posture_label) OVER (
             PARTITION BY hospital_id, division_id, monitor_id, patient_id, ts_local
             ORDER BY history_ts_local
           )
        THEN 1.0
      ELSE 0.0
    END AS posture_switch_flag
  FROM panel_with_velocity
),
panel_summary AS (
  SELECT
    panel.hospital_id,
    panel.division_id,
    panel.monitor_id,
    panel.patient_id,
    panel.ts_local,
    panel.hour_ts,
    COUNTIF(panel.in_anchor_window AND panel.frame_has_location_signal) AS anchor_visible_frames,
    COUNTIF(panel.in_tail_window AND panel.frame_has_location_signal) AS tail_visible_frames,
    ABS(MAX(IF(panel.frame_has_location_signal, panel.relative_second, NULL))) AS last_visible_gap_seconds,
    normalize_visible(
      SUM(IF(panel.in_anchor_window AND panel.frame_has_location_signal, panel.recency_weight / GREATEST(panel.patient_chair_distance, 1e-6), 0.0))
        + 0.25 * safe_ratio(
          COUNTIF(panel.in_anchor_window AND panel.frame_has_location_signal AND panel.dominant_location_label = 'chair'),
          COUNTIF(panel.in_anchor_window AND panel.frame_has_location_signal)
        ),
      SUM(IF(panel.in_anchor_window AND panel.frame_has_location_signal, panel.recency_weight / GREATEST(panel.patient_bed_distance, 1e-6), 0.0))
        + 0.25 * safe_ratio(
          COUNTIF(panel.in_anchor_window AND panel.frame_has_location_signal AND panel.dominant_location_label = 'bed'),
          COUNTIF(panel.in_anchor_window AND panel.frame_has_location_signal)
        ),
      SUM(IF(panel.in_anchor_window AND panel.frame_has_location_signal, panel.recency_weight / GREATEST(panel.patient_room_distance, 1e-6), 0.0))
        + 0.25 * safe_ratio(
          COUNTIF(panel.in_anchor_window AND panel.frame_has_location_signal AND panel.dominant_location_label = 'room'),
          COUNTIF(panel.in_anchor_window AND panel.frame_has_location_signal)
        )
    ) AS anchor_visible_probs,
    normalize_visible(
      SUM(IF(panel.in_tail_window AND panel.frame_has_location_signal, panel.recency_weight / GREATEST(panel.patient_chair_distance, 1e-6), 0.0))
        + 0.25 * safe_ratio(
          COUNTIF(panel.in_tail_window AND panel.frame_has_location_signal AND panel.dominant_location_label = 'chair'),
          COUNTIF(panel.in_tail_window AND panel.frame_has_location_signal)
        ),
      SUM(IF(panel.in_tail_window AND panel.frame_has_location_signal, panel.recency_weight / GREATEST(panel.patient_bed_distance, 1e-6), 0.0))
        + 0.25 * safe_ratio(
          COUNTIF(panel.in_tail_window AND panel.frame_has_location_signal AND panel.dominant_location_label = 'bed'),
          COUNTIF(panel.in_tail_window AND panel.frame_has_location_signal)
        ),
      SUM(IF(panel.in_tail_window AND panel.frame_has_location_signal, panel.recency_weight / GREATEST(panel.patient_room_distance, 1e-6), 0.0))
        + 0.25 * safe_ratio(
          COUNTIF(panel.in_tail_window AND panel.frame_has_location_signal AND panel.dominant_location_label = 'room'),
          COUNTIF(panel.in_tail_window AND panel.frame_has_location_signal)
        )
    ) AS tail_visible_probs,
    safe_ratio(COUNTIF(panel.in_tail_window AND panel.dominant_location_label = 'room'), COUNTIF(panel.in_tail_window)) AS tail_share_room,
    safe_ratio(COUNTIF(panel.in_tail_window AND panel.dominant_location_label = 'no_patient'), COUNTIF(panel.in_tail_window)) AS tail_share_no_patient,
    AVG(IF(panel.in_tail_window, panel.patient_staff_iou, NULL)) AS tail_overlap_mean,
    MAX(IF(panel.in_tail_window, panel.patient_staff_iou, NULL)) AS tail_overlap_max,
    AVG(IF(panel.in_anchor_window, panel.nudge_score_normalized, NULL)) AS anchor_nudge_mean,
    AVG(IF(panel.in_tail_window, panel.nudge_score_normalized, NULL)) AS tail_nudge_mean,
    MAX(IF(panel.in_tail_window, panel.nudge_score_normalized, NULL)) AS tail_nudge_max,
    AVG(IF(panel.in_anchor_window, panel.bed_in_bed_score, NULL)) AS anchor_bed_in_bed_score_mean,
    AVG(IF(panel.in_tail_window, panel.bed_in_bed_score, NULL)) AS tail_bed_in_bed_score_mean,
    MAX(IF(panel.in_tail_window, panel.bed_in_bed_score, NULL)) AS tail_bed_in_bed_score_max,
    AVG(IF(panel.in_anchor_window, panel.motion_bac, NULL)) AS anchor_motion_bac_mean,
    AVG(IF(panel.in_tail_window, panel.motion_bac, NULL)) AS tail_motion_bac_mean,
    MAX(IF(panel.in_tail_window, panel.motion_bac, NULL)) AS tail_motion_bac_max,
    AVG(IF(panel.in_anchor_window, panel.patient_candidate_count, NULL)) AS anchor_patient_candidate_count_mean,
    AVG(IF(panel.in_tail_window, panel.patient_candidate_count, NULL)) AS tail_patient_candidate_count_mean,
    AVG(IF(panel.in_tail_window, panel.staff_candidate_count, NULL)) AS tail_staff_candidate_count_mean,
    AVG(IF(panel.in_tail_window, panel.other_candidate_count, NULL)) AS tail_other_candidate_count_mean,
    AVG(IF(panel.in_tail_window, panel.bed_candidate_count, NULL)) AS tail_bed_candidate_count_mean,
    AVG(IF(panel.in_tail_window, panel.chair_candidate_count, NULL)) AS tail_chair_candidate_count_mean,
    safe_ratio(
      COUNTIF(
        panel.in_anchor_window
        AND panel.frame_has_location_signal
        AND panel.patient_room_distance <= panel.patient_chair_distance
        AND panel.patient_room_distance <= panel.patient_bed_distance
      ),
      COUNTIF(panel.in_anchor_window AND panel.frame_has_location_signal)
    ) AS anchor_room_nearest_share,
    safe_ratio(
      COUNTIF(
        panel.in_tail_window
        AND panel.frame_has_location_signal
        AND panel.patient_room_distance <= panel.patient_chair_distance
        AND panel.patient_room_distance <= panel.patient_bed_distance
      ),
      COUNTIF(panel.in_tail_window AND panel.frame_has_location_signal)
    ) AS tail_room_nearest_share,
    safe_ratio(
      COUNTIF(
        panel.in_tail_window
        AND panel.frame_has_location_signal
        AND panel.patient_bed_distance <= panel.patient_chair_distance
        AND panel.patient_bed_distance <= panel.patient_room_distance
      ),
      COUNTIF(panel.in_tail_window AND panel.frame_has_location_signal)
    ) AS tail_bed_nearest_share,
    AVG(IF(panel.in_anchor_window AND panel.frame_has_location_signal, panel.patient_bed_distance - panel.patient_room_distance, NULL)) AS anchor_room_bed_margin_mean,
    AVG(IF(panel.in_tail_window AND panel.frame_has_location_signal, panel.patient_bed_distance - panel.patient_room_distance, NULL)) AS tail_room_bed_margin_mean,
    AVG(IF(panel.in_tail_window, panel.chair_distance_velocity, NULL)) AS tail_chair_distance_velocity_mean,
    AVG(IF(panel.in_tail_window, panel.bed_distance_velocity, NULL)) AS tail_bed_distance_velocity_mean,
    AVG(IF(panel.in_tail_window, panel.room_distance_velocity, NULL)) AS tail_room_distance_velocity_mean,
    COUNTIF(panel.in_anchor_window AND panel.frame_has_posture_signal) AS anchor_posture_observed_frames,
    COUNTIF(panel.in_tail_window AND panel.frame_has_posture_signal) AS tail_posture_observed_frames,
    COUNTIF(panel.frame_has_posture_signal) AS posture_observed_frames,
    normalize_posture(
      SUM(IF(panel.in_anchor_window AND panel.frame_has_posture_signal, panel.posture_score_sitting * panel.recency_weight, 0.0)),
      SUM(IF(panel.in_anchor_window AND panel.frame_has_posture_signal, panel.posture_score_standing * panel.recency_weight, 0.0)),
      SUM(IF(panel.in_anchor_window AND panel.frame_has_posture_signal, panel.posture_score_lying * panel.recency_weight, 0.0))
    ) AS anchor_posture_score_probs,
    normalize_posture(
      SUM(IF(panel.in_tail_window AND panel.frame_has_posture_signal, panel.posture_score_sitting * panel.recency_weight, 0.0)),
      SUM(IF(panel.in_tail_window AND panel.frame_has_posture_signal, panel.posture_score_standing * panel.recency_weight, 0.0)),
      SUM(IF(panel.in_tail_window AND panel.frame_has_posture_signal, panel.posture_score_lying * panel.recency_weight, 0.0))
    ) AS tail_posture_score_probs,
    normalize_posture(
      COUNTIF(panel.in_anchor_window AND panel.frame_has_posture_signal AND panel.primary_patient_posture_label = 'sitting'),
      COUNTIF(panel.in_anchor_window AND panel.frame_has_posture_signal AND panel.primary_patient_posture_label = 'standing'),
      COUNTIF(panel.in_anchor_window AND panel.frame_has_posture_signal AND panel.primary_patient_posture_label = 'lying')
    ) AS anchor_posture_shares,
    normalize_posture(
      COUNTIF(panel.in_tail_window AND panel.frame_has_posture_signal AND panel.primary_patient_posture_label = 'sitting'),
      COUNTIF(panel.in_tail_window AND panel.frame_has_posture_signal AND panel.primary_patient_posture_label = 'standing'),
      COUNTIF(panel.in_tail_window AND panel.frame_has_posture_signal AND panel.primary_patient_posture_label = 'lying')
    ) AS tail_posture_shares,
    SUM(panel.posture_switch_flag) AS posture_switch_count
  FROM panel_with_posture_switches AS panel
  GROUP BY 1, 2, 3, 4, 5, 6
),
panel_summary_final AS (
  SELECT
    panel.*,
    IF(
      panel.anchor_room_bed_margin_mean IS NOT NULL
      AND panel.tail_room_bed_margin_mean IS NOT NULL,
      panel.tail_room_bed_margin_mean - panel.anchor_room_bed_margin_mean,
      NULL
    ) AS room_bed_margin_delta,
    dominant_visible_label(
      panel.anchor_visible_probs.chair,
      panel.anchor_visible_probs.bed,
      panel.anchor_visible_probs.room,
      0.40
    ) AS anchor_label,
    dominant_visible_label(
      panel.tail_visible_probs.chair,
      panel.tail_visible_probs.bed,
      panel.tail_visible_probs.room,
      0.40
    ) AS tail_label,
    dominant_posture_label(
      panel.anchor_posture_score_probs.sitting,
      panel.anchor_posture_score_probs.standing,
      panel.anchor_posture_score_probs.lying,
      panel.anchor_posture_shares.sitting,
      panel.anchor_posture_shares.standing,
      panel.anchor_posture_shares.lying,
      0.45
    ) AS anchor_posture_label,
    dominant_posture_label(
      panel.tail_posture_score_probs.sitting,
      panel.tail_posture_score_probs.standing,
      panel.tail_posture_score_probs.lying,
      panel.tail_posture_shares.sitting,
      panel.tail_posture_shares.standing,
      panel.tail_posture_shares.lying,
      0.45
    ) AS tail_posture_label
  FROM panel_summary AS panel
),
phase_blend AS (
  SELECT
    src.hospital_id,
    src.division_id,
    src.monitor_id,
    src.patient_id,
    src.ts_local,
    src.hour_ts,
    src.base_probs,
    src.state_visible_prob,
    src.state_not_visible_prob,
    src.state_out_of_room_prob,
    src.dropout_visibility_ratio,
    src.prob_not_visible,
    src.prob_out_of_room,
    src.out_of_room_share,
    src.pre_distance_signal_count,
    src.base_label,
    panel.* EXCEPT (hospital_id, division_id, monitor_id, patient_id, ts_local, hour_ts),
    normalize_visible(src.base_probs.chair, src.base_probs.bed, src.base_probs.room) AS base_visible_probs,
    normalize_visible(src.base_probs.chair, src.base_probs.bed, src.base_probs.room) AS instant_visible_prior,
    constants.*
  FROM source_normalized AS src
  JOIN panel_summary_final AS panel
    USING (hospital_id, division_id, monitor_id, patient_id, ts_local, hour_ts)
  CROSS JOIN constants
),
phase_blend_scored AS (
  SELECT
    *,
    GREATEST(anchor_visible_probs.chair, anchor_visible_probs.bed, anchor_visible_probs.room)
      * LEAST(1.0, safe_ratio(anchor_visible_frames, 4.0)) AS anchor_conf,
    GREATEST(tail_visible_probs.chair, tail_visible_probs.bed, tail_visible_probs.room)
      * LEAST(1.0, safe_ratio(tail_visible_frames, 4.0)) AS tail_conf,
    CASE
      WHEN tail_visible_frames <= 0 THEN 1.0
      WHEN anchor_visible_frames <= 0 THEN 0.0
      ELSE LEAST(
        0.80,
        GREATEST(
          0.20,
          anchor_weight_base
          + CASE
              WHEN GREATEST(tail_visible_probs.chair, tail_visible_probs.bed, tail_visible_probs.room)
                   * LEAST(1.0, safe_ratio(tail_visible_frames, 4.0))
                   > GREATEST(anchor_visible_probs.chair, anchor_visible_probs.bed, anchor_visible_probs.room)
                     * LEAST(1.0, safe_ratio(anchor_visible_frames, 4.0))
                     + 0.15
                THEN -0.15
              WHEN GREATEST(anchor_visible_probs.chair, anchor_visible_probs.bed, anchor_visible_probs.room)
                   * LEAST(1.0, safe_ratio(anchor_visible_frames, 4.0))
                   > GREATEST(tail_visible_probs.chair, tail_visible_probs.bed, tail_visible_probs.room)
                     * LEAST(1.0, safe_ratio(tail_visible_frames, 4.0))
                     + 0.15
                THEN 0.15
              ELSE 0.0
            END
          + 0.15 * LEAST(1.0, safe_ratio(last_visible_gap_seconds, recent_visible_gap_seconds))
          + CASE
              WHEN anchor_label != 'none'
                   AND tail_label != 'none'
                   AND anchor_label != tail_label
                   AND tail_visible_frames < 3
                THEN 0.10
              WHEN anchor_label != 'none'
                   AND tail_label != 'none'
                   AND anchor_label != tail_label
                THEN -0.05
              ELSE 0.0
            END
        )
      )
    END AS dynamic_anchor_weight
  FROM phase_blend
),
phase_blend_posture AS (
  SELECT
    *,
    normalize_posture(
      0.75 * (
        dynamic_anchor_weight * anchor_posture_score_probs.sitting
        + (1.0 - dynamic_anchor_weight) * tail_posture_score_probs.sitting
      ) + 0.25 * (
        dynamic_anchor_weight * anchor_posture_shares.sitting
        + (1.0 - dynamic_anchor_weight) * tail_posture_shares.sitting
      ),
      0.75 * (
        dynamic_anchor_weight * anchor_posture_score_probs.standing
        + (1.0 - dynamic_anchor_weight) * tail_posture_score_probs.standing
      ) + 0.25 * (
        dynamic_anchor_weight * anchor_posture_shares.standing
        + (1.0 - dynamic_anchor_weight) * tail_posture_shares.standing
      ),
      0.75 * (
        dynamic_anchor_weight * anchor_posture_score_probs.lying
        + (1.0 - dynamic_anchor_weight) * tail_posture_score_probs.lying
      ) + 0.25 * (
        dynamic_anchor_weight * anchor_posture_shares.lying
        + (1.0 - dynamic_anchor_weight) * tail_posture_shares.lying
      )
    ) AS posture_score_probs
  FROM phase_blend_scored
),
phase_classify AS (
  SELECT
    *,
    normalize_visible(
      dynamic_anchor_weight * anchor_visible_probs.chair + (1.0 - dynamic_anchor_weight) * tail_visible_probs.chair,
      dynamic_anchor_weight * anchor_visible_probs.bed + (1.0 - dynamic_anchor_weight) * tail_visible_probs.bed,
      dynamic_anchor_weight * anchor_visible_probs.room + (1.0 - dynamic_anchor_weight) * tail_visible_probs.room
    ) AS dynamic_panel_visible,
    scaled_support(GREATEST(COALESCE(tail_motion_bac_max, 0.0), COALESCE(tail_motion_bac_mean, 0.0)), motion_presence_scale) AS motion_support,
    GREATEST(
      scaled_support(anchor_patient_candidate_count_mean, 1.0),
      scaled_support(tail_patient_candidate_count_mean, 1.0)
    ) AS patient_candidate_support,
    scaled_support(tail_other_candidate_count_mean, room_object_mean_min) AS other_candidate_support,
    scaled_support(tail_bed_candidate_count_mean, 1.0) AS bed_candidate_support,
    scaled_support(tail_chair_candidate_count_mean, 1.0) AS chair_candidate_support,
    scaled_support(tail_posture_observed_frames, posture_min_frames) AS posture_support,
    GREATEST(COALESCE(tail_bed_in_bed_score_max, 0.0), COALESCE(tail_bed_in_bed_score_mean, 0.0)) AS bed_in_bed_support
  FROM phase_blend_posture
),
phase_classify_scored AS (
  SELECT
    *,
    normalize_visible(
      0.70 * base_visible_probs.chair + 0.20 * dynamic_panel_visible.chair + 0.10 * instant_visible_prior.chair,
      0.70 * base_visible_probs.bed + 0.20 * dynamic_panel_visible.bed + 0.10 * instant_visible_prior.bed,
      0.70 * base_visible_probs.room + 0.20 * dynamic_panel_visible.room + 0.10 * instant_visible_prior.room
    ) AS v3_visible_probs,
    GREATEST(
      patient_candidate_support,
      posture_support,
      0.75 * bed_in_bed_support,
      0.60 * motion_support,
      0.50 * GREATEST(other_candidate_support, bed_candidate_support, chair_candidate_support)
    ) AS presence_support,
    GREATEST(
      COALESCE(tail_share_room, 0.0),
      COALESCE(tail_room_nearest_share, 0.0),
      0.60 * other_candidate_support,
      0.40 * motion_support
    ) AS room_presence_support,
    GREATEST(
      COALESCE(tail_visible_probs.bed, 0.0),
      COALESCE(tail_bed_nearest_share, 0.0),
      0.65 * bed_in_bed_support,
      0.50 * bed_candidate_support
    ) AS bed_presence_support,
    GREATEST(
      COALESCE(tail_visible_probs.chair, 0.0),
      0.50 * chair_candidate_support,
      0.35 * patient_candidate_support
    ) AS chair_presence_support
  FROM phase_classify
),
phase_classify_regime AS (
  SELECT
    *,
    normalize_probs(
      (1.0 - base_probs.no_patient) * v3_visible_probs.chair,
      (1.0 - base_probs.no_patient) * v3_visible_probs.bed,
      (1.0 - base_probs.no_patient) * v3_visible_probs.room,
      base_probs.no_patient
    ) AS initial_v3_probs,
    GREATEST(
      base_visible_probs.chair,
      base_visible_probs.bed,
      base_visible_probs.room,
      instant_visible_prior.chair,
      instant_visible_prior.bed,
      instant_visible_prior.room,
      COALESCE(state_visible_prob, 0.0),
      COALESCE(dropout_visibility_ratio, 0.0),
      LEAST(1.0, safe_ratio(pre_distance_signal_count, 20.0))
    ) AS visible_support,
    GREATEST(
      COALESCE(state_not_visible_prob, 0.0),
      1.0 - COALESCE(dropout_visibility_ratio, 0.0),
      COALESCE(prob_not_visible, 0.0)
    ) AS transient_support,
    GREATEST(
      COALESCE(state_out_of_room_prob, 0.0),
      COALESCE(out_of_room_share, 0.0),
      COALESCE(prob_out_of_room, 0.0)
    ) AS out_of_room_support,
    GREATEST(
      scaled_support(tail_visible_frames, 2.0),
      scaled_support(pre_distance_signal_count, 2.0)
    ) AS visible_corroboration,
    scaled_support(tail_patient_candidate_count_mean, 1.0) AS tail_patient_corroboration,
    scaled_support(GREATEST(COALESCE(tail_motion_bac_max, 0.0), COALESCE(tail_motion_bac_mean, 0.0)), motion_presence_scale) AS tail_motion_corroboration,
    scaled_support(tail_posture_observed_frames, posture_min_frames) AS tail_posture_corroboration
  FROM phase_classify_scored
),
phase_adjust_no_patient AS (
  SELECT
    *,
    GREATEST(
      visible_corroboration,
      tail_patient_corroboration,
      tail_posture_corroboration,
      0.75 * tail_motion_corroboration
    ) AS sparse_corroboration,
    GREATEST(
      visible_corroboration,
      tail_patient_corroboration,
      0.75 * tail_motion_corroboration
    ) AS room_sparse_corroboration,
    CASE
      WHEN visible_support >= GREATEST(transient_support, out_of_room_support)
           AND visible_support >= 0.35
        THEN 'visible'
      WHEN out_of_room_support >= GREATEST(visible_support, transient_support)
           AND out_of_room_support >= 0.45
        THEN 'out_of_room'
      WHEN transient_support >= GREATEST(visible_support, out_of_room_support)
           AND transient_support >= 0.35
           AND presence_support >= presence_min_support
        THEN 'present_but_sparse'
      WHEN transient_support >= GREATEST(visible_support, out_of_room_support)
           AND transient_support >= 0.35
        THEN 'temporarily_not_visible'
      ELSE 'unknown'
    END AS signal_regime
  FROM phase_classify_regime
),
phase_adjust_no_patient_scored AS (
  SELECT
    *,
    sparse_corroboration >= sparse_corroboration_min AS sparse_corroborated,
    room_sparse_corroboration >= sparse_corroboration_min AS room_sparse_corroborated,
    visible_support >= 0.80
      AND GREATEST(instant_visible_prior.chair, instant_visible_prior.bed, instant_visible_prior.room) >= 0.60
      AS strong_visible_carry,
    CASE
      WHEN signal_regime = 'visible'
           AND sparse_corroboration >= sparse_corroboration_min
           AND visible_support >= 0.80
           AND GREATEST(instant_visible_prior.chair, instant_visible_prior.bed, instant_visible_prior.room) >= 0.60
        THEN LEAST(initial_v3_probs.no_patient, visible_no_patient_cap)
      WHEN signal_regime = 'visible'
           AND sparse_corroboration >= sparse_corroboration_min
        THEN LEAST(initial_v3_probs.no_patient, transient_no_patient_cap)
      WHEN signal_regime = 'temporarily_not_visible'
           AND GREATEST(instant_visible_prior.chair, instant_visible_prior.bed, instant_visible_prior.room) >= 0.45
           AND sparse_corroboration >= sparse_corroboration_min
        THEN LEAST(initial_v3_probs.no_patient, transient_no_patient_cap)
      WHEN signal_regime = 'present_but_sparse'
           AND sparse_corroboration >= sparse_corroboration_min
        THEN LEAST(initial_v3_probs.no_patient, present_sparse_no_patient_cap)
      WHEN signal_regime = 'out_of_room'
        THEN GREATEST(initial_v3_probs.no_patient, out_of_room_no_patient_floor)
      ELSE initial_v3_probs.no_patient
    END AS target_no_patient
  FROM phase_adjust_no_patient
),
phase_correct_location AS (
  SELECT
    *,
    CASE
      WHEN base_label = 'room'
        THEN base_probs
      WHEN ABS(target_no_patient - initial_v3_probs.no_patient) > 0.02
        THEN set_no_patient_probability(
          initial_v3_probs.chair,
          initial_v3_probs.bed,
          initial_v3_probs.room,
          initial_v3_probs.no_patient,
          target_no_patient,
          0.60 * normalize_visible(initial_v3_probs.chair, initial_v3_probs.bed, initial_v3_probs.room).chair
            + 0.40 * instant_visible_prior.chair,
          0.60 * normalize_visible(initial_v3_probs.chair, initial_v3_probs.bed, initial_v3_probs.room).bed
            + 0.40 * instant_visible_prior.bed,
          0.60 * normalize_visible(initial_v3_probs.chair, initial_v3_probs.bed, initial_v3_probs.room).room
            + 0.40 * instant_visible_prior.room
        )
      ELSE initial_v3_probs
    END AS no_patient_adjusted_probs
  FROM phase_adjust_no_patient_scored
),
phase_correct_location_room AS (
  SELECT
    *,
    GREATEST(
      COALESCE(tail_visible_probs.room, 0.0),
      COALESCE(tail_share_room, 0.0),
      COALESCE(tail_overlap_max, 0.0),
      COALESCE(tail_nudge_max, 0.0)
    ) >= 0.35
      AND COALESCE(tail_share_room, 0.0) >= 0.10 AS departure_detected,
    0.30 * IF(GREATEST(
      COALESCE(tail_visible_probs.room, 0.0),
      COALESCE(tail_share_room, 0.0),
      COALESCE(tail_overlap_max, 0.0),
      COALESCE(tail_nudge_max, 0.0)
    ) >= 0.35 AND COALESCE(tail_share_room, 0.0) >= 0.10, 1.0, 0.0)
    + 0.10 * IF(anchor_label IN ('chair', 'bed') AND tail_label = 'room', 1.0, 0.0)
    + 0.25 * LEAST(1.0, safe_ratio(tail_share_room, 0.25))
    + 0.20 * LEAST(1.0, safe_ratio(tail_room_nearest_share, 0.35))
    + 0.10 * positive_margin_score(room_bed_margin_delta)
    + 0.03 * LEAST(1.0, safe_ratio(tail_overlap_max, 0.25))
    + 0.02 * LEAST(1.0, safe_ratio(tail_nudge_max, 0.70)) AS departure_confidence,
    (
      (COALESCE(tail_bed_nearest_share, 0.0) >= 0.75
       AND COALESCE(tail_share_room, 0.0) < 0.10
       AND COALESCE(tail_room_nearest_share, 0.0) < 0.15)
      OR
      (bed_in_bed_support >= bed_in_bed_min_score
       AND COALESCE(tail_share_room, 0.0) < 0.20
       AND COALESCE(tail_room_nearest_share, 0.0) < 0.25)
    ) AS bed_lock,
    COALESCE(tail_room_distance_velocity_mean, 0.0) <= -velocity_min_approach AS room_velocity_approach
  FROM phase_correct_location
),
phase_correct_location_room_scored AS (
  SELECT
    *,
    (
      COALESCE(tail_room_nearest_share, 0.0) >= 0.35
      OR room_velocity_approach
      OR (
        signal_regime = 'present_but_sparse'
        AND room_presence_support >= presence_min_support
        AND room_sparse_corroborated
      )
      OR (
        departure_detected
        AND (
          COALESCE(tail_room_nearest_share, 0.0) >= 0.35
          OR COALESCE(tail_share_room, 0.0) >= 0.35
          OR room_velocity_approach
        )
      )
    ) AS room_override_evidence
  FROM phase_correct_location_room
),
phase_correct_location_bed AS (
  SELECT
    *,
    (
      NOT bed_lock
      AND (
        (
          departure_confidence >= departure_min_confidence
          AND room_override_evidence
          AND room_sparse_corroborated
        )
        OR (
          signal_regime = 'present_but_sparse'
          AND room_presence_support >= presence_min_support
          AND room_sparse_corroborated
        )
      )
    ) AS room_override_applied,
    CASE
      WHEN (
        NOT bed_lock
        AND (
          (
            departure_confidence >= departure_min_confidence
            AND room_override_evidence
            AND room_sparse_corroborated
          )
          OR (
            signal_regime = 'present_but_sparse'
            AND room_presence_support >= presence_min_support
            AND room_sparse_corroborated
          )
        )
      )
        THEN shift_probability_to(
          no_patient_adjusted_probs.chair,
          no_patient_adjusted_probs.bed,
          no_patient_adjusted_probs.room,
          no_patient_adjusted_probs.no_patient,
          'room',
          LEAST(
            0.70,
            GREATEST(
              no_patient_adjusted_probs.room,
              0.18 + 0.60 * GREATEST(departure_confidence, room_presence_support)
            )
          )
        )
      ELSE no_patient_adjusted_probs
    END AS room_adjusted_probs
  FROM phase_correct_location_room_scored
),
phase_correct_location_bed_scored AS (
  SELECT
    *,
    (anchor_visible_frames > 0 OR tail_visible_frames > 0) AS has_visible_tail_support,
    GREATEST(
      patient_candidate_support,
      posture_support,
      0.75 * motion_support,
      0.75 * chair_candidate_support
    ) AS invisible_bed_rescue_corroboration
  FROM phase_correct_location_bed
),
phase_rebalance_support AS (
  SELECT
    *,
    (
      has_visible_tail_support
      OR (
        bed_in_bed_support >= bed_in_bed_invisible_min_score
        AND invisible_bed_rescue_corroboration >= sparse_corroboration_min
      )
    ) AS invisible_bed_rescue_allowed
  FROM phase_correct_location_bed_scored
),
phase_rebalance AS (
  SELECT
    *,
    CASE
      WHEN NOT room_override_applied
           AND bed_in_bed_support >= bed_in_bed_min_score
           AND bed_presence_support >= GREATEST(room_presence_support, chair_presence_support)
           AND room_adjusted_probs.room < 0.30
           AND (
             has_visible_tail_support
             OR (invisible_bed_rescue_allowed AND bed_candidate_support >= presence_min_support)
           )
        THEN shift_probability_to(
          room_adjusted_probs.chair,
          room_adjusted_probs.bed,
          room_adjusted_probs.room,
          room_adjusted_probs.no_patient,
          'bed',
          LEAST(
            0.80,
            GREATEST(room_adjusted_probs.bed, 0.20 + 0.60 * bed_in_bed_support)
          )
        )
      ELSE room_adjusted_probs
    END AS bed_adjusted_probs
  FROM phase_rebalance_support
),
phase_rebalance_velocity AS (
  SELECT
    *,
    normalize_visible(bed_adjusted_probs.chair, bed_adjusted_probs.bed, bed_adjusted_probs.room) AS visible_after_corrections,
    GREATEST(bed_adjusted_probs.chair, bed_adjusted_probs.bed, bed_adjusted_probs.room) AS top_visible_probability,
    (
      bed_adjusted_probs.chair
      + bed_adjusted_probs.bed
      + bed_adjusted_probs.room
      - GREATEST(bed_adjusted_probs.chair, bed_adjusted_probs.bed, bed_adjusted_probs.room)
      - LEAST(bed_adjusted_probs.chair, bed_adjusted_probs.bed, bed_adjusted_probs.room)
    ) AS second_visible_probability,
    GREATEST(0.0, -COALESCE(tail_chair_distance_velocity_mean, 0.0)) AS chair_approach_score,
    GREATEST(0.0, -COALESCE(tail_bed_distance_velocity_mean, 0.0)) AS bed_approach_score,
    GREATEST(0.0, -COALESCE(tail_room_distance_velocity_mean, 0.0)) AS room_approach_score
  FROM phase_rebalance
),
phase_rebalance_velocity_scored AS (
  SELECT
    *,
    CASE
      WHEN room_approach_score >= bed_approach_score AND room_approach_score >= chair_approach_score THEN 'room'
      WHEN bed_approach_score >= chair_approach_score THEN 'bed'
      ELSE 'chair'
    END AS velocity_winner,
    GREATEST(chair_approach_score, bed_approach_score, room_approach_score) AS velocity_winner_score
  FROM phase_rebalance_velocity
),
phase_rebalance_posture AS (
  SELECT
    *,
    CASE
      WHEN top_visible_probability - second_visible_probability <= velocity_tiebreak_margin
           AND velocity_winner_score >= velocity_min_approach
           AND (
             velocity_winner != 'room'
             OR GREATEST(COALESCE(tail_share_room, 0.0), COALESCE(tail_room_nearest_share, 0.0)) >= 0.10
           )
        THEN shift_probability_to(
          bed_adjusted_probs.chair,
          bed_adjusted_probs.bed,
          bed_adjusted_probs.room,
          bed_adjusted_probs.no_patient,
          velocity_winner,
          LEAST(0.70, GREATEST(top_visible_probability + 0.03, 0.40 + velocity_winner_score))
        )
      ELSE bed_adjusted_probs
    END AS velocity_adjusted_probs
  FROM phase_rebalance_velocity_scored
),
phase_rebalance_posture_scored AS (
  SELECT
    *,
    normalize_visible(
      velocity_adjusted_probs.chair,
      velocity_adjusted_probs.bed,
      velocity_adjusted_probs.room
    ) AS visible_before_posture,
    CASE
      WHEN velocity_adjusted_probs.chair >= velocity_adjusted_probs.bed
           AND velocity_adjusted_probs.chair >= velocity_adjusted_probs.room
           AND velocity_adjusted_probs.chair >= velocity_adjusted_probs.no_patient
        THEN 'chair'
      WHEN velocity_adjusted_probs.bed >= velocity_adjusted_probs.room
           AND velocity_adjusted_probs.bed >= velocity_adjusted_probs.no_patient
        THEN 'bed'
      WHEN velocity_adjusted_probs.room >= velocity_adjusted_probs.no_patient
        THEN 'room'
      ELSE 'no_patient'
    END AS current_v3_label,
    posture_score_probs.lying AS lying_probability,
    posture_score_probs.sitting AS sitting_probability,
    posture_score_probs.standing AS standing_probability,
    ABS(velocity_adjusted_probs.chair - velocity_adjusted_probs.bed) AS chair_bed_gap
  FROM phase_rebalance_posture
),
phase_rebalance_posture_metrics AS (
  SELECT
    *,
    GREATEST(0.0, lying_probability - GREATEST(sitting_probability, standing_probability)) AS lying_dominance,
    LEAST(1.0, safe_ratio(posture_observed_frames, posture_min_frames)) AS posture_support_weight,
    visible_before_posture.chair + visible_before_posture.bed AS non_room_visible
  FROM phase_rebalance_posture_scored
),
phase_rebalance_posture_visible AS (
  SELECT
    *,
    LEAST(1.0, GREATEST(0.0, safe_ratio(lying_dominance, 0.25))) * posture_support_weight AS posture_blend,
    visible_before_posture.chair AS chair_signal,
    visible_before_posture.bed + lying_dominance * non_room_visible AS bed_signal
  FROM phase_rebalance_posture_metrics
),
phase_final AS (
  SELECT
    hospital_id,
    division_id,
    monitor_id,
    patient_id,
    ts_local,
    hour_ts,
    CASE
      WHEN current_v3_label IN ('chair', 'bed')
           AND GREATEST(velocity_adjusted_probs.room, velocity_adjusted_probs.no_patient) < 0.35
           AND departure_confidence < departure_min_confidence
           AND posture_observed_frames >= posture_min_frames
           AND posture_switch_count <= posture_max_switch_count
           AND NOT (
             anchor_posture_label != 'none'
             AND tail_posture_label != 'none'
             AND anchor_posture_label != tail_posture_label
           )
           AND lying_probability >= posture_bed_min_lying_prob
           AND chair_bed_gap <= posture_max_chair_bed_gap
        THEN normalize_probs(
          (1.0 - velocity_adjusted_probs.no_patient) * normalize_visible(
            (1.0 - posture_blend) * visible_before_posture.chair
              + posture_blend * non_room_visible * safe_ratio(chair_signal, chair_signal + bed_signal),
            (1.0 - posture_blend) * visible_before_posture.bed
              + posture_blend * non_room_visible * safe_ratio(bed_signal, chair_signal + bed_signal),
            (1.0 - posture_blend) * visible_before_posture.room
              + posture_blend * visible_before_posture.room
          ).chair,
          (1.0 - velocity_adjusted_probs.no_patient) * normalize_visible(
            (1.0 - posture_blend) * visible_before_posture.chair
              + posture_blend * non_room_visible * safe_ratio(chair_signal, chair_signal + bed_signal),
            (1.0 - posture_blend) * visible_before_posture.bed
              + posture_blend * non_room_visible * safe_ratio(bed_signal, chair_signal + bed_signal),
            (1.0 - posture_blend) * visible_before_posture.room
              + posture_blend * visible_before_posture.room
          ).bed,
          (1.0 - velocity_adjusted_probs.no_patient) * normalize_visible(
            (1.0 - posture_blend) * visible_before_posture.chair
              + posture_blend * non_room_visible * safe_ratio(chair_signal, chair_signal + bed_signal),
            (1.0 - posture_blend) * visible_before_posture.bed
              + posture_blend * non_room_visible * safe_ratio(bed_signal, chair_signal + bed_signal),
            (1.0 - posture_blend) * visible_before_posture.room
              + posture_blend * visible_before_posture.room
          ).room,
          velocity_adjusted_probs.no_patient
        )
      ELSE velocity_adjusted_probs
    END AS final_probs
  FROM phase_rebalance_posture_visible
),
hourly AS (
  SELECT
    hospital_id,
    division_id,
    monitor_id,
    patient_id,
    hour_ts,
    AVG(final_probs.chair) AS pct_chair,
    AVG(final_probs.bed) AS pct_bed,
    AVG(final_probs.room) AS pct_ambulatory,
    AVG(final_probs.no_patient) AS pct_not_located
  FROM phase_final
  GROUP BY 1, 2, 3, 4, 5
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
  pct_not_located
FROM hourly
WHERE ABS((pct_chair + pct_bed + pct_ambulatory + pct_not_located) - 1.0) <= 0.02;
