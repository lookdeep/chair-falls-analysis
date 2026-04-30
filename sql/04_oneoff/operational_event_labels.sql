-- One-off event-level posture labels for talk, alarm, and exploratory safety-zone onsets.
CREATE OR REPLACE TABLE `{{event_label_table}}` AS
WITH retained_sessions AS (
  SELECT *
  FROM `{{session_table}}`
  WHERE retained_for_sitter_analysis
),
window_widths AS (
  SELECT {{primary_event_window_half_width_seconds}} AS window_half_width_seconds
  UNION ALL
  SELECT {{sensitivity_event_window_half_width_seconds}} AS window_half_width_seconds
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
    TIMESTAMP_TRUNC(second_ts_utc, MINUTE) AS minute_ts_utc
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
    ) AS patient_room_distance,
    MAX(
      CASE
        WHEN SAFE_CAST(d.frame.nudge.score AS FLOAT64) IS NULL THEN 0.0
        WHEN SAFE_CAST(d.frame.nudge.score AS FLOAT64) > 1.0
             AND SAFE_CAST(d.frame.nudge.score AS FLOAT64) <= 100.0
          THEN GREATEST(0.0, LEAST(1.0, SAFE_CAST(d.frame.nudge.score AS FLOAT64) / 100.0))
        ELSE GREATEST(0.0, LEAST(1.0, SAFE_CAST(d.frame.nudge.score AS FLOAT64)))
      END
    ) AS nudge_score_normalized
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
second_posture AS (
  SELECT
    rs.hospital_id,
    rs.division_id,
    rs.monitor_id,
    rs.patient_id,
    rs.second_ts_utc,
    rs.minute_ts_utc,
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
    END AS dominant_location_label,
    COALESCE(p.nudge_score_normalized, 0.0) AS nudge_score_normalized
  FROM retained_seconds rs
  LEFT JOIN posture_second_candidates p
    USING (hospital_id, division_id, monitor_id, patient_id, second_ts_utc)
),
talk_events_raw AS (
  SELECT
    CAST(a.id AS STRING) AS source_event_id,
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
    ORDER BY source_event_id
  ) = 1
),
talk_event_session_matches AS (
  SELECT
    tr.source_event_id,
    tr.hospital_id,
    rs.division_id,
    tr.monitor_id,
    rs.patient_id,
    tr.event_ts_utc,
    DATETIME(tr.event_ts_utc, "{{hospital_timezone}}") AS event_dt_local,
    COUNT(*) OVER (PARTITION BY tr.source_event_id) AS session_match_count,
    ROW_NUMBER() OVER (
      PARTITION BY tr.source_event_id
      ORDER BY rs.session_start_dt_local, rs.patient_id
    ) AS session_match_rank
  FROM talk_events_raw tr
  JOIN retained_sessions rs
    ON rs.hospital_id = tr.hospital_id
   AND rs.monitor_id = tr.monitor_id
   AND (tr.division_id IS NULL OR rs.division_id = tr.division_id)
   AND DATETIME(tr.event_ts_utc, "{{hospital_timezone}}") >= rs.session_start_dt_local
   AND DATETIME(tr.event_ts_utc, "{{hospital_timezone}}") < rs.session_end_exclusive_dt_local
),
talk_events_session_scoped AS (
  SELECT
    "talk_event" AS metric,
    source_event_id,
    hospital_id,
    division_id,
    monitor_id,
    patient_id,
    event_ts_utc,
    event_dt_local,
    TIMESTAMP_TRUNC(event_ts_utc, SECOND) AS event_second_utc,
    CASE
      WHEN session_match_count > 1 THEN "ambiguous_session"
      ELSE "resolved"
    END AS session_scope_status
  FROM talk_event_session_matches
  WHERE session_match_rank = 1
),
talk_events_scoped AS (
  SELECT
    ts.metric,
    ts.source_event_id,
    ts.hospital_id,
    ts.division_id,
    ts.monitor_id,
    ts.patient_id,
    ts.event_ts_utc,
    ts.event_dt_local,
    ts.event_second_utc,
    CASE
      WHEN ts.session_scope_status != "resolved" THEN ts.session_scope_status
      WHEN rhl.hour_dt_local IS NULL THEN "unmatched_hour"
      WHEN rhl.unit_count != 1 THEN "ambiguous_hour"
      ELSE "resolved"
    END AS event_scope_status
  FROM talk_events_session_scoped ts
  LEFT JOIN retained_hour_lookup rhl
    ON rhl.hospital_id = ts.hospital_id
   AND rhl.monitor_id = ts.monitor_id
   AND rhl.hour_dt_local = DATETIME_TRUNC(
     ts.event_dt_local,
     HOUR
   )
),
alarm_events_raw AS (
  SELECT
    CAST(sa.id AS STRING) AS source_event_id,
    sa.hospital_id,
    CAST(sa.patient_monitor_id AS INT64) AS monitor_id,
    CAST(COALESCE(sa.observed_at, sa.created_at) AS TIMESTAMP) AS event_ts_utc,
    LOWER(TRIM(COALESCE(CAST(sa.event_type AS STRING), ""))) AS event_type,
    LOWER(TRIM(COALESCE(CAST(sa.alarm_type AS STRING), ""))) AS alarm_type,
    LOWER(TRIM(COALESCE(CAST(sa.source AS STRING), ""))) AS source_type,
    LOWER(TRIM(COALESCE(CAST(sa.description AS STRING), ""))) AS description_text
  FROM `{{project}}.hospital_api.stat_alarms` sa
  WHERE sa.hospital_id = {{study_hospital_id}}
    AND sa.patient_monitor_id IS NOT NULL
    AND COALESCE(sa.observed_at, sa.created_at) IS NOT NULL
    AND DATE(CAST(COALESCE(sa.observed_at, sa.created_at) AS TIMESTAMP), "{{hospital_timezone}}")
      BETWEEN DATE "{{study_start_date}}" AND DATE "{{study_end_date}}"
    AND LOWER(TRIM(COALESCE(CAST(sa.alarm_type AS STRING), ""))) = "alarm"
    AND LOWER(TRIM(COALESCE(CAST(sa.event_type AS STRING), ""))) IN ("prevented", "occurred")
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY
      hospital_id,
      monitor_id,
      event_ts_utc,
      event_type,
      alarm_type,
      source_type,
      description_text
    ORDER BY source_event_id
  ) = 1
),
alarm_event_session_matches AS (
  SELECT
    ar.source_event_id,
    ar.hospital_id,
    rs.division_id,
    ar.monitor_id,
    rs.patient_id,
    ar.event_ts_utc,
    DATETIME(ar.event_ts_utc, "{{hospital_timezone}}") AS event_dt_local,
    COUNT(*) OVER (PARTITION BY ar.source_event_id) AS session_match_count,
    ROW_NUMBER() OVER (
      PARTITION BY ar.source_event_id
      ORDER BY rs.session_start_dt_local, rs.patient_id
    ) AS session_match_rank
  FROM alarm_events_raw ar
  JOIN retained_sessions rs
    ON rs.hospital_id = ar.hospital_id
   AND rs.monitor_id = ar.monitor_id
   AND DATETIME(ar.event_ts_utc, "{{hospital_timezone}}") >= rs.session_start_dt_local
   AND DATETIME(ar.event_ts_utc, "{{hospital_timezone}}") < rs.session_end_exclusive_dt_local
),
alarm_events_session_scoped AS (
  SELECT
    "alarm_trigger" AS metric,
    source_event_id,
    hospital_id,
    division_id,
    monitor_id,
    patient_id,
    event_ts_utc,
    event_dt_local,
    TIMESTAMP_TRUNC(event_ts_utc, SECOND) AS event_second_utc,
    CASE
      WHEN session_match_count > 1 THEN "ambiguous_session"
      ELSE "resolved"
    END AS session_scope_status
  FROM alarm_event_session_matches
  WHERE session_match_rank = 1
),
alarm_events_scoped AS (
  SELECT
    ars.metric,
    ars.source_event_id,
    ars.hospital_id,
    ars.division_id,
    ars.monitor_id,
    ars.patient_id,
    ars.event_ts_utc,
    ars.event_dt_local,
    ars.event_second_utc,
    CASE
      WHEN ars.session_scope_status != "resolved" THEN ars.session_scope_status
      WHEN rhl.hour_dt_local IS NULL THEN "unmatched_hour"
      WHEN rhl.unit_count != 1 THEN "ambiguous_hour"
      ELSE "resolved"
    END AS event_scope_status
  FROM alarm_events_session_scoped ars
  LEFT JOIN retained_hour_lookup rhl
    ON rhl.hospital_id = ars.hospital_id
   AND rhl.monitor_id = ars.monitor_id
   AND rhl.hour_dt_local = DATETIME_TRUNC(
     ars.event_dt_local,
     HOUR
   )
),
safety_zone_second_flags AS (
  SELECT
    hospital_id,
    division_id,
    monitor_id,
    patient_id,
    second_ts_utc,
    nudge_score_normalized >= {{prefall_panel_safety_zone_threshold}} AS safety_zone_active,
    LAG(nudge_score_normalized >= {{prefall_panel_safety_zone_threshold}}) OVER (
      PARTITION BY hospital_id, division_id, monitor_id, patient_id
      ORDER BY second_ts_utc
    ) AS previous_safety_zone_active
  FROM second_posture
),
safety_zone_events_scoped AS (
  SELECT
    "safety_zone_onset" AS metric,
    CONCAT(
      CAST(hospital_id AS STRING), "|",
      CAST(division_id AS STRING), "|",
      CAST(monitor_id AS STRING), "|",
      CAST(patient_id AS STRING), "|",
      FORMAT_TIMESTAMP("%Y-%m-%dT%H:%M:%S%Ez", second_ts_utc)
    ) AS source_event_id,
    hospital_id,
    division_id,
    monitor_id,
    patient_id,
    second_ts_utc AS event_ts_utc,
    DATETIME(second_ts_utc, "{{hospital_timezone}}") AS event_dt_local,
    second_ts_utc AS event_second_utc,
    "resolved" AS event_scope_status
  FROM safety_zone_second_flags
  WHERE safety_zone_active
    AND NOT COALESCE(previous_safety_zone_active, FALSE)
),
raw_events AS (
  SELECT * FROM talk_events_scoped
  UNION ALL
  SELECT * FROM alarm_events_scoped
  UNION ALL
  SELECT * FROM safety_zone_events_scoped
),
resolved_event_windows AS (
  SELECT
    r.metric,
    r.source_event_id,
    w.window_half_width_seconds,
    r.hospital_id,
    r.division_id,
    r.monitor_id,
    r.patient_id,
    r.event_ts_utc,
    r.event_dt_local,
    r.event_second_utc
  FROM raw_events r
  CROSS JOIN window_widths w
  WHERE r.event_scope_status = "resolved"
),
event_window_observations AS (
  SELECT
    e.metric,
    e.source_event_id,
    e.window_half_width_seconds,
    e.hospital_id,
    e.division_id,
    e.monitor_id,
    e.patient_id,
    e.event_ts_utc,
    e.event_dt_local,
    e.event_second_utc,
    window_second_utc,
    ABS(TIMESTAMP_DIFF(window_second_utc, e.event_second_utc, SECOND)) AS abs_second_distance,
    COALESCE(sp.dominant_location_label, "missing_posture") AS dominant_location_label
  FROM resolved_event_windows e
  CROSS JOIN UNNEST(
    GENERATE_TIMESTAMP_ARRAY(
      TIMESTAMP_SUB(e.event_second_utc, INTERVAL e.window_half_width_seconds SECOND),
      TIMESTAMP_ADD(e.event_second_utc, INTERVAL e.window_half_width_seconds SECOND),
      INTERVAL 1 SECOND
    )
  ) AS window_second_utc
  LEFT JOIN second_posture sp
    ON sp.hospital_id = e.hospital_id
   AND sp.division_id = e.division_id
   AND sp.monitor_id = e.monitor_id
   AND sp.patient_id = e.patient_id
   AND sp.second_ts_utc = window_second_utc
),
window_second_counts AS (
  SELECT
    metric,
    source_event_id,
    window_half_width_seconds,
    hospital_id,
    division_id,
    monitor_id,
    patient_id,
    event_ts_utc,
    event_dt_local,
    event_second_utc,
    COUNTIF(dominant_location_label IN ("chair", "bed", "room", "no_patient"))
      AS window_non_missing_seconds,
    COUNTIF(dominant_location_label = "missing_posture") AS window_missing_seconds
  FROM event_window_observations
  GROUP BY
    metric,
    source_event_id,
    window_half_width_seconds,
    hospital_id,
    division_id,
    monitor_id,
    patient_id,
    event_ts_utc,
    event_dt_local,
    event_second_utc
),
non_missing_votes AS (
  SELECT
    metric,
    source_event_id,
    window_half_width_seconds,
    dominant_location_label,
    COUNT(*) AS second_count
  FROM event_window_observations
  WHERE dominant_location_label IN ("chair", "bed", "room", "no_patient")
  GROUP BY metric, source_event_id, window_half_width_seconds, dominant_location_label
),
top_vote_counts AS (
  SELECT
    metric,
    source_event_id,
    window_half_width_seconds,
    MAX(second_count) AS max_non_missing_vote
  FROM non_missing_votes
  GROUP BY metric, source_event_id, window_half_width_seconds
),
tied_top_votes AS (
  SELECT
    v.metric,
    v.source_event_id,
    v.window_half_width_seconds,
    t.max_non_missing_vote,
    ARRAY_AGG(v.dominant_location_label ORDER BY v.dominant_location_label) AS tied_top_labels,
    COUNT(*) AS tied_top_label_count
  FROM non_missing_votes v
  JOIN top_vote_counts t
    USING (metric, source_event_id, window_half_width_seconds)
  WHERE v.second_count = t.max_non_missing_vote
  GROUP BY v.metric, v.source_event_id, v.window_half_width_seconds, t.max_non_missing_vote
),
event_second_label AS (
  SELECT
    metric,
    source_event_id,
    window_half_width_seconds,
    dominant_location_label AS event_second_location_label
  FROM event_window_observations
  WHERE window_second_utc = event_second_utc
),
nearest_non_missing_label AS (
  SELECT
    metric,
    source_event_id,
    window_half_width_seconds,
    dominant_location_label AS nearest_non_missing_location_label
  FROM event_window_observations
  WHERE dominant_location_label IN ("chair", "bed", "room", "no_patient")
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY metric, source_event_id, window_half_width_seconds
    ORDER BY abs_second_distance, window_second_utc
  ) = 1
),
resolved_event_labels AS (
  SELECT
    counts.metric,
    counts.source_event_id,
    counts.window_half_width_seconds,
    counts.hospital_id,
    counts.division_id,
    counts.monitor_id,
    counts.patient_id,
    counts.event_ts_utc,
    counts.event_dt_local,
    "resolved" AS event_scope_status,
    CASE
      WHEN COALESCE(tied.tied_top_label_count, 0) = 0 THEN "missing_posture"
      WHEN tied.tied_top_label_count = 1 THEN tied.tied_top_labels[OFFSET(0)]
      WHEN event_second.event_second_location_label IN UNNEST(tied.tied_top_labels) THEN event_second.event_second_location_label
      WHEN nearest.nearest_non_missing_location_label IN UNNEST(tied.tied_top_labels) THEN nearest.nearest_non_missing_location_label
      ELSE tied.tied_top_labels[OFFSET(0)]
    END AS assigned_location_label,
    COALESCE(event_second.event_second_location_label, "missing_posture") AS event_second_location_label,
    nearest.nearest_non_missing_location_label,
    counts.window_non_missing_seconds,
    counts.window_missing_seconds,
    COALESCE(tied.max_non_missing_vote, 0) AS max_non_missing_vote,
    COALESCE(tied.tied_top_label_count, 0) AS tied_top_label_count,
    COALESCE(tied.tied_top_label_count, 0) > 1
      AND event_second.event_second_location_label IN UNNEST(COALESCE(tied.tied_top_labels, []))
      AS used_event_second_tiebreaker,
    COALESCE(tied.tied_top_label_count, 0) > 1
      AND NOT (
        event_second.event_second_location_label IN UNNEST(COALESCE(tied.tied_top_labels, []))
      )
      AND nearest.nearest_non_missing_location_label IN UNNEST(COALESCE(tied.tied_top_labels, []))
      AS used_nearest_second_tiebreaker
  FROM window_second_counts counts
  LEFT JOIN tied_top_votes tied
    USING (metric, source_event_id, window_half_width_seconds)
  LEFT JOIN event_second_label event_second
    USING (metric, source_event_id, window_half_width_seconds)
  LEFT JOIN nearest_non_missing_label nearest
    USING (metric, source_event_id, window_half_width_seconds)
),
unresolved_event_labels AS (
  SELECT
    r.metric,
    r.source_event_id,
    w.window_half_width_seconds,
    r.hospital_id,
    r.division_id,
    r.monitor_id,
    r.patient_id,
    r.event_ts_utc,
    r.event_dt_local,
    r.event_scope_status,
    CAST(NULL AS STRING) AS assigned_location_label,
    CAST(NULL AS STRING) AS event_second_location_label,
    CAST(NULL AS STRING) AS nearest_non_missing_location_label,
    0 AS window_non_missing_seconds,
    0 AS window_missing_seconds,
    0 AS max_non_missing_vote,
    0 AS tied_top_label_count,
    FALSE AS used_event_second_tiebreaker,
    FALSE AS used_nearest_second_tiebreaker
  FROM raw_events r
  CROSS JOIN window_widths w
  WHERE r.event_scope_status != "resolved"
)
SELECT *
FROM resolved_event_labels
UNION ALL
SELECT *
FROM unresolved_event_labels
ORDER BY
  CASE metric
    WHEN "talk_event" THEN 1
    WHEN "alarm_trigger" THEN 2
    WHEN "safety_zone_onset" THEN 3
    ELSE 99
  END,
  event_ts_utc,
  source_event_id,
  window_half_width_seconds;
