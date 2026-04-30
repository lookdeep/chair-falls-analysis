-- One-off operational summary for talk-confirmed sitter sessions.
CREATE OR REPLACE TABLE `{{summary_table}}` AS
WITH minute_base AS (
  SELECT *
  FROM `{{minute_table}}`
),
event_base AS (
  SELECT *
  FROM `{{event_label_table}}`
),
exposure_rows AS (
  SELECT "chair" AS position, SUM(seconds_chair) / 3600.0 AS exposure_hours FROM minute_base
  UNION ALL
  SELECT "bed" AS position, SUM(seconds_bed) / 3600.0 AS exposure_hours FROM minute_base
  UNION ALL
  SELECT "room" AS position, SUM(seconds_room) / 3600.0 AS exposure_hours FROM minute_base
  UNION ALL
  SELECT "no_patient" AS position, SUM(seconds_no_patient) / 3600.0 AS exposure_hours FROM minute_base
),
event_totals AS (
  SELECT
    metric,
    window_half_width_seconds,
    COUNT(*) AS source_value_in_scope,
    COUNTIF(event_scope_status != "resolved") AS unresolved_events,
    COUNTIF(event_scope_status = "resolved" AND assigned_location_label = "missing_posture")
      AS missing_posture_events,
    COUNTIF(event_scope_status = "resolved" AND assigned_location_label = "chair") AS chair_events,
    COUNTIF(event_scope_status = "resolved" AND assigned_location_label = "bed") AS bed_events,
    COUNTIF(event_scope_status = "resolved" AND assigned_location_label = "room") AS room_events,
    COUNTIF(event_scope_status = "resolved" AND assigned_location_label = "no_patient")
      AS no_patient_events
  FROM event_base
  GROUP BY metric, window_half_width_seconds
),
event_position_expanded AS (
  SELECT
    "talk_confirmed_intervention_oneoff" AS scope,
    metric,
    "event_count" AS metric_type,
    "count" AS metric_unit,
    window_half_width_seconds,
    exposure.position,
    exposure.exposure_hours,
    source_value_in_scope,
    CASE exposure.position
      WHEN "chair" THEN chair_events
      WHEN "bed" THEN bed_events
      WHEN "room" THEN room_events
      ELSE no_patient_events
    END AS positioned_numerator_value,
    CASE exposure.position
      WHEN "chair" THEN bed_events + room_events + no_patient_events
      WHEN "bed" THEN chair_events + room_events + no_patient_events
      WHEN "room" THEN chair_events + bed_events + no_patient_events
      ELSE chair_events + bed_events + room_events
    END AS excluded_non_target_position_value,
    missing_posture_events AS excluded_missing_posture_value,
    unresolved_events AS excluded_ambiguous_events
  FROM event_totals
  CROSS JOIN exposure_rows exposure
),
event_rows AS (
  SELECT
    scope,
    metric,
    metric_type,
    metric_unit,
    window_half_width_seconds,
    position,
    exposure_hours,
    source_value_in_scope,
    positioned_numerator_value,
    excluded_non_target_position_value,
    excluded_missing_posture_value,
    excluded_ambiguous_events,
    SAFE_DIVIDE(100.0 * positioned_numerator_value, NULLIF(exposure_hours, 0.0))
      AS rate_per_100_exposure_hours,
    CAST(NULL AS FLOAT64) AS active_seconds_per_exposure_hour,
    CASE metric
      WHEN "talk_event" THEN FORMAT(
        "Talk clicks with +/- %d second state association per %s exposure-hours.",
        window_half_width_seconds,
        position
      )
      WHEN "alarm_trigger" THEN FORMAT(
        "Manual alarm triggers with +/- %d second state association per %s exposure-hours.",
        window_half_width_seconds,
        position
      )
      ELSE FORMAT(
        "Exploratory safety-zone onsets with +/- %d second state association per %s exposure-hours.",
        window_half_width_seconds,
        position
      )
    END AS notes
  FROM event_position_expanded
),
nudge_totals AS (
  SELECT
    SUM(nudge_active_seconds_total_in_scope) AS source_value_in_scope,
    SUM(nudge_active_seconds_missing_posture) AS missing_posture_seconds,
    SUM(nudge_active_seconds_chair) AS chair_seconds,
    SUM(nudge_active_seconds_bed) AS bed_seconds,
    SUM(nudge_active_seconds_room) AS room_seconds,
    SUM(nudge_active_seconds_no_patient) AS no_patient_seconds
  FROM minute_base
),
nudge_position_expanded AS (
  SELECT
    "talk_confirmed_intervention_oneoff" AS scope,
    "nudge_active_seconds" AS metric,
    "duration" AS metric_type,
    "seconds" AS metric_unit,
    CAST(NULL AS INT64) AS window_half_width_seconds,
    exposure.position,
    exposure.exposure_hours,
    source_value_in_scope,
    CASE exposure.position
      WHEN "chair" THEN chair_seconds
      WHEN "bed" THEN bed_seconds
      WHEN "room" THEN room_seconds
      ELSE no_patient_seconds
    END AS positioned_numerator_value,
    CASE exposure.position
      WHEN "chair" THEN bed_seconds + room_seconds + no_patient_seconds
      WHEN "bed" THEN chair_seconds + room_seconds + no_patient_seconds
      WHEN "room" THEN chair_seconds + bed_seconds + no_patient_seconds
      ELSE chair_seconds + bed_seconds + room_seconds
    END AS excluded_non_target_position_value,
    missing_posture_seconds AS excluded_missing_posture_value,
    0 AS excluded_ambiguous_events
  FROM nudge_totals
  CROSS JOIN exposure_rows exposure
),
nudge_rows AS (
  SELECT
    scope,
    metric,
    metric_type,
    metric_unit,
    window_half_width_seconds,
    position,
    exposure_hours,
    source_value_in_scope,
    positioned_numerator_value,
    excluded_non_target_position_value,
    excluded_missing_posture_value,
    excluded_ambiguous_events,
    CAST(NULL AS FLOAT64) AS rate_per_100_exposure_hours,
    SAFE_DIVIDE(positioned_numerator_value, NULLIF(exposure_hours, 0.0))
      AS active_seconds_per_exposure_hour,
    FORMAT("Active nudge-state seconds per %s exposure-hour.", position) AS notes
  FROM nudge_position_expanded
)
SELECT *
FROM event_rows
UNION ALL
SELECT *
FROM nudge_rows;
