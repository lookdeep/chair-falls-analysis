# Analysis Base Data Dictionary (Aggregate Reproduction)

This dictionary lists the minimum fields needed to reproduce Tables 1-3, the promoted secondary operational-signal figure, and primary/sensitivity inferential outputs.

## Analysis Base
- `hospital_id`: Integer health-system identifier (`5` for this study).
- `division_id`: Site/building identifier used as fixed effect and clustering variable.
- `monitor_id`: Monitoring unit identifier.
- `patient_id`: Patient identifier within monitoring system.
- `hour_ts`: UTC hour bucket key for exposure/event linkage.
- `cohort_type`: Cohort assignment (`intervention`, `control`, `observational`).
- `pct_chair`: Fractional chair exposure in the hour (`0..1`).
- `pct_bed`: Fractional bed exposure in the hour (`0..1`).
- `pct_ambulatory`: Fractional ambulatory exposure in the hour (`0..1`).
- `num_alarms`: Legacy hourly alarm-hit count from the monitoring cache, retained for QA but not used in the current secondary event-window study.
- `num_nudges`: Legacy hourly nudge/escalation count from the monitoring cache, retained for QA but not used in the current secondary nudge-burden metric.
- `num_announcements`: Legacy hourly announcement/talk-event count from the monitoring cache, retained for QA but not used in the current secondary talk-click numerator.
- `falls_in_hour`: Event count linked into hour bucket.
- `fall_event_flag`: Boolean indicator (`falls_in_hour > 0`).
- `daypart`: Derived categorical time band.
- `day_of_week`: Derived weekday label.
- `calendar_quarter`: Derived quarter (`YYYYQn`).

## Eligible Event Linkage Fields
- `fall_ts_utc`: Event timestamp (UTC) from livestream event windows.
- `prefall_location_label`: Hard label (`chair`, `bed`, `room`, `no_patient`).
- `pre_prob_chair`: Pre-fall probability mass for chair.
- `pre_prob_bed`: Pre-fall probability mass for bed.

## Derived Modeling Outputs
- `sensitivity_label`: Analysis slice identifier (for example `primary_adjusted`, `position_certain_only`).
- `rr`: Rate ratio (`chair` vs `bed`).
- `ci_lower`, `ci_upper`: 95% CI bounds on RR.
- `p_value`: Wald p-value for position coefficient.
- `n_events`: Effective modeled event total in that slice.
- `deviance_df_ratio`, `pearson_df_ratio`: Dispersion diagnostics.

## Derived Operational Signal Outputs
- `scope`: Secondary operational-study slice identifier (`talk_confirmed_intervention_oneoff` in the current public bundle).
- `source_study_id`: Internal one-off study identifier used to derive the manuscript-facing secondary-analysis export.
- `metric`: Operational signal family (`talk_event`, `alarm_trigger`, `nudge_active_seconds`).
- `metric_type`: Measurement type (`event_count` or `duration`).
- `metric_unit`: Underlying numerator unit (`count` or `seconds`).
- `window_half_width_seconds`: Event-label half-window for talk/alarm rows (`3` primary, `5` sensitivity); blank for duration rows.
- `positioned_numerator_value`: Count of chair- or bed-labeled events, or active nudge seconds, attributed to that posture.
- `excluded_missing_posture_value`: Numerator value excluded because the attribution window had no non-missing posture labels.
- `excluded_ambiguous_events`: Event count excluded because no unique chair/bed assignment was available after tie-breaking; zero for duration rows.
- `rate_per_100_exposure_hours`: Event rate scaled to 100 exposure-hours for talk/alarm rows.
- `active_seconds_per_exposure_hour`: Duration burden for `nudge_active_seconds` rows.
- `notes`: Plain-language description of the numerator and denominator used for that row.
