SELECT
  timestamp,
  timestamp_local,
  date_local,
  time_local,
  hospital_id,
  hospital_system_name,
  division_id,
  hospital_name,
  patient_id,
  monitor_id,
  user_name,
  summary
FROM {{fall_events_compatible_subquery_sql}};
