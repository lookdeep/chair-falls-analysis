SELECT DISTINCT
  hospital_id,
  division_id,
  monitor_id,
  patient_id
FROM {{fall_events_compatible_subquery_sql}}
WHERE monitor_id IS NOT NULL;
