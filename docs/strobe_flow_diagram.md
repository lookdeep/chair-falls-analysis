# STROBE Item 13 Cohort Flow

**run_id:** `consensus_v3_refresh_20260320T215557Z`

```
Hourly monitors in source window (hospital_id=5): 5,531
  -> Passed eligibility gates (min_observed_hours>=4; coverage>=0.95): 3,980
     -> Intervention eligible: 42
     -> Control eligible: 3,938

Study-window adjudicated events (August 2024-December 2025): 43
  -> Linked to eligible analysis base: 40
     -> Chair hard-label events: 5
     -> Bed hard-label events: 23
     -> Room / no_patient labels: 12

Broader monitoring feed (2022-2026, descriptive reference): 91 deduplicated events
  -> Bed: 48
  -> no_patient: 23
  -> Room: 12
  -> Chair: 8

Broader observation cohort (descriptive mechanism coding only):
  37 source annotations -> 32 included fall rows -> 32 deduplicated events

Departure-aware benchmark subset (descriptive diagnostics only):
  30 truth rows -> 31 scored sequences
```

Notes:
- The inferential denominator is exposure-normalized hourly chair/bed time among eligible units.
- Intervention/control membership is outcome-defined at the monitor level: `monitor_id in fall_events_source -> intervention`; `monitor_id only in hourly_location_aggregation -> control`.
- The 43-event study-window cohort comes from the adjudicated reconciliation pathway, not from the broader 91-event monitoring feed.
- The 91-event broader monitoring feed reflects inference-derived alarm-time hard labels.
- The legacy v2 broader-observation raw consensus table (`backups/v2_consensus/falls-observations-v2 - consensus.csv`; 85 annotations: room 30, bed 26, chair 14, blank 11, no_patient 4) is retained for annotation context only and is not part of this STROBE flow.
- The public adjudicated source for this release is `data/public/falls-observations-v3-consensus.csv`.
- The QA reconciliation CSV also reports `consensus_source_rows_study_window = 34` and `consensus_included_fall_rows_study_window = 27`; those are source-row counts from the consensus annotation CSV, not the manuscript-facing adjudicated funnel used in this STROBE supplement.
- The broader observation cohort is a separate descriptive convenience sample and is not used for inferential rate estimation.
- The departure-aware benchmark subset supports furniture-origin, post-departure, and label-evaluation diagnostics only.
