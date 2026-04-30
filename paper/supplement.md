---
title: "Supplementary Material — Exposure-Normalized Bed and Chair Fall Rates via Continuous AI Monitoring"
---

# S1. Auxiliary Annotation Cohorts

Four non-interchangeable data layers support different descriptive tasks. (1) Study-window adjudicated cohort: 43 consensus-matched events (August 2024–December 2025). (2) Inferential event base: 40 eligible linked falls entering the Poisson model (this is the only cohort that contributes to the primary RR estimate). (3) v3 adjudicated descriptive source: 37 source rows yielding 32 deduplicated mechanism-coded events; the same source yielded the departure-aware benchmark subset (30 truth rows, 31 scored sequences) for furniture-origin and label-evaluation analyses. (4) Broader monitoring feed (2022–2026, n=91 events): raw hard-label and site-summary context only.

# S2. Full Sensitivity Panel

Pre-specified sensitivity analyses addressed four domains: (i) time-window restriction across seven local-time strata (00:00–05:59 through 21:00–23:59) plus weekday-only and weekend-only stratifications; (ii) position certainty, restricting to hourly rows where chair or bed fraction ≥0.50; (iii) systematic label-swap misclassification scenarios — symmetric 10%, 20%, and 30% chair/bed reassignment and one-sided 20% swaps in each direction; and (iv) eligibility-threshold variation at `min_observed_hours` ∈ {4, 12, 24, 48}. A furniture-origin reclassification test reassigned room/no_patient events with confirmed chair provenance (via the `last_furniture` field) to 100% chair attribution.

**Table S1. Full sensitivity panel for the adjusted chair-vs-bed rate ratio.**

  ------------------------------------- -------- -------------- ------------- ---------------- -----------------------------------------------------------------------------
  **Analysis**                          **RR**   **95% CI**     **p-value**   **Events (n)**   **Notes**

  Primary — HC3 SE (all hours)          2.35     0.93 to 5.94   0.071         40               estimable; HC3 robust SE primary

  Reference — Wald SE                   2.35     0.87 to 6.33   0.091         40               Wald SE companion

  Exploratory clustered SE (division)   2.35     1.89 to 2.92   <0.0001       40               9 intervention divisions (< 30-cluster threshold); underdispersed fit; not a primary estimate

  Position-certainty hours only         NE       NE             ---           38               insufficient_data

  Chair-origin reclassified             2.35     0.87 to 6.33   0.091         40               furniture-origin sensitivity

  Eligibility threshold ≥ 12 h          2.39     0.88 to 6.47   0.086         38               estimable

  Eligibility threshold ≥ 24 h          2.43     0.89 to 6.63   0.083         36               estimable

  Eligibility threshold ≥ 48 h          2.42     0.84 to 6.98   0.101         30               estimable
  ------------------------------------- -------- -------------- ------------- ---------------- -----------------------------------------------------------------------------

> Note: All seven local-time window stratifications were non-estimable (<5 events per arm) and are omitted. Misclassification scenarios yielded RR 4.49–13.17 across estimable swaps; the one-sided 20% chair-to-bed scenario was non-estimable. These ranges reflect denominator sensitivity to label-assumption rather than a plausible effect distribution.

# S3. Label-Quality Evaluation

Automated label quality was evaluated against the departure-aware benchmark subset (30 truth rows spanning 31 scored sequences). Macro F1 = 0.528; 10-bin expected calibration error (ECE) = 0.450 (0 = perfect calibration; 1 = worst); event-level detection F1 = 0.846 (precision 1.000 / recall 0.733); latency mean absolute error 37.9 s (p50 22 s, p90 83 s). Perfect detection precision indicates no false-positive alarms; recall of 0.733 means roughly one in four falls is missed. A confidence-filtered descriptive recomputation yielded chair 17.1 vs 17.8 and bed 4.2 vs 4.3 per 1,000 h, indicating rank-ordering robustness without resolving the calibration error.

# S4. Multimodal-LLM Benchmark

As an ancillary validation, three commercial LLMs (Gemini 2.5 Flash, 2.5 Pro, 3.1 Pro Preview) independently classified de-identified adjudicated event clips (81 clips from the v2 consensus package) for fall detection, pre-fall location, last-furniture provenance, and fall timing. The primary comparison used a 3-way overlap subset (n=42 visible rows). Results are precomputed (run timestamp 20260313T180809Z); commercial model weights update silently and exact results are not expected to regenerate. API model identifiers: gemini-2.5-flash, gemini-2.5-pro, gemini-3.1-pro-preview (as used at benchmark run time, 2026-03-13). This benchmark uses different denominators (3-way overlap, n=42 primary rows) than the primary pipeline; direct ranking is not appropriate.

**Table S2. Multimodal LLM fall-detection performance (3-way overlap subset, n=42 visible primary rows).**

  ------------------------ ---------------------- -------------------------- ---------------------------------- -----------------------
  **Model**                **Fall Sensitivity**   **Non-fall Specificity**   **Location Accuracy (detected)**   **Fall-time MAE (s)**

  Gemini 2.5 Flash         0.842                  0.750                      0.531                              1,199

  Gemini 2.5 Pro           0.684                  1.000                      0.654                              521

  Gemini 3.1 Pro Preview   0.447                  1.000                      0.588                              106
  ------------------------ ---------------------- -------------------------- ---------------------------------- -----------------------

The collective finding is a sensitivity–specificity / sensitivity–timing tradeoff: no model achieved high recall, high specificity, and high location accuracy simultaneously. Most-missed falls were subtle or furniture-proximal events. The benchmark reinforces the primary pipeline's label-quality narrative without entering the rate model.

# S5. Mechanism-Coded Observation Cohort

A separate observation cohort of 32 deduplicated fall events (from 32 included fall rows across 37 source rows; 4 no_fall rows and 1 unclassifiable empty event excluded) was assembled for mechanism coding via structured dual review by two co-author team members (ZD and PG); no independent external annotation was performed. Each event was classified into one of three mechanism categories: footrest/positioning, transfer failure, or other/unclassified. Among the 7 direct chair falls in this observation cohort, 6 carried a footrest/positioning tag. This cohort is not denominator-linked to the study-window adjudicated cohort and is reported for descriptive mechanism characterization only; mechanism proportions should not be interpreted as population prevalence estimates.

**Table S3. Fall mechanism taxonomy by pre-fall location (observation cohort, n=32 events).**

  -------------------------------- ----------------- ---------------- ----------------- -----------
  **Mechanism category**           **Chair (n=7)**   **Bed (n=12)**   **Room (n=13)**   **Total**

  Footrest / positioning failure   6 (86%)           ---              ---               6

  Transfer failure                 ---               6 (50%)          2 (15%)           8

  Other / unclassified             1 (14%)           6 (50%)          11 (85%)          18
  -------------------------------- ----------------- ---------------- ----------------- -----------

Last-furniture provenance was populated in 13/13 room events (bed 10, chair 3); where departure and fall clocks were available, time-since-departure had median 0.30 minutes (IQR 0.13–1.15; n=13).

# S6. Furniture-Origin Chain Analysis and Reclassification Sensitivity

Among the 30 evaluable events in the departure-aware benchmark subset, furniture-origin chain classification identified 6 direct chair falls, 11 direct bed falls, 3 chair-origin room falls (events where the patient was last observed in a chair before falling in the room), and 10 bed-origin room falls. Including the 3 chair-origin room events with the 6 direct chair falls yields 9 chair-associated events vs 6 when using only direct chair classification.

**Table S4. Event classification by furniture-origin chain (departure-aware benchmark subset, n=30).**

  -------------------- ------------ ----------------
  **Chain Category**   **Events**   **% of Total**

  direct_bed           11           36.7%

  bed_origin_room      10           33.3%

  direct_chair         6            20.0%

  chair_origin_room    3            10.0%
  -------------------- ------------ ----------------

Reassigning the 3 chair-origin room falls to 100% chair attribution did not materially change event allocation; the adjusted RR remained 2.35.

Post-departure latency (furniture departure → fall): bed-origin median 18 s (IQR 7.0–56.5; max 136; n=10); chair-origin events at 8 s, 50 s, and 150 s (n=3).

# S7. Furniture-Exit Detection Concordance — Disclosure

For events with known last-furniture provenance and valid consensus departure timestamps, two AI departure signals (nudge-boundary crossing in the operational `nudge_state` indicator, and location-label transition off the patient's last-furniture label) were to be evaluated against the consensus reference as bias and MAE within a ±5-minute search window. The pre-specified evaluation was non-estimable on this extraction: no events met the evaluability gate after linkage and filtering. This is a data-availability gap (departure timestamps were not populated for events in the study-window adjudicated cohort after linkage), not evidence of zero timing error.

# S8. Hard-Label Position Distribution at Alarm — Broader Monitoring Feed

Within the broader monitoring feed (2022-2026, n=91 deduplicated events), the AI-derived hard-label position distribution at alarm time was: bed (48; 52.7%), no_patient (23; 25.3%), room (12; 13.2%), chair (8; 8.8%). Hard labels are descriptive; the primary inference uses probability-weighted allocation inside the 40-event base.

# S9. Site Distribution

Falls in the broader monitoring feed (n=91) spanned 10 divisions; site names are withheld per de-identification requirements. The top seven divisions accounted for 85 of 91 events.

# S10. Reproducibility Bundle

The accompanying reproducibility bundle contains aggregate output tables and analysis scripts; see `docs/reproduce_this_paper.md`. Frozen run identifier: `consensus_v3_refresh_20260320T215557Z` (fixture `frozen_v1`). Numbers reported in the main manuscript and this supplement trace deterministically to that run.
