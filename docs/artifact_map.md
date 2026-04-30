# Artifact Map

**Current manuscript run:** `consensus_v3_refresh_20260320T215557Z`

Public-release use: the portable evidence bundle is the canonical manuscript source plus the curated aggregate data under `data/public/` and the manuscript-support documents in `docs/`. Machine-generated manifest and audit files are intentionally excluded from this repo because they encode local filesystem provenance rather than portable scientific evidence. The mechanism taxonomy export remains a descriptive packet artifact regenerated from the adjudicated consensus CSV and labeled with the current `run_id` for packet grouping. The current manuscript packet uses an outcome-defined monitor cohort: `monitor_id in fall_events_source -> intervention`; `monitor_id only in hourly_location_aggregation -> control`.

## Packet Lineage

- Canonical packet run: `consensus_v3_refresh_20260320T215557Z`. Manuscript-facing docs should not cite the partial March 24 outcome bundle.
- Legacy v2 broader-observation raw consensus table: `backups/v2_consensus/falls-observations-v2 - consensus.csv` (85 annotations: room 30, bed 26, chair 14, blank 11, no_patient 4). Context only; not hard-label output and not benchmark truth.
- V3 adjudicated descriptive / benchmark source: `data/public/falls-observations-v3-consensus.csv` (37 source rows -> 32 included fall rows / 32 mechanism-coded events; 30 truth rows -> 31 scored sequences for benchmark diagnostics).
- Broader monitoring feed: 91 inference-derived alarm-time hard labels summarized from `chair_bed_analysis_reconciliation_<run_id>.csv` and `falls_prefall_location_<run_id>.csv`.
- Inferential base: 40 linked events used in the adjusted chair-vs-bed model.

## Public Data Bundle
- `data/public/falls-observations-v3-consensus.csv`
- `data/public/derived/chair_bed_risk_rates_<run_id>.csv`
- `data/public/derived/chair_bed_risk_rates_confidence_sensitivity_<run_id>.csv`
- `data/public/derived/chair_bed_operational_event_rates_<run_id>.csv`
- `data/public/derived/chair_bed_adjusted_rr_<run_id>.csv`
- `data/public/derived/chair_bed_adjusted_rr_covariates_<run_id>.csv`
- `data/public/derived/chair_bed_adjusted_rr_robustness_<run_id>.csv`
- `data/public/derived/chair_bed_misclassification_sensitivity_<run_id>.csv`
- `data/public/derived/chair_bed_threshold_sensitivity_<run_id>.csv`
- `data/public/derived/chair_bed_analysis_reconciliation_<run_id>.csv`
- `data/public/derived/consensus_adjudication_summary_<run_id>.csv`
- `data/public/derived/falls_prefall_location_probability_breakdown_<run_id>.csv`
- `data/public/derived/label_eval_furniture_origin_chain_<run_id>.csv`
- `data/public/derived/label_eval_post_departure_latency_<run_id>.csv`
- `data/public/derived/label_eval_benchmark_label_metrics_<run_id>.csv`
- `data/public/derived/label_eval_benchmark_sequence_metrics_<run_id>.csv`
- `data/public/derived/mechanism_taxonomy_<run_id>.csv`
- `data/public/gemini/gemini_model_characterization_metrics_20260313T180809Z.csv`
- `data/public/gemini/gemini_model_characterization_detection_20260313T180809Z.csv`
- `data/public/gemini/gemini_model_characterization_summary_20260313T180809Z.md`

## Restricted Internal Artifacts Not Shipped
- `data/raw/<run_id>/...`
- `data/staged/<run_id>/...`
- `outputs/manifests/*.yaml`
- `outputs/audit/*.json`
- Internal dashboards, submission checklists, review notes, and site builds removed from the public tree

## Reviewer Bundle
- `paper/manuscript.md`
- `docs/strobe_checklist.md`
- `docs/strobe_flow_diagram.md`
- `docs/reproduce_this_paper.md`
- `docs/analysis_base_data_dictionary.md`
- `figures/chair_panel_risk_simulator_static.svg`
- `data/public/falls-observations-v3-consensus.csv`
- `data/public/derived/chair_bed_risk_rates_<run_id>.csv`
- `data/public/derived/chair_bed_operational_event_rates_<run_id>.csv`
- `data/public/derived/chair_bed_adjusted_rr_<run_id>.csv`
- `data/public/derived/chair_bed_adjusted_rr_robustness_<run_id>.csv`
- `data/public/derived/chair_bed_misclassification_sensitivity_<run_id>.csv`
- `data/public/derived/chair_bed_threshold_sensitivity_<run_id>.csv`
- `data/public/derived/chair_bed_analysis_reconciliation_<run_id>.csv`
- `data/public/derived/label_eval_furniture_origin_chain_<run_id>.csv`
- `data/public/derived/label_eval_post_departure_latency_<run_id>.csv`
- `data/public/derived/label_eval_benchmark_label_metrics_<run_id>.csv`
- `data/public/derived/label_eval_benchmark_sequence_metrics_<run_id>.csv`
- `data/public/derived/mechanism_taxonomy_<run_id>.csv` (descriptive export from fixed consensus annotations unless `--consensus-path` is overridden)
- `data/public/gemini/gemini_model_characterization_metrics_20260313T180809Z.csv`
- `data/public/gemini/gemini_model_characterization_detection_20260313T180809Z.csv`
- `data/public/gemini/gemini_model_characterization_summary_20260313T180809Z.md`

## Gemini Characterization Bundle

Precomputed ancillary benchmark (3 models × 81 clips against v2 consensus GT; v2 ground truth retained for this ancillary study). Not regenerated by the primary pipeline; aggregate outputs included in the public bundle as-is.

- `data/public/gemini/gemini_model_characterization_metrics_20260313T180809Z.csv` — aggregate model metrics (sensitivity, specificity, location accuracy, timing)
- `data/public/gemini/gemini_model_characterization_detection_20260313T180809Z.csv` — aggregate detection confusion matrix (TP/FN/FP/TN)
- `data/public/gemini/gemini_model_characterization_summary_20260313T180809Z.md` — aggregate study summary

## Manuscript Mapping
- Table 1 cohort counts/exposure base <- reconciliation CSV and `mechanism_taxonomy_<run_id>.csv`.
- Table 2 unadjusted chair/bed rates <- `chair_bed_risk_rates_<run_id>.csv`.
- Main-text Figure 4 operational signal rates <- `chair_bed_operational_event_rates_<run_id>.csv` (talk-confirmed secondary operational subset; chair/bed summary rows).
- Appendix Figure A1 qualitative review schematic <- `paper/assets/figure5_context_schematic.png` (appendix-only context figure).
- Table 3 adjusted RR and sensitivity rows <- `chair_bed_adjusted_rr_<run_id>.csv`.
- Dispersion and robustness statements <- `chair_bed_adjusted_rr_<run_id>.csv` + robustness CSV.
- Misclassification range statement <- `chair_bed_misclassification_sensitivity_<run_id>.csv`.
- Table 4 mechanism display <- `mechanism_taxonomy_<run_id>.csv` + `data/public/falls-observations-v3-consensus.csv` (or explicit `--consensus-path` override).
- Table 5a furniture-origin chain <- `label_eval_furniture_origin_chain_<run_id>.csv`.
- Post-departure latency paragraph <- `label_eval_post_departure_latency_<run_id>.csv`.
- Label-evaluation metrics paragraph <- `label_eval_benchmark_label_metrics_<run_id>.csv` + `label_eval_benchmark_sequence_metrics_<run_id>.csv` + `chair_bed_risk_rates_confidence_sensitivity_<run_id>.csv`.
- Table 6 ancillary LLM benchmark <- `data/public/gemini/gemini_model_characterization_metrics_20260313T180809Z.csv` (scope=`three_way_overlap`).
- CIN static simulator figure <- `figures/chair_panel_risk_simulator_static.svg` (docs-layer aggregate figure derived from internal simulator defaults; not machine-generated).

## Numeric Claim Crosswalk

| Claim | Value | Artifact | Notes |
|---|---:|---|---|
| Study-window adjudicated events matched to pipeline | 43 | `data/public/derived/chair_bed_analysis_reconciliation_<run_id>.csv` | Manuscript-side source of truth for the linked-event funnel |
| Inferential events linked to analysis base | 40 | `data/public/derived/chair_bed_analysis_reconciliation_<run_id>.csv` | `eligible_events_analysis_base` |
| Broader monitoring feed | 91 | `data/public/derived/chair_bed_analysis_reconciliation_<run_id>.csv` | `raw_events_total` |
| Broader observation cohort | 32 | `data/public/derived/mechanism_taxonomy_<run_id>.csv` | Descriptive mechanism export row count |
| Benchmark subset | 30 truth rows / 31 scored sequences | `data/public/derived/consensus_adjudication_summary_<run_id>.csv` plus `data/public/derived/label_eval_benchmark_sequence_metrics_<run_id>.csv` | Benchmark diagnostics cohort |
| Chair rate | 17.8 per 1,000 exposure-hours | `data/public/derived/chair_bed_risk_rates_<run_id>.csv` | Expected rate, `position=chair` |
| Bed rate | 4.3 per 1,000 exposure-hours | `data/public/derived/chair_bed_risk_rates_<run_id>.csv` | Expected rate, `position=bed` |
| Main-text Figure 4 talk-click rates | chair 73.68 / bed 54.06 per 100 exposure-hours | `data/public/derived/chair_bed_operational_event_rates_<run_id>.csv` | `metric=talk_event`, `window_half_width_seconds=3`, `scope=talk_confirmed_intervention_oneoff` |
| Main-text Figure 4 alarm-trigger rates | chair 19.50 / bed 9.19 per 100 exposure-hours | `data/public/derived/chair_bed_operational_event_rates_<run_id>.csv` | `metric=alarm_trigger`, `window_half_width_seconds=3`, `scope=talk_confirmed_intervention_oneoff` |
| Main-text Figure 4 nudge burden | chair 279.37 / bed 247.42 active seconds per exposure-hour | `data/public/derived/chair_bed_operational_event_rates_<run_id>.csv` | `metric=nudge_active_seconds`, duration rows in the talk-confirmed secondary operational subset |
| Primary RR (HC3) | 2.35 (HC3-corrected Wald 95% CI 0.93–5.94; p=0.0709) | `data/public/derived/chair_bed_adjusted_rr_<run_id>.csv` | `sensitivity_label=primary_adjusted_hc3`; Wald companion (0.87–6.33) in `sensitivity_label=primary_adjusted` row |

## Reconciliation Notes

- The manuscript-level 43-event study-window statement should be reconciled to `data/public/derived/chair_bed_analysis_reconciliation_<run_id>.csv`.
- The staged extraction begins on `2024-08-13`, but this does not alter the study-window cohort counts because neither the staged fall-events parquet nor the adjudicated consensus CSV contains events dated August 1-12, 2024.
