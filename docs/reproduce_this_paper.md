# Reproduce This Paper

## Public Release Scope
- `run_id`: `consensus_v3_refresh_20260320T215557Z`
- Canonical manuscript source: `paper/manuscript.md`
- Public adjudicated annotation source: `data/public/falls-observations-v3-consensus.csv`
- Curated aggregate results bundle: `data/public/derived/`
- Ancillary multimodal benchmark bundle: `data/public/gemini/`

This public repository is source-first. It does not include restricted raw/staged parquet, patient-level video, or tracked machine-local `outputs/manifests/` and `outputs/audit/` provenance files.

## Environment
- Python: `>=3.12`
- Package manager: `uv`
- Dependency lock: `uv.lock`

## Public-Bundle Verification

```bash
uv sync --project . --extra dev
uv run --project . python scripts/generate_figures.py --run-id consensus_v3_refresh_20260320T215557Z
uv run --project . python scripts/compile_chair_fall_dashboard.py --run-id consensus_v3_refresh_20260320T215557Z
```

If `pandoc` is available, you can also build the manuscript packaging surfaces locally:

```bash
uv run --project . python scripts/build_dual_track_paper.py --run-id consensus_v3_refresh_20260320T215557Z --skip-tex-verify
```

`compile_mechanism_taxonomy.py` remains anchored to the adjudicated consensus annotation file (`data/public/falls-observations-v3-consensus.csv` by default). Its `--run-id` groups the descriptive export with the same public manuscript bundle, but it does **not** select a different consensus dataset unless `--consensus-path` is also provided.

## Public Bundle Contents

- `data/public/falls-observations-v3-consensus.csv`
- `data/public/derived/chair_bed_risk_rates_consensus_v3_refresh_20260320T215557Z.csv`
- `data/public/derived/chair_bed_risk_rates_confidence_sensitivity_consensus_v3_refresh_20260320T215557Z.csv`
- `data/public/derived/chair_bed_operational_event_rates_consensus_v3_refresh_20260320T215557Z.csv`
- `data/public/derived/chair_bed_adjusted_rr_consensus_v3_refresh_20260320T215557Z.csv`
- `data/public/derived/chair_bed_adjusted_rr_covariates_consensus_v3_refresh_20260320T215557Z.csv`
- `data/public/derived/chair_bed_adjusted_rr_robustness_consensus_v3_refresh_20260320T215557Z.csv`
- `data/public/derived/chair_bed_misclassification_sensitivity_consensus_v3_refresh_20260320T215557Z.csv`
- `data/public/derived/chair_bed_threshold_sensitivity_consensus_v3_refresh_20260320T215557Z.csv`
- `data/public/derived/chair_bed_analysis_reconciliation_consensus_v3_refresh_20260320T215557Z.csv`
- `data/public/derived/consensus_adjudication_summary_consensus_v3_refresh_20260320T215557Z.csv`
- `data/public/derived/label_eval_furniture_origin_chain_consensus_v3_refresh_20260320T215557Z.csv`
- `data/public/derived/label_eval_post_departure_latency_consensus_v3_refresh_20260320T215557Z.csv`
- `data/public/derived/label_eval_benchmark_label_metrics_consensus_v3_refresh_20260320T215557Z.csv`
- `data/public/derived/label_eval_benchmark_sequence_metrics_consensus_v3_refresh_20260320T215557Z.csv`
- `data/public/derived/mechanism_taxonomy_consensus_v3_refresh_20260320T215557Z.csv`
- `data/public/gemini/gemini_model_characterization_metrics_20260313T180809Z.csv`
- `data/public/gemini/gemini_model_characterization_detection_20260313T180809Z.csv`
- `data/public/gemini/gemini_model_characterization_summary_20260313T180809Z.md`

## Restricted Full Rerun

If you have approved access to the restricted source systems and patient-level inputs, the internal end-to-end pipeline still runs with the legacy commands:

```bash
just run consensus_v3_refresh_20260320T215557Z true true
just model consensus_v3_refresh_20260320T215557Z
just reports consensus_v3_refresh_20260320T215557Z
```

Those commands regenerate local `outputs/` artifacts for internal QA. They are intentionally not tracked in this public release.

## Numeric Claim Crosswalk

| Manuscript / packet claim | Value | Source artifact | Provenance note |
|---|---:|---|---|
| Study-window adjudicated events retained in the manuscript supplement | 43 | `docs/strobe_flow_diagram.md` | Curated STROBE supplement count used by the manuscript packet; the run QA files separately expose the 40-event linked inferential base |
| Inferential events linked to eligible analysis base | 40 | `data/public/derived/chair_bed_analysis_reconciliation_consensus_v3_refresh_20260320T215557Z.csv` | `eligible_events_analysis_base` |
| Broader monitoring feed | 91 | `data/public/derived/chair_bed_analysis_reconciliation_consensus_v3_refresh_20260320T215557Z.csv` | `raw_events_total`; this is the inference-derived hard-label feed |
| Broader observation cohort | 32 | `data/public/derived/mechanism_taxonomy_consensus_v3_refresh_20260320T215557Z.csv` | Row count of the descriptive mechanism export derived from the v3 adjudicated source |
| Benchmark subset | 30 truth rows / 31 scored sequences | `data/public/derived/consensus_adjudication_summary_consensus_v3_refresh_20260320T215557Z.csv` plus `data/public/derived/label_eval_benchmark_sequence_metrics_consensus_v3_refresh_20260320T215557Z.csv` | Truth rows come from the adjudication audit; scored sequences come from the primary departure-aware benchmark split |
| Probability-weighted chair rate | 17.8 per 1,000 exposure-hours | `data/public/derived/chair_bed_risk_rates_consensus_v3_refresh_20260320T215557Z.csv` | `rate_per_1000_exposure_hours_expected` for `position=chair` |
| Probability-weighted bed rate | 4.3 per 1,000 exposure-hours | `data/public/derived/chair_bed_risk_rates_consensus_v3_refresh_20260320T215557Z.csv` | `rate_per_1000_exposure_hours_expected` for `position=bed` |
| Main-text Figure 4 talk-click rates | chair 73.68 / bed 54.06 per 100 exposure-hours | `data/public/derived/chair_bed_operational_event_rates_consensus_v3_refresh_20260320T215557Z.csv` | `metric=talk_event`, `window_half_width_seconds=3`, `scope=talk_confirmed_intervention_oneoff` |
| Main-text Figure 4 alarm-trigger rates | chair 19.50 / bed 9.19 per 100 exposure-hours | `data/public/derived/chair_bed_operational_event_rates_consensus_v3_refresh_20260320T215557Z.csv` | `metric=alarm_trigger`, `window_half_width_seconds=3`, `scope=talk_confirmed_intervention_oneoff` |
| Main-text Figure 4 nudge burden | chair 279.37 / bed 247.42 active seconds per exposure-hour | `data/public/derived/chair_bed_operational_event_rates_consensus_v3_refresh_20260320T215557Z.csv` | `metric=nudge_active_seconds`, duration rows in the talk-confirmed secondary operational subset |
| Primary adjusted RR (HC3) | 2.35 (HC3-corrected Wald 95% CI 0.93–5.94; p=0.0709) | `data/public/derived/chair_bed_adjusted_rr_consensus_v3_refresh_20260320T215557Z.csv` | `sensitivity_label=primary_adjusted_hc3` row; Wald companion (0.87–6.33) in `sensitivity_label=primary_adjusted` row |

## Manuscript Lineage

- Legacy v2 broader-observation raw consensus table: `backups/v2_consensus/falls-observations-v2 - consensus.csv` (85 annotations: room 30, bed 26, chair 14, blank 11, no_patient 4). Context only; not hard-label output and not benchmark truth.
- The 91-event broader monitoring feed is the inference-derived alarm-time hard-label source. It supports descriptive hard-label and site-summary displays only.
- The v3 adjudicated source (`data/public/falls-observations-v3-consensus.csv`) supports both the 32-event mechanism taxonomy and the benchmark diagnostics (`37 source rows -> 30 truth rows / 31 scored sequences`).
- The 40-event inferential base is the only cohort that contributes to the primary RR estimate.
- Table 1 cohort counts come from `data/public/derived/chair_bed_analysis_reconciliation_<run_id>.csv` and `data/public/derived/mechanism_taxonomy_<run_id>.csv`.
- Table 2 and the opening Results rate statement come from `data/public/derived/chair_bed_risk_rates_<run_id>.csv`.
- Main-text Figure 4 and the secondary operational-signal Results paragraph come from `data/public/derived/chair_bed_operational_event_rates_<run_id>.csv`, a curated chair/bed summary of the talk-confirmed secondary operational subset.
- Table 3, the primary RR statement, the clustered-SE note, and the misclassification range statement come from `data/public/derived/chair_bed_adjusted_rr_<run_id>.csv`, `data/public/derived/chair_bed_adjusted_rr_robustness_<run_id>.csv`, and `data/public/derived/chair_bed_misclassification_sensitivity_<run_id>.csv`.
- Table 4 comes from `data/public/derived/mechanism_taxonomy_<run_id>.csv`, which is regenerated from the fixed consensus annotation source unless `--consensus-path` is overridden.
- Table 5a and the post-departure latency paragraph come from `data/public/derived/label_eval_furniture_origin_chain_<run_id>.csv` and `data/public/derived/label_eval_post_departure_latency_<run_id>.csv`.
- The label-evaluation metrics paragraph comes from `data/public/derived/label_eval_benchmark_label_metrics_<run_id>.csv`, `data/public/derived/label_eval_benchmark_sequence_metrics_<run_id>.csv`, and `data/public/derived/chair_bed_risk_rates_confidence_sensitivity_<run_id>.csv`.
- The static CIN figure (`figures/chair_panel_risk_simulator_static.svg`) is a docs-layer submission artifact derived from the manuscript’s internal simulator defaults; it is not a machine-generated pipeline output.

### Ancillary Gemini Benchmark (precomputed)

The Gemini characterization study is a precomputed ancillary artifact. It is not regenerated by the primary pipeline; the aggregate outputs are included in the public bundle as-is.

- Table 6 <- `data/public/gemini/gemini_model_characterization_metrics_20260313T180809Z.csv` (scope=`three_way_overlap`)
- Ancillary Methods/Results/Discussion text <- same bundle plus `data/public/gemini/gemini_model_characterization_detection_20260313T180809Z.csv`

## Notes
- `just` recipe arguments in this repository are positional. Use `just run <RUN_ID> <GATE_1> <GATE_2>`, `just reports <RUN_ID>`, and `just model <RUN_ID>` only when you have access to the restricted internal inputs.
- Patient-level raw data are restricted under DUA and are not publicly shareable.
- Public reproducibility is based on the aggregate `data/public/` bundle, the manuscript/docs, and the figure-generation/build scripts included here.
- The current packet uses an outcome-defined monitor cohort: `monitor_id in fall_events_source -> intervention`; `monitor_id only in hourly_location_aggregation -> control`.
- The legacy v2 broader-observation raw consensus table is retained only for annotation context; it should not be relabeled as hard-label output in manuscript-facing materials.
- The hourly extraction begins on `2024-08-13`, but this does not change the 43-event study-window narrative: the staged fall-events parquet and the adjudicated consensus CSV contain no events dated August 1-12, 2024.
