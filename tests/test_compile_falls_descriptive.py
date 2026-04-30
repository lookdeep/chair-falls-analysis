# ruff: noqa: E402

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.paper_defaults import CURRENT_MANUSCRIPT_RUN_ID  # noqa: E402
from ld_chair_falls.report_descriptive import (  # noqa: E402
    compare_descriptive_runs,
    load_chair_bed_inference_inputs,
    load_descriptive_inputs,
    load_full_cohort_hourly_inputs,
    resolve_chair_bed_inference_run_id,
    resolve_full_cohort_run_id,
    resolve_run_id,
    write_chair_bed_inference_report,
    write_falls_only_descriptive,
    write_full_cohort_hourly_descriptive,
)
from ld_chair_falls.utils import write_json, write_yaml  # noqa: E402


def _write_run_fixture(project_root: Path, run_id: str) -> None:
    qa_dir = project_root / "outputs" / "qa"
    manifests_dir = project_root / "outputs" / "manifests"
    qa_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    write_yaml(
        manifests_dir / f"run_manifest_{run_id}.yaml",
        {
            "run_id": run_id,
            "run_mode": "descriptive_only",
            "requested_run_mode": "descriptive_only",
            "final_publication_artifact_target": "paper/manuscript.md",
        },
    )
    write_json(
        qa_dir / f"source_profile_{run_id}.json",
        {
            "case_crossover_source_status": "ready",
            "negative_control_source_status": "pending_external",
            "nonfall_control_source_status": "pending_external",
            "fall_events_source": {
                "rows": 93,
                "distinct_patient_id": 86,
                "distinct_monitor_id": 89,
                "distinct_hospital_id": 2,
                "distinct_division_id": 11,
            },
            "negative_control_source_inventory": {
                "chunk_count": 0,
                "history_status": "unavailable",
                "effective_start_ts_utc": None,
            },
            "control_denominator_completeness": {
                "rows_with_valid_pct_sum": 0,
                "total_rows": 0,
            },
        },
    )
    write_json(
        qa_dir / f"qa_summary_{run_id}.json",
        {
            "run_id": run_id,
            "effective_run_mode": "descriptive_only",
            "case_crossover_source_status": "ready",
            "negative_control_source_status": "pending_external",
            "nonfall_control_source_status": "pending_external",
            "eligibility_units": 0,
            "eligibility_units_passed": 0,
            "analysis_base_rows": 0,
        },
    )

    (qa_dir / f"falls_by_site_{run_id}.csv").write_text(
        "hospital_id,hospital_system_name,division_id,hospital_name,falls\n"
        "5,Baptist,5,NEA,31\n",
        encoding="utf-8",
    )
    (qa_dir / f"falls_site_name_aliases_{run_id}.csv").write_text(
        "hospital_id,division_id,canonical_hospital_name,alias_hospital_name,alias_rows\n",
        encoding="utf-8",
    )
    (qa_dir / f"falls_by_weekday_{run_id}.csv").write_text(
        "weekday,falls\nSaturday,18\n",
        encoding="utf-8",
    )
    (qa_dir / f"falls_by_daypart_{run_id}.csv").write_text(
        "daypart,falls\nwindow_12_14,31\n",
        encoding="utf-8",
    )
    (qa_dir / f"falls_by_month_{run_id}.csv").write_text(
        "month,falls\n2026-02,1\n",
        encoding="utf-8",
    )
    (qa_dir / f"falls_month_of_year_{run_id}.csv").write_text(
        "month_num,month_of_year,falls,pct_of_falls\n2,Feb,1,1.0\n",
        encoding="utf-8",
    )
    (qa_dir / f"falls_prefall_location_{run_id}.csv").write_text(
        "prefall_location_label,falls,pct_of_falls,median_margin\nroom,1,1.0,0.11\n",
        encoding="utf-8",
    )
    (qa_dir / f"falls_prefall_location_probabilities_{run_id}.csv").write_text(
        "prefall_location_label,expected_falls,pct_of_falls\nchair,0.2,0.2\nbed,0.5,0.5\nroom,0.2,0.2\nno_patient,0.1,0.1\n",
        encoding="utf-8",
    )
    (qa_dir / f"falls_prefall_location_probability_breakdown_{run_id}.csv").write_text(
        (
            "grouping,group_value,prefall_location_label,falls_in_group,expected_falls,mean_probability,median_probability,pct_expected_in_group\n"
            "daypart,window_12_14,bed,1,0.5,0.5,0.5,0.5\n"
        ),
        encoding="utf-8",
    )
    (qa_dir / f"falls_response_latency_{run_id}.csv").write_text(
        "metric,value\ntotal_falls,1\nresponses_detected,1\nresponse_rate,1.0\n",
        encoding="utf-8",
    )
    (qa_dir / f"falls_patient_day_hour_location_{run_id}.csv").write_text(
        (
            "patient_id,fall_date_local,fall_hour_local,prefall_location_label,falls,bucket_total_falls,pct_in_bucket\n"
            "101,2026-02-14,10,room,1,1,1.0\n"
        ),
        encoding="utf-8",
    )


def _write_full_cohort_fixture(project_root: Path, run_id: str) -> None:
    _write_run_fixture(project_root, run_id)
    qa_dir = project_root / "outputs" / "qa"
    (qa_dir / f"cohort_composition_{run_id}.csv").write_text(
        "cohort_type,hospital_id,division_id,monitor_id,hour_rows\n"
        "control,5,,1001,120\n"
        "intervention,5,,2002,80\n",
        encoding="utf-8",
    )
    (qa_dir / f"cohort_duration_{run_id}.csv").write_text(
        "cohort_type,count,mean,std,min,25%,50%,75%,max\n"
        "control,10,50,5,40,45,50,55,60\n"
        "intervention,5,80,10,60,70,80,90,100\n",
        encoding="utf-8",
    )
    (qa_dir / f"cohort_eligibility_rates_{run_id}.csv").write_text(
        "cohort_type,eligible,units\n"
        "control,True,8\n"
        "control,False,2\n"
        "intervention,True,4\n"
        "intervention,False,1\n",
        encoding="utf-8",
    )
    (qa_dir / f"cohort_exclusions_{run_id}.csv").write_text(
        "hospital_id,division_id,monitor_id,patient_id,exclusion_reason\n"
        "5,,1001,11,insufficient_hours\n",
        encoding="utf-8",
    )
    (qa_dir / f"fall_density_{run_id}.csv").write_text(
        "cohort_type,daypart,falls\n"
        "control,window_06_08,0\n"
        "intervention,window_06_08,2\n",
        encoding="utf-8",
    )
    (qa_dir / f"control_denominator_coverage_{run_id}.csv").write_text(
        "cohort_type,hospital_id,division_id,rows,valid_rows\n"
        "control,5,,1000,900\n"
        "intervention,5,,200,180\n",
        encoding="utf-8",
    )
    (qa_dir / f"chair_bed_risk_rates_{run_id}.csv").write_text(
        "scope,daypart,position,eligible_event_count,exposure_hours,expected_falls,hard_label_falls,rate_per_1000_exposure_hours_expected,rate_per_1000_exposure_hours_hard_label\n"
        "intervention_eligible,all_dayparts,chair,4,100.0,1.2,1,12.0,10.0\n",
        encoding="utf-8",
    )
    (qa_dir / f"chair_bed_risk_rates_confidence_sensitivity_{run_id}.csv").write_text(
        (
            "scope,confidence_segment,position,eligible_event_count,mean_confidence_weight,exposure_hours,expected_falls,confidence_weighted_expected_falls,hard_label_falls,rate_per_1000_exposure_hours_expected,rate_per_1000_exposure_hours_confidence_weighted,rate_per_1000_exposure_hours_hard_label\n"
            "intervention_eligible,all_events,chair,4,0.75,100.0,1.2,0.9,1,12.0,9.0,10.0\n"
        ),
        encoding="utf-8",
    )
    (qa_dir / f"chair_bed_missingness_stress_{run_id}.csv").write_text(
        (
            "scope,excluded_low_confidence_pct,retained_event_count,excluded_event_count,retained_fraction,mean_confidence_weight_retained,position,exposure_hours,expected_falls,confidence_weighted_expected_falls,hard_label_falls,rate_per_1000_exposure_hours_expected,rate_per_1000_exposure_hours_confidence_weighted,rate_per_1000_exposure_hours_hard_label,chair_to_bed_rate_ratio_expected,chair_to_bed_rate_ratio_confidence_weighted,chair_to_bed_rate_ratio_hard_label\n"
            "intervention_eligible,0,12,0,1.0,0.78,chair,100.0,1.2,0.9,1,12.0,9.0,10.0,1.5,1.4,1.3\n"
            "intervention_eligible,0,12,0,1.0,0.78,bed,150.0,1.0,0.8,1,8.0,6.4,6.7,1.5,1.4,1.3\n"
            "intervention_eligible,30,8,4,0.6667,0.86,chair,100.0,0.9,0.75,1,9.0,7.5,10.0,1.35,1.28,1.2\n"
            "intervention_eligible,30,8,4,0.6667,0.86,bed,150.0,1.0,0.88,1,6.7,5.9,6.7,1.35,1.28,1.2\n"
        ),
        encoding="utf-8",
    )
    write_json(
        qa_dir / f"fall_negative_control_diagnostics_{run_id}.json",
        {
            "run_id": run_id,
            "status": "pending_external_source",
            "control_source_status": "pending_external",
            "events_with_hazard": 0,
            "events_with_negative_control": 0,
            "events_with_negative_control_pair": 0,
        },
    )
    write_json(
        qa_dir / f"fall_negative_control_coverage_{run_id}.json",
        {
            "run_id": run_id,
            "history_status": "pending_external_source",
            "source_chunk_count": 0,
            "fall_events_eligible_for_matching": 0,
            "fall_events_with_selected_controls": 0,
            "effective_source_start_ts_utc": None,
        },
    )
    (qa_dir / f"fall_negative_control_match_quality_{run_id}.csv").write_text(
        (
            "fall_event_id,hazard_anchor_ts_utc,division_id,source_monitor_id,hazard_eligible,"
            "after_effective_source_start,division_available,division_chunk_count,matching_candidates,"
            "selected_control_count,best_match_tier,excluded_reason\n"
        ),
        encoding="utf-8",
    )
    (qa_dir / f"cohort_analysis_{run_id}.md").write_text(
        f"# Cohort Analysis {run_id}\n\n## Summary\n- Hourly rows: 1200\n",
        encoding="utf-8",
    )
    (qa_dir / f"transform_metrics_{run_id}.json").write_text(
        (
            "{"
            "\"run_id\": \"" + run_id + "\","
            "\"mapping_report\": {\"hourly_mapped_rate\": 1.0, \"events_mapped_rate\": 1.0},"
            "\"row_counts\": {\"patient_hour_analysis_base\": 12, \"patient_hour_eligibility\": 15}"
            "}"
        ),
        encoding="utf-8",
    )


def test_resolve_run_id_prefers_latest_manifest(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_run_fixture(project_root, "falls_only_old")
    _write_run_fixture(project_root, "falls_only_new")

    old_path = project_root / "outputs" / "manifests" / "run_manifest_falls_only_old.yaml"
    new_path = project_root / "outputs" / "manifests" / "run_manifest_falls_only_new.yaml"
    os.utime(old_path, (100, 100))
    os.utime(new_path, (200, 200))

    assert resolve_run_id(project_root, None) == "falls_only_new"
    assert resolve_run_id(project_root, "falls_only_old") == "falls_only_old"


def test_resolve_run_id_prefers_locked_manuscript_bundle_when_available(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_run_fixture(project_root, "falls_only_new")
    _write_run_fixture(project_root, CURRENT_MANUSCRIPT_RUN_ID)

    default_path = (
        project_root
        / "outputs"
        / "manifests"
        / f"run_manifest_{CURRENT_MANUSCRIPT_RUN_ID}.yaml"
    )
    latest_path = project_root / "outputs" / "manifests" / "run_manifest_falls_only_new.yaml"
    os.utime(default_path, (100, 100))
    os.utime(latest_path, (200, 200))

    assert resolve_run_id(project_root, None) == CURRENT_MANUSCRIPT_RUN_ID


def test_resolve_run_id_rejects_partial_locked_manuscript_bundle(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_run_fixture(project_root, CURRENT_MANUSCRIPT_RUN_ID)

    (
        project_root
        / "outputs"
        / "qa"
        / f"falls_response_latency_{CURRENT_MANUSCRIPT_RUN_ID}.csv"
    ).unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Default manuscript run_id=.*is incomplete for this surface",
    ):
        resolve_run_id(project_root, None)


def test_write_falls_only_descriptive_generates_html(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_id = "falls_only_fixture"
    _write_run_fixture(project_root, run_id)

    output = write_falls_only_descriptive(project_root, run_id)
    html = output.read_text(encoding="utf-8")

    assert output.exists()
    assert str(output).endswith("outputs/falls_only_descriptive.html")
    assert "Falls-Only Descriptive Report" in html
    assert run_id in html
    assert "93" in html
    assert "descriptive" in html.lower()
    assert "Conclusions" in html
    assert "Visual Summary" in html
    assert "Appendix Visuals" in html
    assert "Minimal Table Appendix" in html
    assert "KPI Snapshot" in html
    assert "Interactive Location Drilldown" in html
    assert "Response Rate" in html
    assert "Site Pareto" in html
    assert "Observed vs Probability-Weighted Location Burden" in html
    assert "Smoothed Hourly Density by Pre-Fall Location (KDE)" in html
    assert "Weekday x Daypart Heatmap" in html
    assert "Observed Pre-Fall Location Burden" in html
    assert "Expected Pre-Fall Location Burden" in html
    assert "Pre-Fall Probability Breakdown Small Multiples" in html
    assert "Response Latency (Livestream)" in html
    assert "<h2>Falls by Month</h2>" not in html
    assert "<h2>Patient-Day-Hour Location (Livestream)</h2>" not in html
    assert "room" in html.lower()
    assert "outputs/qa/falls_by_site_" in html
    assert str(project_root) not in html
    assert "1.0000" not in html


def test_resolve_full_cohort_run_id_prefers_latest_manifest(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_full_cohort_fixture(project_root, "cohort_old")
    _write_full_cohort_fixture(project_root, "cohort_new")

    old_path = project_root / "outputs" / "manifests" / "run_manifest_cohort_old.yaml"
    new_path = project_root / "outputs" / "manifests" / "run_manifest_cohort_new.yaml"
    os.utime(old_path, (100, 100))
    os.utime(new_path, (200, 200))

    assert resolve_full_cohort_run_id(project_root, None) == "cohort_new"


def test_resolve_full_cohort_run_id_prefers_locked_manuscript_bundle_when_available(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_full_cohort_fixture(project_root, "cohort_new")
    _write_full_cohort_fixture(project_root, CURRENT_MANUSCRIPT_RUN_ID)

    default_path = (
        project_root
        / "outputs"
        / "manifests"
        / f"run_manifest_{CURRENT_MANUSCRIPT_RUN_ID}.yaml"
    )
    latest_path = project_root / "outputs" / "manifests" / "run_manifest_cohort_new.yaml"
    os.utime(default_path, (100, 100))
    os.utime(latest_path, (200, 200))

    assert resolve_full_cohort_run_id(project_root, None) == CURRENT_MANUSCRIPT_RUN_ID


def test_write_full_cohort_descriptive_generates_html(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_id = "cohort_fixture"
    _write_full_cohort_fixture(project_root, run_id)

    output = write_full_cohort_hourly_descriptive(project_root, run_id)
    html = output.read_text(encoding="utf-8")

    assert output.exists()
    assert str(output).endswith("outputs/full_cohort_hourly_descriptive.html")
    assert "Full-Cohort Hourly Descriptive Report" in html
    assert run_id in html
    assert "Eligibility Mix by Cohort" in html
    assert "Control Denominator Coverage" in html
    assert "Confidence Sensitivity Small Multiples" in html
    assert "Visual Summary" in html
    assert "Appendix Visuals" in html
    assert "Minimal Table Appendix" in html
    assert "Cohort Duration Distribution" in html
    assert "Confidence Sensitivity Small Multiples" in html
    assert "Interactive Eligibility Lens" in html
    assert "Intervention cohort:" in html
    assert "patients who experienced at least one fall during their stay." in html
    assert "Control cohort:" in html
    assert "patients with no documented fall during their stay." in html


def test_full_cohort_context_uses_gate_preflight_mode_and_warning(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_id = "cohort_gate_mismatch"
    _write_full_cohort_fixture(project_root, run_id)
    manifests_dir = project_root / "outputs" / "manifests"
    write_yaml(
        manifests_dir / f"run_manifest_{run_id}.yaml",
        {
            "run_id": run_id,
            "run_mode": "inferential_ready",
            "requested_run_mode": "descriptive_only",
            "final_publication_artifact_target": "paper/manuscript.md",
            "gate_preflight": {
                "effective_run_mode": "descriptive_only",
                "gate_1_pass": True,
                "gate_2_pass": False,
            },
        },
    )

    output = write_full_cohort_hourly_descriptive(project_root, run_id)
    html = output.read_text(encoding="utf-8")

    assert "Gate preflight mode:</strong> descriptive_only" in html
    assert "Manifest mode:</strong> inferential_ready" in html
    assert "Consistency warning" in html
    assert "Gate 1=pass, Gate 2=fail" in html
    assert "Gate 2 is not passing" in html


def test_falls_site_pareto_reports_other_sites_share(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_id = "falls_many_sites"
    _write_run_fixture(project_root, run_id)
    qa_dir = project_root / "outputs" / "qa"
    (qa_dir / f"falls_by_site_{run_id}.csv").write_text(
        (
            "hospital_id,hospital_system_name,division_id,hospital_name,falls\n"
            "5,Baptist,1,Site A,31\n"
            "5,Baptist,2,Site B,12\n"
            "5,Baptist,3,Site C,9\n"
            "5,Baptist,4,Site D,8\n"
            "5,Baptist,5,Site E,7\n"
            "5,Baptist,6,Site F,7\n"
            "5,Baptist,7,Site G,5\n"
            "5,Baptist,8,Site H,3\n"
            "5,Baptist,9,Site I,2\n"
            "5,Baptist,10,Site J,1\n"
        ),
        encoding="utf-8",
    )

    output = write_falls_only_descriptive(project_root, run_id)
    html = output.read_text(encoding="utf-8")
    assert "other 2 site(s) contribute 3/93 (3.2%)" in html


def test_full_cohort_daypart_heatmap_uses_display_labels(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_id = "cohort_daypart_labels"
    _write_full_cohort_fixture(project_root, run_id)
    qa_dir = project_root / "outputs" / "qa"
    (qa_dir / f"fall_density_{run_id}.csv").write_text(
        (
            "cohort_type,daypart,falls\n"
            "control,window_00_05,0\n"
            "control,window_06_08,0\n"
            "control,window_12_14,0\n"
            "intervention,window_00_05,8\n"
            "intervention,window_06_08,7\n"
            "intervention,window_12_14,12\n"
        ),
        encoding="utf-8",
    )

    output = write_full_cohort_hourly_descriptive(project_root, run_id)
    html = output.read_text(encoding="utf-8")
    assert "12:00-14:59" in html
    assert "window_12_14" not in html


def test_resolve_chair_bed_inference_run_id_prefers_latest_manifest(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_full_cohort_fixture(project_root, "inference_old")
    _write_full_cohort_fixture(project_root, "inference_new")

    old_path = project_root / "outputs" / "manifests" / "run_manifest_inference_old.yaml"
    new_path = project_root / "outputs" / "manifests" / "run_manifest_inference_new.yaml"
    os.utime(old_path, (100, 100))
    os.utime(new_path, (200, 200))

    assert resolve_chair_bed_inference_run_id(project_root, None) == "inference_new"


def test_resolve_chair_bed_inference_run_id_prefers_locked_manuscript_bundle_when_available(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _write_full_cohort_fixture(project_root, "inference_new")
    _write_full_cohort_fixture(project_root, CURRENT_MANUSCRIPT_RUN_ID)

    default_path = (
        project_root
        / "outputs"
        / "manifests"
        / f"run_manifest_{CURRENT_MANUSCRIPT_RUN_ID}.yaml"
    )
    latest_path = project_root / "outputs" / "manifests" / "run_manifest_inference_new.yaml"
    os.utime(default_path, (100, 100))
    os.utime(latest_path, (200, 200))

    assert resolve_chair_bed_inference_run_id(project_root, None) == CURRENT_MANUSCRIPT_RUN_ID


def test_write_chair_bed_inference_report_generates_html(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_id = "inference_fixture"
    _write_full_cohort_fixture(project_root, run_id)

    output = write_chair_bed_inference_report(project_root, run_id)
    html = output.read_text(encoding="utf-8")

    assert output.exists()
    assert str(output).endswith("outputs/chair_bed_inference_descriptive.html")
    assert "Chair-Bed Inference Support Report" in html
    assert run_id in html
    assert "Expected vs Hard-Label Falls" in html
    assert "Confidence Sensitivity Small Multiples" in html
    assert "Visual Summary" in html
    assert "Appendix Visuals" in html
    assert "Minimal Table Appendix" not in html
    assert "Confidence Sensitivity Small Multiples" in html
    assert "Missingness Stress (Chair/Bed RR)" in html
    assert "Non-Fall Controls Pending" in html
    assert "Pre-Fall Location Probabilities" in html
    assert "Interactive Confidence Segment Lens" in html
    assert "Interpretation Guardrails" in html
    assert "Conclusions" in html
    assert "outputs/qa/chair_bed_risk_rates_" in html
    assert str(project_root) not in html


def test_write_chair_bed_inference_report_handles_empty_optional_csvs(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_id = "inference_optional_empty"
    _write_full_cohort_fixture(project_root, run_id)
    qa_dir = project_root / "outputs" / "qa"
    (qa_dir / f"cohort_eligibility_rates_{run_id}.csv").write_text("", encoding="utf-8")
    (qa_dir / f"control_denominator_coverage_{run_id}.csv").write_text("", encoding="utf-8")
    (qa_dir / f"falls_prefall_location_probabilities_{run_id}.csv").write_text("", encoding="utf-8")

    output = write_chair_bed_inference_report(project_root, run_id)
    html = output.read_text(encoding="utf-8")

    assert output.exists()
    assert "Chair-Bed Inference Support Report" in html
    assert "Expected vs Hard-Label Falls" in html
    assert "Appendix Visuals" in html


def test_load_chair_bed_inference_inputs_missing_artifacts_raise_clear_error(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_id = "inference_missing"
    _write_full_cohort_fixture(project_root, run_id)
    missing = project_root / "outputs" / "qa" / f"chair_bed_risk_rates_{run_id}.csv"
    missing.unlink()

    with pytest.raises(FileNotFoundError) as exc_info:
        load_chair_bed_inference_inputs(project_root, run_id)

    assert run_id in str(exc_info.value)
    assert "chair_bed_risk_rates" in str(exc_info.value)


def test_load_full_cohort_inputs_missing_artifacts_raise_clear_error(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_id = "cohort_missing"
    _write_full_cohort_fixture(project_root, run_id)
    missing = project_root / "outputs" / "qa" / f"cohort_duration_{run_id}.csv"
    missing.unlink()

    with pytest.raises(FileNotFoundError) as exc_info:
        load_full_cohort_hourly_inputs(project_root, run_id)

    assert run_id in str(exc_info.value)
    assert "cohort_duration" in str(exc_info.value)


def test_compare_descriptive_runs_flags_major_disagreement(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_full_cohort_fixture(project_root, "baseline")
    _write_full_cohort_fixture(project_root, "candidate")

    qa_dir = project_root / "outputs" / "qa"
    write_json(
        qa_dir / "qa_summary_candidate.json",
        {
            "run_id": "candidate",
            "effective_run_mode": "descriptive_only",
            "eligibility_units": 15,
            "eligibility_units_passed": 14,
            "analysis_base_rows": 70000,
        },
    )
    (qa_dir / "transform_metrics_candidate.json").write_text(
        (
            "{"
            "\"run_id\": \"candidate\","
            "\"mapping_report\": {\"hourly_mapped_rate\": 0.0, \"events_mapped_rate\": 0.0},"
            "\"row_counts\": {\"patient_hour_analysis_base\": 70000, \"patient_hour_eligibility\": 15}"
            "}"
        ),
        encoding="utf-8",
    )

    comparison = compare_descriptive_runs(project_root, "baseline", "candidate")
    major_metrics = {row["metric"] for row in comparison["major_disagreements"]}
    assert comparison["major_disagreement_count"] > 0
    assert "analysis_base_rows" in major_metrics
    assert "hourly_mapped_rate" in major_metrics


def test_missing_artifacts_raise_clear_error(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_id = "falls_only_missing"
    _write_run_fixture(project_root, run_id)

    missing = project_root / "outputs" / "qa" / f"falls_prefall_location_{run_id}.csv"
    missing.unlink()

    with pytest.raises(FileNotFoundError) as exc_info:
        load_descriptive_inputs(project_root, run_id)

    message = str(exc_info.value)
    assert run_id in message
    assert "falls_prefall_location" in message
