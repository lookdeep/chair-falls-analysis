# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.config import load_settings
from ld_chair_falls.extract import run_extract_pipeline
from ld_chair_falls.manifest import write_run_manifest
from ld_chair_falls.qa import run_qa_pipeline
from ld_chair_falls.transform import run_transform_pipeline


@pytest.mark.smoke
def test_end_to_end_dry_run_smoke() -> None:
    run_id = f"smoke_pipeline_{uuid4().hex[:8]}"
    settings = load_settings(run_id=run_id, dry_run=True)

    run_extract_pipeline(settings)
    transform_manifest = run_transform_pipeline(settings)
    run_qa_pipeline(settings)
    manifest = write_run_manifest(settings)
    assert manifest.exists()
    assert (settings.paths.qa_dir / f"falls_prefall_location_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"falls_prefall_location_probabilities_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"falls_prefall_location_probability_breakdown_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"falls_response_latency_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"falls_patient_day_hour_location_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"fall_second_level_panel_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"fall_event_localized_metrics_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"fall_response_curve_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"fall_case_crossover_diagnostics_{run_id}.json").exists()
    assert (settings.paths.qa_dir / f"fall_negative_control_diagnostics_{run_id}.json").exists()
    assert (settings.paths.qa_dir / f"fall_case_crossover_sets_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"fall_case_crossover_effects_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"fall_negative_control_sets_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"fall_negative_control_effects_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"fall_negative_control_coverage_{run_id}.json").exists()
    assert (settings.paths.qa_dir / f"fall_negative_control_match_quality_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"falls_month_of_year_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"label_eval_truth_prefall_location_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"label_eval_tag_location_association_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"label_eval_signal_location_profile_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"label_eval_association_summary_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"label_eval_shadow_model_metrics_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"label_eval_shadow_model_predictions_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"label_eval_shadow_model_comparison_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"label_eval_benchmark_sequence_metrics_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"label_eval_benchmark_label_metrics_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"label_eval_benchmark_confusion_matrix_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"label_eval_benchmark_probability_quality_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"label_eval_sequence_predictions_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"label_eval_candidate_comparison_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"label_eval_benchmark_status_{run_id}.json").exists()
    assert (settings.paths.qa_dir / f"onset_events_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"onset_case_crossover_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"shadow_feature_importance_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"shadow_ablation_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"shadow_window_sensitivity_{run_id}.csv").exists()
    assert (settings.paths.qa_dir / f"shadow_cv_fold_metrics_{run_id}.csv").exists()
    assert not (settings.paths.qa_dir / f"chair_bed_risk_rates_{run_id}.csv").exists()
    assert not (
        settings.paths.qa_dir / f"chair_bed_risk_rates_confidence_sensitivity_{run_id}.csv"
    ).exists()
    assert not (settings.paths.qa_dir / f"chair_bed_operational_event_rates_{run_id}.csv").exists()
    assert transform_manifest["hospital_timezone"] == "America/Chicago"
    assert "cohort_validation_status" in transform_manifest
    assert "cohort_fallback_used" in transform_manifest
