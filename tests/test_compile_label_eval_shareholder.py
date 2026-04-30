# ruff: noqa: E402

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.paper_defaults import CURRENT_MANUSCRIPT_RUN_ID  # noqa: E402
from ld_chair_falls.report_descriptive import (  # noqa: E402
    resolve_label_eval_shareholder_run_id,
    write_label_eval_shareholder_report,
)
from ld_chair_falls.utils import write_json, write_yaml  # noqa: E402


def _write_label_eval_fixture(project_root: Path, run_id: str) -> None:
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
            "fall_events_source": {
                "rows": 93,
                "distinct_patient_id": 86,
                "distinct_monitor_id": 89,
            },
            "control_denominator_completeness": {
                "rows_with_valid_pct_sum": 100,
                "total_rows": 120,
            },
        },
    )
    write_json(
        qa_dir / f"qa_summary_{run_id}.json",
        {
            "run_id": run_id,
            "effective_run_mode": "descriptive_only",
            "analysis_base_rows": 500,
        },
    )

    (qa_dir / f"label_eval_sequence_metrics_{run_id}.csv").write_text(
        "metric,value\nscored_sequences,47\naccuracy,0.41\nmacro_f1,0.35\n",
        encoding="utf-8",
    )
    (qa_dir / f"label_eval_confusion_matrix_{run_id}.csv").write_text(
        (
            "truth_label,predicted_label,count\n"
            "chair,chair,3\n"
            "chair,bed,2\n"
            "bed,bed,8\n"
            "room,bed,4\n"
            "no_patient,no_patient,1\n"
        ),
        encoding="utf-8",
    )
    (qa_dir / f"label_eval_probability_quality_{run_id}.csv").write_text(
        "metric,value\nlog_loss,1.2\nece_10_bin,0.2\n",
        encoding="utf-8",
    )
    (qa_dir / f"label_eval_response_timing_{run_id}.csv").write_text(
        "metric,value\ndetection_f1,0.84\nlatency_mae_seconds,83\n",
        encoding="utf-8",
    )
    (qa_dir / f"label_eval_population_stats_{run_id}.csv").write_text(
        "metric,value\ntruth_rows,53\nsequence_count,50\nsequence_offscreen_count,3\ntruth_unique_monitors,47\n",
        encoding="utf-8",
    )
    (qa_dir / f"label_eval_truth_prefall_location_{run_id}.csv").write_text(
        "prefall_location,truth_rows,pct_of_truth_rows\nroom,22,0.4151\nbed,18,0.3396\nchair,9,0.1698\nno_patient,4,0.0755\n",
        encoding="utf-8",
    )
    (qa_dir / f"label_eval_tag_location_association_{run_id}.csv").write_text(
        (
            "prefall_location,fall_tag,annotations,annotations_for_location,annotations_for_tag,pct_of_location,pct_global,lift_vs_global\n"
            "bed,slip,7,21,24,0.3333,0.2667,1.2499\n"
            "room,slip,11,37,24,0.2973,0.2667,1.1149\n"
        ),
        encoding="utf-8",
    )
    (qa_dir / f"label_eval_signal_location_profile_{run_id}.csv").write_text(
        (
            "truth_label,sequences,accuracy_within_label,mean_prob_chair,mean_prob_bed,mean_prob_room,mean_prob_no_patient,truth_response_detect_rate,derived_response_detect_rate,mean_truth_latency_seconds,mean_derived_latency_seconds\n"
            "room,21,0.0952,0.057,0.556,0.1032,0.2361,1.0,0.6667,35.57,89.14\n"
            "bed,18,0.7222,0.0436,0.6591,0.1102,0.1871,0.9444,0.8333,63.0,90.8\n"
        ),
        encoding="utf-8",
    )
    (qa_dir / f"label_eval_association_summary_{run_id}.csv").write_text(
        "metric,value\ntag_location_cramers_v,0.641\ntop_truth_location_share,0.4151\n",
        encoding="utf-8",
    )
    (qa_dir / f"label_eval_shadow_model_metrics_{run_id}.csv").write_text(
        (
            "model,metric,value\n"
            "baseline_heuristic,macro_f1,0.35\n"
            "shadow_mnlogit,macro_f1,0.42\n"
        ),
        encoding="utf-8",
    )
    (qa_dir / f"label_eval_shadow_model_comparison_{run_id}.csv").write_text(
        (
            "metric,baseline_value,shadow_value,delta\n"
            "accuracy,0.41,0.48,0.07\n"
            "macro_f1,0.35,0.42,0.07\n"
            "ece_10_bin,0.20,0.16,-0.04\n"
        ),
        encoding="utf-8",
    )
    write_json(
        qa_dir / f"label_eval_threshold_checks_{run_id}.json",
        {
            "enabled": True,
            "mode": "dual",
            "evaluated": True,
            "overall_pass": False,
            "checks": [
                {
                    "name": "macro_f1",
                    "operator": ">=",
                    "threshold": 0.55,
                    "actual": 0.35,
                    "pass": False,
                }
            ],
        },
    )


def test_resolve_label_eval_run_id_prefers_latest_manifest(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_label_eval_fixture(project_root, "label_eval_old")
    _write_label_eval_fixture(project_root, "label_eval_new")

    old_path = project_root / "outputs" / "manifests" / "run_manifest_label_eval_old.yaml"
    new_path = project_root / "outputs" / "manifests" / "run_manifest_label_eval_new.yaml"
    os.utime(old_path, (100, 100))
    os.utime(new_path, (200, 200))

    assert resolve_label_eval_shareholder_run_id(project_root, None) == "label_eval_new"


def test_resolve_label_eval_run_id_prefers_locked_manuscript_bundle_when_available(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_label_eval_fixture(project_root, "label_eval_new")
    _write_label_eval_fixture(project_root, CURRENT_MANUSCRIPT_RUN_ID)

    default_path = (
        project_root
        / "outputs"
        / "manifests"
        / f"run_manifest_{CURRENT_MANUSCRIPT_RUN_ID}.yaml"
    )
    latest_path = project_root / "outputs" / "manifests" / "run_manifest_label_eval_new.yaml"
    os.utime(default_path, (100, 100))
    os.utime(latest_path, (200, 200))

    assert resolve_label_eval_shareholder_run_id(project_root, None) == CURRENT_MANUSCRIPT_RUN_ID


def test_write_label_eval_shareholder_report_generates_html(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_id = "label_eval_fixture"
    _write_label_eval_fixture(project_root, run_id)

    output = write_label_eval_shareholder_report(project_root, run_id)
    html = output.read_text(encoding="utf-8")

    assert output.exists()
    assert str(output).endswith("outputs/label_eval_shareholder_report.html")
    assert "Shareholder Report: Livestream Label Evaluation" in html
    assert "KPI Scorecards" in html
    assert "Visual Summary" in html
    assert "Confusion Matrix (Sequence-Level)" in html
    assert "Threshold Gate" in html
    assert "Labeled Cohort Snapshot" in html
    assert "Shadow Model Delta (Report-Only)" in html
    assert "Tag-to-Location Association Table" in html
    assert "Signal Profile by Truth Location" in html
    assert "Minimal Table Appendix" in html
