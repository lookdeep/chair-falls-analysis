# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.config import Settings
from ld_chair_falls.extract import EXTRACT_SPECS
from ld_chair_falls.label_eval import (
    BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S,
    BENCHMARK_SPLIT_PRIMARY_4CLASS_V2_RAW_AUDIT,
    _benchmark_status,
    _build_sequence_truth_predictions,
    _combine_shadow_panels,
    _detect_onset_events,
    _match_with_derived,
    _trajectory_feature_rows,
    evaluate_livestream_derivations_against_truth,
)


def _make_settings(
    tmp_path: Path,
    *,
    thresholds_enabled: bool = False,
    shadow_cv_folds: int = 5,
    shadow_cv_repeats: int = 3,
) -> Settings:
    public_dir = tmp_path / "data" / "public"
    public_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        project_root=tmp_path,
        run_id="label_eval_test",
        dry_run=True,
        label_eval_mode="dual",
        label_eval_thresholds_enabled=thresholds_enabled,
        label_eval_shadow_cv_folds=shadow_cv_folds,
        label_eval_shadow_cv_repeats=shadow_cv_repeats,
        fall_labels_consensus_csv_path=Path("data/public/falls-observations-v3-consensus.csv"),
        fall_labels_raw_logs_csv_path=Path("docs/falls-observations-v1 - raw_logs.csv"),
        fall_labels_rubric_csv_path=Path("docs/falls-observations-v1 - rubric.csv"),
    )


def _truth_instance_row(
    *,
    event_instance_id: str = "evt-1",
    event_key: str = "111|2025-01-01|10:00",
    sequence_id: str = "111|2025-01-01|10:00#S01",
    event_instance_ordinal: int = 1,
    prefall_location: str = "chair",
) -> dict[str, object]:
    return {
        "event_instance_id": event_instance_id,
        "event_key": event_key,
        "sequence_id": sequence_id,
        "event_instance_ordinal": event_instance_ordinal,
        "prefall_location": prefall_location,
        "benchmark_truth_label": prefall_location,
        "benchmark_truth_label_raw": prefall_location,
        "departure_aware_truth_applied": False,
        "departure_aware_source_furniture": pd.NA,
        "departure_aware_latency_seconds": pd.NA,
        "offscreen_flag": False,
        "fall_time_seconds": 10.0,
        "response_time_seconds": pd.NA,
        "monitor_id": 111,
        "date_local": "2025-01-01",
        "minute_local": "10:00",
    }


def test_sequence_level_evaluation_uses_last_instance_label_and_scores_probabilities(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    consensus = pd.DataFrame(
        {
            "event_key": ["111|2025-01-01|10:00", "111|2025-01-01|10:00", "111|2025-01-01|10:00"],
            "prefall_location": ["chair", "room", "bed"],
            "fall_time_consensus": ["10:00:05", "10:00:30", "10:01:10"],
            "response_time_consensus": ["", "10:00:40", "10:01:40"],
            "fall_tags": ["slip", "trip", "collapse"],
        }
    )
    consensus.to_csv(tmp_path / settings.fall_labels_consensus_csv_path, index=False)

    event_windows = pd.DataFrame(
        {
            "monitor_id": [111, 111, 111],
            "fall_ts_local": [
                "2025-01-01T10:00:05Z",
                "2025-01-01T10:00:30Z",
                "2025-01-01T10:01:10Z",
            ],
            "fall_ts_utc": [
                "2025-01-01T10:00:05Z",
                "2025-01-01T10:00:30Z",
                "2025-01-01T10:01:10Z",
            ],
            "pre_prob_chair": [0.9, 0.2, 0.1],
            "pre_prob_bed": [0.05, 0.2, 0.8],
            "pre_prob_room": [0.05, 0.6, 0.1],
            "pre_prob_no_patient": [0.0, 0.0, 0.0],
            "response_detected": [False, True, True],
            "response_latency_seconds": [None, 12.0, 18.0],
        }
    )

    artifacts = evaluate_livestream_derivations_against_truth(settings, event_windows)

    assert not artifacts.sequence_metrics.empty
    assert not artifacts.confusion_matrix.empty
    assert not artifacts.probability_quality.empty
    assert not artifacts.auc_summary.empty
    assert not artifacts.threshold_sweep.empty
    assert not artifacts.calibration_curve.empty
    assert not artifacts.response_timing.empty
    assert not artifacts.population_stats.empty
    assert not artifacts.truth_prefall_location.empty
    assert not artifacts.signal_location_profile.empty
    assert not artifacts.association_summary.empty
    assert artifacts.operating_point["objective"] == "balanced_macro_f1"

    scored_sequences = artifacts.sequence_metrics.loc[
        artifacts.sequence_metrics["metric"] == "scored_sequences", "value"
    ].iloc[0]
    assert scored_sequences == 2.0


def test_offscreen_rows_are_excluded_from_baseline_truth(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    consensus = pd.DataFrame(
        {
            "event_key": ["222|2025-01-02|11:00"],
            "prefall_location": ["no_patient"],
            "fall_time_consensus": [""],
            "response_time_consensus": ["11:00:20"],
            "fall_tags": ["bathroom"],
        }
    )
    consensus.to_csv(tmp_path / settings.fall_labels_consensus_csv_path, index=False)

    event_windows = pd.DataFrame(
        {
            "monitor_id": [222],
            "fall_ts_local": ["2025-01-02T11:00:00Z"],
            "fall_ts_utc": ["2025-01-02T11:00:00Z"],
            "pre_prob_chair": [0.0],
            "pre_prob_bed": [0.0],
            "pre_prob_room": [0.0],
            "pre_prob_no_patient": [1.0],
            "response_detected": [True],
            "response_latency_seconds": [10.0],
        }
    )

    artifacts = evaluate_livestream_derivations_against_truth(settings, event_windows)
    scored_sequences = artifacts.sequence_metrics.loc[
        artifacts.sequence_metrics["metric"] == "scored_sequences", "value"
    ]
    assert scored_sequences.empty or float(scored_sequences.iloc[0]) == 0.0

    truth_rows = artifacts.population_stats.loc[
        artifacts.population_stats["metric"] == "truth_rows", "value"
    ].iloc[0]
    assert truth_rows == 0.0
    assert artifacts.signal_location_profile.empty
    primary_rows = artifacts.candidate_comparison.loc[
        artifacts.candidate_comparison["benchmark_split"] == BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S
    ]
    assert not primary_rows.empty
    no_patient_f1 = primary_rows.loc[
        primary_rows["model_version"] == "legacy", "f1_no_patient"
    ].iloc[0]
    assert float(no_patient_f1) >= 0.0


def test_match_with_derived_falls_back_to_single_event_key_match_when_exact_ordinal_is_missing() -> None:
    truth_instances = pd.DataFrame(
        [
            _truth_instance_row(
                event_instance_id="evt-1",
                event_key="2834|2023-12-29|11:11",
                sequence_id="2834|2023-12-29|11:11#S02",
                event_instance_ordinal=2,
                prefall_location="room",
            )
        ]
    )
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [16],
            "monitor_id": [2834],
            "fall_ts_local": ["2023-12-29T11:11:08Z"],
            "fall_ts_utc": ["2023-12-29T11:11:08Z"],
            "pre_prob_chair": [0.60],
            "pre_prob_bed": [0.05],
            "pre_prob_room": [0.34],
            "pre_prob_no_patient": [0.01],
            "response_detected": [False],
            "response_latency_seconds": [None],
        }
    )

    matched = _match_with_derived(truth_instances, event_windows)
    row = matched.iloc[0]

    assert row["match_source"] == "single_event_key_fallback"
    assert row["fall_event_id"] == 16
    assert float(row["pre_prob_chair"]) == 0.60


def test_sequence_predictions_mark_unmatched_zero_prob_sequences_unscored() -> None:
    matched = pd.DataFrame(
        [
            {
                **_truth_instance_row(prefall_location="no_patient"),
                "pre_prob_chair": pd.NA,
                "pre_prob_bed": pd.NA,
                "pre_prob_room": pd.NA,
                "pre_prob_no_patient": pd.NA,
                "response_detected": pd.NA,
                "response_latency_seconds": pd.NA,
                "prediction_reason": pd.NA,
                "match_source": "unmatched",
            }
        ]
    )

    sequence_predictions = _build_sequence_truth_predictions(matched, allow_offscreen_scoring=True)
    row = sequence_predictions.iloc[0]

    assert row["derived_label"] == "no_patient"
    assert row["prediction_reason"] == "no_derived_match"
    assert row["match_source"] == "unmatched"
    assert bool(row["has_derived_match"]) is False
    assert float(row["derived_prob_sum"]) == 0.0
    assert bool(row["scored_for_location"]) is False


def test_no_fall_rows_are_excluded_from_baseline_truth(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    consensus = pd.DataFrame(
        {
            "event_key": ["223|2025-01-02|11:01"],
            "prefall_location": ["bed"],
            "fall_time_consensus": [""],
            "response_time_consensus": [""],
            "fall_tags": ["no_fall, camera_side"],
        }
    )
    consensus.to_csv(tmp_path / settings.fall_labels_consensus_csv_path, index=False)

    event_windows = pd.DataFrame(
        {
            "monitor_id": [223],
            "fall_ts_local": ["2025-01-02T11:01:00Z"],
            "fall_ts_utc": ["2025-01-02T11:01:00Z"],
            "pre_prob_chair": [0.0],
            "pre_prob_bed": [1.0],
            "pre_prob_room": [0.0],
            "pre_prob_no_patient": [0.0],
            "response_detected": [False],
            "response_latency_seconds": [None],
        }
    )

    artifacts = evaluate_livestream_derivations_against_truth(settings, event_windows)
    truth_rows = artifacts.population_stats.loc[
        artifacts.population_stats["metric"] == "truth_rows", "value"
    ].iloc[0]
    assert truth_rows == 0.0
    assert artifacts.sequence_metrics.empty


def test_threshold_checks_emit_when_enabled(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, thresholds_enabled=True)
    consensus = pd.DataFrame(
        {
            "event_key": ["333|2025-01-03|09:10"],
            "prefall_location": ["chair"],
            "fall_time_consensus": ["09:10:10"],
            "response_time_consensus": ["09:10:40"],
            "fall_tags": ["slip"],
        }
    )
    consensus.to_csv(tmp_path / settings.fall_labels_consensus_csv_path, index=False)

    event_windows = pd.DataFrame(
        {
            "monitor_id": [333],
            "fall_ts_local": ["2025-01-03T09:10:10Z"],
            "fall_ts_utc": ["2025-01-03T09:10:10Z"],
            "pre_prob_chair": [1.0],
            "pre_prob_bed": [0.0],
            "pre_prob_room": [0.0],
            "pre_prob_no_patient": [0.0],
            "response_detected": [True],
            "response_latency_seconds": [30.0],
        }
    )

    artifacts = evaluate_livestream_derivations_against_truth(settings, event_windows)
    assert artifacts.threshold_checks["enabled"] is True
    assert artifacts.threshold_checks["evaluated"] is True
    assert isinstance(artifacts.threshold_checks["checks"], list)


def test_auc_and_operating_point_outputs_are_computed(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, thresholds_enabled=True)
    consensus = pd.DataFrame(
        {
            "event_key": [
                "900|2025-01-01|10:00",
                "901|2025-01-01|10:01",
                "902|2025-01-01|10:02",
                "903|2025-01-01|10:03",
            ],
            "prefall_location": ["chair", "bed", "room", "no_patient"],
            "fall_time_consensus": ["10:00:10", "10:01:10", "10:02:10", "10:03:10"],
            "response_time_consensus": ["10:00:40", "10:01:40", "10:02:40", "10:03:40"],
            "fall_tags": ["a", "b", "c", "d"],
        }
    )
    consensus.to_csv(tmp_path / settings.fall_labels_consensus_csv_path, index=False)

    event_windows = pd.DataFrame(
        {
            "monitor_id": [900, 901, 902, 903],
            "fall_ts_local": [
                "2025-01-01T10:00:10Z",
                "2025-01-01T10:01:10Z",
                "2025-01-01T10:02:10Z",
                "2025-01-01T10:03:10Z",
            ],
            "fall_ts_utc": [
                "2025-01-01T10:00:10Z",
                "2025-01-01T10:01:10Z",
                "2025-01-01T10:02:10Z",
                "2025-01-01T10:03:10Z",
            ],
            "pre_prob_chair": [0.9, 0.1, 0.1, 0.1],
            "pre_prob_bed": [0.05, 0.8, 0.1, 0.1],
            "pre_prob_room": [0.03, 0.05, 0.7, 0.1],
            "pre_prob_no_patient": [0.02, 0.05, 0.1, 0.7],
            "response_detected": [True, True, True, True],
            "response_latency_seconds": [30.0, 30.0, 30.0, 30.0],
        }
    )

    artifacts = evaluate_livestream_derivations_against_truth(settings, event_windows)
    assert set(artifacts.auc_summary["target_class"].astype(str)) == {"chair", "bed", "room", "no_patient"}
    assert len(artifacts.operating_point["selected"]) == 4
    assert all(0.0 <= float(row["threshold"]) <= 1.0 for row in artifacts.operating_point["selected"])
    assert not artifacts.benchmark_sequence_metrics.empty
    assert not artifacts.benchmark_label_metrics.empty
    assert not artifacts.sequence_predictions.empty


def test_primary_benchmark_maps_offscreen_rows_to_no_patient(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    consensus = pd.DataFrame(
        {
            "event_key": ["910|2025-01-01|10:00", "911|2025-01-01|10:01"],
            "prefall_location": ["chair", "room"],
            "fall_time_consensus": ["10:00:10", ""],
            "response_time_consensus": ["10:00:40", ""],
            "fall_tags": ["a", "offscreen"],
        }
    )
    consensus.to_csv(tmp_path / settings.fall_labels_consensus_csv_path, index=False)

    event_windows = pd.DataFrame(
        {
            "monitor_id": [910, 911],
            "fall_ts_local": ["2025-01-01T10:00:10Z", "2025-01-01T10:01:10Z"],
            "fall_ts_utc": ["2025-01-01T10:00:10Z", "2025-01-01T10:01:10Z"],
            "pre_prob_chair": [0.9, 0.0],
            "pre_prob_bed": [0.1, 0.0],
            "pre_prob_room": [0.0, 0.0],
            "pre_prob_no_patient": [0.0, 1.0],
            "response_detected": [True, False],
            "response_latency_seconds": [30.0, None],
        }
    )

    artifacts = evaluate_livestream_derivations_against_truth(settings, event_windows)
    primary_predictions = artifacts.sequence_predictions.loc[
        (
            artifacts.sequence_predictions["benchmark_split"]
            == BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S
        )
        & (artifacts.sequence_predictions["model_version"] == "legacy")
    ].copy()
    assert len(primary_predictions.index) == 2
    offscreen = primary_predictions.loc[
        primary_predictions["sequence_id"].astype(str) == "911|2025-01-01|10:01#S01"
    ].iloc[0]
    assert offscreen["truth_label"] == "no_patient"
    assert bool(offscreen["scored_for_location"]) is True


def test_departure_aware_primary_relabels_recent_room_departure_to_last_furniture(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    consensus = pd.DataFrame(
        {
            "event_key": ["930|2025-01-01|10:00"],
            "prefall_location": ["room"],
            "fall_time_consensus": ["10:00:08"],
            "response_time_consensus": ["10:00:40"],
            "fall_tags": ["departure"],
            "last_furniture": ["bed"],
            "furniture_departure_time": ["10:00:02"],
        }
    )
    consensus.to_csv(tmp_path / settings.fall_labels_consensus_csv_path, index=False)

    event_windows = pd.DataFrame(
        {
            "monitor_id": [930],
            "fall_ts_local": ["2025-01-01T10:00:08Z"],
            "fall_ts_utc": ["2025-01-01T10:00:08Z"],
            "pre_prob_chair": [0.0],
            "pre_prob_bed": [1.0],
            "pre_prob_room": [0.0],
            "pre_prob_no_patient": [0.0],
            "response_detected": [True],
            "response_latency_seconds": [32.0],
        }
    )

    artifacts = evaluate_livestream_derivations_against_truth(settings, event_windows)

    assert artifacts.benchmark_status["primary_split"] == BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S

    primary_row = artifacts.sequence_predictions.loc[
        (artifacts.sequence_predictions["benchmark_split"] == BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S)
        & (artifacts.sequence_predictions["model_version"] == "legacy")
    ].iloc[0]
    assert primary_row["truth_label"] == "bed"
    assert primary_row["benchmark_truth_label"] == "bed"
    assert primary_row["benchmark_truth_label_raw"] == "room"
    assert bool(primary_row["departure_aware_truth_applied"]) is True
    assert primary_row["departure_aware_source_furniture"] == "bed"
    assert float(primary_row["departure_aware_latency_seconds"]) == 6.0

    raw_audit_row = artifacts.sequence_predictions.loc[
        (artifacts.sequence_predictions["benchmark_split"] == BENCHMARK_SPLIT_PRIMARY_4CLASS_V2_RAW_AUDIT)
        & (artifacts.sequence_predictions["model_version"] == "legacy")
    ].iloc[0]
    assert raw_audit_row["truth_label"] == "room"
    assert raw_audit_row["benchmark_truth_label"] == "room"
    assert raw_audit_row["benchmark_truth_label_raw"] == "room"
    assert bool(raw_audit_row["departure_aware_truth_applied"]) is False


def test_departure_aware_primary_does_not_relabel_at_ten_seconds_or_no_patient(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    consensus = pd.DataFrame(
        {
            "event_key": ["931|2025-01-01|10:00", "932|2025-01-01|10:01"],
            "prefall_location": ["room", "no_patient"],
            "fall_time_consensus": ["10:00:12", "10:01:08"],
            "response_time_consensus": ["10:00:40", "10:01:40"],
            "fall_tags": ["departure", "offscreen"],
            "last_furniture": ["chair", "bed"],
            "furniture_departure_time": ["10:00:02", "10:01:02"],
        }
    )
    consensus.to_csv(tmp_path / settings.fall_labels_consensus_csv_path, index=False)

    event_windows = pd.DataFrame(
        {
            "monitor_id": [931, 932],
            "fall_ts_local": ["2025-01-01T10:00:12Z", "2025-01-01T10:01:08Z"],
            "fall_ts_utc": ["2025-01-01T10:00:12Z", "2025-01-01T10:01:08Z"],
            "pre_prob_chair": [0.0, 0.0],
            "pre_prob_bed": [0.0, 0.0],
            "pre_prob_room": [1.0, 0.0],
            "pre_prob_no_patient": [0.0, 1.0],
            "response_detected": [True, False],
            "response_latency_seconds": [28.0, None],
        }
    )

    artifacts = evaluate_livestream_derivations_against_truth(settings, event_windows)
    primary_predictions = artifacts.sequence_predictions.loc[
        (artifacts.sequence_predictions["benchmark_split"] == BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S)
        & (artifacts.sequence_predictions["model_version"] == "legacy")
    ].copy()

    ten_second_room = primary_predictions.loc[
        primary_predictions["sequence_id"].astype(str) == "931|2025-01-01|10:00#S01"
    ].iloc[0]
    assert ten_second_room["truth_label"] == "room"
    assert bool(ten_second_room["departure_aware_truth_applied"]) is False

    no_patient_row = primary_predictions.loc[
        primary_predictions["sequence_id"].astype(str) == "932|2025-01-01|10:01#S01"
    ].iloc[0]
    assert no_patient_row["truth_label"] == "no_patient"
    assert bool(no_patient_row["departure_aware_truth_applied"]) is False


def test_benchmark_status_selects_v2_from_departure_aware_comparative_checks(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    comparison = pd.DataFrame(
        {
            "benchmark_split": [
                BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S,
                BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S,
            ],
            "model_version": ["legacy", "v2"],
            "macro_f1": [0.40, 0.42],
            "f1_chair": [0.45, 0.41],
            "f1_bed": [0.60, 0.60],
            "f1_room": [0.45, 0.46],
            "f1_no_patient": [0.17, 0.17],
        }
    )

    status = _benchmark_status(settings, comparison)

    assert status["primary_split"] == BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S
    assert status["overall_pass"] is False
    assert status["selection_pass"] is True
    assert status["selected_model_version"] == "v2"


def test_benchmark_status_prefers_v3_when_it_clears_selection_checks(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    comparison = pd.DataFrame(
        {
            "benchmark_split": [BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S] * 3,
            "model_version": ["legacy", "v2", "v3"],
            "macro_f1": [0.40, 0.42, 0.46],
            "f1_chair": [0.45, 0.41, 0.46],
            "f1_bed": [0.60, 0.60, 0.59],
            "f1_room": [0.45, 0.46, 0.50],
            "f1_no_patient": [0.17, 0.17, 0.20],
        }
    )

    status = _benchmark_status(settings, comparison)

    assert status["selection_pass"] is True
    assert status["selected_model_version"] == "v3"


def test_v2_recent_visible_fallback_reassigns_focus_missing_event(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    consensus = pd.DataFrame(
        {
            "event_key": ["920|2025-01-01|10:00"],
            "prefall_location": ["chair"],
            "fall_time_consensus": ["10:00:10"],
            "response_time_consensus": ["10:00:40"],
            "fall_tags": ["a"],
        }
    )
    consensus.to_csv(tmp_path / settings.fall_labels_consensus_csv_path, index=False)

    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "monitor_id": [920],
            "fall_ts_local": ["2025-01-01T10:00:10Z"],
            "fall_ts_utc": ["2025-01-01T10:00:10Z"],
            "pre_focus_visible_frames": [0],
            "pre_dropout_visible_frames": [12],
            "pre_state_visible_prob": [0.25],
            "pre_dropout_last_visible_gap_seconds": [20.0],
            "pre_prob_chair": [0.0],
            "pre_prob_bed": [0.0],
            "pre_prob_room": [0.0],
            "pre_prob_no_patient": [1.0],
            "response_detected": [True],
            "response_latency_seconds": [30.0],
        }
    )
    second_level = pd.DataFrame(
        {
            "fall_event_id": [1, 1, 1],
            "second_offset": [-50, -30, -10],
            "frame_has_location_signal": [True, True, True],
            "patient_chair_distance": [0.2, 0.25, 0.3],
            "patient_bed_distance": [1.2, 1.0, 0.9],
            "patient_room_distance": [2.0, 2.0, 2.0],
            "dominant_location_label": ["chair", "chair", "chair"],
        }
    )

    artifacts = evaluate_livestream_derivations_against_truth(settings, event_windows, second_level)
    v2_row = artifacts.sequence_predictions.loc[
        (artifacts.sequence_predictions["benchmark_split"] == "secondary_inframe_legacy")
        & (artifacts.sequence_predictions["model_version"] == "v2")
    ].iloc[0]
    assert v2_row["derived_label"] == "chair"
    assert v2_row["prediction_reason"] == "recent_visible_fallback"


def test_tag_location_association_and_effect_size_emit_for_repeated_tags(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    consensus = pd.DataFrame(
        {
            "event_key": [
                "700|2025-01-01|09:00",
                "701|2025-01-01|09:01",
                "702|2025-01-01|09:02",
                "703|2025-01-01|09:03",
                "704|2025-01-01|09:04",
                "705|2025-01-01|09:05",
            ],
            "prefall_location": ["bed", "bed", "bed", "chair", "chair", "room"],
            "fall_time_consensus": ["09:00:10", "09:01:10", "09:02:10", "09:03:10", "09:04:10", "09:05:10"],
            "response_time_consensus": ["09:00:30", "09:01:30", "09:02:30", "09:03:30", "09:04:30", "09:05:30"],
            "fall_tags": ["slip", "slip", "slip", "tumble", "tumble", "slip"],
        }
    )
    consensus.to_csv(tmp_path / settings.fall_labels_consensus_csv_path, index=False)

    event_windows = pd.DataFrame(
        {
            "monitor_id": [700, 701, 702, 703, 704, 705],
            "fall_ts_local": [
                "2025-01-01T09:00:10Z",
                "2025-01-01T09:01:10Z",
                "2025-01-01T09:02:10Z",
                "2025-01-01T09:03:10Z",
                "2025-01-01T09:04:10Z",
                "2025-01-01T09:05:10Z",
            ],
            "fall_ts_utc": [
                "2025-01-01T09:00:10Z",
                "2025-01-01T09:01:10Z",
                "2025-01-01T09:02:10Z",
                "2025-01-01T09:03:10Z",
                "2025-01-01T09:04:10Z",
                "2025-01-01T09:05:10Z",
            ],
            "pre_prob_chair": [0.2, 0.2, 0.2, 0.8, 0.8, 0.1],
            "pre_prob_bed": [0.7, 0.7, 0.7, 0.1, 0.1, 0.2],
            "pre_prob_room": [0.05, 0.05, 0.05, 0.05, 0.05, 0.6],
            "pre_prob_no_patient": [0.05, 0.05, 0.05, 0.05, 0.05, 0.1],
            "response_detected": [True, True, True, True, True, True],
            "response_latency_seconds": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        }
    )

    artifacts = evaluate_livestream_derivations_against_truth(settings, event_windows)
    assert not artifacts.tag_location_association.empty

    cv = artifacts.association_summary.loc[
        artifacts.association_summary["metric"] == "tag_location_cramers_v", "value"
    ]
    assert not cv.empty


def test_shadow_model_outputs_emit_with_second_level_panel(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    consensus = pd.DataFrame(
        {
            "event_key": [
                "810|2025-01-01|09:00",
                "811|2025-01-01|09:01",
                "812|2025-01-01|09:02",
                "813|2025-01-01|09:03",
            ],
            "prefall_location": ["chair", "bed", "room", "no_patient"],
            "fall_time_consensus": ["09:00:10", "09:01:10", "09:02:10", "09:03:10"],
            "response_time_consensus": ["09:00:30", "09:01:35", "09:02:33", "09:03:45"],
            "fall_tags": ["a", "b", "c", "d"],
        }
    )
    consensus.to_csv(tmp_path / settings.fall_labels_consensus_csv_path, index=False)

    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1, 2, 3, 4],
            "monitor_id": [810, 811, 812, 813],
            "fall_ts_local": [
                "2025-01-01T09:00:10Z",
                "2025-01-01T09:01:10Z",
                "2025-01-01T09:02:10Z",
                "2025-01-01T09:03:10Z",
            ],
            "fall_ts_utc": [
                "2025-01-01T09:00:10Z",
                "2025-01-01T09:01:10Z",
                "2025-01-01T09:02:10Z",
                "2025-01-01T09:03:10Z",
            ],
            "pre_prob_chair": [0.9, 0.1, 0.1, 0.1],
            "pre_prob_bed": [0.05, 0.8, 0.1, 0.1],
            "pre_prob_room": [0.03, 0.05, 0.7, 0.1],
            "pre_prob_no_patient": [0.02, 0.05, 0.1, 0.7],
            "response_detected": [True, True, True, True],
            "response_latency_seconds": [20.0, 25.0, 23.0, 35.0],
        }
    )

    second_level = pd.DataFrame(
        {
            "fall_event_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "second_offset": [-120, -60, -1] * 4,
            "dominant_location_label": [
                "chair",
                "chair",
                "chair",
                "bed",
                "bed",
                "room",
                "room",
                "room",
                "room",
                "no_patient",
                "bed",
                "no_patient",
            ],
            "frame_has_location_signal": [True, True, True, True, True, True, True, True, True, False, True, False],
            "patient_chair_distance": [0.2, 0.3, 0.4, 2.5, 2.2, 2.8, 3.1, 3.0, 2.9, None, 4.0, None],
            "patient_bed_distance": [2.7, 2.5, 2.3, 0.3, 0.4, 0.5, 2.0, 1.9, 1.8, None, 0.8, None],
            "patient_room_distance": [3.0, 2.9, 2.8, 1.5, 1.6, 1.7, 0.4, 0.3, 0.2, None, 2.0, None],
        }
    )

    artifacts = evaluate_livestream_derivations_against_truth(settings, event_windows, second_level)

    assert not artifacts.shadow_model_metrics.empty
    assert not artifacts.shadow_model_predictions.empty
    assert not artifacts.shadow_model_comparison.empty
    assert "macro_f1" in set(artifacts.shadow_model_comparison["metric"].astype(str))
    assert not artifacts.shadow_ablation.empty
    assert not artifacts.shadow_cv_fold_metrics.empty


def test_dt_aware_trajectory_features_use_true_second_deltas(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    second_level = pd.DataFrame(
        {
            "fall_event_id": [1, 1, 1, 1] + [2] * 60,
            "window_role": ["hazard"] * 64,
            "second_offset": [-60, -58, -55, -50] + list(range(-60, 0)),
            "dominant_location_label": ["chair", "chair", "chair", "chair"] + ["bed"] * 60,
            "frame_has_location_signal": [True] * 64,
            "patient_chair_distance": [0.0, 2.0, 5.0, 10.0] + [float((idx + 60) * 2) for idx in range(-60, 0)],
            "patient_bed_distance": [5.0, 5.0, 5.0, 5.0] + [float((idx + 60) * 3 + 1) for idx in range(-60, 0)],
            "patient_room_distance": [8.0, 8.0, 8.0, 8.0] + [float((idx + 60) * 4 + 2) for idx in range(-60, 0)],
        }
    )

    combined = _combine_shadow_panels(second_level, pd.DataFrame())
    trajectory = _trajectory_feature_rows(settings, combined)
    row = trajectory.loc[
        (trajectory["fall_event_id"] == 1)
        & (trajectory["window_role"] == "hazard")
        & (trajectory["window_seconds"] == 60)
    ].iloc[0]

    assert row["window_reason_code"] == "insufficient_sample_presence"
    valid = trajectory.loc[
        (trajectory["fall_event_id"] == 2)
        & (trajectory["window_role"] == "hazard")
        & (trajectory["window_seconds"] == 60)
    ].iloc[0]
    assert pd.isna(valid["window_reason_code"])
    assert valid["patient_chair_distance_velocity_mean"] == 2.0


def test_onset_detection_handles_no_onset_and_sparse_windows(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    no_onset_rows = pd.DataFrame(
        {
            "fall_event_id": [1] * 300,
            "window_role": ["hazard"] * 300,
            "anchor_ts_utc": ["2025-01-01T10:00:00Z"] * 300,
            "second_offset": list(range(-300, 0)),
            "dominant_location_label": ["chair"] * 300,
            "frame_has_location_signal": [True] * 300,
            "patient_chair_distance": [0.5] * 300,
            "patient_bed_distance": [2.5] * 300,
            "patient_room_distance": [3.0] * 300,
            "nudge_score": [0.1] * 300,
            "nudge_state": ["inactive"] * 300,
        }
    )
    sparse_rows = pd.DataFrame(
        {
            "fall_event_id": [2],
            "window_role": ["hazard"],
            "anchor_ts_utc": ["2025-01-01T11:00:00Z"],
            "second_offset": [-10],
            "dominant_location_label": ["bed"],
            "frame_has_location_signal": [True],
            "patient_chair_distance": [2.0],
            "patient_bed_distance": [0.4],
            "patient_room_distance": [2.5],
            "nudge_score": [0.0],
            "nudge_state": ["inactive"],
        }
    )

    onset = _detect_onset_events(settings, pd.concat([no_onset_rows, sparse_rows], ignore_index=True))

    no_onset = onset.loc[onset["fall_event_id"] == 1].iloc[0]
    sparse = onset.loc[onset["fall_event_id"] == 2].iloc[0]
    assert bool(no_onset["onset_detected"]) is False
    assert pd.isna(no_onset["onset_channel"])
    assert sparse["eligibility_reason_code"] == "insufficient_sample_presence"


def test_second_level_extract_schema_parity_is_wired_for_hazard_and_controls() -> None:
    spec_map = {spec.name: spec for spec in EXTRACT_SPECS}
    hazard_cols = spec_map["fall_livestream_second_level"].dry_columns
    crossover_cols = spec_map["fall_case_crossover_second_level"].dry_columns
    negative_control_cols = spec_map["fall_negative_control_second_level"].dry_columns

    assert crossover_cols[:3] == ["fall_event_id", "window_role", "anchor_ts_utc"]
    assert crossover_cols[3:5] == ["fall_ts_utc", "fall_ts_local"]
    assert hazard_cols[1:] == crossover_cols[3:]
    assert "primary_patient_posture_label" in hazard_cols
    assert "primary_patient_posture_score_sitting" in hazard_cols
    assert "patient_posture_source" in hazard_cols
    for column in [
        "bed_in_bed_score",
        "motion_bac",
        "patient_candidate_count",
        "staff_candidate_count",
        "other_candidate_count",
        "bed_candidate_count",
        "chair_candidate_count",
    ]:
        assert column in hazard_cols
        assert column in crossover_cols
        assert column in negative_control_cols
    assert negative_control_cols[:3] == ["fall_event_id", "window_role", "anchor_ts_utc"]
    assert "source_chunk_id" in negative_control_cols
    assert "primary_patient_posture_label" in negative_control_cols


def test_shadow_ablation_outputs_include_expected_models_and_columns(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    consensus = pd.DataFrame(
        {
            "event_key": [f"{900 + idx}|2025-01-01|09:0{idx}" for idx in range(4)],
            "prefall_location": ["chair", "bed", "room", "no_patient"],
            "fall_time_consensus": ["09:00:10", "09:01:10", "09:02:10", "09:03:10"],
            "response_time_consensus": ["09:00:30", "09:01:40", "09:02:40", "09:03:35"],
            "fall_tags": ["a", "b", "c", "d"],
        }
    )
    consensus.to_csv(tmp_path / settings.fall_labels_consensus_csv_path, index=False)

    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1, 2, 3, 4],
            "monitor_id": [900, 901, 902, 903],
            "fall_ts_local": [f"2025-01-01T09:0{idx}:10Z" for idx in range(4)],
            "fall_ts_utc": [f"2025-01-01T09:0{idx}:10Z" for idx in range(4)],
            "pre_prob_chair": [0.85, 0.05, 0.1, 0.1],
            "pre_prob_bed": [0.05, 0.8, 0.1, 0.1],
            "pre_prob_room": [0.05, 0.1, 0.7, 0.1],
            "pre_prob_no_patient": [0.05, 0.05, 0.1, 0.7],
            "response_detected": [True, True, True, True],
            "response_latency_seconds": [10.0, 12.0, 14.0, 16.0],
        }
    )

    rows: list[dict[str, object]] = []
    for fall_event_id, role_offset in [(1, 0.0), (2, 1.0), (3, 2.0), (4, 3.0)]:
        for role, shift in [("hazard", 0.0), ("control_1d", 0.2), ("control_2d", -0.1)]:
            for second_offset in range(-300, 0):
                rows.append(
                    {
                        "fall_event_id": fall_event_id,
                        "window_role": role,
                        "anchor_ts_utc": f"2025-01-01T09:0{fall_event_id - 1}:10Z",
                        "second_offset": second_offset,
                        "dominant_location_label": ["chair", "bed", "room", "no_patient"][fall_event_id - 1],
                        "frame_has_location_signal": True,
                        "patient_chair_distance": role_offset + shift + abs(second_offset) / 300.0,
                        "patient_bed_distance": abs((role_offset + 1.0) - shift) + abs(second_offset) / 400.0,
                        "patient_room_distance": abs((role_offset + 2.0) + shift) + abs(second_offset) / 500.0,
                        "nudge_score": 0.1 if role != "hazard" else 0.2,
                        "nudge_state": "inactive",
                    }
                )
    crossover_panel = pd.DataFrame(rows)

    artifacts = evaluate_livestream_derivations_against_truth(
        settings,
        event_windows,
        pd.DataFrame(),
        crossover_panel,
    )

    expected_cols = {
        "model",
        "metric",
        "mean",
        "ci_low",
        "ci_high",
        "feature_family_set",
        "feature_count",
        "calibration_status",
    }
    assert expected_cols.issubset(set(artifacts.shadow_ablation.columns))
    assert {
        "baseline_heuristic",
        "stacked_baseline_probs_trajectory_posture_onset",
        "random_forest",
    }.issubset(
        set(artifacts.shadow_ablation["model"].astype(str))
    )
    assert not artifacts.shadow_window_sensitivity.empty
    assert not artifacts.shadow_feature_importance.empty


def test_onset_detection_emits_posture_transition_before_location_departure(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    rows: list[dict[str, object]] = []
    for second_offset in range(-300, 0):
        posture_label = "sitting"
        posture_scores = {
            "primary_patient_posture_score_sitting": 0.95,
            "primary_patient_posture_score_standing": 0.03,
            "primary_patient_posture_score_lying": 0.02,
        }
        dominant_location_label = "chair"
        if second_offset >= -8:
            posture_label = "standing"
            posture_scores = {
                "primary_patient_posture_score_sitting": 0.10,
                "primary_patient_posture_score_standing": 0.85,
                "primary_patient_posture_score_lying": 0.05,
            }
        if second_offset >= -6:
            dominant_location_label = "room"
        rows.append(
            {
                "fall_event_id": 1,
                "window_role": "hazard",
                "anchor_ts_utc": "2025-01-01T10:00:00Z",
                "second_offset": second_offset,
                "dominant_location_label": dominant_location_label,
                "frame_has_location_signal": True,
                "patient_chair_distance": 0.5 if dominant_location_label == "chair" else 2.0,
                "patient_bed_distance": 2.0,
                "patient_room_distance": 2.0 if dominant_location_label == "chair" else 0.5,
                "nudge_score": 0.0,
                "nudge_state": "inactive",
                "frame_has_posture_signal": True,
                "primary_patient_posture_label": posture_label,
                **posture_scores,
            }
        )

    onset = _detect_onset_events(settings, pd.DataFrame(rows))
    hazard = onset.loc[onset["fall_event_id"] == 1].iloc[0]

    assert bool(hazard["onset_detected"]) is True
    assert hazard["onset_channel"] == "posture_transition"


def test_shadow_cv_logs_missing_class_fallback(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, shadow_cv_folds=2, shadow_cv_repeats=1)
    consensus = pd.DataFrame(
        {
            "event_key": ["100|2025-01-01|10:00", "101|2025-01-01|10:01"],
            "prefall_location": ["chair", "bed"],
            "fall_time_consensus": ["10:00:10", "10:01:10"],
            "response_time_consensus": ["10:00:20", "10:01:20"],
            "fall_tags": ["a", "b"],
        }
    )
    consensus.to_csv(tmp_path / settings.fall_labels_consensus_csv_path, index=False)

    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1, 2],
            "monitor_id": [100, 101],
            "fall_ts_local": ["2025-01-01T10:00:10Z", "2025-01-01T10:01:10Z"],
            "fall_ts_utc": ["2025-01-01T10:00:10Z", "2025-01-01T10:01:10Z"],
            "pre_prob_chair": [0.9, 0.1],
            "pre_prob_bed": [0.1, 0.8],
            "pre_prob_room": [0.0, 0.1],
            "pre_prob_no_patient": [0.0, 0.0],
            "response_detected": [True, True],
            "response_latency_seconds": [10.0, 10.0],
        }
    )

    crossover_rows = []
    for fall_event_id, label in [(1, "chair"), (2, "bed")]:
        for role in ["hazard", "control_1d", "control_2d"]:
            for second_offset in range(-300, 0):
                crossover_rows.append(
                    {
                        "fall_event_id": fall_event_id,
                        "window_role": role,
                        "anchor_ts_utc": "2025-01-01T10:00:10Z",
                        "second_offset": second_offset,
                        "dominant_location_label": label,
                        "frame_has_location_signal": True,
                        "patient_chair_distance": 0.5 if label == "chair" else 2.0,
                        "patient_bed_distance": 0.5 if label == "bed" else 2.0,
                        "patient_room_distance": 3.0,
                        "nudge_score": 0.1,
                        "nudge_state": "inactive",
                    }
                )

    artifacts = evaluate_livestream_derivations_against_truth(
        settings,
        event_windows,
        pd.DataFrame(),
        pd.DataFrame(crossover_rows),
    )

    fallback_rows = artifacts.shadow_cv_fold_metrics.loc[
        artifacts.shadow_cv_fold_metrics["fallback_reason"].astype(str) == "insufficient_training_classes"
    ]
    assert not fallback_rows.empty
