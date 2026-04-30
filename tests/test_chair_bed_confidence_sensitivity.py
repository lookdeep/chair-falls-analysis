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
from ld_chair_falls.qa import (
    _build_chair_bed_confidence_sensitivity_table,
    _build_chair_bed_missingness_stress_table,
    _build_chair_bed_operational_event_rates_table,
    _build_negative_control_match_quality,
    _build_negative_control_outputs,
)


def _base_analysis_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_base = pd.DataFrame(
        {
            "cohort_type": ["intervention", "intervention"],
            "hospital_id": [5, 5],
            "division_id": [12, 12],
            "monitor_id": [101, 101],
            "patient_id": [1001, 1001],
            "hour_ts": ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"],
            "pct_chair": [0.8, 0.7],
            "pct_bed": [0.2, 0.3],
        }
    )
    eligibility = pd.DataFrame(
        {
            "cohort_type": ["intervention"],
            "eligible": [True],
            "hospital_id": [5],
            "division_id": [12],
            "monitor_id": [101],
            "patient_id": [1001],
        }
    )
    return analysis_base, eligibility


def test_confidence_sensitivity_segments_and_rates() -> None:
    analysis_base, eligibility = _base_analysis_inputs()
    event_windows = pd.DataFrame(
        {
            "hospital_id": [5, 5, 5],
            "division_id": [12, 12, 12],
            "monitor_id": [101, 101, 101],
            "patient_id": [1001, 1001, 1001],
            "fall_ts_utc": ["2025-01-01T00:12:00Z", "2025-01-01T00:30:00Z", "2025-01-01T01:15:00Z"],
            "fall_ts_local": ["2025-01-01T00:12:00Z", "2025-01-01T00:30:00Z", "2025-01-01T01:15:00Z"],
            "prefall_location_label": ["chair", "bed", "bed"],
            "pre_prob_chair": [0.8, 0.4, 0.1],
            "pre_prob_bed": [0.1, 0.5, 0.7],
            "pre_state_visible_prob": [0.9, 0.6, 0.2],
            "pre_dropout_visibility_ratio": [0.9, 0.6, 0.3],
            "pre_dropout_last_visible_gap_seconds": [30.0, 120.0, 240.0],
        }
    )

    table = _build_chair_bed_confidence_sensitivity_table(
        analysis_base,
        eligibility,
        {"fall_livestream_event_windows": event_windows},
        "America/Chicago",
    )

    assert not table.empty
    assert set(table["confidence_segment"].unique().tolist()) == {
        "all_events",
        "high_confidence",
        "medium_confidence",
        "low_confidence",
    }

    chair_all = table.loc[
        (table["confidence_segment"] == "all_events") & (table["position"] == "chair")
    ].iloc[0]
    bed_all = table.loc[
        (table["confidence_segment"] == "all_events") & (table["position"] == "bed")
    ].iloc[0]
    high_chair = table.loc[
        (table["confidence_segment"] == "high_confidence") & (table["position"] == "chair")
    ].iloc[0]
    low_chair = table.loc[
        (table["confidence_segment"] == "low_confidence") & (table["position"] == "chair")
    ].iloc[0]

    assert chair_all["eligible_event_count"] == 3
    assert bed_all["eligible_event_count"] == 3
    assert chair_all["expected_falls"] == 1.3
    assert bed_all["expected_falls"] == 1.3
    assert chair_all["rate_per_1000_exposure_hours_expected"] == 866.6667
    assert bed_all["rate_per_1000_exposure_hours_expected"] == 2600.0
    assert chair_all["confidence_weighted_expected_falls"] < chair_all["expected_falls"]
    assert low_chair["mean_confidence_weight"] < high_chair["mean_confidence_weight"]


def test_confidence_sensitivity_returns_empty_when_no_eligible_events() -> None:
    analysis_base, eligibility = _base_analysis_inputs()
    table = _build_chair_bed_confidence_sensitivity_table(
        analysis_base,
        eligibility,
        {"fall_livestream_event_windows": pd.DataFrame()},
        "America/Chicago",
    )
    assert table.empty
    assert "confidence_segment" in table.columns


def test_confidence_sensitivity_requires_temporal_alignment() -> None:
    analysis_base, eligibility = _base_analysis_inputs()
    event_windows = pd.DataFrame(
        {
            "hospital_id": [5],
            "division_id": [12],
            "monitor_id": [101],
            "patient_id": [1001],
            "fall_ts_utc": ["2025-01-02T08:00:00Z"],
            "prefall_location_label": ["chair"],
            "pre_prob_chair": [1.0],
            "pre_prob_bed": [0.0],
            "pre_state_visible_prob": [1.0],
            "pre_dropout_visibility_ratio": [1.0],
            "pre_dropout_last_visible_gap_seconds": [10.0],
        }
    )
    table = _build_chair_bed_confidence_sensitivity_table(
        analysis_base,
        eligibility,
        {"fall_livestream_event_windows": event_windows},
        "America/Chicago",
    )
    assert table.empty


def test_operational_event_rates_weight_hourly_counts_by_position() -> None:
    analysis_base = pd.DataFrame(
        {
            "cohort_type": ["intervention", "intervention"],
            "hospital_id": [5, 5],
            "division_id": [12, 12],
            "monitor_id": [101, 101],
            "patient_id": [1001, 1001],
            "hour_ts": ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"],
            "pct_chair": [0.8, 0.2],
            "pct_bed": [0.2, 0.8],
            "num_alarms": [4, 6],
            "num_nudges": [10, 20],
            "num_announcements": [2, 8],
        }
    )
    eligibility = pd.DataFrame(
        {
            "cohort_type": ["intervention"],
            "eligible": [True],
            "hospital_id": [5],
            "division_id": [12],
            "monitor_id": [101],
            "patient_id": [1001],
        }
    )

    table = _build_chair_bed_operational_event_rates_table(analysis_base, eligibility)

    assert not table.empty
    assert table["metric"].tolist() == [
        "alarms",
        "alarms",
        "nudges",
        "nudges",
        "announcements",
        "announcements",
    ]
    chair_alarm = table.loc[(table["metric"] == "alarms") & (table["position"] == "chair")].iloc[0]
    bed_alarm = table.loc[(table["metric"] == "alarms") & (table["position"] == "bed")].iloc[0]
    chair_nudge = table.loc[(table["metric"] == "nudges") & (table["position"] == "chair")].iloc[0]

    assert chair_alarm["exposure_hours"] == 1.0
    assert bed_alarm["exposure_hours"] == 1.0
    assert chair_alarm["weighted_event_count"] == 4.4
    assert bed_alarm["weighted_event_count"] == 5.6
    assert chair_alarm["rate_per_exposure_hour"] == 4.4
    assert chair_alarm["rate_per_100_exposure_hours"] == 440.0
    assert bed_alarm["chair_to_bed_rate_ratio"] == 0.785714
    assert chair_nudge["weighted_event_count"] == 12.0


def test_missingness_stress_outputs_cutpoint_rows() -> None:
    analysis_base, eligibility = _base_analysis_inputs()
    event_windows = pd.DataFrame(
        {
            "hospital_id": [5, 5, 5, 5],
            "division_id": [12, 12, 12, 12],
            "monitor_id": [101, 101, 101, 101],
            "patient_id": [1001, 1001, 1001, 1001],
            "fall_ts_utc": [
                "2025-01-01T00:05:00Z",
                "2025-01-01T00:20:00Z",
                "2025-01-01T00:35:00Z",
                "2025-01-01T01:10:00Z",
            ],
            "fall_ts_local": [
                "2025-01-01T00:05:00Z",
                "2025-01-01T00:20:00Z",
                "2025-01-01T00:35:00Z",
                "2025-01-01T01:10:00Z",
            ],
            "prefall_location_label": ["chair", "bed", "bed", "chair"],
            "pre_prob_chair": [0.7, 0.2, 0.1, 0.6],
            "pre_prob_bed": [0.2, 0.7, 0.8, 0.3],
            "pre_state_visible_prob": [0.95, 0.7, 0.4, 0.9],
            "pre_dropout_visibility_ratio": [0.9, 0.65, 0.3, 0.88],
            "pre_dropout_last_visible_gap_seconds": [20.0, 60.0, 240.0, 30.0],
        }
    )

    table = _build_chair_bed_missingness_stress_table(
        analysis_base,
        eligibility,
        {"fall_livestream_event_windows": event_windows},
        "America/Chicago",
        (0, 25, 50),
    )

    assert not table.empty
    assert set(table["excluded_low_confidence_pct"].astype(int).tolist()) == {0, 25, 50}
    chair_rows = table.loc[table["position"] == "chair"].copy()
    assert not chair_rows.empty
    assert chair_rows["chair_to_bed_rate_ratio_expected"].notna().all()
    assert chair_rows["chair_to_bed_rate_ratio_confidence_weighted"].notna().all()


def test_negative_control_outputs_include_pairs_and_effects() -> None:
    case_windows = pd.DataFrame(
        {
            "fall_event_id": [1, 2],
            "window_role": ["hazard", "hazard"],
            "anchor_ts_utc": ["2025-01-01T10:00:00Z", "2025-01-01T10:05:00Z"],
            "monitor_id": [1001, 1002],
            "frame_count": [120, 120],
            "visibility_ratio": [0.9, 0.8],
            "chair_share": [0.7, 0.2],
            "bed_share": [0.2, 0.6],
            "room_share": [0.1, 0.2],
            "no_patient_share": [0.0, 0.0],
            "mean_patient_chair_distance": [0.5, 2.1],
            "mean_patient_bed_distance": [1.2, 0.6],
            "mean_patient_room_distance": [2.5, 1.7],
            "mean_patient_staff_iou": [0.2, 0.3],
            "max_patient_staff_iou": [0.5, 0.7],
            "dominant_switch_count": [2, 5],
            "nearby_fall_count_30m": [0, 0],
            "eligible_window": [True, True],
        }
    )
    negative_windows = pd.DataFrame(
        {
            "fall_event_id": [1, 2],
            "window_role": ["nonfall_control", "nonfall_control"],
            "anchor_ts_utc": ["2025-01-01T10:00:00Z", "2025-01-01T10:05:00Z"],
            "source_monitor_id": [1001, 1002],
            "monitor_id": [2001, 2002],
            "matching_candidates": [4, 5],
            "frame_count": [120, 120],
            "visibility_ratio": [0.8, 0.75],
            "chair_share": [0.4, 0.15],
            "bed_share": [0.4, 0.7],
            "room_share": [0.2, 0.15],
            "no_patient_share": [0.0, 0.0],
            "mean_patient_chair_distance": [1.1, 2.5],
            "mean_patient_bed_distance": [0.9, 0.5],
            "mean_patient_room_distance": [1.8, 1.6],
            "mean_patient_staff_iou": [0.15, 0.25],
            "max_patient_staff_iou": [0.4, 0.6],
            "dominant_switch_count": [1, 4],
            "nearby_fall_count_30m": [0, 0],
            "eligible_window": [True, True],
        }
    )

    sets_df, effects_df, diagnostics = _build_negative_control_outputs(case_windows, negative_windows)

    assert not sets_df.empty
    assert set(sets_df["window_role"].astype(str).unique().tolist()) == {"hazard", "nonfall_control"}
    assert int(diagnostics["events_with_hazard"]) == 2
    assert int(diagnostics["events_with_negative_control_pair"]) == 2
    assert not effects_df.empty
    assert "chair_share" in set(effects_df["metric"].astype(str).tolist())


def test_negative_control_match_quality_flags_pre_source_and_unmatched_falls(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, run_id="negative_control_quality_test")
    case_windows = pd.DataFrame(
        {
            "fall_event_id": [1, 2, 3],
            "window_role": ["hazard", "hazard", "hazard"],
            "anchor_ts_utc": ["2025-01-04T10:00:00Z", "2025-01-10T10:00:00Z", "2025-01-10T11:00:00Z"],
            "division_id": [21, 21, 48],
            "monitor_id": [1001, 1002, 1003],
            "eligible_window": [True, True, True],
        }
    )
    negative_windows = pd.DataFrame(
        {
            "fall_event_id": [2, 2],
            "window_role": ["nonfall_control", "nonfall_control"],
            "anchor_ts_utc": ["2025-01-06T10:59:05Z", "2025-01-07T10:59:05Z"],
            "source_monitor_id": [1002, 1002],
            "monitor_id": [2001, 2002],
            "matching_candidates": [5, 5],
            "control_rank": [1, 2],
            "match_tier": ["same_local_hour", "same_local_hour"],
            "eligible_window": [True, True],
        }
    )
    source_inventory = pd.DataFrame(
        {
            "division_id": [21, 21],
            "monitor_id": [2001, 2002],
            "patient_id": [pd.NA, 333],
            "anchor_ts_utc": ["2025-01-05T10:59:05Z", "2025-01-06T10:59:05Z"],
        }
    )

    match_quality, coverage = _build_negative_control_match_quality(
        case_windows,
        negative_windows,
        source_inventory,
        settings,
    )

    assert int(coverage["fall_events_with_selected_controls"]) == 1
    assert int(coverage["fall_events_eligible_for_matching"]) == 1
    reasons = match_quality.set_index("fall_event_id")["excluded_reason"].to_dict()
    assert reasons[1] == "before_source_start"
    assert reasons[2] == ""
    assert reasons[3] == "division_unavailable"
