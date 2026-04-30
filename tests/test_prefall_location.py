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
from ld_chair_falls.prefall_location import (
    _posture_adjusted_visible_probs,
    build_panel_prefall_event_windows,
)


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(project_root=tmp_path, run_id="prefall_location_test", dry_run=True)


def test_panel_inference_defaults_to_no_patient_without_visible_signal(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "pre_prob_chair": [0.0],
            "pre_prob_bed": [0.0],
            "pre_prob_room": [0.0],
            "pre_prob_no_patient": [1.0],
        }
    )

    enriched = build_panel_prefall_event_windows(settings, event_windows, pd.DataFrame())
    row = enriched.iloc[0]

    assert row["prefall_location_label"] == "no_patient"
    assert row["prefall_location_reason"] == "zero_signal_default"
    assert row["pre_prob_no_patient"] == 1.0


def test_panel_inference_uses_sparse_sql_fallback_without_visible_signal(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "pre_prob_chair": [0.05],
            "pre_prob_bed": [0.90],
            "pre_prob_room": [0.00],
            "pre_prob_no_patient": [0.05],
        }
    )

    enriched = build_panel_prefall_event_windows(settings, event_windows, pd.DataFrame())
    row = enriched.iloc[0]

    assert row["prefall_location_label"] == "bed"
    assert row["prefall_location_reason"] == "sparse_panel_sql_fallback"
    assert row["pre_prob_bed"] > row["pre_prob_no_patient"]


def test_panel_inference_v3_uses_visible_regime_carry_forward_without_panel_signal(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "pre_prob_chair": [0.0],
            "pre_prob_bed": [0.0],
            "pre_prob_room": [0.0],
            "pre_prob_no_patient": [1.0],
            "pre_focus_weight_chair": [0.70],
            "pre_focus_weight_bed": [0.20],
            "pre_focus_weight_room": [0.10],
            "pre_cond_prob_chair": [0.65],
            "pre_cond_prob_bed": [0.20],
            "pre_cond_prob_room": [0.15],
            "pre_state_visible_prob": [0.85],
            "pre_state_not_visible_prob": [0.10],
            "pre_state_out_of_room_prob": [0.05],
            "pre_dropout_visibility_ratio": [0.70],
            "pre_dropout_gap_ratio": [0.30],
            "pre_prob_not_visible": [0.10],
            "pre_prob_out_of_room": [0.05],
            "pre_distance_signal_count": [18],
        }
    )

    enriched = build_panel_prefall_event_windows(settings, event_windows, pd.DataFrame())
    row = enriched.iloc[0]

    assert row["prefall_location_v2_label"] == "no_patient"
    assert row["prefall_location_v3_label"] == "chair"
    assert row["prefall_location_reason"] == "sparse_signal_carry_forward_v3"
    assert row["prefall_location_v3_signal_regime"] == "visible"
    assert bool(row["prefall_location_v3_carry_forward_used"])
    assert row["pre_prob_v3_no_patient"] <= settings.prefall_panel_v3_visible_no_patient_cap


def test_panel_inference_uses_recent_visible_tail_to_override_stale_no_patient_prior(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "pre_prob_chair": [0.0],
            "pre_prob_bed": [0.0],
            "pre_prob_room": [0.0],
            "pre_prob_no_patient": [1.0],
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
            "patient_staff_iou": [0.0, 0.0, 0.0],
            "dominant_location_label": ["chair", "chair", "chair"],
            "nudge_score": [0.0, 0.0, 0.0],
        }
    )

    enriched = build_panel_prefall_event_windows(settings, event_windows, second_level)
    row = enriched.iloc[0]

    assert row["prefall_location_label"] == "chair"
    assert row["prefall_location_reason"] == "recent_visible_fallback"
    assert row["pre_prob_no_patient"] < 0.5


def test_panel_inference_uses_overlap_and_safety_zone_as_direct_room_evidence(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "pre_prob_chair": [0.10],
            "pre_prob_bed": [0.75],
            "pre_prob_room": [0.10],
            "pre_prob_no_patient": [0.05],
        }
    )
    second_level = pd.DataFrame(
        {
            "fall_event_id": [1, 1, 1, 1],
            "second_offset": [-150, -110, -40, -10],
            "frame_has_location_signal": [True, True, True, True],
            "patient_chair_distance": [1.5, 1.8, 2.4, 2.8],
            "patient_bed_distance": [0.3, 0.4, 1.6, 2.0],
            "patient_room_distance": [1.6, 1.7, 0.4, 0.3],
            "patient_staff_iou": [0.0, 0.0, 0.22, 0.35],
            "dominant_location_label": ["bed", "bed", "room", "room"],
            "nudge_score": [0.0, 5.0, 60.0, 100.0],
        }
    )

    enriched = build_panel_prefall_event_windows(settings, event_windows, second_level)
    row = enriched.iloc[0]

    assert row["prefall_location_label"] == "room"
    assert row["prefall_location_reason"] == "room_recovery_v2"
    assert row["prefall_location_v2_departure_detected"] is True
    assert row["pre_prob_room"] > row["pre_prob_bed"]


def test_panel_inference_recovers_room_when_tail_distance_improves_even_if_labels_stay_bed(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "pre_prob_chair": [0.05],
            "pre_prob_bed": [0.80],
            "pre_prob_room": [0.10],
            "pre_prob_no_patient": [0.05],
            "prefall_location_label": ["room"],
        }
    )
    second_level = pd.DataFrame(
        {
            "fall_event_id": [1, 1, 1, 1],
            "second_offset": [-150, -120, -30, -5],
            "frame_has_location_signal": [True, True, True, True],
            "patient_chair_distance": [2.4, 2.3, 2.6, 2.5],
            "patient_bed_distance": [0.35, 0.40, 0.70, 1.10],
            "patient_room_distance": [1.40, 1.30, 0.30, 0.22],
            "patient_staff_iou": [0.0, 0.0, 0.0, 0.0],
            "dominant_location_label": ["bed", "bed", "bed", "bed"],
            "nudge_score": [0.0, 0.0, 0.70, 0.95],
        }
    )

    enriched = build_panel_prefall_event_windows(settings, event_windows, second_level)
    row = enriched.iloc[0]

    assert row["prefall_location_label"] == "room"
    assert row["prefall_location_reason"] == "room_recovery_v2"
    assert row["panel_tail_room_nearest_share"] == 1.0
    assert row["pre_prob_room"] > row["pre_prob_bed"]


def test_panel_inference_keeps_bed_when_room_distance_lacks_support_gate(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "pre_prob_chair": [0.05],
            "pre_prob_bed": [0.80],
            "pre_prob_room": [0.10],
            "pre_prob_no_patient": [0.05],
            "prefall_location_label": ["room"],
        }
    )
    second_level = pd.DataFrame(
        {
            "fall_event_id": [1] * 10,
            "second_offset": [-150, -130, -110, -90, -70, -50, -40, -30, -20, -10],
            "frame_has_location_signal": [True] * 10,
            "patient_chair_distance": [2.2] * 10,
            "patient_bed_distance": [0.35, 0.35, 0.35, 0.35, 0.35, 0.60, 0.62, 0.64, 0.66, 0.68],
            "patient_room_distance": [1.10, 1.10, 1.10, 1.10, 1.10, 0.65, 0.67, 0.69, 0.71, 0.67],
            "patient_staff_iou": [0.0] * 10,
            "dominant_location_label": ["bed"] * 9 + ["room"],
            "nudge_score": [0.0] * 10,
        }
    )

    enriched = build_panel_prefall_event_windows(settings, event_windows, second_level)
    row = enriched.iloc[0]

    assert row["panel_tail_room_nearest_share"] == 0.2
    assert row["prefall_location_label"] == "bed"
    assert row["pre_prob_bed"] > row["pre_prob_room"]


def test_panel_inference_caps_no_patient_when_overlap_preserves_visible_presence(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "pre_prob_chair": [0.25],
            "pre_prob_bed": [0.05],
            "pre_prob_room": [0.10],
            "pre_prob_no_patient": [0.60],
        }
    )
    second_level = pd.DataFrame(
        {
            "fall_event_id": [1, 1, 1],
            "second_offset": [-55, -25, -5],
            "frame_has_location_signal": [True, True, True],
            "patient_chair_distance": [0.25, 0.25, 0.3],
            "patient_bed_distance": [0.45, 0.5, 0.55],
            "patient_room_distance": [1.5, 1.4, 1.5],
            "patient_staff_iou": [0.12, 0.18, 0.25],
            "dominant_location_label": ["chair", "chair", "chair"],
            "nudge_score": [0.0, 0.0, 0.0],
        }
    )

    enriched = build_panel_prefall_event_windows(settings, event_windows, second_level)
    row = enriched.iloc[0]

    assert row["pre_prob_no_patient"] <= settings.prefall_panel_overlap_no_patient_cap
    assert row["prefall_location_label"] == "chair"


def test_panel_inference_v3_uses_posture_to_shift_visible_state_mass(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "pre_prob_chair": [0.45],
            "pre_prob_bed": [0.40],
            "pre_prob_room": [0.10],
            "pre_prob_no_patient": [0.05],
        }
    )
    second_level = pd.DataFrame(
        {
            "fall_event_id": [1, 1, 1, 1, 1, 1],
            "second_offset": [-150, -120, -90, -60, -30, -10],
            "frame_has_location_signal": [True, True, True, True, True, True],
            "patient_chair_distance": [0.8, 0.8, 0.8, 0.9, 0.9, 1.0],
            "patient_bed_distance": [0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
            "patient_room_distance": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            "patient_staff_iou": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "dominant_location_label": ["chair", "chair", "chair", "bed", "bed", "bed"],
            "nudge_score": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "primary_patient_posture_label": ["lying", "lying", "lying", "lying", "lying", "lying"],
            "primary_patient_posture_score_sitting": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            "primary_patient_posture_score_standing": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            "primary_patient_posture_score_lying": [0.90, 0.90, 0.90, 0.90, 0.90, 0.90],
        }
    )

    enriched = build_panel_prefall_event_windows(settings, event_windows, second_level)
    row = enriched.iloc[0]

    assert row["prefall_location_v2_label"] in {"chair", "bed"}
    assert row["prefall_location_v3_label"] == "bed"
    assert row["prefall_location_label"] == "bed"
    assert row["prefall_location_reason"] == "posture_state_collapse_v3"
    assert row["pre_prob_v3_bed"] > row["pre_prob_v2_bed"]


def test_panel_inference_v3_falls_back_when_posture_is_ambiguous(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "pre_prob_chair": [0.70],
            "pre_prob_bed": [0.20],
            "pre_prob_room": [0.05],
            "pre_prob_no_patient": [0.05],
        }
    )
    second_level = pd.DataFrame(
        {
            "fall_event_id": [1, 1, 1],
            "second_offset": [-80, -40, -10],
            "frame_has_location_signal": [True, True, True],
            "patient_chair_distance": [0.4, 0.4, 0.4],
            "patient_bed_distance": [1.5, 1.5, 1.5],
            "patient_room_distance": [2.0, 2.0, 2.0],
            "patient_staff_iou": [0.0, 0.0, 0.0],
            "dominant_location_label": ["chair", "chair", "chair"],
            "nudge_score": [0.0, 0.0, 0.0],
            "primary_patient_posture_label": ["sitting", "standing", "lying"],
            "primary_patient_posture_score_sitting": [0.34, 0.33, 0.33],
            "primary_patient_posture_score_standing": [0.33, 0.34, 0.33],
            "primary_patient_posture_score_lying": [0.33, 0.33, 0.34],
        }
    )

    enriched = build_panel_prefall_event_windows(settings, event_windows, second_level)
    row = enriched.iloc[0]

    assert row["prefall_location_v2_label"] == "chair"
    assert row["prefall_location_v3_label"] == "chair"
    assert row["prefall_location_label"] == "chair"
    assert row["prefall_location_reason"] != "posture_state_collapse_v3"


def test_posture_adjustment_preserves_room_mass_while_rebalancing_chair_bed() -> None:
    adjusted = _posture_adjusted_visible_probs(
        {
            "chair": 0.325169,
            "bed": 0.25,
            "room": 0.424831,
        },
        {
            "sitting": 0.05,
            "standing": 0.000003,
            "lying": 0.949997,
        },
    )

    assert adjusted["room"] == 0.424831
    assert adjusted["chair"] < 0.325169
    assert adjusted["bed"] > 0.25


def test_panel_inference_v3_uses_departure_override_when_room_signal_is_moderate(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "pre_prob_chair": [0.05],
            "pre_prob_bed": [0.75],
            "pre_prob_room": [0.05],
            "pre_prob_no_patient": [0.15],
        }
    )
    second_level = pd.DataFrame(
        {
            "fall_event_id": [1] * 12,
            "second_offset": [-170, -140, -58, -52, -46, -40, -34, -28, -22, -16, -10, -4],
            "frame_has_location_signal": [True] * 12,
            "patient_chair_distance": [2.0] * 12,
            "patient_bed_distance": [0.35, 0.38, 0.42, 0.46, 0.50, 0.55, 0.70, 0.75, 0.80, 0.85, 0.95, 1.10],
            "patient_room_distance": [1.80, 1.75, 0.95, 0.90, 0.80, 0.76, 0.68, 0.64, 0.60, 0.55, 0.50, 0.45],
            "patient_staff_iou": [0.0, 0.0, 0.08, 0.08, 0.08, 0.08, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
            "dominant_location_label": ["bed", "bed", "bed", "bed", "bed", "bed", "bed", "bed", "bed", "bed", "bed", "room"],
            "nudge_score": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.0],
        }
    )

    enriched = build_panel_prefall_event_windows(settings, event_windows, second_level)
    row = enriched.iloc[0]

    assert row["prefall_location_v2_label"] == "bed"
    assert row["prefall_location_v3_label"] == "room"
    assert row["prefall_location_reason"] == "room_departure_recovery_v3"
    assert row["prefall_location_v3_departure_confidence"] >= settings.prefall_panel_v3_departure_min_confidence
    assert row["pre_prob_v3_room"] > row["pre_prob_v2_room"]


def test_panel_inference_v3_does_not_override_room_with_nonstanding_posture(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "pre_prob_chair": [0.30],
            "pre_prob_bed": [0.20],
            "pre_prob_room": [0.40],
            "pre_prob_no_patient": [0.10],
        }
    )
    second_level = pd.DataFrame(
        {
            "fall_event_id": [1, 1, 1, 1],
            "second_offset": [-120, -90, -30, -10],
            "frame_has_location_signal": [True, True, True, True],
            "patient_chair_distance": [0.4, 0.4, 1.5, 1.5],
            "patient_bed_distance": [1.8, 1.8, 2.0, 2.0],
            "patient_room_distance": [2.5, 2.5, 0.5, 0.5],
            "patient_staff_iou": [0.0, 0.0, 0.0, 0.0],
            "dominant_location_label": ["chair", "chair", "room", "room"],
            "nudge_score": [0.0, 0.0, 0.0, 0.0],
            "primary_patient_posture_label": ["sitting", "sitting", "sitting", "sitting"],
            "primary_patient_posture_score_sitting": [0.99, 0.99, 0.99, 0.99],
            "primary_patient_posture_score_standing": [0.005, 0.005, 0.005, 0.005],
            "primary_patient_posture_score_lying": [0.005, 0.005, 0.005, 0.005],
        }
    )

    enriched = build_panel_prefall_event_windows(settings, event_windows, second_level)
    row = enriched.iloc[0]

    assert row["prefall_location_v2_label"] == "room"
    assert row["prefall_location_v3_label"] == "room"
    assert row["pre_prob_v3_room"] >= row["pre_prob_v2_room"]
    assert row["prefall_location_reason"] != "posture_state_collapse_v3"


def test_panel_inference_v3_recovers_sparse_room_presence_from_cache_support(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "pre_prob_chair": [0.10],
            "pre_prob_bed": [0.10],
            "pre_prob_room": [0.10],
            "pre_prob_no_patient": [0.70],
            "pre_focus_weight_chair": [0.34],
            "pre_focus_weight_bed": [0.33],
            "pre_focus_weight_room": [0.33],
            "pre_cond_prob_chair": [0.33],
            "pre_cond_prob_bed": [0.33],
            "pre_cond_prob_room": [0.34],
            "pre_state_visible_prob": [0.20],
            "pre_state_not_visible_prob": [0.85],
            "pre_state_out_of_room_prob": [0.10],
            "pre_dropout_visibility_ratio": [0.20],
            "pre_prob_not_visible": [0.80],
            "pre_prob_out_of_room": [0.10],
            "pre_distance_signal_count": [2],
        }
    )
    second_level = pd.DataFrame(
        {
            "fall_event_id": [1, 1, 1, 1],
            "second_offset": [-55, -35, -15, -5],
            "frame_has_location_signal": [False, False, False, False],
            "patient_chair_distance": [pd.NA, pd.NA, pd.NA, pd.NA],
            "patient_bed_distance": [pd.NA, pd.NA, pd.NA, pd.NA],
            "patient_room_distance": [pd.NA, pd.NA, pd.NA, pd.NA],
            "patient_staff_iou": [0.0, 0.0, 0.0, 0.0],
            "dominant_location_label": ["room", "room", "room", "room"],
            "nudge_score": [0.0, 0.0, 0.0, 0.0],
            "motion_bac": [0.16, 0.18, 0.17, 0.19],
            "patient_candidate_count": [1, 1, 1, 1],
            "other_candidate_count": [1, 1, 1, 1],
            "bed_candidate_count": [0, 0, 0, 0],
            "chair_candidate_count": [0, 0, 0, 0],
        }
    )

    enriched = build_panel_prefall_event_windows(settings, event_windows, second_level)
    row = enriched.iloc[0]

    assert row["prefall_location_v2_label"] == "no_patient"
    assert row["prefall_location_v3_signal_regime"] == "present_but_sparse"
    assert row["prefall_location_label"] == "room"
    assert row["prefall_location_reason"] == "room_departure_recovery_v3"
    assert row["prefall_location_v3_presence_support"] >= settings.prefall_panel_v3_presence_min_support
    assert row["prefall_location_v3_room_presence_support"] >= settings.prefall_panel_v3_presence_min_support
    assert row["pre_prob_no_patient"] <= settings.prefall_panel_v3_present_sparse_no_patient_cap


def test_panel_inference_v3_recovers_bed_from_bed_in_bed_cache_support(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "pre_prob_chair": [0.10],
            "pre_prob_bed": [0.10],
            "pre_prob_room": [0.10],
            "pre_prob_no_patient": [0.70],
            "pre_focus_weight_chair": [0.34],
            "pre_focus_weight_bed": [0.33],
            "pre_focus_weight_room": [0.33],
            "pre_cond_prob_chair": [0.33],
            "pre_cond_prob_bed": [0.34],
            "pre_cond_prob_room": [0.33],
            "pre_state_visible_prob": [0.20],
            "pre_state_not_visible_prob": [0.85],
            "pre_state_out_of_room_prob": [0.10],
            "pre_dropout_visibility_ratio": [0.20],
            "pre_prob_not_visible": [0.80],
            "pre_prob_out_of_room": [0.10],
            "pre_distance_signal_count": [2],
        }
    )
    second_level = pd.DataFrame(
        {
            "fall_event_id": [1, 1, 1, 1],
            "second_offset": [-55, -35, -15, -5],
            "frame_has_location_signal": [False, False, False, False],
            "patient_chair_distance": [pd.NA, pd.NA, pd.NA, pd.NA],
            "patient_bed_distance": [pd.NA, pd.NA, pd.NA, pd.NA],
            "patient_room_distance": [pd.NA, pd.NA, pd.NA, pd.NA],
            "patient_staff_iou": [0.0, 0.0, 0.0, 0.0],
            "dominant_location_label": ["bed", "bed", "bed", "bed"],
            "nudge_score": [0.0, 0.0, 0.0, 0.0],
            "bed_in_bed_score": [0.90, 0.92, 0.94, 0.95],
            "motion_bac": [0.10, 0.12, 0.11, 0.12],
            "patient_candidate_count": [1, 1, 1, 1],
            "other_candidate_count": [0, 0, 0, 0],
            "bed_candidate_count": [1, 1, 1, 1],
            "chair_candidate_count": [0, 0, 0, 0],
        }
    )

    enriched = build_panel_prefall_event_windows(settings, event_windows, second_level)
    row = enriched.iloc[0]

    assert row["prefall_location_v2_label"] == "no_patient"
    assert row["prefall_location_v3_signal_regime"] == "present_but_sparse"
    assert row["prefall_location_label"] == "bed"
    assert row["prefall_location_reason"] == "bed_in_bed_recovery_v3"
    assert row["prefall_location_v3_bed_in_bed_support"] >= settings.prefall_panel_v3_bed_in_bed_min_score
    assert row["pre_prob_bed"] > row["pre_prob_no_patient"]


def test_panel_inference_v3_does_not_rescue_bed_from_lone_invisible_bed_score(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "pre_prob_chair": [0.0],
            "pre_prob_bed": [0.0],
            "pre_prob_room": [0.0],
            "pre_prob_no_patient": [1.0],
            "pre_state_visible_prob": [0.0],
            "pre_state_not_visible_prob": [0.5],
            "pre_state_out_of_room_prob": [0.5],
            "pre_dropout_visibility_ratio": [0.0],
            "pre_prob_not_visible": [0.5],
            "pre_prob_out_of_room": [0.5],
            "pre_distance_signal_count": [0],
        }
    )
    second_level = pd.DataFrame(
        {
            "fall_event_id": [1, 1, 1, 1],
            "second_offset": [-40, -25, -10, -5],
            "frame_has_location_signal": [False, False, False, False],
            "patient_chair_distance": [pd.NA, pd.NA, pd.NA, pd.NA],
            "patient_bed_distance": [pd.NA, pd.NA, pd.NA, pd.NA],
            "patient_room_distance": [pd.NA, pd.NA, pd.NA, pd.NA],
            "patient_staff_iou": [0.0, 0.0, 0.0, 0.0],
            "dominant_location_label": ["no_patient"] * 4,
            "nudge_score": [0.0, 0.0, 0.0, 0.0],
            "bed_in_bed_score": [0.72, 0.78, 0.81, 0.76],
            "motion_bac": [0.0, 0.0, 0.0, 0.0],
            "patient_candidate_count": [0, 0, 0, 0],
            "staff_candidate_count": [0, 0, 0, 0],
            "other_candidate_count": [0, 0, 0, 0],
            "bed_candidate_count": [1, 1, 1, 1],
            "chair_candidate_count": [0, 0, 0, 0],
        }
    )

    enriched = build_panel_prefall_event_windows(settings, event_windows, second_level)
    row = enriched.iloc[0]

    assert row["prefall_location_v3_bed_in_bed_support"] < settings.prefall_panel_v3_bed_in_bed_invisible_min_score
    assert row["prefall_location_label"] == "no_patient"
    assert row["prefall_location_reason"] == "zero_signal_default"
    assert row["pre_prob_bed"] == 0.0


def test_panel_inference_v3_does_not_recover_room_from_sparse_nonpatient_context(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "pre_prob_chair": [0.0],
            "pre_prob_bed": [0.0],
            "pre_prob_room": [0.0],
            "pre_prob_no_patient": [1.0],
            "pre_state_visible_prob": [0.20],
            "pre_state_not_visible_prob": [0.45],
            "pre_state_out_of_room_prob": [0.35],
            "pre_dropout_visibility_ratio": [0.10],
            "pre_prob_not_visible": [0.45],
            "pre_prob_out_of_room": [0.35],
            "pre_distance_signal_count": [0],
        }
    )
    second_level = pd.DataFrame(
        {
            "fall_event_id": [1, 1, 1, 1],
            "second_offset": [-20, -15, -10, -5],
            "frame_has_location_signal": [False, False, False, False],
            "patient_chair_distance": [pd.NA, pd.NA, pd.NA, pd.NA],
            "patient_bed_distance": [pd.NA, pd.NA, pd.NA, pd.NA],
            "patient_room_distance": [pd.NA, pd.NA, pd.NA, pd.NA],
            "patient_staff_iou": [0.0, 0.0, 0.0, 0.0],
            "dominant_location_label": ["no_patient"] * 4,
            "nudge_score": [100.0, 100.0, 100.0, 100.0],
            "motion_bac": [0.0, 0.0, 0.0, 0.0],
            "patient_candidate_count": [0, 0, 0, 0],
            "staff_candidate_count": [1, 1, 1, 1],
            "other_candidate_count": [1, 1, 1, 1],
            "bed_candidate_count": [1, 1, 1, 1],
            "chair_candidate_count": [1, 1, 1, 1],
        }
    )

    enriched = build_panel_prefall_event_windows(settings, event_windows, second_level)
    row = enriched.iloc[0]

    assert row["prefall_location_v3_signal_regime"] == "present_but_sparse"
    assert row["prefall_location_v3_room_presence_support"] >= settings.prefall_panel_v3_presence_min_support
    assert row["prefall_location_label"] == "no_patient"
    assert row["prefall_location_reason"] == "zero_signal_default"
    assert row["pre_prob_room"] == 0.0


def test_panel_inference_v3_does_not_carry_forward_anchor_only_visible_history(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "pre_prob_chair": [0.0],
            "pre_prob_bed": [0.20],
            "pre_prob_room": [0.0],
            "pre_prob_no_patient": [0.80],
            "pre_focus_weight_chair": [0.0],
            "pre_focus_weight_bed": [1.0],
            "pre_focus_weight_room": [0.0],
            "pre_cond_prob_chair": [0.0],
            "pre_cond_prob_bed": [1.0],
            "pre_cond_prob_room": [0.0],
            "pre_state_visible_prob": [0.85],
            "pre_state_not_visible_prob": [0.10],
            "pre_state_out_of_room_prob": [0.05],
            "pre_dropout_visibility_ratio": [0.80],
            "pre_prob_not_visible": [0.10],
            "pre_prob_out_of_room": [0.05],
            "pre_distance_signal_count": [1],
        }
    )
    second_level = pd.DataFrame(
        {
            "fall_event_id": [1, 1, 1, 1],
            "second_offset": [-120, -90, -10, -5],
            "frame_has_location_signal": [True, True, False, False],
            "patient_chair_distance": [1.5, 1.6, pd.NA, pd.NA],
            "patient_bed_distance": [0.10, 0.12, pd.NA, pd.NA],
            "patient_room_distance": [2.0, 2.0, pd.NA, pd.NA],
            "patient_staff_iou": [0.0, 0.0, 0.0, 0.0],
            "dominant_location_label": ["bed", "bed", "no_patient", "no_patient"],
            "nudge_score": [0.0, 0.0, 0.0, 0.0],
            "patient_candidate_count": [1, 1, 0, 0],
            "bed_candidate_count": [1, 1, 0, 0],
            "chair_candidate_count": [0, 0, 0, 0],
            "primary_patient_posture_label": ["lying", "lying", None, None],
            "primary_patient_posture_score_sitting": [0.02, 0.02, pd.NA, pd.NA],
            "primary_patient_posture_score_standing": [0.02, 0.02, pd.NA, pd.NA],
            "primary_patient_posture_score_lying": [0.96, 0.96, pd.NA, pd.NA],
        }
    )

    enriched = build_panel_prefall_event_windows(settings, event_windows, second_level)
    row = enriched.iloc[0]

    assert row["prefall_location_v2_label"] == "no_patient"
    assert row["prefall_location_label"] == "no_patient"
    assert row["prefall_location_reason"] != "sparse_signal_carry_forward_v3"


def test_panel_inference_v3_does_not_use_single_tail_room_frame_for_departure_override(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    event_windows = pd.DataFrame(
        {
            "fall_event_id": [1],
            "pre_prob_chair": [0.0],
            "pre_prob_bed": [0.0],
            "pre_prob_room": [0.0],
            "pre_prob_no_patient": [1.0],
            "pre_state_visible_prob": [0.20],
            "pre_state_not_visible_prob": [0.45],
            "pre_state_out_of_room_prob": [0.35],
            "pre_dropout_visibility_ratio": [0.10],
            "pre_prob_not_visible": [0.45],
            "pre_prob_out_of_room": [0.35],
            "pre_distance_signal_count": [0],
        }
    )
    second_level = pd.DataFrame(
        {
            "fall_event_id": [1, 1, 1, 1],
            "second_offset": [-20, -15, -10, -5],
            "frame_has_location_signal": [False, False, False, False],
            "patient_chair_distance": [pd.NA, pd.NA, pd.NA, pd.NA],
            "patient_bed_distance": [pd.NA, pd.NA, pd.NA, pd.NA],
            "patient_room_distance": [pd.NA, pd.NA, pd.NA, pd.NA],
            "patient_staff_iou": [0.0, 0.0, 0.0, 0.0],
            "dominant_location_label": ["room", "room", "room", "room"],
            "nudge_score": [100.0, 100.0, 100.0, 100.0],
            "patient_candidate_count": [0, 0, 0, 0],
            "staff_candidate_count": [1, 1, 1, 1],
            "other_candidate_count": [1, 1, 1, 1],
            "bed_candidate_count": [1, 1, 1, 1],
            "chair_candidate_count": [1, 1, 1, 1],
        }
    )

    enriched = build_panel_prefall_event_windows(settings, event_windows, second_level)
    row = enriched.iloc[0]

    assert row["prefall_location_v2_label"] == "no_patient"
    assert row["prefall_location_label"] != "room"
    assert row["prefall_location_reason"] != "room_departure_recovery_v3"
