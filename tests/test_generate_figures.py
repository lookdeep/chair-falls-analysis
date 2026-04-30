# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import generate_figures


def _write_required_csvs(project_root: Path, run_id: str, *, public_bundle: bool) -> Path:
    if public_bundle:
        derived_dir = project_root / "data" / "public" / "derived"
        mechanism_path = derived_dir / f"mechanism_taxonomy_{run_id}.csv"
    else:
        derived_dir = project_root / "outputs" / "qa"
        mechanism_path = project_root / "outputs" / f"mechanism_taxonomy_{run_id}.csv"

    derived_dir.mkdir(parents=True, exist_ok=True)
    mechanism_path.parent.mkdir(parents=True, exist_ok=True)

    (derived_dir / f"chair_bed_risk_rates_{run_id}.csv").write_text(
        "scope,position,rate_per_1000_exposure_hours_expected,expected_falls,exposure_hours\n"
        "intervention_eligible,chair,19.2,6.16,320.51\n"
        "intervention_eligible,bed,4.33,22.16,5121.42\n",
        encoding="utf-8",
    )
    (derived_dir / f"chair_bed_adjusted_rr_{run_id}.csv").write_text(
        "sensitivity_label,rr,ci_lower,ci_upper,p_value,n_events,model_notes\n"
        "primary_adjusted,2.35,0.87,6.33,0.0907,40,estimable\n",
        encoding="utf-8",
    )
    (derived_dir / f"chair_bed_operational_event_rates_{run_id}.csv").write_text(
        "scope,source_study_id,metric,metric_type,metric_unit,window_half_width_seconds,position,exposure_hours,source_value_in_scope,positioned_numerator_value,excluded_missing_posture_value,excluded_ambiguous_events,rate_per_100_exposure_hours,active_seconds_per_exposure_hour,notes\n"
        "talk_confirmed_intervention_oneoff,operational_position_study_20260325_talkgate,talk_event,event_count,count,3,chair,348.788333,3637,257,21,53,73.683657,,Talk clicks with +/- 3 second state association.\n"
        "talk_confirmed_intervention_oneoff,operational_position_study_20260325_talkgate,talk_event,event_count,count,3,bed,5351.695000,3637,2893,21,53,54.057640,,Talk clicks with +/- 3 second state association.\n"
        "talk_confirmed_intervention_oneoff,operational_position_study_20260325_talkgate,talk_event,event_count,count,5,chair,348.788333,3637,248,18,53,71.103296,,Talk clicks with +/- 5 second state association.\n"
        "talk_confirmed_intervention_oneoff,operational_position_study_20260325_talkgate,talk_event,event_count,count,5,bed,5351.695000,3637,2912,18,53,54.412667,,Talk clicks with +/- 5 second state association.\n"
        "talk_confirmed_intervention_oneoff,operational_position_study_20260325_talkgate,alarm_trigger,event_count,count,3,chair,348.788333,644,68,1,3,19.496065,,Manual alarm triggers with +/- 3 second state association.\n"
        "talk_confirmed_intervention_oneoff,operational_position_study_20260325_talkgate,alarm_trigger,event_count,count,3,bed,5351.695000,644,492,1,3,9.193349,,Manual alarm triggers with +/- 3 second state association.\n"
        "talk_confirmed_intervention_oneoff,operational_position_study_20260325_talkgate,alarm_trigger,event_count,count,5,chair,348.788333,644,63,1,3,18.062531,,Manual alarm triggers with +/- 5 second state association.\n"
        "talk_confirmed_intervention_oneoff,operational_position_study_20260325_talkgate,alarm_trigger,event_count,count,5,bed,5351.695000,644,497,1,3,9.286777,,Manual alarm triggers with +/- 5 second state association.\n"
        "talk_confirmed_intervention_oneoff,operational_position_study_20260325_talkgate,nudge_active_seconds,duration,seconds,,chair,348.788333,1586458,97442,2599,0,,279.372877,Active nudge-state seconds per chair exposure-hour.\n"
        "talk_confirmed_intervention_oneoff,operational_position_study_20260325_talkgate,nudge_active_seconds,duration,seconds,,bed,5351.695000,1586458,1324103,2599,0,,247.417500,Active nudge-state seconds per bed exposure-hour.\n",
        encoding="utf-8",
    )
    (derived_dir / f"chair_bed_misclassification_sensitivity_{run_id}.csv").write_text(
        "scenario_label,rr,ci_lower,ci_upper\n"
        "symmetric_swap_10pct,2.20,0.80,6.10\n"
        "symmetric_swap_20pct,2.05,0.76,5.90\n"
        "symmetric_swap_30pct,1.90,0.70,5.60\n"
        "chair_to_bed_20pct,1.70,0.60,4.90\n"
        "bed_to_chair_20pct,2.80,1.00,7.50\n",
        encoding="utf-8",
    )
    (derived_dir / f"chair_bed_threshold_sensitivity_{run_id}.csv").write_text(
        "duration_threshold_hours,rr,ci_lower,ci_upper\n"
        "4,2.35,0.87,6.33\n"
        "12,2.40,0.90,6.40\n"
        "24,2.15,0.79,5.95\n"
        "48,2.05,0.75,5.70\n",
        encoding="utf-8",
    )
    (derived_dir / f"falls_prefall_location_probability_breakdown_{run_id}.csv").write_text(
        "grouping,group_value,prefall_location_label,expected_falls\n"
        "daypart,window_06_08,chair,1.0\n",
        encoding="utf-8",
    )
    mechanism_path.write_text(
        "event_key,prefall_location,charter_mechanism,posture_bucket,count\n"
        "1,chair,footrest_positioning,chair,1\n",
        encoding="utf-8",
    )
    return derived_dir


def test_discover_csvs_prefers_public_bundle_artifacts(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_id = "my_run_20260312T000000Z"
    public_dir = _write_required_csvs(project_root, run_id, public_bundle=True)
    _write_required_csvs(project_root, run_id, public_bundle=False)

    csv_map = generate_figures.discover_csvs(run_id, project_root)
    assert csv_map["risk_rates"].parent == public_dir
    assert csv_map["mechanism_taxonomy"].parent == public_dir
    assert csv_map["operational_event_rates"].parent == public_dir


def test_make_fig2_allows_non_estimable_model_specs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(generate_figures, "OUT", out_dir)

    adjusted_rr = pd.DataFrame(
        [
            {
                "sensitivity_label": "primary_adjusted",
                "rr": 2.35,
                "ci_lower": 0.87,
                "ci_upper": 6.33,
            },
            {
                "sensitivity_label": "primary_adjusted_hc3",
                "rr": 2.35,
                "ci_lower": 0.93,
                "ci_upper": 5.94,
            },
            {
                "sensitivity_label": "primary_adjusted_clustered",
                "rr": 2.35,
                "ci_lower": 1.89,
                "ci_upper": 2.92,
            },
            {
                "sensitivity_label": "position_certain_only",
                "rr": None,
                "ci_lower": None,
                "ci_upper": None,
                "model_notes": "insufficient_data",
            },
            {
                "sensitivity_label": "furniture_origin_reclassified",
                "rr": 2.35,
                "ci_lower": 0.87,
                "ci_upper": 6.33,
            },
        ]
    )
    misclassification = pd.DataFrame(
        [
            {"scenario_label": "symmetric_swap_10pct", "rr": 2.20, "ci_lower": 0.80, "ci_upper": 6.10},
            {"scenario_label": "symmetric_swap_20pct", "rr": 2.05, "ci_lower": 0.76, "ci_upper": 5.90},
            {"scenario_label": "symmetric_swap_30pct", "rr": 1.90, "ci_lower": 0.70, "ci_upper": 5.60},
            {"scenario_label": "chair_to_bed_20pct", "rr": 1.70, "ci_lower": 0.60, "ci_upper": 4.90},
            {"scenario_label": "bed_to_chair_20pct", "rr": 2.80, "ci_lower": 1.00, "ci_upper": 7.50},
        ]
    )
    threshold = pd.DataFrame(
        [
            {"threshold_hours": 4, "rr": 2.35, "ci_lower": 0.87, "ci_upper": 6.33},
            {"threshold_hours": 12, "rr": 2.40, "ci_lower": 0.90, "ci_upper": 6.40},
            {"threshold_hours": 24, "rr": 2.15, "ci_lower": 0.79, "ci_upper": 5.95},
            {"threshold_hours": 48, "rr": 2.05, "ci_lower": 0.75, "ci_upper": 5.70},
        ]
    )

    generate_figures.make_fig2(adjusted_rr, misclassification, threshold)

    assert (out_dir / "fig2_sensitivity_forest.svg").exists()
    assert (out_dir / "fig2_sensitivity_forest.png").exists()


def test_make_fig5_creates_daypart_context_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(generate_figures, "OUT", out_dir)

    block_order = [
        "window_00_05",
        "window_06_08",
        "window_09_11",
        "window_12_14",
        "window_15_17",
        "window_18_20",
        "window_21_23",
    ]
    rows: list[dict[str, object]] = []
    for idx, block in enumerate(block_order):
        chair_prob = 0.15 + idx * 0.05
        rows.extend(
            [
                {
                    "grouping": "daypart",
                    "group_value": block,
                    "prefall_location_label": "chair",
                    "expected_falls": 1.5 + idx,
                    "mean_probability": chair_prob,
                },
                {
                    "grouping": "daypart",
                    "group_value": block,
                    "prefall_location_label": "bed",
                    "expected_falls": 2.0 + idx,
                    "mean_probability": 0.0,
                },
                {
                    "grouping": "daypart",
                    "group_value": block,
                    "prefall_location_label": "room",
                    "expected_falls": 0.5 + idx * 0.1,
                    "mean_probability": 0.0,
                },
                {
                    "grouping": "daypart",
                    "group_value": block,
                    "prefall_location_label": "no_patient",
                    "expected_falls": 0.25 + idx * 0.05,
                    "mean_probability": 0.0,
                },
            ]
        )

    generate_figures.make_fig5(pd.DataFrame(rows))

    assert (out_dir / "fig5_chair_occupancy_daypart.svg").exists()
    assert (out_dir / "fig5_chair_occupancy_daypart.png").exists()


def test_make_fig6_creates_public_context_schematic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(generate_figures, "OUT", out_dir)

    generate_figures.make_fig6()

    assert (out_dir / "fig6_context_schematic.svg").exists()
    assert (out_dir / "fig6_context_schematic.png").exists()


def test_fig7_footer_text_uses_main_text_secondary_analysis_copy() -> None:
    legacy = generate_figures._fig7_footer_text(talk_confirmed=False)
    talk_confirmed = generate_figures._fig7_footer_text(
        talk_confirmed=True,
        chair_exposure=348.788333,
        bed_exposure=5351.695,
    )

    assert "Appendix figure" not in legacy
    assert "Appendix figure" not in talk_confirmed
    assert "Main-text secondary-analysis figure." in legacy
    assert "Main-text secondary-analysis figure." in talk_confirmed
    assert "chair 348.8 h" in talk_confirmed
    assert "bed 5,351.7 h" in talk_confirmed


def test_make_fig7_creates_operational_signal_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(generate_figures, "OUT", out_dir)

    operational = pd.DataFrame(
        [
            {
                "scope": "talk_confirmed_intervention_oneoff",
                "metric": "talk_event",
                "metric_type": "event_count",
                "metric_unit": "count",
                "window_half_width_seconds": 3.0,
                "position": "chair",
                "exposure_hours": 348.788333,
                "positioned_numerator_value": 257,
                "rate_per_100_exposure_hours": 73.683657,
                "active_seconds_per_exposure_hour": None,
            },
            {
                "scope": "talk_confirmed_intervention_oneoff",
                "metric": "talk_event",
                "metric_type": "event_count",
                "metric_unit": "count",
                "window_half_width_seconds": 3.0,
                "position": "bed",
                "exposure_hours": 5351.695000,
                "positioned_numerator_value": 2893,
                "rate_per_100_exposure_hours": 54.057640,
                "active_seconds_per_exposure_hour": None,
            },
            {
                "scope": "talk_confirmed_intervention_oneoff",
                "metric": "alarm_trigger",
                "metric_type": "event_count",
                "metric_unit": "count",
                "window_half_width_seconds": 3.0,
                "position": "chair",
                "exposure_hours": 348.788333,
                "positioned_numerator_value": 68,
                "rate_per_100_exposure_hours": 19.496065,
                "active_seconds_per_exposure_hour": None,
            },
            {
                "scope": "talk_confirmed_intervention_oneoff",
                "metric": "alarm_trigger",
                "metric_type": "event_count",
                "metric_unit": "count",
                "window_half_width_seconds": 3.0,
                "position": "bed",
                "exposure_hours": 5351.695000,
                "positioned_numerator_value": 492,
                "rate_per_100_exposure_hours": 9.193349,
                "active_seconds_per_exposure_hour": None,
            },
            {
                "scope": "talk_confirmed_intervention_oneoff",
                "metric": "nudge_active_seconds",
                "metric_type": "duration",
                "metric_unit": "seconds",
                "window_half_width_seconds": None,
                "position": "chair",
                "exposure_hours": 348.788333,
                "positioned_numerator_value": 97442,
                "rate_per_100_exposure_hours": None,
                "active_seconds_per_exposure_hour": 279.372877,
            },
            {
                "scope": "talk_confirmed_intervention_oneoff",
                "metric": "nudge_active_seconds",
                "metric_type": "duration",
                "metric_unit": "seconds",
                "window_half_width_seconds": None,
                "position": "bed",
                "exposure_hours": 5351.695000,
                "positioned_numerator_value": 1324103,
                "rate_per_100_exposure_hours": None,
                "active_seconds_per_exposure_hour": 247.417500,
            },
        ]
    )

    generate_figures.make_fig7(operational)

    assert (out_dir / "fig7_operational_signals.svg").exists()
    assert (out_dir / "fig7_operational_signals.png").exists()
