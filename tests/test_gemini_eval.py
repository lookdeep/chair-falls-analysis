# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.consensus import (
    CONSENSUS_STATUS_ACCEPTED_NONFALL,
    CONSENSUS_STATUS_EXCLUDED_OFFSCREEN,
    CONSENSUS_STATUS_INCLUDED_FALL,
)
from ld_chair_falls.gemini_eval import (
    CONSENSUS_STATUS_EXCLUDED_NO_SIGNAL,
    build_coverage_table,
    build_event_level_comparison,
    compare_gemini_summary_against_truth,
    gemini_file_to_event_key,
    load_gemini_summary,
    load_truth_annotations,
    normalize_gemini_tags,
    relative_seconds_from_event_key,
)


def test_gemini_file_to_event_key_parses_expected_filename() -> None:
    assert gemini_file_to_event_key("1456_2023-05-04_23-07.mp4") == "1456|2023-05-04|23:07"
    assert gemini_file_to_event_key("bad_name.mp4") is None


def test_relative_seconds_from_event_key_uses_event_hour_and_wraps_midnight() -> None:
    assert relative_seconds_from_event_key("10275|2025-10-15|18:47", "18:48:10") == 2890.0
    assert relative_seconds_from_event_key("999|2025-01-01|23:55", "00:00:05") == 3605.0


def test_normalize_gemini_tags_handles_pipe_delimited_values() -> None:
    assert normalize_gemini_tags("tumble|support_bed|OFF_SCREEN") == [
        "offscreen",
        "support_bed",
        "tumble",
    ]


def test_build_event_level_comparison_uses_nearest_truth_row_for_duplicate_event_key(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.csv"
    consensus_path = tmp_path / "consensus.csv"

    pd.DataFrame(
        [
            {
                "file": "2834_2023-12-29_11-11.mp4",
                "fall_detected": True,
                "prefall_location": "room",
                "last_furniture": "chair",
                "fall_time_seconds": 75,
                "fall_time_wallclock": "",
                "staff_entry_time_seconds": 78,
                "staff_entry_time_wallclock": "",
                "staff_response_time_seconds": 3,
                "fall_tags": "slip|back",
                "confidence": "high",
            }
        ]
    ).to_csv(summary_path, index=False)

    pd.DataFrame(
        [
            {
                "event_key": "2834|2023-12-29|11:11",
                "last_furniture": "",
                "furniture_departure_time": "",
                "prefall_location": "chair",
                "fall_time_consensus": "11:00:30",
                "response_time_consensus": "",
                "fall_tags": "slip",
            },
            {
                "event_key": "2834|2023-12-29|11:11",
                "last_furniture": "chair",
                "furniture_departure_time": "11:01:10",
                "prefall_location": "room",
                "fall_time_consensus": "11:01:15",
                "response_time_consensus": "11:01:18",
                "fall_tags": "slip, back",
            },
        ]
    ).to_csv(consensus_path, index=False)

    summary = load_gemini_summary(summary_path)
    truth = load_truth_annotations(consensus_path)

    comparison = build_event_level_comparison(summary, truth)
    row = comparison.iloc[0]

    assert row["match_source"] == "nearest_fall_time"
    assert row["gt_truth_row_ordinal"] == 2
    assert row["gt_prefall_location"] == "room"
    assert bool(row["location_match"]) is True
    assert row["fall_time_abs_error_seconds"] == 0.0


def test_load_gemini_summary_allows_positive_detection_without_visible_fall_time(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.csv"
    consensus_path = tmp_path / "consensus.csv"

    pd.DataFrame(
        [
            {
                "file": "100_2025-01-01_10-00.mp4",
                "fall_detected": True,
                "prefall_location": "room",
                "last_furniture": "bed",
                "fall_time_seconds": "",
                "fall_time_wallclock": "",
                "staff_entry_time_seconds": 22,
                "staff_entry_time_wallclock": "10:00:22",
                "staff_response_time_seconds": "",
                "fall_tags": "collapse",
                "confidence": "medium",
            }
        ]
    ).to_csv(summary_path, index=False)

    pd.DataFrame(
        [
            {
                "event_key": "100|2025-01-01|10:00",
                "last_furniture": "bed",
                "furniture_departure_time": "10:00:05",
                "prefall_location": "room",
                "fall_time_consensus": "10:00:12",
                "response_time_consensus": "10:00:22",
                "fall_tags": "collapse",
            }
        ]
    ).to_csv(consensus_path, index=False)

    summary = load_gemini_summary(summary_path)
    truth = load_truth_annotations(consensus_path)
    comparison = build_event_level_comparison(summary, truth)
    row = comparison.iloc[0]

    assert bool(row["gemini_fall_detected"]) is True
    assert pd.isna(summary.iloc[0]["gemini_fall_time_seconds"])
    assert row["match_source"] == "direct_event_key"
    assert pd.isna(row["fall_time_abs_error_seconds"])
    assert row["gemini_staff_entry_time_seconds"] == 22.0


def test_compare_gemini_summary_against_truth_scopes_headline_metrics_and_coverage(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.csv"
    consensus_path = tmp_path / "consensus.csv"
    results_path = tmp_path / "results.jsonl"

    pd.DataFrame(
        [
            {
                "file": "100_2025-01-01_10-00.mp4",
                "fall_detected": True,
                "prefall_location": "chair",
                "last_furniture": "chair",
                "fall_time_seconds": 12,
                "fall_time_wallclock": "10:00:12",
                "staff_entry_time_seconds": 20,
                "staff_entry_time_wallclock": "10:00:20",
                "staff_response_time_seconds": 8,
                "fall_tags": "slip",
                "confidence": "high",
            },
            {
                "file": "101_2025-01-01_10-01.mp4",
                "fall_detected": False,
                "prefall_location": "",
                "last_furniture": "",
                "fall_time_seconds": "",
                "fall_time_wallclock": "",
                "staff_entry_time_seconds": "",
                "staff_entry_time_wallclock": "",
                "staff_response_time_seconds": "",
                "fall_tags": "no_fall",
                "confidence": "high",
            },
            {
                "file": "102_2025-01-01_10-02.mp4",
                "fall_detected": True,
                "prefall_location": "room",
                "last_furniture": "",
                "fall_time_seconds": 30,
                "fall_time_wallclock": "10:00:30",
                "staff_entry_time_seconds": 45,
                "staff_entry_time_wallclock": "10:00:45",
                "staff_response_time_seconds": 15,
                "fall_tags": "tumble",
                "confidence": "medium",
            },
        ]
    ).to_csv(summary_path, index=False)

    pd.DataFrame(
        [
            {
                "event_key": "100|2025-01-01|10:00",
                "last_furniture": "chair",
                "furniture_departure_time": "10:00:10",
                "prefall_location": "chair",
                "fall_time_consensus": "10:00:12",
                "response_time_consensus": "10:00:20",
                "fall_tags": "slip",
            },
            {
                "event_key": "101|2025-01-01|10:01",
                "last_furniture": "",
                "furniture_departure_time": "",
                "prefall_location": "bed",
                "fall_time_consensus": "",
                "response_time_consensus": "",
                "fall_tags": "no_fall",
            },
            {
                "event_key": "102|2025-01-01|10:02",
                "last_furniture": "",
                "furniture_departure_time": "",
                "prefall_location": "room",
                "fall_time_consensus": "",
                "response_time_consensus": "",
                "fall_tags": "offscreen",
            },
            {
                "event_key": "103|2025-01-01|10:03",
                "last_furniture": "",
                "furniture_departure_time": "",
                "prefall_location": "bed",
                "fall_time_consensus": "10:03:10",
                "response_time_consensus": "10:03:20",
                "fall_tags": "slip",
            },
            {
                "event_key": "854|2022-12-23|11:07",
                "last_furniture": "",
                "furniture_departure_time": "",
                "prefall_location": "room",
                "fall_time_consensus": "11:07:40",
                "response_time_consensus": "11:09:00",
                "fall_tags": "slip",
            },
        ]
    ).to_csv(consensus_path, index=False)

    results_path.write_text(
        "\n".join(
            [
                '{"file":"100_2025-01-01_10-00.mp4","model":"gemini-2.5-flash","result":{"ok":true}}',
                '{"file":"101_2025-01-01_10-01.mp4","model":"gemini-2.5-flash","result":{"ok":true}}',
                '{"file":"102_2025-01-01_10-02.mp4","model":"gemini-2.5-flash","result":{"ok":true}}',
                '{"file":"103_2025-01-01_10-03.mp4","model":"gemini-2.5-flash","error":"timeout"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = compare_gemini_summary_against_truth(
        summary_csv=summary_path,
        consensus_csv=consensus_path,
        results_jsonl=results_path,
    )

    headline = artifacts.metrics.loc[artifacts.metrics["metric"] == "headline_denominator", "value"].iloc[0]
    assert headline == 2

    coverage_all = artifacts.coverage.loc[artifacts.coverage["consensus_status"] == "all"].iloc[0]
    assert int(coverage_all["total_gt_event_keys"]) == 5
    assert int(coverage_all["successful_event_keys"]) == 3
    assert int(coverage_all["failed_event_keys"]) == 1
    assert int(coverage_all["missing_event_keys"]) == 2

    offscreen_row = artifacts.coverage.loc[
        artifacts.coverage["consensus_status"] == CONSENSUS_STATUS_EXCLUDED_OFFSCREEN
    ].iloc[0]
    assert int(offscreen_row["successful_event_keys"]) == 1

    no_signal_row = artifacts.coverage.loc[
        artifacts.coverage["consensus_status"] == CONSENSUS_STATUS_EXCLUDED_NO_SIGNAL
    ].iloc[0]
    assert int(no_signal_row["missing_event_keys"]) == 1

    failure_keys = set(artifacts.failures["event_key"].tolist())
    assert failure_keys == {"103|2025-01-01|10:03", "854|2022-12-23|11:07"}

    false_positive_rate = artifacts.metrics.loc[
        artifacts.metrics["metric"] == "false_positive_rate", "value"
    ].iloc[0]
    assert float(false_positive_rate) == 0.0


def test_load_gemini_records_marks_error_rows() -> None:
    records = pd.DataFrame(
        [
            {
                "file": "100_2025-01-01_10-00.mp4",
                "event_key": "100|2025-01-01|10:00",
                "model": "gemini-2.5-flash",
                "record_status": "success",
                "error": None,
            },
            {
                "file": "101_2025-01-01_10-01.mp4",
                "event_key": "101|2025-01-01|10:01",
                "model": "gemini-2.5-flash",
                "record_status": "error",
                "error": "timeout",
            },
        ]
    )

    comparison = pd.DataFrame({"event_key": ["100|2025-01-01|10:00"]})
    truth = pd.DataFrame(
        {
            "event_key": ["100|2025-01-01|10:00", "101|2025-01-01|10:01"],
            "consensus_status": [CONSENSUS_STATUS_INCLUDED_FALL, CONSENSUS_STATUS_ACCEPTED_NONFALL],
        }
    )

    coverage = build_coverage_table(truth, comparison, records.loc[records["record_status"] == "error"])
    nonfall_row = coverage.loc[coverage["consensus_status"] == CONSENSUS_STATUS_ACCEPTED_NONFALL].iloc[0]

    assert int(nonfall_row["failed_event_keys"]) == 1
