# ruff: noqa: E402

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.consensus import (
    CONSENSUS_STATUS_ACCEPTED_NONFALL,
    CONSENSUS_STATUS_EXCLUDED_BAD_OR_NO_VIDEO,
    CONSENSUS_STATUS_EXCLUDED_NO_SIGNAL,
    CONSENSUS_STATUS_EXCLUDED_OFFSCREEN,
    CONSENSUS_STATUS_INCLUDED_FALL,
    build_consensus_adjudication_tables,
    prepare_consensus_annotations,
)


def test_prepare_consensus_annotations_applies_status_precedence_and_tag_normalization() -> None:
    frame = pd.DataFrame(
        {
            "event_key": [
                "100|2025-01-01|10:00",
                "101|2025-01-01|10:01",
                "102|2025-01-01|10:02",
                "103|2025-01-01|10:03",
            ],
            "prefall_location": ["chair", "room", "bed", "bed"],
            "fall_time_consensus": ["10:00:05", "", "", "10:03:05"],
            "response_time_consensus": ["10:00:30", "", "", "10:03:40"],
            "fall_tags": [
                "NO_FALL, off_screen",
                "bad_video, offscreen",
                "bathroom, off_screen",
                "slip",
            ],
        }
    )

    prepared = prepare_consensus_annotations(frame)

    assert prepared["fall_tags"].tolist() == [
        "no_fall, offscreen",
        "bad_video, offscreen",
        "bathroom, offscreen",
        "slip",
    ]
    assert prepared["consensus_status"].tolist() == [
        CONSENSUS_STATUS_ACCEPTED_NONFALL,
        CONSENSUS_STATUS_EXCLUDED_BAD_OR_NO_VIDEO,
        CONSENSUS_STATUS_EXCLUDED_OFFSCREEN,
        CONSENSUS_STATUS_INCLUDED_FALL,
    ]


def test_build_consensus_adjudication_tables_reports_included_sequences_and_chair_origin_counts() -> None:
    frame = pd.DataFrame(
        {
            "event_key": [
                "111|2025-01-01|10:00",
                "111|2025-01-01|10:00",
                "111|2025-01-01|10:00",
                "222|2025-01-01|11:00",
                "333|2025-01-01|12:00",
                "444|2025-01-01|13:00",
                "854|2022-12-23|11:07",
            ],
            "last_furniture": ["", "", "", "chair", "", "", ""],
            "furniture_departure_time": ["", "", "", "10:59:50", "", "", ""],
            "prefall_location": ["chair", "room", "bed", "room", "bed", "room", "chair"],
            "fall_time_consensus": ["10:00:05", "10:00:30", "10:01:10", "11:00:10", "", "", "11:07:40"],
            "response_time_consensus": ["", "10:00:40", "10:01:40", "11:00:50", "", "", "11:08:00"],
            "fall_tags": ["slip", "trip", "collapse", "support_table", "no_fall", "offscreen", "slip"],
        }
    )

    prepared = prepare_consensus_annotations(frame)
    status_counts, summary = build_consensus_adjudication_tables(prepared)
    summary_map = dict(zip(summary["metric"], summary["value"], strict=True))
    status_map = {
        row["consensus_status"]: (int(row["row_count"]), int(row["unique_event_keys"]))
        for row in status_counts.to_dict(orient="records")
    }

    assert status_map[CONSENSUS_STATUS_INCLUDED_FALL] == (4, 2)
    assert status_map[CONSENSUS_STATUS_ACCEPTED_NONFALL] == (1, 1)
    assert status_map[CONSENSUS_STATUS_EXCLUDED_OFFSCREEN] == (1, 1)
    assert status_map[CONSENSUS_STATUS_EXCLUDED_NO_SIGNAL] == (1, 1)
    assert summary_map["included_fall_rows"] == 4
    assert summary_map["included_fall_unique_event_keys"] == 2
    assert summary_map["included_fall_sequences"] == 3
    assert summary_map["source_unique_monitors"] == 5
    assert summary_map["excluded_no_signal_rows"] == 1
    assert summary_map["excluded_no_signal_unique_event_keys"] == 1
    assert summary_map["mechanism_eligible_event_keys"] == 2
    assert summary_map["chair_origin_room_or_no_patient_event_keys"] == 1


def test_prepare_consensus_annotations_excludes_known_no_signal_events_from_fall_cohort() -> None:
    frame = pd.DataFrame(
        {
            "event_key": ["854|2022-12-23|11:07"],
            "prefall_location": ["chair"],
            "fall_time_consensus": ["11:07:40"],
            "response_time_consensus": ["11:08:00"],
            "fall_tags": ["slip"],
        }
    )

    prepared = prepare_consensus_annotations(frame)
    row = prepared.iloc[0]

    assert row["consensus_status"] == CONSENSUS_STATUS_EXCLUDED_NO_SIGNAL
    assert bool(row["consensus_include_in_fall_cohort"]) is False


def test_build_consensus_adjudication_tables_appends_study_window_counts() -> None:
    frame = pd.DataFrame(
        {
            "event_key": [
                "100|2024-07-31|23:59",
                "101|2024-08-01|00:01",
                "102|2025-06-01|12:00",
                "103|2025-12-31|23:58",
                "104|2026-01-01|00:01",
            ],
            "last_furniture": ["", "", "", "", ""],
            "furniture_departure_time": ["", "", "", "", ""],
            "prefall_location": ["chair", "chair", "room", "bed", "bed"],
            "fall_time_consensus": ["23:59:10", "00:01:10", "", "23:58:10", "00:01:10"],
            "response_time_consensus": ["", "00:01:40", "", "23:58:40", "00:01:40"],
            "fall_tags": ["slip", "slip", "no_fall", "support_bed", "slip"],
        }
    )

    prepared = prepare_consensus_annotations(frame)
    _, summary = build_consensus_adjudication_tables(
        prepared,
        study_start_date=date(2024, 8, 1),
        study_end_date=date(2025, 12, 31),
    )
    summary_map = dict(zip(summary["metric"], summary["value"], strict=True))

    assert summary_map["source_rows"] == 5
    assert summary_map["included_fall_unique_monitors"] == 4
    assert summary_map["study_window_start_date"] == "2024-08-01"
    assert summary_map["study_window_end_date"] == "2025-12-31"
    assert summary_map["study_window_source_rows"] == 3
    assert summary_map["study_window_source_unique_event_keys"] == 3
    assert summary_map["study_window_source_unique_monitors"] == 3
    assert summary_map["study_window_included_fall_rows"] == 2
    assert summary_map["study_window_included_fall_unique_event_keys"] == 2
    assert summary_map["study_window_included_fall_sequences"] == 2
    assert summary_map["study_window_included_fall_unique_monitors"] == 2
    assert summary_map["study_window_accepted_nonfall_rows"] == 1
