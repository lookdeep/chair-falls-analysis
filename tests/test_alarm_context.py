# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from ld_chair_falls import alarm_context


def test_load_consensus_alarm_video_base_assigns_ordinals_and_row_numbers(tmp_path: Path) -> None:
    consensus = pd.DataFrame(
        {
            "event_key": [
                "111|2025-01-01|10:00",
                "111|2025-01-01|10:00",
                "222|2025-01-02|11:00",
            ],
            "prefall_location": ["chair", "chair", "bed"],
            "fall_time_consensus": ["10:00:05", "10:00:30", ""],
            "response_time_consensus": ["", "10:00:40", ""],
            "fall_tags": ["slip", "slip", "offscreen"],
        }
    )
    path = tmp_path / "consensus.csv"
    consensus.to_csv(path, index=False)

    frame, summary = alarm_context.load_consensus_alarm_video_base(path, "America/Chicago")

    assert frame["source_row_number"].tolist() == [1, 2, 3]
    assert frame["event_instance_ordinal"].tolist() == [1, 2, 1]
    assert frame["sequence_id"].tolist() == [
        "111|2025-01-01|10:00#S01",
        "111|2025-01-01|10:00#S01",
        "222|2025-01-02|11:00#S01",
    ]
    assert frame["fall_ts_utc"].notna().tolist() == [True, True, False]
    assert summary == {"rows": 3, "unique_event_keys": 2, "rows_with_fall_time": 2}


def test_normalize_alarm_matches_parses_report_and_event_descriptions() -> None:
    raw = pd.DataFrame(
        {
            "source_row_number": [1, 2],
            "event_key": ["100|2025-01-01|10:00", "101|2025-01-01|10:01"],
            "event_instance_ordinal": [1, 1],
            "sequence_id": ["100|2025-01-01|10:00#S01", "101|2025-01-01|10:01#S01"],
            "monitor_id": [100, 101],
            "recording_id": [10, 11],
            "created_at_utc": ["2025-01-01T16:00:10Z", "2025-01-01T16:01:10Z"],
            "sec_from_fall": [10, 10],
            "recording_source": ["device", "device"],
            "recording_path": ["a.mp4", "b.mp4"],
            "recording_status": ["deleted", "deleted"],
            "analysis_status": ["completed", "completed"],
            "analysis_result": [
                json.dumps({"report": "Patient fell to the floor.", "events": []}),
                json.dumps(
                    {
                        "report": "Patient is assisted by staff.",
                        "events": [{"description": "Patient slipped to the floor"}],
                    }
                ),
            ],
        }
    )

    normalized = alarm_context.normalize_alarm_matches(raw)

    assert normalized["report_explicit_fall"].tolist() == [True, False]
    assert normalized["events_explicit_fall"].tolist() == [False, True]
    assert normalized["explicit_fall_any"].tolist() == [True, True]
    assert normalized["explicit_fall_match_source"].tolist() == ["report", "events_description"]


def test_summarize_alarm_video_context_applies_coverage_counts_and_reasons() -> None:
    base = pd.DataFrame(
        {
            "source_row_number": [1, 2, 3],
            "event_key": ["100|2025-01-01|10:00", "101|2025-01-02|10:00", "102|2024-01-01|10:00"],
            "event_instance_ordinal": [1, 1, 1],
            "sequence_id": [
                "100|2025-01-01|10:00#S01",
                "101|2025-01-02|10:00#S01",
                "102|2024-01-01|10:00#S01",
            ],
            "monitor_id": [100, 101, 102],
            "date_local": ["2025-01-01", "2025-01-02", "2024-01-01"],
            "fall_time_consensus": ["10:00:05", "", "10:00:05"],
            "response_time_consensus": ["10:00:30", "", "10:00:30"],
            "consensus_status": ["included_fall", "excluded_offscreen", "included_fall"],
            "fall_ts_utc": [
                pd.Timestamp("2025-01-01T16:00:05Z"),
                pd.NaT,
                pd.Timestamp("2024-01-01T16:00:05Z"),
            ],
        }
    )
    matches = pd.DataFrame(
        {
            "source_row_number": [1, 1],
            "event_key": ["100|2025-01-01|10:00", "100|2025-01-01|10:00"],
            "event_instance_ordinal": [1, 1],
            "sequence_id": ["100|2025-01-01|10:00#S01", "100|2025-01-01|10:00#S01"],
            "monitor_id": [100, 100],
            "recording_id": [10, 11],
            "created_at_utc": ["2025-01-01T16:00:10Z", "2025-01-01T16:03:00Z"],
            "sec_from_fall": [5, 175],
            "recording_source": ["device", "device"],
            "recording_path": ["a.mp4", "b.mp4"],
            "recording_status": ["deleted", "deleted"],
            "analysis_status": ["completed", "completed"],
            "analysis_result": [
                json.dumps({"report": "Patient fell to the floor.", "events": []}),
                json.dumps({"report": "Patient is assisted by staff.", "events": []}),
            ],
        }
    )
    source_range = pd.DataFrame(
        {
            "source_min_created_at": [pd.Timestamp("2025-01-01T00:00:00Z")],
            "source_max_created_at": [pd.Timestamp("2025-12-31T23:59:59Z")],
            "source_row_count": [100],
        }
    )

    summary = alarm_context.summarize_alarm_video_context(base, matches, source_range)

    row1 = summary.loc[summary["source_row_number"] == 1].iloc[0]
    assert bool(row1["alarm_video_applicable"]) is True
    assert row1["alarm_video_reason"] == "matched_alarm_within_1m"
    assert int(row1["alarm_count_within_1m"]) == 1
    assert int(row1["alarm_count_within_3m"]) == 2
    assert bool(row1["nearest_alarm_report_explicit_fall"]) is True
    assert row1["nearest_alarm_path"] == "a.mp4"

    row2 = summary.loc[summary["source_row_number"] == 2].iloc[0]
    assert bool(row2["alarm_video_applicable"]) is False
    assert row2["alarm_video_reason"] == "no_fall_time"

    row3 = summary.loc[summary["source_row_number"] == 3].iloc[0]
    assert bool(row3["alarm_video_applicable"]) is False
    assert row3["alarm_video_reason"] == "outside_type7_source_range"
