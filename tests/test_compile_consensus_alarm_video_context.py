# ruff: noqa: E402, I001

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import pandas as pd

import compile_consensus_alarm_video_context as script


class FakeBigQueryClientWrapper:
    def __init__(self, _settings) -> None:
        self.calls: list[str] = []

    def query_to_dataframe(self, *, name: str, sql: str, dry_run_columns=None, destination="memory"):
        self.calls.append(name)
        if name == "consensus_alarm_video_source_range":
            return pd.DataFrame(
                {
                    "source_min_created_at": [pd.Timestamp("2025-07-21T00:00:00Z")],
                    "source_max_created_at": [pd.Timestamp("2026-03-14T00:00:00Z")],
                    "source_row_count": [10],
                }
            )
        if name == "consensus_alarm_video_matches":
            return pd.DataFrame(
                {
                    "source_row_number": [1],
                    "event_key": ["7814|2025-07-29|10:42"],
                    "event_instance_ordinal": [1],
                    "sequence_id": ["7814|2025-07-29|10:42#S01"],
                    "monitor_id": [7814],
                    "recording_id": [152624],
                    "created_at_utc": ["2025-07-29T15:42:18Z"],
                    "sec_from_fall": [27],
                    "recording_source": ["device"],
                    "recording_path": ["archive/recorder/7814/alarms/alarm_152624.mp4"],
                    "recording_status": ["deleted"],
                    "analysis_status": ["completed"],
                    "analysis_result": [
                        json.dumps(
                            {
                                "report": "The patient is seen falling out of bed.",
                                "events": [{"description": "Patient falls out of bed"}],
                            }
                        )
                    ],
                }
            )
        raise AssertionError(f"Unexpected query name: {name}")

    def flush_query_log(self) -> Path:
        return Path("/tmp/query-log.json")


def test_main_writes_summary_and_details_csv(tmp_path: Path, monkeypatch) -> None:
    consensus = pd.DataFrame(
        {
            "event_key": ["7814|2025-07-29|10:42", "9151|2025-08-22|08:49"],
            "prefall_location": ["room", "bed"],
            "fall_time_consensus": ["10:41:51", ""],
            "response_time_consensus": ["10:43:15", ""],
            "fall_tags": ["slip", "offscreen"],
        }
    )
    consensus_path = tmp_path / "consensus.csv"
    summary_path = tmp_path / "alarm-video-context.csv"
    details_path = tmp_path / "alarm-video-details.csv"
    consensus.to_csv(consensus_path, index=False)

    monkeypatch.setattr(script, "BigQueryClientWrapper", FakeBigQueryClientWrapper)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile_consensus_alarm_video_context.py",
            "--project-root",
            str(tmp_path),
            "--run-id",
            "test_run",
            "--consensus-path",
            str(consensus_path),
            "--output-path",
            str(summary_path),
            "--details-output-path",
            str(details_path),
        ],
    )

    script.main()

    summary = pd.read_csv(summary_path)
    details = pd.read_csv(details_path)

    assert summary["source_row_number"].tolist() == [1, 2]
    assert summary.loc[0, "alarm_video_reason"] == "matched_alarm_within_1m"
    assert int(summary.loc[0, "alarm_count_within_1m"]) == 1
    assert bool(summary.loc[0, "nearest_alarm_report_explicit_fall"]) is True
    assert summary.loc[0, "nearest_alarm_source"] == "device"
    assert summary.loc[1, "alarm_video_reason"] == "no_fall_time"

    assert details["recording_id"].tolist() == [152624]
    assert bool(details.loc[0, "report_explicit_fall"]) is True
    assert details.loc[0, "explicit_fall_match_source"] == "report"
