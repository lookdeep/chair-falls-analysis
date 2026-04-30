# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import compile_mechanism_taxonomy as mechanism
import pandas as pd


def test_read_consensus_filters_out_nonfall_offscreen_and_video_rows(tmp_path: Path) -> None:
    consensus = pd.DataFrame(
        {
            "event_key": [
                "100|2025-01-01|10:00",
                "101|2025-01-01|10:01",
                "102|2025-01-01|10:02",
                "103|2025-01-01|10:03",
            ],
            "last_furniture": ["", "", "", "chair"],
            "furniture_departure_time": ["", "", "", "10:02:50"],
            "prefall_location": ["chair", "room", "room", "room"],
            "fall_time_consensus": ["10:00:05", "", "", "10:03:05"],
            "response_time_consensus": ["10:00:30", "", "", "10:03:40"],
            "fall_tags": ["slip", "no_fall", "bad_video, offscreen", "support_table"],
        }
    )
    path = tmp_path / "consensus.csv"
    consensus.to_csv(path, index=False)

    rows = mechanism.read_consensus(path)

    assert [row["event_key"] for row in rows] == ["100|2025-01-01|10:00", "103|2025-01-01|10:03"]


def test_deduplicate_events_prefers_highest_priority_mechanism() -> None:
    deduped = mechanism.deduplicate_events(
        [
            {
                "event_key": "200|2025-01-01|10:00",
                "prefall_location": "chair",
                "fall_tags": "slip",
                "last_furniture": "",
                "furniture_departure_time": "",
                "fall_time_consensus": "10:00:05",
                "response_time_consensus": "",
            },
            {
                "event_key": "200|2025-01-01|10:00",
                "prefall_location": "chair",
                "fall_tags": "footrest, slip",
                "last_furniture": "",
                "furniture_departure_time": "",
                "fall_time_consensus": "10:00:10",
                "response_time_consensus": "10:00:40",
            },
        ]
    )

    assert len(deduped) == 1
    assert deduped[0]["charter_mechanism"] == "footrest_positioning"
