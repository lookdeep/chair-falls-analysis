# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.config import Settings
from ld_chair_falls.qa import write_source_profile
from ld_chair_falls.transform import (
    build_analysis_base,
    build_fall_events_clean,
    build_hourly_location,
)


def test_build_hourly_location_uses_four_component_sum() -> None:
    raw = pd.DataFrame(
        {
            "hospital_id": [5, 5],
            "division_id": [pd.NA, pd.NA],
            "monitor_id": [101, 101],
            "patient_id": [1001, 1001],
            "hour_ts": ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"],
            "pct_chair": [0.6, 0.6],
            "pct_bed": [0.3, 0.3],
            "pct_ambulatory": [0.0, 0.0],
            "pct_not_located": [0.1, 0.25],
        }
    )

    transformed = build_hourly_location(raw)

    assert transformed["row_valid_pct"].tolist() == [True, False]
    assert transformed["pct_sum"].tolist() == pytest.approx([1.0, 1.15])


def test_build_hourly_location_backfills_missing_not_located() -> None:
    raw = pd.DataFrame(
        {
            "hospital_id": [5],
            "division_id": [pd.NA],
            "monitor_id": [101],
            "patient_id": [1001],
            "hour_ts": ["2025-01-01T00:00:00Z"],
            "pct_chair": [0.7],
            "pct_bed": [0.2],
            "pct_ambulatory": [0.1],
        }
    )

    transformed = build_hourly_location(raw)

    assert transformed["pct_not_located"].tolist() == [0.0]
    assert transformed["row_valid_pct"].tolist() == [True]


def test_source_profile_completeness_uses_not_located_component(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, run_id="source_profile_four_component")
    raw_tables = {
        "fall_events_source": pd.DataFrame(
            {
                "timestamp": ["2025-01-01T00:00:00Z"],
                "hospital_id": [5],
                "division_id": [pd.NA],
                "monitor_id": [1],
                "patient_id": [1],
            }
        ),
        "hourly_location_aggregation": pd.DataFrame(
            {
                "hour_ts": ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"],
                "hospital_id": [5, 5],
                "division_id": [pd.NA, pd.NA],
                "monitor_id": [101, 102],
                "patient_id": [1001, 1002],
                "pct_chair": [0.6, 0.5],
                "pct_bed": [0.3, 0.2],
                "pct_ambulatory": [0.0, 0.0],
                "pct_not_located": [0.1, 0.3],
            }
        ),
        "key_dimensions": pd.DataFrame(
            {
                "hospital_id": [5],
                "division_id": [pd.NA],
                "monitor_id": [101],
                "patient_id": [1001],
            }
        ),
    }

    output_path = write_source_profile(settings, raw_tables=raw_tables)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["control_denominator_completeness"]["rows_with_valid_pct_sum"] == 2
    assert payload["control_denominator_completeness"]["total_rows"] == 2


def test_source_profile_scopes_to_study_hospital_id(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, run_id="scope_profile", study_hospital_id=5)
    raw_tables = {
        "fall_events_source": pd.DataFrame(
            {
                "timestamp": ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"],
                "hospital_id": [5, 11],
                "division_id": [1, 111],
                "monitor_id": [1, 2],
                "patient_id": [1, 2],
            }
        ),
        "hourly_location_aggregation": pd.DataFrame(
            {
                "hour_ts": ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"],
                "hospital_id": [5, 11],
                "division_id": [1, 111],
                "monitor_id": [101, 202],
                "patient_id": [1001, 2002],
                "pct_chair": [0.6, 0.6],
                "pct_bed": [0.3, 0.3],
                "pct_ambulatory": [0.0, 0.0],
                "pct_not_located": [0.1, 0.1],
            }
        ),
        "key_dimensions": pd.DataFrame(
            {
                "hospital_id": [5, 11],
                "division_id": [1, 111],
                "monitor_id": [101, 202],
                "patient_id": [1001, 2002],
            }
        ),
    }

    payload = json.loads(write_source_profile(settings, raw_tables=raw_tables).read_text(encoding="utf-8"))
    assert payload["fall_events_source"]["rows"] == 1
    assert payload["hourly_location_aggregation"]["rows"] == 1
    assert payload["key_dimensions"]["rows"] == 1


def test_source_profile_negative_control_inventory_detects_partial_history(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        run_id="negative_control_profile",
        negative_control_expected_start_date=pd.Timestamp("2024-01-01").date(),
    )
    raw_tables = {
        "fall_events_source": pd.DataFrame(
            {
                "timestamp": ["2025-01-10T00:00:00Z"],
                "hospital_id": [5],
                "division_id": [21],
                "monitor_id": [1001],
                "patient_id": [111],
            }
        ),
        "hourly_location_aggregation": pd.DataFrame(),
        "key_dimensions": pd.DataFrame(),
        "negative_control_source_inventory": pd.DataFrame(
            {
                "hospital_id": [5, 5],
                "division_id": [21, 22],
                "monitor_id": [2001, 2002],
                "patient_id": [pd.NA, 333],
                "chunk_start_ts_utc": ["2025-01-05T10:55:05Z", "2025-01-06T10:55:05Z"],
                "chunk_end_ts_utc": ["2025-01-05T10:59:04Z", "2025-01-06T10:59:04Z"],
                "anchor_ts_utc": ["2025-01-05T10:59:05Z", "2025-01-06T10:59:05Z"],
                "frame_count": [300, 300],
                "duration_seconds": [299, 299],
            }
        ),
    }

    payload = json.loads(write_source_profile(settings, raw_tables=raw_tables).read_text(encoding="utf-8"))
    inventory = payload["negative_control_source_inventory"]

    assert payload["case_crossover_source_status"] == "ready"
    assert payload["negative_control_source_status"] == "ready"
    assert inventory["chunk_count"] == 2
    assert inventory["null_patient_id"] == 1
    assert inventory["exact_five_minute_chunks"] == 2
    assert inventory["history_status"] == "partial_history"
    assert inventory["monitor_overlap_with_fall_cohort"] == 0


def test_build_fall_events_clean_dedupes_on_event_timestamp_not_hour() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:05:00Z", "2025-01-01T00:25:00Z"],
            "timestamp_local": ["2024-12-31T18:05:00Z", "2024-12-31T18:25:00Z"],
            "hospital_id": [5, 5],
            "division_id": [1, 1],
            "patient_id": [1001, 1001],
            "monitor_id": [101, 101],
            "summary": ["same", "same"],
        }
    )
    cleaned, metrics = build_fall_events_clean(raw)
    assert metrics["fall_events_after_dedupe"] == 2
    assert len(cleaned.index) == 2


def test_build_fall_events_clean_uses_hospital_timezone_fallback_for_local_time() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2025-01-01T12:30:00Z"],
            "timestamp_local": [pd.NA],
            "hospital_id": [5],
            "division_id": [1],
            "patient_id": [1001],
            "monitor_id": [101],
            "summary": ["fallback"],
        }
    )
    cleaned, _ = build_fall_events_clean(raw, hospital_timezone="America/Chicago")
    row = cleaned.iloc[0]
    assert str(row["event_ts_local"]) == "2025-01-01 06:30:00"
    assert row["event_local_timezone_source"] == "hospital_timezone"


def test_build_analysis_base_falls_back_to_monitor_hour_when_event_patient_missing() -> None:
    hourly = pd.DataFrame(
        {
            "hospital_id": [5],
            "division_id": [1],
            "monitor_id": [101],
            "patient_id": [1001],
            "cohort_type": ["intervention"],
            "hour_ts": ["2025-01-01T06:00:00Z"],
            "pct_chair": [1.0],
            "pct_bed": [0.0],
            "pct_ambulatory": [0.0],
            "pct_not_located": [0.0],
            "row_valid_pct": [True],
        }
    )
    events = pd.DataFrame(
        {
            "hospital_id": [5],
            "division_id": [1],
            "monitor_id": [101],
            "patient_id": [pd.NA],
            "cohort_type": ["intervention"],
            "event_ts_local": ["2025-01-01 06:30:00"],
            "event_ts_utc": [pd.NA],
        }
    )
    eligibility = pd.DataFrame(
        {
            "hospital_id": [5],
            "division_id": [1],
            "monitor_id": [101],
            "patient_id": [1001],
            "cohort_type": ["intervention"],
            "eligible": [True],
        }
    )

    analysis_base = build_analysis_base(hourly, events, eligibility)

    assert len(analysis_base.index) == 1
    assert analysis_base.iloc[0]["falls_in_hour"] == 1
    assert bool(analysis_base.iloc[0]["fall_event_flag"]) is True
