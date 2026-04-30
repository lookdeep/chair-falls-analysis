# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.cohort_map import load_and_validate_cohort_map
from ld_chair_falls.config import Settings


def _settings() -> Settings:
    return Settings(project_root=PROJECT_ROOT, run_id="test_cohort_derivation")


def test_derives_cohort_map_from_raw_tables_when_file_not_provided() -> None:
    settings = _settings()
    raw_tables = {
        "fall_events_source": pd.DataFrame({"monitor_id": [11, 22, 22]}),
        "hourly_location_aggregation": pd.DataFrame({"monitor_id": [22, 33, 33, 44]}),
    }

    cohort_map, report = load_and_validate_cohort_map(settings, raw_tables=raw_tables)
    mapping = dict(zip(cohort_map["monitor_id"].tolist(), cohort_map["cohort_type"].tolist(), strict=True))

    assert report["status"] == "valid"
    assert report["derived"] is True
    assert report["source"] == "derived_from_raw_tables"
    assert report["derivation"]["cohort_basis"] == "outcome_defined_monitor_membership"
    assert mapping[11] == "intervention"
    assert mapping[22] == "intervention"
    assert mapping[33] == "control"
    assert mapping[44] == "control"


def test_missing_when_no_file_and_no_raw_tables() -> None:
    settings = _settings()
    cohort_map, report = load_and_validate_cohort_map(settings, raw_tables=None)

    assert cohort_map.empty
    assert report["status"] == "missing"


def test_applies_manual_intervention_override_for_monitor_2834() -> None:
    settings = Settings(
        project_root=PROJECT_ROOT,
        run_id="test_cohort_derivation",
        manual_intervention_monitor_ids=(2834,),
    )
    raw_tables = {
        "fall_events_source": pd.DataFrame({"monitor_id": [11, 22]}),
        "hourly_location_aggregation": pd.DataFrame({"monitor_id": [22, 33]}),
    }

    cohort_map, report = load_and_validate_cohort_map(settings, raw_tables=raw_tables)
    mapping = dict(zip(cohort_map["monitor_id"].tolist(), cohort_map["cohort_type"].tolist(), strict=True))

    assert mapping[2834] == "intervention"
    assert report["derivation"]["manual_intervention_monitor_ids"] == [2834]
