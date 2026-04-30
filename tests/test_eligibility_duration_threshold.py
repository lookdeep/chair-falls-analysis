# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.transform import build_eligibility


def _hourly_unit(observed_hours: int) -> pd.DataFrame:
    base = pd.Timestamp("2025-01-01T00:00:00Z")
    return pd.DataFrame(
        {
            "hospital_id": [5] * observed_hours,
            "division_id": [11] * observed_hours,
            "monitor_id": [101] * observed_hours,
            "patient_id": [1001] * observed_hours,
            "cohort_type": ["intervention"] * observed_hours,
            "hour_ts": [base + pd.Timedelta(hours=offset) for offset in range(observed_hours)],
            "row_valid_pct": [True] * observed_hours,
        }
    )


def test_build_eligibility_default_threshold_allows_four_hours() -> None:
    eligibility = build_eligibility(_hourly_unit(4))
    row = eligibility.iloc[0]

    assert bool(row["eligible"]) is True
    assert bool(row["duration_requirement_met"]) is True
    assert row["min_observed_hours_threshold"] == 4
    assert row["exclusion_reason"] == ""
    assert "duration_gte_48h" not in eligibility.columns


def test_build_eligibility_default_threshold_rejects_below_four_hours() -> None:
    eligibility = build_eligibility(_hourly_unit(3))
    row = eligibility.iloc[0]

    assert bool(row["eligible"]) is False
    assert bool(row["duration_requirement_met"]) is False
    assert row["exclusion_reason"] == "duration_lt_min_hours"


def test_build_eligibility_respects_configured_threshold_override() -> None:
    eligibility = build_eligibility(_hourly_unit(5), min_observed_hours=6)
    row = eligibility.iloc[0]

    assert bool(row["eligible"]) is False
    assert bool(row["duration_requirement_met"]) is False
    assert row["min_observed_hours_threshold"] == 6
    assert row["exclusion_reason"] == "duration_lt_min_hours"
