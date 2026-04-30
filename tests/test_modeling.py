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
from ld_chair_falls.dayparts import DAYPART_SENSITIVITY_LABELS
from ld_chair_falls.modeling import (
    _build_position_cells,
    _fit_rate_model,
    _load_consensus_furniture_origin,
    _slice_daypart_window,
    run_adjusted_rr_model,
)


def _analysis_base() -> pd.DataFrame:
    rows = []
    strata = [
        ("1", "window_06_08", "Monday", "2025Q1", "2025-01-01T06:00:00Z"),
        ("1", "window_12_14", "Tuesday", "2025Q1", "2025-01-02T12:00:00Z"),
        ("2", "window_21_23", "Wednesday", "2025Q2", "2025-01-03T21:00:00Z"),
        ("2", "window_09_11", "Thursday", "2025Q2", "2025-01-04T09:00:00Z"),
    ]
    for idx, (division_id, daypart, day_of_week, quarter, hour_ts) in enumerate(strata):
        rows.append(
            {
                "division_id": division_id,
                "daypart": daypart,
                "day_of_week": day_of_week,
                "calendar_quarter": quarter,
                "pct_chair": 0.6 if idx % 2 == 0 else 0.4,
                "pct_bed": 0.4 if idx % 2 == 0 else 0.6,
                "pct_ambulatory": 0.0,
                "falls_in_hour": 6 + idx,
                "hour_ts": pd.Timestamp(hour_ts),
            }
        )
    return pd.DataFrame(rows)


def test_build_position_cells_proportional_allocation() -> None:
    df = pd.DataFrame(
        [
            {
                "division_id": "1",
                "daypart": "window_06_08",
                "day_of_week": "Monday",
                "calendar_quarter": "2025Q1",
                "pct_chair": 0.75,
                "pct_bed": 0.25,
                "falls_in_hour": 4,
            }
        ]
    )
    cells = _build_position_cells(df)
    chair = float(cells.loc[cells["position"] == "chair", "fall_count"].iloc[0])
    bed = float(cells.loc[cells["position"] == "bed", "fall_count"].iloc[0])
    assert chair == 3.0
    assert bed == 1.0


def test_fit_rate_model_emits_dispersion_and_cluster_metadata() -> None:
    cells = _build_position_cells(_analysis_base())
    result = _fit_rate_model(cells, "primary_adjusted_clustered", se_type="cluster")
    assert result["model_family"] == "poisson"
    assert result["se_type"] == "cluster"
    assert result["cluster_variable"] == "division_id"
    if not result["model_notes"]:
        assert result["deviance_df_ratio"] is not None
        assert result["pearson_df_ratio"] is not None
    else:
        assert str(result["model_notes"]).startswith("model_error:")


def test_canonical_time_slices() -> None:
    df = pd.DataFrame(
        {
            "hour_ts": pd.to_datetime(
                [
                    "2025-01-01T05:00:00Z",
                    "2025-01-01T06:00:00Z",
                    "2025-01-01T09:00:00Z",
                    "2025-01-01T12:00:00Z",
                    "2025-01-01T15:00:00Z",
                    "2025-01-01T18:00:00Z",
                    "2025-01-01T21:00:00Z",
                    "2025-01-01T23:00:00Z",
                ],
                utc=True,
            )
        }
    )
    assert len(_slice_daypart_window(df, "window_00_05").index) == 1
    assert len(_slice_daypart_window(df, "window_06_08").index) == 1
    assert len(_slice_daypart_window(df, "window_09_11").index) == 1
    assert len(_slice_daypart_window(df, "window_12_14").index) == 1
    assert len(_slice_daypart_window(df, "window_15_17").index) == 1
    assert len(_slice_daypart_window(df, "window_18_20").index) == 1
    assert len(_slice_daypart_window(df, "window_21_23").index) == 2
    assert pd.Timestamp("2025-01-01T06:00:00Z") in set(_slice_daypart_window(df, "window_06_08")["hour_ts"])
    assert pd.Timestamp("2025-01-01T21:00:00Z") in set(_slice_daypart_window(df, "window_21_23")["hour_ts"])


def test_run_adjusted_rr_model_writes_sensitivity_artifacts(tmp_path: Path) -> None:
    run_id = "modeling_test"
    settings = Settings(project_root=tmp_path, run_id=run_id)
    staged = settings.paths.staged_run_dir
    qa_dir = settings.paths.qa_dir
    staged.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    base = _analysis_base()
    base.to_parquet(staged / "prep.patient_hour_analysis_base_v1.parquet", index=False)

    hourly = base.copy()
    hourly["hospital_id"] = 5
    hourly["monitor_id"] = [100, 100, 200, 200]
    hourly["patient_id"] = [1000, 1000, 2000, 2000]
    hourly["cohort_type"] = "intervention"
    hourly["row_valid_pct"] = True
    hourly.to_parquet(staged / "prep.patient_hour_location_with_cohort_v1.parquet", index=False)

    events = pd.DataFrame(
        {
            "hospital_id": [5, 5, 5, 5],
            "division_id": [1, 1, 2, 2],
            "monitor_id": [100, 100, 200, 200],
            "patient_id": [1000, 1000, 2000, 2000],
            "cohort_type": ["intervention"] * 4,
            "event_hour_ts": pd.to_datetime(
                [
                    "2025-01-01T06:00:00Z",
                    "2025-01-02T07:00:00Z",
                    "2025-01-03T08:00:00Z",
                    "2025-01-04T09:00:00Z",
                ],
                utc=True,
            ),
        }
    )
    events.to_parquet(staged / "prep.fall_events_with_cohort_v1.parquet", index=False)

    adjusted_rr_path, covariates_path = run_adjusted_rr_model(settings)
    assert adjusted_rr_path.exists()
    assert covariates_path.exists()

    rr_df = pd.read_csv(adjusted_rr_path)
    assert "deviance_df_ratio" in rr_df.columns
    assert "se_type" in rr_df.columns
    assert "primary_adjusted_clustered" in rr_df["sensitivity_label"].tolist()
    for label in DAYPART_SENSITIVITY_LABELS.values():
        assert label in rr_df["sensitivity_label"].tolist()
    assert "position_certain_only" in rr_df["sensitivity_label"].tolist()
    assert "high_confidence_only" not in rr_df["sensitivity_label"].tolist()

    misclassification_path = qa_dir / f"chair_bed_misclassification_sensitivity_{run_id}.csv"
    threshold_path = qa_dir / f"chair_bed_threshold_sensitivity_{run_id}.csv"
    robustness_path = qa_dir / f"chair_bed_adjusted_rr_robustness_{run_id}.csv"
    assert misclassification_path.exists()
    assert threshold_path.exists()
    assert robustness_path.exists()
    threshold_df = pd.read_csv(threshold_path)
    assert set(threshold_df["threshold_hours"]) == {4, 12, 24, 48}


def test_load_consensus_furniture_origin_uses_included_fall_rows_only(tmp_path: Path) -> None:
    public_dir = tmp_path / "data" / "public"
    public_dir.mkdir(parents=True, exist_ok=True)
    consensus = pd.DataFrame(
        {
            "event_key": [
                "100|2025-01-01|10:00",
                "101|2025-01-01|11:00",
                "102|2025-01-01|12:00",
                "103|2025-01-01|13:00",
            ],
            "last_furniture": ["chair", "chair", "chair", "bed"],
            "furniture_departure_time": ["09:59:50", "", "", "12:59:55"],
            "prefall_location": ["room", "room", "room", "room"],
            "fall_time_consensus": ["10:00:05", "", "", "13:00:10"],
            "response_time_consensus": ["10:00:20", "", "", "13:00:20"],
            "fall_tags": ["support_table", "no_fall", "offscreen", "support_bed"],
        }
    )
    consensus_path = public_dir / "falls-observations-v3-consensus.csv"
    consensus.to_csv(consensus_path, index=False)

    settings = Settings(
        project_root=tmp_path,
        run_id="modeling_consensus_test",
        fall_labels_consensus_csv_path=Path("data/public/falls-observations-v3-consensus.csv"),
    )

    result = _load_consensus_furniture_origin(settings)

    assert result["event_key"].tolist() == ["100|2025-01-01|10:00"]
