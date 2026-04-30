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

import run_operational_position_study as script
from ld_chair_falls.operational_study import OPERATIONAL_MINUTE_SQL_PATH, OPERATIONAL_SUMMARY_SQL_PATH


class FakeBigQueryClientWrapper:
    last_instance: FakeBigQueryClientWrapper | None = None

    def __init__(self, settings) -> None:
        self.settings = settings
        self.execute_calls: list[tuple[str, str]] = []
        self.query_calls: list[str] = []
        FakeBigQueryClientWrapper.last_instance = self

    def execute_query(self, *, name: str, sql: str, destination: str = "query_job") -> None:
        self.execute_calls.append((name, destination))
        assert "CREATE OR REPLACE TABLE" in sql

    def query_to_dataframe(self, *, name: str, sql: str, dry_run_columns=None, destination="memory"):
        self.query_calls.append(name)
        if name == "oneoff_operational_session_audit_export":
            assert "oneoff_operational_session_audit" in sql
            return pd.DataFrame(
                {
                    "hospital_id": [5],
                    "division_id": [37],
                    "monitor_id": [2001],
                    "patient_id": [3001],
                    "session_start_dt_local": ["2025-07-01 08:00:00"],
                    "session_end_exclusive_dt_local": ["2025-07-01 12:00:00"],
                    "observed_hours": [4],
                    "expected_hours": [4],
                    "coverage_ratio": [1.0],
                    "valid_pct_ratio": [1.0],
                    "talk_clicked_events": [2],
                    "meets_observed_hours_gate": [True],
                    "meets_coverage_gate": [True],
                    "meets_valid_pct_gate": [True],
                    "meets_talk_gate": [True],
                    "retained_for_sitter_analysis": [True],
                    "period_prepost_2025_06_01": ["post_2025_06_01"],
                }
            )
        if name == "oneoff_operational_event_labels_export":
            assert "oneoff_operational_event_labels" in sql
            return pd.DataFrame(
                {
                    "metric": ["talk_event"],
                    "source_event_id": ["evt-1"],
                    "window_half_width_seconds": [3],
                    "hospital_id": [5],
                    "division_id": [37],
                    "monitor_id": [2001],
                    "patient_id": [3001],
                    "event_ts_utc": ["2025-07-01T15:03:00+00:00"],
                    "event_dt_local": ["2025-07-01 10:03:00"],
                    "event_scope_status": ["resolved"],
                    "assigned_location_label": ["chair"],
                    "event_second_location_label": ["chair"],
                    "nearest_non_missing_location_label": ["chair"],
                    "window_non_missing_seconds": [7],
                    "window_missing_seconds": [0],
                    "max_non_missing_vote": [7],
                    "tied_top_label_count": [1],
                    "used_event_second_tiebreaker": [False],
                    "used_nearest_second_tiebreaker": [False],
                }
            )
        assert "oneoff_operational_position_rates" in sql
        return pd.DataFrame(
            {
                "scope": ["talk_confirmed_intervention_oneoff"],
                "metric": ["talk_event"],
                "metric_type": ["event_count"],
                "metric_unit": ["count"],
                "window_half_width_seconds": [3],
                "position": ["chair"],
                "exposure_hours": [12.0],
                "source_value_in_scope": [18.0],
                "positioned_numerator_value": [7.0],
                "excluded_non_target_position_value": [10.0],
                "excluded_missing_posture_value": [1.0],
                "excluded_ambiguous_events": [0],
                "rate_per_100_exposure_hours": [58.33],
                "active_seconds_per_exposure_hour": [pd.NA],
                "notes": ["Talk clicks with +/- 3 second state association per chair exposure-hours."],
            }
        )

    def flush_query_log(self) -> Path:
        path = self.settings.project_root / "outputs/manifests/query_log_study_run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")
        return path


def test_main_writes_sql_csv_and_metadata(tmp_path: Path, monkeypatch) -> None:
    session_template = (PROJECT_ROOT / "sql/04_oneoff/operational_session_audit.sql").read_text(
        encoding="utf-8"
    )
    minute_template = (PROJECT_ROOT / OPERATIONAL_MINUTE_SQL_PATH).read_text(encoding="utf-8")
    event_label_template = (PROJECT_ROOT / "sql/04_oneoff/operational_event_labels.sql").read_text(
        encoding="utf-8"
    )
    summary_template = (PROJECT_ROOT / OPERATIONAL_SUMMARY_SQL_PATH).read_text(encoding="utf-8")
    (tmp_path / "sql/04_oneoff/operational_session_audit.sql").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / OPERATIONAL_MINUTE_SQL_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "sql/04_oneoff/operational_event_labels.sql").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / OPERATIONAL_SUMMARY_SQL_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "sql/04_oneoff/operational_session_audit.sql").write_text(
        session_template,
        encoding="utf-8",
    )
    (tmp_path / OPERATIONAL_MINUTE_SQL_PATH).write_text(minute_template, encoding="utf-8")
    (tmp_path / "sql/04_oneoff/operational_event_labels.sql").write_text(
        event_label_template,
        encoding="utf-8",
    )
    (tmp_path / OPERATIONAL_SUMMARY_SQL_PATH).write_text(summary_template, encoding="utf-8")

    monkeypatch.setattr(script, "BigQueryClientWrapper", FakeBigQueryClientWrapper)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_operational_position_study.py",
            "--project-root",
            str(tmp_path),
            "--study-id",
            "study_run",
            "--no-dry-run",
        ],
    )

    script.main()

    client = FakeBigQueryClientWrapper.last_instance
    assert client is not None
    assert client.execute_calls == [
        (
            "oneoff_operational_session_audit_table",
            "ld-restricted.chair_falls_analysis.oneoff_operational_session_audit_study_run",
        ),
        (
            "oneoff_operational_minute_table",
            "ld-restricted.chair_falls_analysis.oneoff_operational_minute_study_run",
        ),
        (
            "oneoff_operational_event_labels_table",
            "ld-restricted.chair_falls_analysis.oneoff_operational_event_labels_study_run",
        ),
        (
            "oneoff_operational_position_rates_table",
            "ld-restricted.chair_falls_analysis.oneoff_operational_position_rates_study_run",
        ),
    ]
    assert client.query_calls == [
        "oneoff_operational_session_audit_export",
        "oneoff_operational_event_labels_export",
        "oneoff_operational_position_rates_export",
    ]

    session_sql = tmp_path / "outputs/qa/oneoff_operational_session_audit_study_run.sql"
    minute_sql = tmp_path / "outputs/qa/oneoff_operational_minute_study_run.sql"
    event_label_sql = tmp_path / "outputs/qa/oneoff_operational_event_labels_study_run.sql"
    summary_sql = tmp_path / "outputs/qa/oneoff_operational_position_rates_study_run.sql"
    session_csv = tmp_path / "outputs/qa/oneoff_operational_session_audit_study_run.csv"
    event_label_csv = tmp_path / "outputs/qa/oneoff_operational_event_labels_study_run.csv"
    summary_csv = tmp_path / "outputs/qa/oneoff_operational_position_rates_study_run.csv"
    metadata_path = tmp_path / "outputs/qa/oneoff_operational_position_study_study_run.json"

    assert session_sql.exists()
    assert minute_sql.exists()
    assert event_label_sql.exists()
    assert summary_sql.exists()
    assert session_csv.exists()
    assert event_label_csv.exists()
    assert summary_csv.exists()
    assert metadata_path.exists()

    written = pd.read_csv(summary_csv)
    assert written["metric"].tolist() == ["talk_event"]
    assert written["metric_type"].tolist() == ["event_count"]
    assert written["window_half_width_seconds"].tolist() == [3]
    assert written["position"].tolist() == ["chair"]

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["dry_run"] is False
    assert payload["artifacts"]["session_table_fqn"].endswith(
        ".oneoff_operational_session_audit_study_run"
    )
    assert payload["artifacts"]["summary_table_fqn"].endswith(".oneoff_operational_position_rates_study_run")
