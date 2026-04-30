# ruff: noqa: E402, I001
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.config import Settings
from ld_chair_falls.operational_study import (
    OPERATIONAL_EVENT_LABEL_SQL_PATH,
    OPERATIONAL_SESSION_SQL_PATH,
    OPERATIONAL_MINUTE_SQL_PATH,
    OPERATIONAL_SUMMARY_SQL_PATH,
    build_operational_event_label_export_sql,
    build_operational_event_label_sql,
    build_operational_session_export_sql,
    build_operational_session_sql,
    build_operational_minute_sql,
    build_operational_study_artifacts,
    build_operational_summary_sql,
    sanitize_study_id,
    write_operational_study_metadata,
    write_operational_study_sql,
)


def test_sanitize_study_id_and_artifact_paths(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, run_id="render_case", dry_run=True)

    assert sanitize_study_id("study-case/1") == "study_case_1"

    artifacts = build_operational_study_artifacts(settings, "study-case/1")

    assert artifacts.table_suffix == "study_case_1"
    assert artifacts.session_table_fqn.endswith(".oneoff_operational_session_audit_study_case_1")
    assert artifacts.minute_table_fqn.endswith(".oneoff_operational_minute_study_case_1")
    assert artifacts.event_label_table_fqn.endswith(".oneoff_operational_event_labels_study_case_1")
    assert artifacts.summary_table_fqn.endswith(".oneoff_operational_position_rates_study_case_1")
    assert (
        artifacts.session_sql_path
        == tmp_path / "outputs/qa/oneoff_operational_session_audit_study_case_1.sql"
    )
    assert artifacts.minute_sql_path == tmp_path / "outputs/qa/oneoff_operational_minute_study_case_1.sql"
    assert (
        artifacts.event_label_sql_path
        == tmp_path / "outputs/qa/oneoff_operational_event_labels_study_case_1.sql"
    )
    assert (
        artifacts.session_csv_path
        == tmp_path / "outputs/qa/oneoff_operational_session_audit_study_case_1.csv"
    )
    assert (
        artifacts.event_labels_csv_path
        == tmp_path / "outputs/qa/oneoff_operational_event_labels_study_case_1.csv"
    )
    assert artifacts.summary_csv_path == tmp_path / "outputs/qa/oneoff_operational_position_rates_study_case_1.csv"


def test_build_operational_sql_renders_expected_sources() -> None:
    settings = Settings(
        project_root=PROJECT_ROOT,
        run_id="render_case",
        dry_run=True,
        manual_intervention_monitor_ids=(2834, 3001),
    )

    session_sql = build_operational_session_sql(settings, "study-case/1")
    minute_sql = build_operational_minute_sql(settings, "study-case/1")
    event_label_sql = build_operational_event_label_sql(settings, "study-case/1")
    summary_sql = build_operational_summary_sql(settings, "study-case/1")

    assert "{{" not in session_sql
    assert "{{" not in minute_sql
    assert "{{" not in event_label_sql
    assert "{{" not in summary_sql
    assert "hospital_api.audit_logs" in session_sql
    assert "hospital_api.nudge_history" in minute_sql
    assert "hospital_api.stat_alarms" in event_label_sql
    assert "hospital_api.audit_logs" in event_label_sql
    assert "live_stream_falls_derivatives_cache" in minute_sql
    assert "live_stream_falls_derivatives_cache" in event_label_sql
    assert "oneoff_operational_session_audit_study_case_1" in session_sql
    assert "oneoff_operational_minute_study_case_1" in minute_sql
    assert "oneoff_operational_event_labels_study_case_1" in event_label_sql
    assert "oneoff_operational_position_rates_study_case_1" in summary_sql
    assert 'LIKE "%talk_clicked%"' in session_sql
    assert 'LIKE "%talk_clicked%"' in event_label_sql
    assert "talk_clicked_events >= 1 AS meets_talk_gate" in session_sql
    assert "window_half_width_seconds" in event_label_sql
    assert "safety_zone_onset" in event_label_sql
    assert 'COUNTIF(dominant_location_label = "missing_posture") AS nudge_active_seconds_missing_posture' in minute_sql
    assert "Talk clicks with +/- %d second state association per %s exposure-hours." in summary_sql
    assert "Active nudge-state seconds per %s exposure-hour." in summary_sql

    session_export_sql = build_operational_session_export_sql(
        "ld-restricted.chair_falls_analysis.oneoff_operational_session_audit_study_case_1"
    )
    event_export_sql = build_operational_event_label_export_sql(
        "ld-restricted.chair_falls_analysis.oneoff_operational_event_labels_study_case_1"
    )
    assert "period_prepost_2025_06_01" in session_export_sql
    assert "source_event_id" in event_export_sql
    assert "window_half_width_seconds" in event_export_sql


def test_write_operational_sql_and_metadata(tmp_path: Path) -> None:
    session_template = (PROJECT_ROOT / OPERATIONAL_SESSION_SQL_PATH).read_text(encoding="utf-8")
    minute_template = (PROJECT_ROOT / OPERATIONAL_MINUTE_SQL_PATH).read_text(encoding="utf-8")
    event_label_template = (PROJECT_ROOT / OPERATIONAL_EVENT_LABEL_SQL_PATH).read_text(
        encoding="utf-8"
    )
    summary_template = (PROJECT_ROOT / OPERATIONAL_SUMMARY_SQL_PATH).read_text(encoding="utf-8")
    session_template_path = tmp_path / OPERATIONAL_SESSION_SQL_PATH
    minute_template_path = tmp_path / OPERATIONAL_MINUTE_SQL_PATH
    event_label_template_path = tmp_path / OPERATIONAL_EVENT_LABEL_SQL_PATH
    summary_template_path = tmp_path / OPERATIONAL_SUMMARY_SQL_PATH
    session_template_path.parent.mkdir(parents=True, exist_ok=True)
    minute_template_path.parent.mkdir(parents=True, exist_ok=True)
    event_label_template_path.parent.mkdir(parents=True, exist_ok=True)
    summary_template_path.parent.mkdir(parents=True, exist_ok=True)
    session_template_path.write_text(session_template, encoding="utf-8")
    minute_template_path.write_text(minute_template, encoding="utf-8")
    event_label_template_path.write_text(event_label_template, encoding="utf-8")
    summary_template_path.write_text(summary_template, encoding="utf-8")

    settings = Settings(project_root=tmp_path, run_id="study_run", dry_run=True)
    artifacts, session_sql, minute_sql, event_label_sql, summary_sql = write_operational_study_sql(
        settings, "study-run"
    )

    assert artifacts.session_sql_path.exists()
    assert artifacts.minute_sql_path.exists()
    assert artifacts.event_label_sql_path.exists()
    assert artifacts.summary_sql_path.exists()
    assert artifacts.session_sql_path.read_text(encoding="utf-8") == session_sql
    assert artifacts.minute_sql_path.read_text(encoding="utf-8") == minute_sql
    assert artifacts.event_label_sql_path.read_text(encoding="utf-8") == event_label_sql
    assert artifacts.summary_sql_path.read_text(encoding="utf-8") == summary_sql

    metadata_path = write_operational_study_metadata(
        settings=settings,
        artifacts=artifacts,
        dry_run=True,
        query_log_path=tmp_path / "outputs/manifests/query_log_study_run.json",
    )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert payload["settings"]["min_observed_hours"] == 4
    assert payload["settings"]["primary_event_window_half_width_seconds"] == 3
    assert payload["artifacts"]["minute_table_fqn"].endswith(".oneoff_operational_minute_study_run")
    assert payload["artifacts"]["event_label_table_fqn"].endswith(
        ".oneoff_operational_event_labels_study_run"
    )
    assert payload["artifacts"]["query_log_path"].endswith("query_log_study_run.json")
