from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .extract import _read_sql, _render_fall_events_compatible_subquery
from .transform import MIN_COVERAGE_RATIO, MIN_VALID_PCT_RATIO
from .utils import ensure_dir, write_json

OPERATIONAL_SESSION_SQL_PATH = Path("sql/04_oneoff/operational_session_audit.sql")
OPERATIONAL_MINUTE_SQL_PATH = Path("sql/04_oneoff/operational_minute_study.sql")
OPERATIONAL_EVENT_LABEL_SQL_PATH = Path("sql/04_oneoff/operational_event_labels.sql")
OPERATIONAL_SUMMARY_SQL_PATH = Path("sql/04_oneoff/operational_position_rates.sql")

PRIMARY_EVENT_WINDOW_HALF_WIDTH_SECONDS = 3
SENSITIVITY_EVENT_WINDOW_HALF_WIDTH_SECONDS = 5
TALK_GATE_PIVOT_DATE = "2025-06-01"

SESSION_AUDIT_EXPORT_COLUMNS = [
    "hospital_id",
    "division_id",
    "monitor_id",
    "patient_id",
    "session_start_dt_local",
    "session_end_exclusive_dt_local",
    "observed_hours",
    "expected_hours",
    "coverage_ratio",
    "valid_pct_ratio",
    "talk_clicked_events",
    "meets_observed_hours_gate",
    "meets_coverage_gate",
    "meets_valid_pct_gate",
    "meets_talk_gate",
    "retained_for_sitter_analysis",
    "period_prepost_2025_06_01",
]

EVENT_LABEL_EXPORT_COLUMNS = [
    "metric",
    "source_event_id",
    "window_half_width_seconds",
    "hospital_id",
    "division_id",
    "monitor_id",
    "patient_id",
    "event_ts_utc",
    "event_dt_local",
    "event_scope_status",
    "assigned_location_label",
    "event_second_location_label",
    "nearest_non_missing_location_label",
    "window_non_missing_seconds",
    "window_missing_seconds",
    "max_non_missing_vote",
    "tied_top_label_count",
    "used_event_second_tiebreaker",
    "used_nearest_second_tiebreaker",
]

SUMMARY_EXPORT_COLUMNS = [
    "scope",
    "metric",
    "metric_type",
    "metric_unit",
    "window_half_width_seconds",
    "position",
    "exposure_hours",
    "source_value_in_scope",
    "positioned_numerator_value",
    "excluded_non_target_position_value",
    "excluded_missing_posture_value",
    "excluded_ambiguous_events",
    "rate_per_100_exposure_hours",
    "active_seconds_per_exposure_hour",
    "notes",
]


@dataclass(frozen=True)
class OperationalStudyArtifacts:
    study_id: str
    table_suffix: str
    session_table_fqn: str
    minute_table_fqn: str
    event_label_table_fqn: str
    summary_table_fqn: str
    session_sql_path: Path
    minute_sql_path: Path
    event_label_sql_path: Path
    summary_sql_path: Path
    session_csv_path: Path
    event_labels_csv_path: Path
    summary_csv_path: Path
    metadata_path: Path


def sanitize_study_id(study_id: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_]+", "_", study_id).strip("_")
    return token or "study"


def build_operational_study_artifacts(settings: Settings, study_id: str) -> OperationalStudyArtifacts:
    suffix = sanitize_study_id(study_id)
    qa_dir = ensure_dir(settings.paths.qa_dir)
    return OperationalStudyArtifacts(
        study_id=study_id,
        table_suffix=suffix,
        session_table_fqn=(
            f"{settings.google_cloud_project}.{settings.bq_dataset}.oneoff_operational_session_audit_{suffix}"
        ),
        minute_table_fqn=(
            f"{settings.google_cloud_project}.{settings.bq_dataset}.oneoff_operational_minute_{suffix}"
        ),
        event_label_table_fqn=(
            f"{settings.google_cloud_project}.{settings.bq_dataset}.oneoff_operational_event_labels_{suffix}"
        ),
        summary_table_fqn=(
            f"{settings.google_cloud_project}.{settings.bq_dataset}.oneoff_operational_position_rates_{suffix}"
        ),
        session_sql_path=qa_dir / f"oneoff_operational_session_audit_{suffix}.sql",
        minute_sql_path=qa_dir / f"oneoff_operational_minute_{suffix}.sql",
        event_label_sql_path=qa_dir / f"oneoff_operational_event_labels_{suffix}.sql",
        summary_sql_path=qa_dir / f"oneoff_operational_position_rates_{suffix}.sql",
        session_csv_path=qa_dir / f"oneoff_operational_session_audit_{suffix}.csv",
        event_labels_csv_path=qa_dir / f"oneoff_operational_event_labels_{suffix}.csv",
        summary_csv_path=qa_dir / f"oneoff_operational_position_rates_{suffix}.csv",
        metadata_path=qa_dir / f"oneoff_operational_position_study_{suffix}.json",
    )


def _manual_monitor_sql(settings: Settings) -> str:
    if not settings.manual_intervention_monitor_ids:
        return "CAST(NULL AS INT64)"
    return ", ".join(str(int(value)) for value in settings.manual_intervention_monitor_ids)


def _render_template(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def _base_replacements(settings: Settings) -> dict[str, str]:
    return {
        "{{project}}": settings.google_cloud_project,
        "{{dataset}}": settings.bq_dataset,
        "{{study_hospital_id}}": str(settings.study_hospital_id),
        "{{hospital_timezone}}": settings.hospital_timezone,
        "{{study_start_date}}": settings.study_start_date.isoformat(),
        "{{study_end_date}}": settings.study_end_date.isoformat(),
        "{{min_observed_hours}}": str(settings.min_observed_hours),
        "{{min_coverage_ratio}}": str(MIN_COVERAGE_RATIO),
        "{{min_valid_pct_ratio}}": str(MIN_VALID_PCT_RATIO),
        "{{primary_event_window_half_width_seconds}}": str(PRIMARY_EVENT_WINDOW_HALF_WIDTH_SECONDS),
        "{{sensitivity_event_window_half_width_seconds}}": str(
            SENSITIVITY_EVENT_WINDOW_HALF_WIDTH_SECONDS
        ),
        "{{talk_gate_pivot_date}}": TALK_GATE_PIVOT_DATE,
        "{{prefall_panel_safety_zone_threshold}}": str(
            settings.prefall_panel_safety_zone_threshold
        ),
        "{{live_stream_derivatives_table}}": settings.live_stream_derivatives_table,
        "{{manual_intervention_monitor_ids_sql}}": _manual_monitor_sql(settings),
        "{{fall_events_compatible_subquery_sql}}": _render_fall_events_compatible_subquery(settings),
    }


def build_operational_session_sql(settings: Settings, study_id: str) -> str:
    artifacts = build_operational_study_artifacts(settings, study_id)
    template = _read_sql(settings.project_root / OPERATIONAL_SESSION_SQL_PATH)
    replacements = _base_replacements(settings)
    replacements.update({"{{session_table}}": artifacts.session_table_fqn})
    return _render_template(template, replacements)


def build_operational_minute_sql(settings: Settings, study_id: str) -> str:
    artifacts = build_operational_study_artifacts(settings, study_id)
    template = _read_sql(settings.project_root / OPERATIONAL_MINUTE_SQL_PATH)
    replacements = _base_replacements(settings)
    replacements.update(
        {
            "{{session_table}}": artifacts.session_table_fqn,
            "{{minute_table}}": artifacts.minute_table_fqn,
        }
    )
    return _render_template(template, replacements)


def build_operational_event_label_sql(settings: Settings, study_id: str) -> str:
    artifacts = build_operational_study_artifacts(settings, study_id)
    template = _read_sql(settings.project_root / OPERATIONAL_EVENT_LABEL_SQL_PATH)
    replacements = _base_replacements(settings)
    replacements.update(
        {
            "{{session_table}}": artifacts.session_table_fqn,
            "{{event_label_table}}": artifacts.event_label_table_fqn,
        }
    )
    return _render_template(template, replacements)


def build_operational_summary_sql(settings: Settings, study_id: str) -> str:
    artifacts = build_operational_study_artifacts(settings, study_id)
    template = _read_sql(settings.project_root / OPERATIONAL_SUMMARY_SQL_PATH)
    replacements = _base_replacements(settings)
    replacements.update(
        {
            "{{minute_table}}": artifacts.minute_table_fqn,
            "{{event_label_table}}": artifacts.event_label_table_fqn,
            "{{summary_table}}": artifacts.summary_table_fqn,
        }
    )
    return _render_template(template, replacements)


def build_operational_session_export_sql(session_table_fqn: str) -> str:
    return f"""
SELECT
  hospital_id,
  division_id,
  monitor_id,
  patient_id,
  session_start_dt_local,
  session_end_exclusive_dt_local,
  observed_hours,
  expected_hours,
  coverage_ratio,
  valid_pct_ratio,
  talk_clicked_events,
  meets_observed_hours_gate,
  meets_coverage_gate,
  meets_valid_pct_gate,
  meets_talk_gate,
  retained_for_sitter_analysis,
  period_prepost_2025_06_01
FROM `{session_table_fqn}`
ORDER BY
  retained_for_sitter_analysis DESC,
  division_id,
  monitor_id,
  patient_id,
  session_start_dt_local
""".strip()


def build_operational_event_label_export_sql(event_label_table_fqn: str) -> str:
    return f"""
SELECT
  metric,
  source_event_id,
  window_half_width_seconds,
  hospital_id,
  division_id,
  monitor_id,
  patient_id,
  event_ts_utc,
  event_dt_local,
  event_scope_status,
  assigned_location_label,
  event_second_location_label,
  nearest_non_missing_location_label,
  window_non_missing_seconds,
  window_missing_seconds,
  max_non_missing_vote,
  tied_top_label_count,
  used_event_second_tiebreaker,
  used_nearest_second_tiebreaker
FROM `{event_label_table_fqn}`
ORDER BY
  CASE metric
    WHEN "talk_event" THEN 1
    WHEN "alarm_trigger" THEN 2
    WHEN "safety_zone_onset" THEN 3
    ELSE 99
  END,
  event_ts_utc,
  source_event_id,
  window_half_width_seconds
""".strip()


def build_operational_summary_export_sql(summary_table_fqn: str) -> str:
    return f"""
SELECT
  scope,
  metric,
  metric_type,
  metric_unit,
  window_half_width_seconds,
  position,
  exposure_hours,
  source_value_in_scope,
  positioned_numerator_value,
  excluded_non_target_position_value,
  excluded_missing_posture_value,
  excluded_ambiguous_events,
  rate_per_100_exposure_hours,
  active_seconds_per_exposure_hour,
  notes
FROM `{summary_table_fqn}`
ORDER BY
  CASE metric
    WHEN "talk_event" THEN 1
    WHEN "alarm_trigger" THEN 2
    WHEN "safety_zone_onset" THEN 3
    WHEN "nudge_active_seconds" THEN 4
    ELSE 99
  END,
  window_half_width_seconds,
  CASE position
    WHEN "chair" THEN 1
    WHEN "bed" THEN 2
    WHEN "room" THEN 3
    WHEN "no_patient" THEN 4
    ELSE 99
  END
""".strip()


def write_operational_study_sql(
    settings: Settings,
    study_id: str,
) -> tuple[OperationalStudyArtifacts, str, str, str, str]:
    artifacts = build_operational_study_artifacts(settings, study_id)
    session_sql = build_operational_session_sql(settings, study_id)
    minute_sql = build_operational_minute_sql(settings, study_id)
    event_label_sql = build_operational_event_label_sql(settings, study_id)
    summary_sql = build_operational_summary_sql(settings, study_id)
    artifacts.session_sql_path.write_text(session_sql, encoding="utf-8")
    artifacts.minute_sql_path.write_text(minute_sql, encoding="utf-8")
    artifacts.event_label_sql_path.write_text(event_label_sql, encoding="utf-8")
    artifacts.summary_sql_path.write_text(summary_sql, encoding="utf-8")
    return artifacts, session_sql, minute_sql, event_label_sql, summary_sql


def write_operational_study_metadata(
    *,
    settings: Settings,
    artifacts: OperationalStudyArtifacts,
    dry_run: bool,
    query_log_path: Path,
) -> Path:
    artifact_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(artifacts).items()
    }
    payload: dict[str, Any] = {
        "settings": {
            "run_id": settings.run_id,
            "study_hospital_id": settings.study_hospital_id,
            "google_cloud_project": settings.google_cloud_project,
            "bq_dataset": settings.bq_dataset,
            "hospital_timezone": settings.hospital_timezone,
            "study_start_date": settings.study_start_date.isoformat(),
            "study_end_date": settings.study_end_date.isoformat(),
            "min_observed_hours": settings.min_observed_hours,
            "min_coverage_ratio": MIN_COVERAGE_RATIO,
            "min_valid_pct_ratio": MIN_VALID_PCT_RATIO,
            "primary_event_window_half_width_seconds": PRIMARY_EVENT_WINDOW_HALF_WIDTH_SECONDS,
            "sensitivity_event_window_half_width_seconds": (
                SENSITIVITY_EVENT_WINDOW_HALF_WIDTH_SECONDS
            ),
            "talk_gate_pivot_date": TALK_GATE_PIVOT_DATE,
            "prefall_panel_safety_zone_threshold": settings.prefall_panel_safety_zone_threshold,
            "manual_intervention_monitor_ids": list(settings.manual_intervention_monitor_ids),
        },
        "artifacts": artifact_payload | {"query_log_path": str(query_log_path)},
        "dry_run": dry_run,
    }
    return write_json(artifacts.metadata_path, payload)
