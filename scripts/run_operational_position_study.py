#!/usr/bin/env python3
# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.bq_client import BigQueryClientWrapper
from ld_chair_falls.config import load_settings
from ld_chair_falls.operational_study import (
    EVENT_LABEL_EXPORT_COLUMNS,
    SESSION_AUDIT_EXPORT_COLUMNS,
    SUMMARY_EXPORT_COLUMNS,
    build_operational_event_label_export_sql,
    build_operational_session_export_sql,
    build_operational_summary_export_sql,
    write_operational_study_metadata,
    write_operational_study_sql,
)
from ld_chair_falls.utils import default_run_id, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the one-off chair-vs-bed operational position study in BigQuery."
    )
    parser.add_argument(
        "--study-id",
        default=default_run_id(),
        help="Stable suffix used for the minute/summary tables and local artifacts.",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Path to project root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="Override the BigQuery dataset used for one-off tables.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV export path relative to project root.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When true, only render SQL and emit empty placeholder outputs locally.",
    )
    return parser.parse_args()


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root) if args.project_root else PROJECT_ROOT

    settings = load_settings(run_id=args.study_id, dry_run=args.dry_run)
    settings = settings.model_copy(update={"project_root": project_root})
    if args.output_dataset:
        settings = settings.model_copy(update={"bq_dataset": args.output_dataset})

    artifacts, session_sql, minute_sql, event_label_sql, summary_sql = write_operational_study_sql(
        settings, args.study_id
    )
    output_csv_path = (
        _resolve_path(project_root, args.output_csv)
        if args.output_csv
        else artifacts.summary_csv_path
    )
    ensure_dir(output_csv_path.parent)
    ensure_dir(artifacts.session_csv_path.parent)
    ensure_dir(artifacts.event_labels_csv_path.parent)

    client = BigQueryClientWrapper(settings)
    client.execute_query(
        name="oneoff_operational_session_audit_table",
        sql=session_sql,
        destination=artifacts.session_table_fqn,
    )
    client.execute_query(
        name="oneoff_operational_minute_table",
        sql=minute_sql,
        destination=artifacts.minute_table_fqn,
    )
    client.execute_query(
        name="oneoff_operational_event_labels_table",
        sql=event_label_sql,
        destination=artifacts.event_label_table_fqn,
    )
    client.execute_query(
        name="oneoff_operational_position_rates_table",
        sql=summary_sql,
        destination=artifacts.summary_table_fqn,
    )

    session_frame = client.query_to_dataframe(
        name="oneoff_operational_session_audit_export",
        sql=build_operational_session_export_sql(artifacts.session_table_fqn),
        dry_run_columns=SESSION_AUDIT_EXPORT_COLUMNS,
        destination=str(artifacts.session_csv_path),
    )
    session_frame.to_csv(artifacts.session_csv_path, index=False)

    event_labels_frame = client.query_to_dataframe(
        name="oneoff_operational_event_labels_export",
        sql=build_operational_event_label_export_sql(artifacts.event_label_table_fqn),
        dry_run_columns=EVENT_LABEL_EXPORT_COLUMNS,
        destination=str(artifacts.event_labels_csv_path),
    )
    event_labels_frame.to_csv(artifacts.event_labels_csv_path, index=False)

    summary_frame = client.query_to_dataframe(
        name="oneoff_operational_position_rates_export",
        sql=build_operational_summary_export_sql(artifacts.summary_table_fqn),
        dry_run_columns=SUMMARY_EXPORT_COLUMNS,
        destination=str(output_csv_path),
    )
    summary_frame.to_csv(output_csv_path, index=False)

    query_log_path = client.flush_query_log()
    metadata_path = write_operational_study_metadata(
        settings=settings,
        artifacts=artifacts,
        dry_run=settings.dry_run,
        query_log_path=query_log_path,
    )

    print(f"study_id={args.study_id}")
    print(f"dry_run={settings.dry_run}")
    print(f"session_table={artifacts.session_table_fqn}")
    print(f"minute_table={artifacts.minute_table_fqn}")
    print(f"event_label_table={artifacts.event_label_table_fqn}")
    print(f"summary_table={artifacts.summary_table_fqn}")
    print(f"session_sql={artifacts.session_sql_path}")
    print(f"minute_sql={artifacts.minute_sql_path}")
    print(f"event_label_sql={artifacts.event_label_sql_path}")
    print(f"summary_sql={artifacts.summary_sql_path}")
    print(f"session_csv={artifacts.session_csv_path}")
    print(f"event_labels_csv={artifacts.event_labels_csv_path}")
    print(f"summary_csv={output_csv_path}")
    print(f"query_log={query_log_path}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
