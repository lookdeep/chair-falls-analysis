#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Build alarm-video context sidecars for consensus annotations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.alarm_context import (
    DETAIL_COLUMNS,
    build_nearby_alarm_query,
    build_source_coverage_sql,
    load_consensus_alarm_video_base,
    normalize_alarm_matches,
    summarize_alarm_video_context,
)
from ld_chair_falls.bq_client import BigQueryClientWrapper
from ld_chair_falls.config import Settings
from ld_chair_falls.utils import default_run_id, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build alarm-video context sidecars for consensus annotations."
    )
    parser.add_argument(
        "--run-id",
        default=default_run_id(),
        help="Run identifier used for the QA details artifact filename.",
    )
    parser.add_argument(
        "--consensus-path",
        default="data/public/falls-observations-v3-consensus.csv",
        help="Consensus CSV path relative to project root.",
    )
    parser.add_argument(
        "--output-path",
        default="docs/falls-observations-v3 - alarm-video-context.csv",
        help="Summary sidecar CSV path relative to project root.",
    )
    parser.add_argument(
        "--details-output-path",
        default=None,
        help="Detailed alarm-match QA CSV path relative to project root.",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Path to project root (default: parent of scripts/).",
    )
    return parser.parse_args()


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    consensus_path = _resolve_path(project_root, args.consensus_path)
    output_path = _resolve_path(project_root, args.output_path)
    details_output_path = _resolve_path(
        project_root,
        args.details_output_path or f"outputs/qa/consensus_alarm_video_matches_{args.run_id}.csv",
    )

    if not consensus_path.exists():
        print(f"ERROR: consensus CSV not found at {consensus_path}", file=sys.stderr)
        sys.exit(1)

    base, summary = load_consensus_alarm_video_base(consensus_path, hospital_timezone="America/Chicago")
    settings = Settings(project_root=project_root, run_id=args.run_id, dry_run=False)
    client = BigQueryClientWrapper(settings)

    source_range = client.query_to_dataframe(
        name="consensus_alarm_video_source_range",
        sql=build_source_coverage_sql(
            project_id=settings.google_cloud_project,
            hospital_id=settings.study_hospital_id,
        ),
        dry_run_columns=["source_min_created_at", "source_max_created_at", "source_row_count"],
    )

    eligible = base.loc[base["monitor_id"].notna() & base["fall_ts_utc"].notna()].copy()
    if eligible.empty:
        raw_details = pd.DataFrame(columns=DETAIL_COLUMNS)
        details = raw_details.copy()
    else:
        raw_details = client.query_to_dataframe(
            name="consensus_alarm_video_matches",
            sql=build_nearby_alarm_query(
                project_id=settings.google_cloud_project,
                hospital_id=settings.study_hospital_id,
                events=eligible,
            ),
            dry_run_columns=[
                "source_row_number",
                "event_key",
                "event_instance_ordinal",
                "sequence_id",
                "monitor_id",
                "recording_id",
                "created_at_utc",
                "sec_from_fall",
                "recording_source",
                "recording_path",
                "recording_status",
                "analysis_status",
                "analysis_result",
            ],
        )
        details = normalize_alarm_matches(raw_details)

    summary_frame = summarize_alarm_video_context(base, raw_details, source_range)

    ensure_dir(output_path.parent)
    ensure_dir(details_output_path.parent)
    summary_frame.to_csv(output_path, index=False)
    details.to_csv(details_output_path, index=False)
    client.flush_query_log()

    matched_rows = int(pd.to_numeric(summary_frame["alarm_count_within_1m"], errors="coerce").fillna(0).gt(0).sum())
    explicit_rows = int(summary_frame["nearest_alarm_report_explicit_fall"].astype("boolean").fillna(False).sum())
    print(f"Loaded {summary['rows']} consensus rows from {consensus_path.name}")
    print(f"Wrote summary sidecar to: {output_path}")
    print(f"Wrote detailed QA matches to: {details_output_path}")
    print(f"Rows with >=1 alarm clip within 1 minute: {matched_rows}")
    print(f"Rows with nearest report explicitly describing fall: {explicit_rows}")
    print(f"run_id={args.run_id}")


if __name__ == "__main__":
    main()
