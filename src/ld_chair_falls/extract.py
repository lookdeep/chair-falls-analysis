from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .bq_client import BigQueryClientWrapper
from .config import Settings
from .utils import ensure_dir, write_yaml


@dataclass(frozen=True)
class ExtractSpec:
    name: str
    sql_path_attr: str
    output_filename: str
    dry_columns: list[str]


EXTRACT_SPECS: tuple[ExtractSpec, ...] = (
    ExtractSpec(
        name="fall_events_source",
        sql_path_attr="fall_events_sql_path",
        output_filename="fall_events_raw.parquet",
        dry_columns=[
            "timestamp",
            "timestamp_local",
            "date_local",
            "time_local",
            "hospital_id",
            "hospital_system_name",
            "division_id",
            "hospital_name",
            "patient_id",
            "monitor_id",
            "user_name",
            "summary",
        ],
    ),
    ExtractSpec(
        name="hourly_location_aggregation",
        sql_path_attr="hourly_location_sql_path",
        output_filename="hourly_location_raw.parquet",
        dry_columns=[
            "hospital_id",
            "division_id",
            "monitor_id",
            "patient_id",
            "hour_ts",
            "pct_chair",
            "pct_bed",
            "pct_ambulatory",
            "pct_not_located",
            "num_alarms",
            "num_nudges",
            "num_announcements",
        ],
    ),
    ExtractSpec(
        name="key_dimensions",
        sql_path_attr="key_dimensions_sql_path",
        output_filename="key_dimensions_raw.parquet",
        dry_columns=["hospital_id", "division_id", "monitor_id", "patient_id"],
    ),
    ExtractSpec(
        name="fall_livestream_event_windows",
        sql_path_attr="fall_livestream_event_windows_sql_path",
        output_filename="fall_livestream_event_windows_raw.parquet",
        dry_columns=[
            "fall_event_id",
            "fall_ts_utc",
            "fall_ts_local",
            "hospital_id",
            "division_id",
            "patient_id",
            "monitor_id",
            "summary",
            "pre_window_frames",
            "post_window_frames",
            "pre_focus_frames",
            "pre_focus_visible_frames",
            "pre_focus_bin_count",
            "pre_focus_visible_bin_count",
            "pre_dropout_frames",
            "pre_dropout_visible_frames",
            "pre_dropout_bin_count",
            "pre_dropout_visible_bin_count",
            "pre_focus_min_patient_chair_distance",
            "pre_focus_min_patient_bed_distance",
            "pre_focus_min_patient_room_distance",
            "pre_dropout_last_visible_ts_utc",
            "pre_focus_weight_chair",
            "pre_focus_weight_bed",
            "pre_focus_weight_room",
            "first_staff_overlap_ts_utc",
            "post_staff_overlap_frames",
            "pre_distance_signal_count",
            "pre_dropout_visibility_ratio",
            "pre_dropout_last_visible_gap_seconds",
            "pre_dropout_gap_ratio",
            "pre_dropout_dropout_ratio",
            "pre_state_visible_prob",
            "pre_state_not_visible_prob",
            "pre_state_out_of_room_prob",
            "pre_state_unknown_prob",
            "pre_focus_weight_sum",
            "pre_cond_prob_chair",
            "pre_cond_prob_bed",
            "pre_cond_prob_room",
            "pre_out_of_room_share",
            "pre_missing_mass",
            "pre_prob_chair",
            "pre_prob_bed",
            "pre_prob_room",
            "pre_prob_no_patient",
            "pre_prob_not_visible",
            "pre_prob_out_of_room",
            "pre_prob_unknown",
            "prefall_location_label",
            "prefall_location_margin",
            "response_detected",
            "response_latency_seconds",
            "pre_posture_observed_frame_count",
            "pre_posture_sitting_score_mean",
            "pre_posture_standing_score_mean",
            "pre_posture_lying_score_mean",
            "pre_posture_sitting_share",
            "pre_posture_standing_share",
            "pre_posture_lying_share",
            "pre_posture_switch_count",
        ],
    ),
    ExtractSpec(
        name="fall_livestream_second_level",
        sql_path_attr="fall_livestream_second_level_sql_path",
        output_filename="fall_livestream_second_level_raw.parquet",
        dry_columns=[
            "fall_event_id",
            "fall_ts_utc",
            "fall_ts_local",
            "hospital_id",
            "division_id",
            "patient_id",
            "monitor_id",
            "frame_ts_utc",
            "second_offset",
            "window_phase",
            "frame_has_location_signal",
            "patient_chair_distance",
            "patient_bed_distance",
            "patient_room_distance",
            "patient_staff_iou",
            "dominant_location_label",
            "nudge_state",
            "nudge_score",
            "bed_in_bed_score",
            "motion_bac",
            "patient_candidate_count",
            "staff_candidate_count",
            "other_candidate_count",
            "bed_candidate_count",
            "chair_candidate_count",
            "primary_patient_posture_label",
            "primary_patient_posture_score_sitting",
            "primary_patient_posture_score_standing",
            "primary_patient_posture_score_lying",
            "patient_posture_candidate_count",
            "patient_posture_source",
        ],
    ),
    ExtractSpec(
        name="negative_control_source_inventory",
        sql_path_attr="negative_control_source_inventory_sql_path",
        output_filename="negative_control_source_inventory_raw.parquet",
        dry_columns=[
            "source_chunk_id",
            "hospital_id",
            "division_id",
            "monitor_id",
            "patient_id",
            "chunk_start_ts_utc",
            "chunk_end_ts_utc",
            "anchor_ts_utc",
            "frame_count",
            "duration_seconds",
            "anchor_hour_local",
            "anchor_daypart_local",
            "has_patient_id",
        ],
    ),
    ExtractSpec(
        name="fall_case_crossover_windows",
        sql_path_attr="fall_case_crossover_windows_sql_path",
        output_filename="fall_case_crossover_windows_raw.parquet",
        dry_columns=[
            "fall_event_id",
            "window_role",
            "anchor_ts_utc",
            "hospital_id",
            "division_id",
            "patient_id",
            "monitor_id",
            "frame_count",
            "visible_frame_count",
            "visibility_ratio",
            "chair_share",
            "bed_share",
            "room_share",
            "no_patient_share",
            "mean_patient_chair_distance",
            "mean_patient_bed_distance",
            "mean_patient_room_distance",
            "min_patient_chair_distance",
            "min_patient_bed_distance",
            "min_patient_room_distance",
            "mean_patient_staff_iou",
            "max_patient_staff_iou",
            "dominant_switch_count",
            "posture_observed_frame_count",
            "posture_sitting_score_mean",
            "posture_standing_score_mean",
            "posture_lying_score_mean",
            "posture_sitting_share",
            "posture_standing_share",
            "posture_lying_share",
            "posture_switch_count",
            "nearby_fall_count_30m",
            "eligible_window",
        ],
    ),
    ExtractSpec(
        name="fall_case_crossover_second_level",
        sql_path_attr="fall_case_crossover_second_level_sql_path",
        output_filename="fall_case_crossover_second_level_raw.parquet",
        dry_columns=[
            "fall_event_id",
            "window_role",
            "anchor_ts_utc",
            "fall_ts_utc",
            "fall_ts_local",
            "hospital_id",
            "division_id",
            "patient_id",
            "monitor_id",
            "frame_ts_utc",
            "second_offset",
            "window_phase",
            "frame_has_location_signal",
            "patient_chair_distance",
            "patient_bed_distance",
            "patient_room_distance",
            "patient_staff_iou",
            "dominant_location_label",
            "nudge_state",
            "nudge_score",
            "bed_in_bed_score",
            "motion_bac",
            "patient_candidate_count",
            "staff_candidate_count",
            "other_candidate_count",
            "bed_candidate_count",
            "chair_candidate_count",
            "primary_patient_posture_label",
            "primary_patient_posture_score_sitting",
            "primary_patient_posture_score_standing",
            "primary_patient_posture_score_lying",
            "patient_posture_candidate_count",
            "patient_posture_source",
        ],
    ),
    ExtractSpec(
        name="fall_negative_control_second_level",
        sql_path_attr="fall_negative_control_second_level_sql_path",
        output_filename="fall_negative_control_second_level_raw.parquet",
        dry_columns=[
            "fall_event_id",
            "window_role",
            "anchor_ts_utc",
            "hospital_id",
            "division_id",
            "patient_id",
            "monitor_id",
            "source_chunk_id",
            "source_chunk_start_ts_utc",
            "source_chunk_end_ts_utc",
            "source_monitor_id",
            "control_rank",
            "match_tier",
            "matching_candidates",
            "frame_ts_utc",
            "second_offset",
            "window_phase",
            "frame_has_location_signal",
            "patient_chair_distance",
            "patient_bed_distance",
            "patient_room_distance",
            "patient_staff_iou",
            "dominant_location_label",
            "nudge_state",
            "nudge_score",
            "bed_in_bed_score",
            "motion_bac",
            "patient_candidate_count",
            "staff_candidate_count",
            "other_candidate_count",
            "bed_candidate_count",
            "chair_candidate_count",
            "primary_patient_posture_label",
            "primary_patient_posture_score_sitting",
            "primary_patient_posture_score_standing",
            "primary_patient_posture_score_lying",
            "patient_posture_candidate_count",
            "patient_posture_source",
        ],
    ),
    ExtractSpec(
        name="fall_negative_control_windows",
        sql_path_attr="fall_negative_control_windows_sql_path",
        output_filename="fall_negative_control_windows_raw.parquet",
        dry_columns=[
            "fall_event_id",
            "window_role",
            "anchor_ts_utc",
            "hospital_id",
            "division_id",
            "patient_id",
            "monitor_id",
            "source_chunk_id",
            "source_chunk_start_ts_utc",
            "source_chunk_end_ts_utc",
            "source_monitor_id",
            "control_rank",
            "match_tier",
            "matching_candidates",
            "frame_count",
            "visible_frame_count",
            "visibility_ratio",
            "chair_share",
            "bed_share",
            "room_share",
            "no_patient_share",
            "mean_patient_chair_distance",
            "mean_patient_bed_distance",
            "mean_patient_room_distance",
            "mean_patient_staff_iou",
            "max_patient_staff_iou",
            "dominant_switch_count",
            "posture_observed_frame_count",
            "posture_sitting_score_mean",
            "posture_standing_score_mean",
            "posture_lying_score_mean",
            "posture_sitting_share",
            "posture_standing_share",
            "posture_lying_share",
            "posture_switch_count",
            "nearby_fall_count_30m",
            "eligible_window",
        ],
    ),
)

CASE_CROSSOVER_EXTRACT_NAMES = frozenset(
    {
        "fall_case_crossover_windows",
        "fall_case_crossover_second_level",
    }
)

NEGATIVE_CONTROL_EXTRACT_NAMES = frozenset(
    {
        "negative_control_source_inventory",
        "fall_negative_control_second_level",
        "fall_negative_control_windows",
    }
)
CONTROL_EXTRACT_NAMES = CASE_CROSSOVER_EXTRACT_NAMES | NEGATIVE_CONTROL_EXTRACT_NAMES


def _extract_enabled(name: str, settings: Settings) -> bool:
    if name in CASE_CROSSOVER_EXTRACT_NAMES:
        return settings.case_crossover_source_ready
    if name in NEGATIVE_CONTROL_EXTRACT_NAMES:
        return settings.negative_control_source_ready
    return True


def _extract_skip_reason(name: str, settings: Settings) -> str:
    if name in CASE_CROSSOVER_EXTRACT_NAMES:
        return f"case_crossover_source_status={settings.case_crossover_source_status}"
    if name in NEGATIVE_CONTROL_EXTRACT_NAMES:
        return f"negative_control_source_status={settings.negative_control_source_status}"
    return "enabled"


def _case_crossover_anchor_days(settings: Settings) -> tuple[int, ...]:
    days: list[int] = []
    for day in settings.fall_case_crossover_anchor_days:
        if day >= 0:
            raise ValueError("fall_case_crossover_anchor_days must contain negative day offsets only")
        if day not in days:
            days.append(day)
    if not days:
        raise ValueError("fall_case_crossover_anchor_days must contain at least one control offset")
    return tuple(days)


def _case_crossover_control_role(day: int) -> str:
    return f"control_{abs(day)}d"


def _render_case_crossover_control_unions(settings: Settings, *, include_fall_ts: bool) -> str:
    blocks: list[str] = []
    for day in _case_crossover_anchor_days(settings):
        abs_day = abs(day)
        role = _case_crossover_control_role(day)
        select_lines = [
            "  UNION ALL",
            "  SELECT",
            "    fall_event_id,",
            f'    "{role}" AS window_role,',
            f"    TIMESTAMP_SUB(fall_ts_utc, INTERVAL {abs_day} DAY) AS anchor_ts_utc,",
        ]
        if include_fall_ts:
            select_lines.extend(
                [
                    "    fall_ts_utc,",
                    "    fall_ts_local,",
                ]
            )
        select_lines.extend(
            [
                "    hospital_id,",
                "    division_id,",
                "    patient_id,",
                "    monitor_id",
                "  FROM falls",
            ]
        )
        blocks.append("\n".join(select_lines))
    return "\n".join(blocks)


def _render_case_crossover_control_role_list(settings: Settings) -> str:
    return ", ".join(f'"{_case_crossover_control_role(day)}"' for day in _case_crossover_anchor_days(settings))


def _render_fall_events_compatible_subquery(settings: Settings) -> str:
    return "\n".join(
        [
            "(",
            "  WITH base_events AS (",
            "    SELECT",
            "      ROW_NUMBER() OVER (",
            "        ORDER BY timestamp_local, hospital_id, division_id, monitor_id, summary, user_name",
            "      ) AS raw_event_id,",
            f'      TIMESTAMP(DATETIME(timestamp_local), "{settings.hospital_timezone}") AS timestamp,',
            "      timestamp_local,",
            "      date_local,",
            "      time_local,",
            "      hospital_id,",
            "      hospital_system_name,",
            "      division_id,",
            "      hospital_name,",
            "      monitor_id,",
            "      user_name,",
            "      summary",
            f"    FROM `{settings.google_cloud_project}.{settings.bq_dataset}.{settings.fall_events_table}`",
            f"    WHERE hospital_id = {settings.study_hospital_id}",
            "      AND timestamp_local IS NOT NULL",
            "      AND division_id IS NOT NULL",
            "      AND monitor_id IS NOT NULL",
            "  ),",
            "  deriv_candidates AS (",
            "    SELECT",
            "      b.raw_event_id,",
            "      d.patient_id,",
            "      ROW_NUMBER() OVER (",
            "        PARTITION BY b.raw_event_id",
            "        ORDER BY ABS(TIMESTAMP_DIFF(d.timestamp, b.timestamp, SECOND)), d.patient_id",
            "      ) AS rn",
            "    FROM base_events b",
            f"    LEFT JOIN `{settings.google_cloud_project}.{settings.bq_dataset}.{settings.live_stream_derivatives_table}` d",
            "      ON d.hospital_id = b.hospital_id",
            "     AND d.division_id = b.division_id",
            "     AND d.monitor_id = b.monitor_id",
            "     AND d.patient_id IS NOT NULL",
            "     AND d.timestamp BETWEEN TIMESTAMP_SUB(b.timestamp, INTERVAL 30 MINUTE)",
            "                         AND TIMESTAMP_ADD(b.timestamp, INTERVAL 30 MINUTE)",
            "  ),",
            "  deriv_best AS (",
            "    SELECT raw_event_id, patient_id",
            "    FROM deriv_candidates",
            "    WHERE rn = 1",
            "  ),",
            "  hourly_candidates AS (",
            "    SELECT",
            "      b.raw_event_id,",
            "      h.patient_id,",
            "      ROW_NUMBER() OVER (",
            "        PARTITION BY b.raw_event_id",
            "        ORDER BY ABS(TIMESTAMP_DIFF(TIMESTAMP_TRUNC(h.hour_bin_local, HOUR), TIMESTAMP_TRUNC(b.timestamp_local, HOUR), SECOND)), h.patient_id",
            "      ) AS rn",
            "    FROM base_events b",
            f"    LEFT JOIN `{settings.google_cloud_project}.{settings.bq_dataset}.hourly_graph_facts_alarms_cache` h",
            "      ON h.hospital_id = b.hospital_id",
            "     AND h.division_id = b.division_id",
            "     AND h.monitor_id = b.monitor_id",
            "     AND h.patient_id IS NOT NULL",
            "     AND h.hour_bin_local BETWEEN TIMESTAMP_SUB(b.timestamp_local, INTERVAL 24 HOUR)",
            "                             AND TIMESTAMP_ADD(b.timestamp_local, INTERVAL 24 HOUR)",
            "  ),",
            "  hourly_best AS (",
            "    SELECT raw_event_id, patient_id",
            "    FROM hourly_candidates",
            "    WHERE rn = 1",
            "  )",
            "  SELECT",
            "    b.timestamp,",
            "    b.timestamp_local,",
            "    b.date_local,",
            "    b.time_local,",
            "    b.hospital_id,",
            "    b.hospital_system_name,",
            "    b.division_id,",
            "    b.hospital_name,",
            "    COALESCE(d.patient_id, h.patient_id) AS patient_id,",
            "    b.monitor_id,",
            "    b.user_name,",
            "    b.summary",
            "  FROM base_events b",
            "  LEFT JOIN deriv_best d USING (raw_event_id)",
            "  LEFT JOIN hourly_best h USING (raw_event_id)",
            ")",
        ]
    )


def _render_negative_control_source_chunk_subquery(settings: Settings) -> str:
    return "\n".join(
        [
            "(",
            "  WITH base AS (",
            "    SELECT",
            "      timestamp,",
            "      hospital_id,",
            "      division_id,",
            "      monitor_id,",
            "      patient_id",
            f"    FROM `{settings.google_cloud_project}.{settings.bq_dataset}.{settings.negative_control_derivatives_table}`",
            f"    WHERE hospital_id = {settings.study_hospital_id}",
            "      AND timestamp IS NOT NULL",
            "      AND division_id IS NOT NULL",
            "      AND monitor_id IS NOT NULL",
            "  ),",
            "  flagged AS (",
            "    SELECT",
            "      *,",
            "      LAG(timestamp) OVER (",
            "        PARTITION BY hospital_id, division_id, monitor_id, patient_id",
            "        ORDER BY timestamp",
            "      ) AS prev_ts",
            "    FROM base",
            "  ),",
            "  chunked AS (",
            "    SELECT",
            "      *,",
            "      SUM(",
            "        IF(prev_ts IS NULL OR TIMESTAMP_DIFF(timestamp, prev_ts, SECOND) > 5, 1, 0)",
            "      ) OVER (",
            "        PARTITION BY hospital_id, division_id, monitor_id, patient_id",
            "        ORDER BY timestamp",
            "      ) AS chunk_id",
            "    FROM flagged",
            "  ),",
            "  aggregated AS (",
            "    SELECT",
            "      CONCAT(",
            "        CAST(hospital_id AS STRING), '|',",
            "        CAST(division_id AS STRING), '|',",
            "        CAST(monitor_id AS STRING), '|',",
            "        IFNULL(CAST(patient_id AS STRING), 'null'), '|',",
            "        CAST(chunk_id AS STRING)",
            "      ) AS source_chunk_id,",
            "      hospital_id,",
            "      division_id,",
            "      monitor_id,",
            "      patient_id,",
            "      MIN(timestamp) AS chunk_start_ts_utc,",
            "      MAX(timestamp) AS chunk_end_ts_utc,",
            "      TIMESTAMP_ADD(MAX(timestamp), INTERVAL 1 SECOND) AS anchor_ts_utc,",
            "      COUNT(*) AS frame_count,",
            "      TIMESTAMP_DIFF(MAX(timestamp), MIN(timestamp), SECOND) AS duration_seconds",
            "    FROM chunked",
            "    GROUP BY hospital_id, division_id, monitor_id, patient_id, chunk_id",
            "  )",
            "  SELECT",
            "    source_chunk_id,",
            "    hospital_id,",
            "    division_id,",
            "    monitor_id,",
            "    patient_id,",
            "    chunk_start_ts_utc,",
            "    chunk_end_ts_utc,",
            "    anchor_ts_utc,",
            "    frame_count,",
            "    duration_seconds,",
            f'    EXTRACT(HOUR FROM anchor_ts_utc AT TIME ZONE "{settings.hospital_timezone}") AS anchor_hour_local,',
            "    CASE",
            f'      WHEN EXTRACT(HOUR FROM anchor_ts_utc AT TIME ZONE "{settings.hospital_timezone}") BETWEEN 0 AND 5 THEN "window_00_05"',
            f'      WHEN EXTRACT(HOUR FROM anchor_ts_utc AT TIME ZONE "{settings.hospital_timezone}") BETWEEN 6 AND 8 THEN "window_06_08"',
            f'      WHEN EXTRACT(HOUR FROM anchor_ts_utc AT TIME ZONE "{settings.hospital_timezone}") BETWEEN 9 AND 11 THEN "window_09_11"',
            f'      WHEN EXTRACT(HOUR FROM anchor_ts_utc AT TIME ZONE "{settings.hospital_timezone}") BETWEEN 12 AND 14 THEN "window_12_14"',
            f'      WHEN EXTRACT(HOUR FROM anchor_ts_utc AT TIME ZONE "{settings.hospital_timezone}") BETWEEN 15 AND 17 THEN "window_15_17"',
            f'      WHEN EXTRACT(HOUR FROM anchor_ts_utc AT TIME ZONE "{settings.hospital_timezone}") BETWEEN 18 AND 20 THEN "window_18_20"',
            '      ELSE "window_21_23"',
            "    END AS anchor_daypart_local,",
            "    patient_id IS NOT NULL AS has_patient_id",
            "  FROM aggregated",
            ")",
        ]
    )


def _render_sql(template: str, settings: Settings) -> str:
    rendered = template
    replacements = {
        "{{project}}": settings.google_cloud_project,
        "{{dataset}}": settings.bq_dataset,
        "{{study_hospital_id}}": str(settings.study_hospital_id),
        "{{fall_events_table}}": settings.fall_events_table,
        "{{live_stream_derivatives_table}}": settings.live_stream_derivatives_table,
        "{{negative_control_source_table}}": settings.negative_control_derivatives_table,
        "{{hospital_timezone}}": settings.hospital_timezone,
        "{{negative_control_match_limit}}": str(settings.negative_control_match_limit),
        "{{fall_events_compatible_subquery_sql}}": _render_fall_events_compatible_subquery(settings),
        "{{negative_control_source_chunk_subquery_sql}}": _render_negative_control_source_chunk_subquery(settings),
        "{{fall_case_crossover_control_unions_sql}}": _render_case_crossover_control_unions(
            settings,
            include_fall_ts=False,
        ),
        "{{fall_case_crossover_control_unions_second_level_sql}}": _render_case_crossover_control_unions(
            settings,
            include_fall_ts=True,
        ),
        "{{fall_case_crossover_control_roles_sql}}": _render_case_crossover_control_role_list(settings),
    }
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def _read_sql(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_parquet_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except TypeError as exc:
        # Some upstream parquet producers persist extension dtype metadata (for example dbdate)
        # that newer pandas cannot resolve. Fall back to Arrow conversion without pandas metadata.
        if "dbdate" not in str(exc):
            raise
        return pq.read_table(path).to_pandas(ignore_metadata=True)


def run_extract_pipeline(settings: Settings) -> dict:
    raw_dir = ensure_dir(settings.paths.raw_run_dir)
    client = BigQueryClientWrapper(settings)

    extracts: list[dict] = []
    skipped_extracts: list[dict[str, str]] = []
    for spec in EXTRACT_SPECS:
        if not _extract_enabled(spec.name, settings):
            skipped_extracts.append(
                {
                    "name": spec.name,
                    "status": "skipped",
                    "reason": _extract_skip_reason(spec.name, settings),
                }
            )
            continue
        sql_path = settings.project_root / getattr(settings, spec.sql_path_attr)
        sql = _render_sql(_read_sql(sql_path), settings)
        destination = raw_dir / spec.output_filename
        client.extract_to_parquet(
            name=spec.name,
            sql=sql,
            destination=destination,
            dry_run_columns=spec.dry_columns,
        )
        extracts.append(
            {
                "name": spec.name,
                "sql_path": str(sql_path),
                "output": str(destination),
            }
        )

    query_log_path = client.flush_query_log()
    manifest = {
        "run_id": settings.run_id,
        "dry_run": settings.dry_run,
        "raw_snapshot_dir": str(raw_dir),
        "extracts": extracts,
        "skipped_extracts": skipped_extracts,
        "query_log_path": str(query_log_path),
        "case_crossover_source_status": settings.case_crossover_source_status,
        "negative_control_source_status": settings.negative_control_source_status,
        "nonfall_control_source_status": settings.negative_control_source_status,
    }
    manifest_path = settings.paths.manifests_dir / f"extract_manifest_{settings.run_id}.yaml"
    manifest["manifest_path"] = str(manifest_path)
    write_yaml(manifest_path, manifest)
    return manifest


def load_raw_tables(settings: Settings) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for spec in EXTRACT_SPECS:
        path = settings.paths.raw_run_dir / spec.output_filename
        if not _extract_enabled(spec.name, settings):
            tables[spec.name] = pd.DataFrame(columns=spec.dry_columns)
        elif path.exists():
            tables[spec.name] = _read_parquet_robust(path)
        else:
            tables[spec.name] = pd.DataFrame(columns=spec.dry_columns)
    return tables
