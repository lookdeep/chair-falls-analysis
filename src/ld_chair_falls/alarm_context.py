from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .consensus import (
    assign_sequence_ids,
    parse_event_key,
    prepare_consensus_annotations,
    time_to_seconds,
)

TYPE7_ANALYSIS_ID = 7
DEFAULT_WINDOW_SECONDS = 300
STRICT_REPORT_RE = re.compile(r"\b(fall|falls|falling|fell|floor|slip|slipped)\b", re.IGNORECASE)

SUMMARY_COLUMNS = [
    "source_row_number",
    "event_key",
    "event_instance_ordinal",
    "sequence_id",
    "monitor_id",
    "date_local",
    "fall_time_consensus",
    "response_time_consensus",
    "consensus_status",
    "alarm_video_applicable",
    "alarm_video_reason",
    "alarm_count_within_1m",
    "alarm_count_within_3m",
    "alarm_count_within_5m",
    "nearest_alarm_created_at_utc",
    "nearest_alarm_sec_from_fall",
    "nearest_alarm_source",
    "nearest_alarm_path",
    "nearest_alarm_recording_status",
    "nearest_alarm_analysis_status",
    "nearest_alarm_report_explicit_fall",
    "nearest_alarm_events_explicit_fall",
    "nearest_alarm_explicit_fall_any",
    "nearest_alarm_report_match_source",
    "nearest_alarm_report",
    "nearest_alarm_events_json",
]

DETAIL_COLUMNS = [
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
    "report",
    "events_json",
    "report_explicit_fall",
    "events_explicit_fall",
    "explicit_fall_any",
    "explicit_fall_match_source",
]


def _clean_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    token = str(value).strip()
    if token.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return token


def _to_local_utc(date_token: Any, wallclock_token: Any, timezone_name: str) -> pd.Timestamp:
    date_text = _clean_string(date_token)
    seconds = time_to_seconds(wallclock_token)
    if not date_text or seconds is None:
        return pd.NaT
    try:
        date_value = datetime.strptime(date_text, "%Y-%m-%d").date()
    except ValueError:
        return pd.NaT

    total_seconds = int(seconds)
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    local_dt = datetime(
        date_value.year,
        date_value.month,
        date_value.day,
        hh,
        mm,
        ss,
        tzinfo=ZoneInfo(timezone_name),
    )
    return pd.Timestamp(local_dt.astimezone(UTC))


def load_consensus_alarm_video_base(consensus_path: Path, hospital_timezone: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(consensus_path)
    raw.insert(0, "source_row_number", range(1, len(raw.index) + 1))

    prepared = prepare_consensus_annotations(raw)
    prepared = parse_event_key(prepared)
    prepared["fall_time_seconds"] = prepared["fall_time_consensus"].apply(time_to_seconds)
    prepared["response_time_seconds"] = prepared["response_time_consensus"].apply(time_to_seconds)
    prepared["fall_ts_utc"] = prepared.apply(
        lambda row: _to_local_utc(row.get("date_local"), row.get("fall_time_consensus"), hospital_timezone),
        axis=1,
    )

    sequenced = assign_sequence_ids(prepared).sort_values("source_row_number", kind="mergesort")
    sequenced = sequenced.reset_index(drop=True)

    summary = {
        "rows": int(len(sequenced.index)),
        "unique_event_keys": int(sequenced["event_key"].nunique(dropna=True)),
        "rows_with_fall_time": int(sequenced["fall_ts_utc"].notna().sum()),
    }
    return sequenced, summary


def build_source_coverage_sql(project_id: str, hospital_id: int, analysis_type_id: int = TYPE7_ANALYSIS_ID) -> str:
    return f"""
SELECT
  MIN(r.created_at) AS source_min_created_at,
  MAX(r.created_at) AS source_max_created_at,
  COUNT(*) AS source_row_count
FROM `{project_id}.hospital_api.recording_analyses` ra
JOIN `{project_id}.hospital_api.recordings` r
  ON ra.recording_id = r.id
WHERE ra.hospital_id = {hospital_id}
  AND ra.recording_analysis_type_id = {analysis_type_id}
""".strip()


def _sql_string(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "''") + "'"


def build_nearby_alarm_query(
    *,
    project_id: str,
    hospital_id: int,
    events: pd.DataFrame,
    analysis_type_id: int = TYPE7_ANALYSIS_ID,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> str:
    eligible = events.loc[events["monitor_id"].notna() & events["fall_ts_utc"].notna()].copy()
    if eligible.empty:
        raise ValueError("No eligible events available for alarm-video query.")

    struct_rows: list[str] = []
    for row in eligible.to_dict(orient="records"):
        fall_ts = pd.Timestamp(row["fall_ts_utc"]).to_pydatetime().astimezone(UTC)
        struct_rows.append(
            "STRUCT("
            f"{int(row['source_row_number'])} AS source_row_number, "
            f"{_sql_string(str(row['event_key']))} AS event_key, "
            f"{int(row['event_instance_ordinal'])} AS event_instance_ordinal, "
            f"{_sql_string(str(row['sequence_id']))} AS sequence_id, "
            f"{int(row['monitor_id'])} AS monitor_id, "
            f"TIMESTAMP {_sql_string(fall_ts.strftime('%Y-%m-%d %H:%M:%S+00:00'))} AS fall_ts_utc"
            ")"
        )

    events_sql = ",\n    ".join(struct_rows)
    return f"""
WITH events AS (
  SELECT * FROM UNNEST([
    {events_sql}
  ])
)
SELECT
  e.source_row_number,
  e.event_key,
  e.event_instance_ordinal,
  e.sequence_id,
  e.monitor_id,
  r.id AS recording_id,
  r.created_at AS created_at_utc,
  TIMESTAMP_DIFF(r.created_at, e.fall_ts_utc, SECOND) AS sec_from_fall,
  r.source AS recording_source,
  r.path AS recording_path,
  r.status AS recording_status,
  ra.status AS analysis_status,
  ra.result AS analysis_result
FROM events e
JOIN `{project_id}.hospital_api.recordings` r
  ON r.patient_monitor_id = e.monitor_id
JOIN `{project_id}.hospital_api.recording_analyses` ra
  ON ra.recording_id = r.id
WHERE ra.hospital_id = {hospital_id}
  AND ra.recording_analysis_type_id = {analysis_type_id}
  AND ABS(TIMESTAMP_DIFF(r.created_at, e.fall_ts_utc, SECOND)) <= {window_seconds}
ORDER BY source_row_number, ABS(sec_from_fall), created_at_utc
""".strip()


def _parse_analysis_payload(value: Any) -> tuple[str, list[dict[str, Any]]]:
    text = _clean_string(value)
    if not text:
        return "", []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return "", []
    report = payload.get("report")
    events = payload.get("events")
    if not isinstance(report, str):
        report = ""
    if not isinstance(events, list):
        events = []
    return report.strip(), [event for event in events if isinstance(event, dict)]


def normalize_alarm_matches(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    if "analysis_result" not in frame.columns and set(DETAIL_COLUMNS).issubset(frame.columns):
        normalized = frame.copy()
        normalized["created_at_utc"] = pd.to_datetime(normalized["created_at_utc"], errors="coerce", utc=True)
        normalized["sec_from_fall"] = pd.to_numeric(normalized["sec_from_fall"], errors="coerce")
        return normalized[DETAIL_COLUMNS].copy()

    normalized = frame.copy()
    normalized["created_at_utc"] = pd.to_datetime(normalized["created_at_utc"], errors="coerce", utc=True)
    normalized["sec_from_fall"] = pd.to_numeric(normalized["sec_from_fall"], errors="coerce")

    reports: list[str] = []
    events_json: list[str] = []
    report_hits: list[bool] = []
    event_hits: list[bool] = []
    match_sources: list[str] = []

    for payload in normalized.get("analysis_result", pd.Series(index=normalized.index, dtype="object")):
        report, events = _parse_analysis_payload(payload)
        event_texts = [
            _clean_string(event.get("description"))
            for event in events
            if _clean_string(event.get("description"))
        ]
        report_hit = bool(STRICT_REPORT_RE.search(report))
        events_hit = any(STRICT_REPORT_RE.search(text) for text in event_texts)
        if report_hit:
            match_source = "report"
        elif events_hit:
            match_source = "events_description"
        else:
            match_source = ""

        reports.append(report)
        events_json.append(json.dumps(events, sort_keys=True))
        report_hits.append(report_hit)
        event_hits.append(events_hit)
        match_sources.append(match_source)

    normalized["report"] = pd.Series(reports, index=normalized.index, dtype="string")
    normalized["events_json"] = pd.Series(events_json, index=normalized.index, dtype="string")
    normalized["report_explicit_fall"] = pd.Series(report_hits, index=normalized.index, dtype="boolean")
    normalized["events_explicit_fall"] = pd.Series(event_hits, index=normalized.index, dtype="boolean")
    normalized["explicit_fall_any"] = (
        normalized["report_explicit_fall"].fillna(False) | normalized["events_explicit_fall"].fillna(False)
    ).astype("boolean")
    normalized["explicit_fall_match_source"] = pd.Series(match_sources, index=normalized.index, dtype="string")

    for column in DETAIL_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    return normalized[DETAIL_COLUMNS].copy()


def _coverage_bounds(source_range: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    if source_range.empty:
        return pd.NaT, pd.NaT
    first = source_range.iloc[0]
    return (
        pd.to_datetime(first.get("source_min_created_at"), errors="coerce", utc=True),
        pd.to_datetime(first.get("source_max_created_at"), errors="coerce", utc=True),
    )


def summarize_alarm_video_context(base: pd.DataFrame, matches: pd.DataFrame, source_range: pd.DataFrame) -> pd.DataFrame:
    normalized_matches = normalize_alarm_matches(matches)
    source_min_ts, source_max_ts = _coverage_bounds(source_range)
    grouped = {
        int(source_row_number): group.sort_values(
            ["sec_from_fall", "created_at_utc"],
            key=lambda series: series.abs() if series.name == "sec_from_fall" else series,
            kind="mergesort",
        ).reset_index(drop=True)
        for source_row_number, group in normalized_matches.groupby("source_row_number", dropna=False)
    }

    rows: list[dict[str, Any]] = []
    for row in base.to_dict(orient="records"):
        source_row_number = int(row["source_row_number"])
        fall_ts_utc = pd.to_datetime(row.get("fall_ts_utc"), errors="coerce", utc=True)
        subset = grouped.get(source_row_number, pd.DataFrame(columns=DETAIL_COLUMNS))

        summary_row = {
            "source_row_number": source_row_number,
            "event_key": row.get("event_key"),
            "event_instance_ordinal": row.get("event_instance_ordinal"),
            "sequence_id": row.get("sequence_id"),
            "monitor_id": row.get("monitor_id"),
            "date_local": row.get("date_local"),
            "fall_time_consensus": row.get("fall_time_consensus"),
            "response_time_consensus": row.get("response_time_consensus"),
            "consensus_status": row.get("consensus_status"),
            "alarm_video_applicable": False,
            "alarm_video_reason": pd.NA,
            "alarm_count_within_1m": pd.NA,
            "alarm_count_within_3m": pd.NA,
            "alarm_count_within_5m": pd.NA,
            "nearest_alarm_created_at_utc": pd.NaT,
            "nearest_alarm_sec_from_fall": pd.NA,
            "nearest_alarm_source": pd.NA,
            "nearest_alarm_path": pd.NA,
            "nearest_alarm_recording_status": pd.NA,
            "nearest_alarm_analysis_status": pd.NA,
            "nearest_alarm_report_explicit_fall": pd.NA,
            "nearest_alarm_events_explicit_fall": pd.NA,
            "nearest_alarm_explicit_fall_any": pd.NA,
            "nearest_alarm_report_match_source": pd.NA,
            "nearest_alarm_report": pd.NA,
            "nearest_alarm_events_json": pd.NA,
        }

        if pd.isna(fall_ts_utc):
            summary_row["alarm_video_reason"] = "no_fall_time"
            rows.append(summary_row)
            continue

        if pd.notna(source_min_ts) and pd.notna(source_max_ts):
            if fall_ts_utc < source_min_ts or fall_ts_utc > source_max_ts:
                summary_row["alarm_video_reason"] = "outside_type7_source_range"
                rows.append(summary_row)
                continue

        summary_row["alarm_video_applicable"] = True
        count_1m = int((subset["sec_from_fall"].abs() <= 60).sum()) if not subset.empty else 0
        count_3m = int((subset["sec_from_fall"].abs() <= 180).sum()) if not subset.empty else 0
        count_5m = int((subset["sec_from_fall"].abs() <= 300).sum()) if not subset.empty else 0
        summary_row["alarm_count_within_1m"] = count_1m
        summary_row["alarm_count_within_3m"] = count_3m
        summary_row["alarm_count_within_5m"] = count_5m
        summary_row["alarm_video_reason"] = "matched_alarm_within_1m" if count_1m > 0 else "no_alarm_within_1m"

        nearest_subset = subset.loc[subset["sec_from_fall"].abs() <= 300].copy()
        if nearest_subset.empty:
            rows.append(summary_row)
            continue

        nearest = nearest_subset.iloc[0]
        summary_row["nearest_alarm_created_at_utc"] = nearest.get("created_at_utc")
        summary_row["nearest_alarm_sec_from_fall"] = nearest.get("sec_from_fall")
        summary_row["nearest_alarm_source"] = nearest.get("recording_source")
        summary_row["nearest_alarm_path"] = nearest.get("recording_path")
        summary_row["nearest_alarm_recording_status"] = nearest.get("recording_status")
        summary_row["nearest_alarm_analysis_status"] = nearest.get("analysis_status")
        summary_row["nearest_alarm_report_explicit_fall"] = bool(nearest.get("report_explicit_fall"))
        summary_row["nearest_alarm_events_explicit_fall"] = bool(nearest.get("events_explicit_fall"))
        summary_row["nearest_alarm_explicit_fall_any"] = bool(nearest.get("explicit_fall_any"))
        summary_row["nearest_alarm_report_match_source"] = nearest.get("explicit_fall_match_source")
        summary_row["nearest_alarm_report"] = nearest.get("report")
        summary_row["nearest_alarm_events_json"] = nearest.get("events_json")
        rows.append(summary_row)

    summary = pd.DataFrame(rows)
    for column in SUMMARY_COLUMNS:
        if column not in summary.columns:
            summary[column] = pd.NA
    return summary[SUMMARY_COLUMNS].copy()
