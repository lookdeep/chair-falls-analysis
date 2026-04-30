from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

CONSENSUS_REQUIRED_COLUMNS = (
    "event_key",
    "prefall_location",
    "fall_time_consensus",
    "response_time_consensus",
    "fall_tags",
)
CONSENSUS_OPTIONAL_COLUMNS = ("last_furniture", "furniture_departure_time")

CONSENSUS_STATUS_INCLUDED_FALL = "included_fall"
CONSENSUS_STATUS_ACCEPTED_NONFALL = "accepted_nonfall"
CONSENSUS_STATUS_EXCLUDED_OFFSCREEN = "excluded_offscreen"
CONSENSUS_STATUS_EXCLUDED_BAD_OR_NO_VIDEO = "excluded_bad_or_no_video"
CONSENSUS_STATUS_EXCLUDED_NO_SIGNAL = "excluded_no_signal"

CONSENSUS_STATUS_PRIORITY = {
    CONSENSUS_STATUS_ACCEPTED_NONFALL: 0,
    CONSENSUS_STATUS_EXCLUDED_BAD_OR_NO_VIDEO: 1,
    CONSENSUS_STATUS_EXCLUDED_NO_SIGNAL: 2,
    CONSENSUS_STATUS_EXCLUDED_OFFSCREEN: 3,
    CONSENSUS_STATUS_INCLUDED_FALL: 4,
}

# Events where both the legacy SQL derivatives and the v2 second-level panel
# returned zero location signal.  These are from monitors/time-ranges where the
# vision pipeline did not produce location data, making the pre-fall location
# unclassifiable by either model.  Excluded from benchmark scoring.
_NO_SIGNAL_EVENT_KEYS: frozenset[str] = frozenset(
    {
        "854|2022-12-23|11:07",
        "854|2022-12-24|05:55",
        "1546|2023-05-30|09:48",
        "1788|2023-07-17|21:03",
        "2410|2023-10-16|15:19",
        "2909|2024-01-02|15:20",
        "2923|2024-01-02|12:58",
        "3226|2024-02-09|14:11",
        "3515|2024-03-14|16:00",
        "4233|2024-06-09|07:07",
        "4550|2024-07-20|06:17",
        "5384|2024-10-21|05:10",
        "6056|2025-01-11|13:58",
    }
)

_LABEL_ALIASES = {"other": "room", "unknown": "no_patient"}
_TAG_ALIASES = {
    "off_screen": "offscreen",
    "no fall": "no_fall",
    "bad video": "bad_video",
    "no video": "no_video",
}
_ACCEPTED_NONFALL_TAGS = {"no_fall"}
_VIDEO_EXCLUSION_TAGS = {"bad_video", "no_video"}
_OFFSCREEN_TAGS = {"offscreen"}


def _clean_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    token = str(value).strip()
    if token.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return token


def normalize_consensus_location(value: Any) -> str | pd.NA:
    token = _clean_string(value).lower()
    if not token:
        return pd.NA
    return _LABEL_ALIASES.get(token, token)


def normalize_consensus_furniture(value: Any) -> str | pd.NA:
    token = _clean_string(value).lower()
    if not token:
        return pd.NA
    return token


def normalize_consensus_tags(value: Any) -> list[str]:
    token = _clean_string(value)
    if not token:
        return []
    normalized: list[str] = []
    for raw in token.split(","):
        tag = raw.strip().lower()
        if not tag:
            continue
        tag = _TAG_ALIASES.get(tag, tag.replace(" ", "_"))
        normalized.append(tag)
    return sorted(set(normalized))


def normalize_consensus_tags_string(value: Any) -> str:
    return ", ".join(normalize_consensus_tags(value))


def time_to_seconds(value: Any) -> float | None:
    token = _clean_string(value)
    if token in {"", "-"}:
        return None
    parts = token.split(":")
    if len(parts) != 3:
        return None
    try:
        hh, mm, ss = [int(part) for part in parts]
    except ValueError:
        return None
    if not (0 <= hh < 24 and 0 <= mm < 60 and 0 <= ss < 60):
        return None
    return float((hh * 3600) + (mm * 60) + ss)


def parse_event_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if frame.empty or "event_key" not in frame.columns:
        out["monitor_id"] = pd.Series(dtype="Int64")
        out["date_local"] = pd.Series(dtype="string")
        out["minute_local"] = pd.Series(dtype="string")
        return out

    event_parts = frame["event_key"].astype("string").str.split("|", expand=True)
    monitor_part = event_parts[0] if 0 in event_parts.columns else pd.Series(pd.NA, index=frame.index)
    date_part = event_parts[1] if 1 in event_parts.columns else pd.Series(pd.NA, index=frame.index)
    minute_part = event_parts[2] if 2 in event_parts.columns else pd.Series(pd.NA, index=frame.index)

    out["monitor_id"] = pd.to_numeric(monitor_part, errors="coerce").astype("Int64")
    out["date_local"] = date_part.astype("string")
    out["minute_local"] = minute_part.astype("string")
    return out


def assign_sequence_ids(consensus: pd.DataFrame) -> pd.DataFrame:
    if consensus.empty:
        return consensus.copy()

    rows: list[dict[str, Any]] = []
    for event_key, group in consensus.groupby("event_key", dropna=False):
        group = group.sort_values(
            ["fall_time_seconds", "response_time_seconds"],
            ascending=[True, True],
            na_position="last",
            kind="mergesort",
        ).reset_index(drop=True)

        seq_num = 1
        for idx, row in group.iterrows():
            response_known = pd.notna(row["response_time_seconds"])
            entry = row.to_dict()
            entry["event_instance_ordinal"] = idx + 1
            entry["sequence_ordinal"] = seq_num
            entry["sequence_id"] = f"{event_key}#S{seq_num:02d}"
            rows.append(entry)
            if response_known:
                seq_num += 1

    return pd.DataFrame(rows)


def _empty_consensus_frame() -> pd.DataFrame:
    columns = list(CONSENSUS_REQUIRED_COLUMNS) + list(CONSENSUS_OPTIONAL_COLUMNS) + [
        "normalized_fall_tags",
        "consensus_status",
        "consensus_include_in_fall_cohort",
    ]
    return pd.DataFrame(columns=columns)


def prepare_consensus_annotations(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: column.strip().lower() for column in frame.columns}
    result = frame.rename(columns=renamed).copy()
    for column in CONSENSUS_REQUIRED_COLUMNS + CONSENSUS_OPTIONAL_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    result["event_key"] = result["event_key"].astype("string").str.strip()
    result["prefall_location"] = pd.Series(
        [normalize_consensus_location(value) for value in result["prefall_location"]],
        index=result.index,
        dtype="object",
    )
    result["last_furniture"] = pd.Series(
        [normalize_consensus_furniture(value) for value in result["last_furniture"]],
        index=result.index,
        dtype="object",
    )
    result["normalized_fall_tags"] = pd.Series(
        [normalize_consensus_tags(value) for value in result["fall_tags"]],
        index=result.index,
        dtype="object",
    )
    result["fall_tags"] = pd.Series(
        [", ".join(tags) for tags in result["normalized_fall_tags"]],
        index=result.index,
        dtype="string",
    )

    statuses: list[str] = []
    for _, row in result.iterrows():
        tags = set(row.get("normalized_fall_tags", []))
        if tags & _ACCEPTED_NONFALL_TAGS:
            statuses.append(CONSENSUS_STATUS_ACCEPTED_NONFALL)
            continue
        if tags & _VIDEO_EXCLUSION_TAGS:
            statuses.append(CONSENSUS_STATUS_EXCLUDED_BAD_OR_NO_VIDEO)
            continue
        if tags & _OFFSCREEN_TAGS or time_to_seconds(row.get("fall_time_consensus")) is None:
            statuses.append(CONSENSUS_STATUS_EXCLUDED_OFFSCREEN)
            continue
        statuses.append(CONSENSUS_STATUS_INCLUDED_FALL)

    result["consensus_status"] = pd.Series(statuses, index=result.index, dtype="string")

    # Override: demote included_fall events with zero location signal.
    no_signal_mask = (
        result["consensus_status"].eq(CONSENSUS_STATUS_INCLUDED_FALL)
        & result["event_key"].isin(_NO_SIGNAL_EVENT_KEYS)
    )
    result.loc[no_signal_mask, "consensus_status"] = CONSENSUS_STATUS_EXCLUDED_NO_SIGNAL

    result["consensus_include_in_fall_cohort"] = result["consensus_status"].eq(
        CONSENSUS_STATUS_INCLUDED_FALL
    )
    return result


def load_consensus_annotations(consensus_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not consensus_path.exists():
        return _empty_consensus_frame(), {
            "status": "missing_consensus_csv",
            "path": str(consensus_path),
        }

    consensus = pd.read_csv(consensus_path)
    if consensus.empty:
        return _empty_consensus_frame(), {
            "status": "empty_consensus_csv",
            "path": str(consensus_path),
        }

    expected_columns = set(CONSENSUS_REQUIRED_COLUMNS)
    missing = sorted(expected_columns - set(column.strip().lower() for column in consensus.columns))
    if missing:
        return _empty_consensus_frame(), {
            "status": "invalid_consensus_schema",
            "missing_columns": missing,
            "path": str(consensus_path),
        }

    prepared = prepare_consensus_annotations(consensus)
    summary = {
        "status": "ok",
        "path": str(consensus_path),
        "rows": int(len(prepared.index)),
        "unique_event_keys": int(prepared["event_key"].nunique(dropna=True)),
    }
    return prepared, summary


def included_fall_annotations(consensus: pd.DataFrame) -> pd.DataFrame:
    if consensus.empty or "consensus_include_in_fall_cohort" not in consensus.columns:
        return consensus.iloc[0:0].copy()
    return consensus.loc[consensus["consensus_include_in_fall_cohort"]].copy()


def filter_consensus_to_study_window(
    consensus: pd.DataFrame,
    study_start_date: date | None = None,
    study_end_date: date | None = None,
) -> pd.DataFrame:
    if consensus.empty or (study_start_date is None and study_end_date is None):
        return consensus.copy()

    filtered = parse_event_key(consensus)
    local_dates = pd.to_datetime(filtered["date_local"], errors="coerce").dt.date
    mask = pd.Series(True, index=filtered.index)
    if study_start_date is not None:
        mask &= local_dates >= study_start_date
    if study_end_date is not None:
        mask &= local_dates <= study_end_date
    return filtered.loc[mask].copy()


def _build_consensus_summary_rows(consensus: pd.DataFrame, *, metric_prefix: str = "") -> list[dict[str, Any]]:
    prefix = f"{metric_prefix}_" if metric_prefix else ""
    parsed = parse_event_key(consensus)
    included = included_fall_annotations(consensus)
    included = parse_event_key(included)
    included["fall_time_seconds"] = included["fall_time_consensus"].apply(time_to_seconds)
    included["response_time_seconds"] = included["response_time_consensus"].apply(time_to_seconds)
    included_sequences = assign_sequence_ids(included)

    mechanism_eligible_event_keys = (
        included.loc[included["prefall_location"].notna(), "event_key"].nunique(dropna=True)
        if not included.empty
        else 0
    )
    chair_origin_event_keys = (
        included.loc[
            included["prefall_location"].isin(["room", "no_patient"])
            & included["last_furniture"].eq("chair"),
            "event_key",
        ].nunique(dropna=True)
        if not included.empty
        else 0
    )

    return [
        {"metric": f"{prefix}source_rows", "value": int(len(consensus.index))},
        {"metric": f"{prefix}source_unique_event_keys", "value": int(consensus["event_key"].nunique(dropna=True))},
        {
            "metric": f"{prefix}source_unique_monitors",
            "value": int(parsed["monitor_id"].nunique(dropna=True)) if not parsed.empty else 0,
        },
        {
            "metric": f"{prefix}included_fall_rows",
            "value": int(len(included.index)),
        },
        {
            "metric": f"{prefix}included_fall_unique_event_keys",
            "value": int(included["event_key"].nunique(dropna=True)) if not included.empty else 0,
        },
        {
            "metric": f"{prefix}included_fall_sequences",
            "value": int(included_sequences["sequence_id"].nunique(dropna=True))
            if not included_sequences.empty
            else 0,
        },
        {
            "metric": f"{prefix}included_fall_unique_monitors",
            "value": int(included["monitor_id"].nunique(dropna=True)) if not included.empty else 0,
        },
        {
            "metric": f"{prefix}accepted_nonfall_rows",
            "value": int(
                consensus["consensus_status"].eq(CONSENSUS_STATUS_ACCEPTED_NONFALL).sum()
            ),
        },
        {
            "metric": f"{prefix}accepted_nonfall_unique_event_keys",
            "value": int(
                consensus.loc[
                    consensus["consensus_status"].eq(CONSENSUS_STATUS_ACCEPTED_NONFALL),
                    "event_key",
                ].nunique(dropna=True)
            ),
        },
        {
            "metric": f"{prefix}excluded_offscreen_rows",
            "value": int(
                consensus["consensus_status"].eq(CONSENSUS_STATUS_EXCLUDED_OFFSCREEN).sum()
            ),
        },
        {
            "metric": f"{prefix}excluded_offscreen_unique_event_keys",
            "value": int(
                consensus.loc[
                    consensus["consensus_status"].eq(CONSENSUS_STATUS_EXCLUDED_OFFSCREEN),
                    "event_key",
                ].nunique(dropna=True)
            ),
        },
        {
            "metric": f"{prefix}excluded_bad_or_no_video_rows",
            "value": int(
                consensus["consensus_status"].eq(CONSENSUS_STATUS_EXCLUDED_BAD_OR_NO_VIDEO).sum()
            ),
        },
        {
            "metric": f"{prefix}excluded_bad_or_no_video_unique_event_keys",
            "value": int(
                consensus.loc[
                    consensus["consensus_status"].eq(CONSENSUS_STATUS_EXCLUDED_BAD_OR_NO_VIDEO),
                    "event_key",
                ].nunique(dropna=True)
            ),
        },
        {
            "metric": f"{prefix}excluded_no_signal_rows",
            "value": int(
                consensus["consensus_status"].eq(CONSENSUS_STATUS_EXCLUDED_NO_SIGNAL).sum()
            ),
        },
        {
            "metric": f"{prefix}excluded_no_signal_unique_event_keys",
            "value": int(
                consensus.loc[
                    consensus["consensus_status"].eq(CONSENSUS_STATUS_EXCLUDED_NO_SIGNAL),
                    "event_key",
                ].nunique(dropna=True)
            ),
        },
        {
            "metric": f"{prefix}mechanism_eligible_event_keys",
            "value": int(mechanism_eligible_event_keys),
        },
        {
            "metric": f"{prefix}chair_origin_room_or_no_patient_event_keys",
            "value": int(chair_origin_event_keys),
        },
    ]


def build_consensus_adjudication_tables(
    consensus: pd.DataFrame,
    *,
    study_start_date: date | None = None,
    study_end_date: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    status_rows = []
    for status, subset in consensus.groupby("consensus_status", dropna=False):
        status_rows.append(
            {
                "consensus_status": str(status),
                "row_count": int(len(subset.index)),
                "unique_event_keys": int(subset["event_key"].nunique(dropna=True)),
            }
        )
    status_counts = pd.DataFrame(status_rows).sort_values(
        "consensus_status",
        key=lambda series: series.map(CONSENSUS_STATUS_PRIORITY).fillna(99),
        kind="mergesort",
    )

    summary_rows = _build_consensus_summary_rows(consensus)
    if study_start_date is not None or study_end_date is not None:
        filtered = filter_consensus_to_study_window(consensus, study_start_date, study_end_date)
        summary_rows.extend(
            [
                {
                    "metric": "study_window_start_date",
                    "value": study_start_date.isoformat() if study_start_date is not None else "",
                },
                {
                    "metric": "study_window_end_date",
                    "value": study_end_date.isoformat() if study_end_date is not None else "",
                },
            ]
        )
        summary_rows.extend(_build_consensus_summary_rows(filtered, metric_prefix="study_window"))
    summary = pd.DataFrame(summary_rows)
    return status_counts.reset_index(drop=True), summary
