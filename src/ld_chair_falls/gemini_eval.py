from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .consensus import (
    CONSENSUS_STATUS_ACCEPTED_NONFALL,
    CONSENSUS_STATUS_EXCLUDED_BAD_OR_NO_VIDEO,
    CONSENSUS_STATUS_EXCLUDED_OFFSCREEN,
    CONSENSUS_STATUS_INCLUDED_FALL,
    CONSENSUS_STATUS_PRIORITY,
    normalize_consensus_furniture,
    normalize_consensus_location,
    normalize_consensus_tags,
    prepare_consensus_annotations,
    time_to_seconds,
)

CONSENSUS_STATUS_EXCLUDED_NO_SIGNAL = "excluded_no_signal"
_LOCAL_STATUS_PRIORITY = {
    **CONSENSUS_STATUS_PRIORITY,
    CONSENSUS_STATUS_EXCLUDED_NO_SIGNAL: 2,
}
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
HEADLINE_CONSENSUS_STATUSES = (
    CONSENSUS_STATUS_INCLUDED_FALL,
    CONSENSUS_STATUS_ACCEPTED_NONFALL,
)
DIAGNOSTIC_ONLY_STATUSES = (
    CONSENSUS_STATUS_EXCLUDED_OFFSCREEN,
    CONSENSUS_STATUS_EXCLUDED_NO_SIGNAL,
    CONSENSUS_STATUS_EXCLUDED_BAD_OR_NO_VIDEO,
)
TIMING_THRESHOLDS_SECONDS = (10, 30, 60)
_EVENT_FILE_RE = re.compile(
    r"^(?P<monitor_id>\d+)_(?P<date_local>\d{4}-\d{2}-\d{2})_(?P<hour>\d{2})-(?P<minute>\d{2})(?:\.\w+)?$"
)


@dataclass(frozen=True)
class GeminiGroundTruthArtifacts:
    comparison: pd.DataFrame
    coverage: pd.DataFrame
    failures: pd.DataFrame
    metrics: pd.DataFrame
    summary: dict[str, Any]


def gemini_file_to_event_key(filename: str) -> str | None:
    token = Path(filename).name.strip()
    match = _EVENT_FILE_RE.match(token)
    if match is None:
        return None
    return (
        f"{match.group('monitor_id')}|{match.group('date_local')}|"
        f"{match.group('hour')}:{match.group('minute')}"
    )


def normalize_gemini_tags(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    token = str(value).strip()
    if not token or token.lower() in {"none", "nan", "<na>"}:
        return []
    return normalize_consensus_tags(token.replace("|", ","))


def relative_seconds_from_event_key(event_key: str, wallclock_value: Any) -> float | None:
    seconds = time_to_seconds(wallclock_value)
    if seconds is None:
        return None
    parts = str(event_key).split("|")
    if len(parts) != 3:
        return None
    hour_token = parts[2].split(":", 1)[0]
    try:
        event_hour = int(hour_token)
    except ValueError:
        return None
    delta = seconds - float(event_hour * 3600)
    if delta < 0:
        delta += 86400.0
    return delta


def _safe_float(value: Any) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    token = str(value).strip().lower()
    if token in {"true", "1", "yes"}:
        return True
    if token in {"false", "0", "no", ""}:
        return False
    return bool(value)


def _wallclock_error_seconds(left: Any, right: Any) -> float | None:
    left_seconds = time_to_seconds(left)
    right_seconds = time_to_seconds(right)
    if left_seconds is None or right_seconds is None:
        return None
    delta = abs(left_seconds - right_seconds)
    return float(min(delta, 86400.0 - delta))


def _exact_match(predicted: Any, truth: Any) -> bool | pd.NA:
    if pd.isna(truth):
        return pd.NA
    if pd.isna(predicted):
        return False
    return bool(str(predicted) == str(truth))


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _metric_row(group: str, metric: str, value: Any, denominator: int | None = None) -> dict[str, Any]:
    return {
        "metric_group": group,
        "metric": metric,
        "value": value,
        "denominator": denominator,
    }


def _frame_to_markdown(frame: pd.DataFrame, *, columns: list[str]) -> str:
    if frame.empty:
        return "_None._"
    subset = frame.loc[:, columns].copy()
    headers = [str(column) for column in subset.columns]
    rows = [
        [_format_metric_value(value) for value in row]
        for row in subset.itertuples(index=False, name=None)
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    header_line = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    rule_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, rule_line, *body_lines])


def load_gemini_summary(summary_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(summary_csv)
    frame["file"] = frame["file"].astype("string").str.strip()
    frame["event_key"] = frame["file"].apply(lambda value: gemini_file_to_event_key(str(value)))
    invalid = frame.loc[frame["event_key"].isna(), "file"].tolist()
    if invalid:
        raise ValueError(f"Unable to derive event_key from Gemini files: {invalid}")

    frame["gemini_fall_detected"] = frame["fall_detected"].apply(_safe_bool)
    frame["gemini_prefall_location"] = frame["prefall_location"].apply(normalize_consensus_location)
    frame["gemini_last_furniture"] = frame["last_furniture"].apply(normalize_consensus_furniture)
    frame["gemini_fall_time_seconds"] = pd.to_numeric(frame.get("fall_time_seconds"), errors="coerce")
    frame["gemini_staff_entry_time_seconds"] = pd.to_numeric(
        frame.get("staff_entry_time_seconds"), errors="coerce"
    )
    frame["gemini_staff_response_time_seconds"] = pd.to_numeric(
        frame.get("staff_response_time_seconds"), errors="coerce"
    )
    frame["gemini_fall_tags_list"] = frame["fall_tags"].apply(normalize_gemini_tags)
    frame["gemini_fall_tags"] = frame["gemini_fall_tags_list"].apply(lambda values: ", ".join(values))
    frame["gemini_confidence"] = frame.get("confidence", pd.Series(index=frame.index, dtype="object")).astype(
        "string"
    )
    frame["gemini_fall_time_wallclock"] = frame.get(
        "fall_time_wallclock", pd.Series(index=frame.index, dtype="object")
    ).astype("string")
    frame["gemini_staff_entry_time_wallclock"] = frame.get(
        "staff_entry_time_wallclock", pd.Series(index=frame.index, dtype="object")
    ).astype("string")
    return frame


def load_gemini_records(results_jsonl: Path | None) -> pd.DataFrame:
    if results_jsonl is None or not results_jsonl.exists():
        return pd.DataFrame(columns=["file", "event_key", "model", "record_status", "error"])

    rows: list[dict[str, Any]] = []
    with results_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if not token:
                continue
            payload = json.loads(token)
            file_name = str(payload.get("file", "")).strip()
            rows.append(
                {
                    "file": file_name,
                    "event_key": gemini_file_to_event_key(file_name) if file_name else None,
                    "model": payload.get("model"),
                    "record_status": "error" if "error" in payload else "success",
                    "error": payload.get("error"),
                }
            )
    return pd.DataFrame(rows)


def load_truth_annotations(consensus_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(consensus_csv)
    truth = prepare_consensus_annotations(raw)
    no_signal_mask = (
        truth["consensus_status"].eq(CONSENSUS_STATUS_INCLUDED_FALL)
        & truth["event_key"].isin(_NO_SIGNAL_EVENT_KEYS)
    )
    if no_signal_mask.any():
        truth.loc[no_signal_mask, "consensus_status"] = CONSENSUS_STATUS_EXCLUDED_NO_SIGNAL
        truth.loc[no_signal_mask, "consensus_include_in_fall_cohort"] = False
    truth["gt_fall_time_seconds"] = truth.apply(
        lambda row: relative_seconds_from_event_key(row["event_key"], row["fall_time_consensus"]),
        axis=1,
    )
    truth["gt_response_time_seconds"] = truth.apply(
        lambda row: relative_seconds_from_event_key(row["event_key"], row["response_time_consensus"]),
        axis=1,
    )
    truth["gt_fall_tags_list"] = truth["fall_tags"].apply(normalize_consensus_tags)
    truth["truth_row_ordinal"] = truth.groupby("event_key", dropna=False).cumcount() + 1
    return truth


def _select_truth_row(summary_row: pd.Series, candidates: pd.DataFrame) -> tuple[pd.Series | None, str]:
    if candidates.empty:
        return None, "missing_truth"
    if len(candidates.index) == 1:
        return candidates.iloc[0], "direct_event_key"

    ranked = candidates.copy()
    summary_fall_seconds = _safe_float(summary_row.get("gemini_fall_time_seconds"))
    if summary_fall_seconds is not None:
        ranked["_distance"] = (
            pd.to_numeric(ranked["gt_fall_time_seconds"], errors="coerce") - summary_fall_seconds
        ).abs()
        ranked["_distance"] = ranked["_distance"].fillna(1e12)
        ranked = ranked.sort_values(
            ["_distance", "gt_fall_time_seconds", "truth_row_ordinal"],
            ascending=[True, True, True],
            na_position="last",
            kind="mergesort",
        )
        return ranked.iloc[0], "nearest_fall_time"

    ranked = ranked.sort_values(
        ["gt_fall_time_seconds", "gt_response_time_seconds", "truth_row_ordinal"],
        ascending=[True, True, True],
        na_position="last",
        kind="mergesort",
    )
    return ranked.iloc[0], "earliest_truth_fallback"


def build_event_level_comparison(
    summary: pd.DataFrame,
    truth: pd.DataFrame,
    records: pd.DataFrame | None = None,
) -> pd.DataFrame:
    record_models: dict[str, Any] = {}
    if records is not None and not records.empty:
        success_records = records.loc[records["record_status"] == "success", ["file", "model"]].drop_duplicates(
            subset=["file"], keep="last"
        )
        record_models = dict(zip(success_records["file"], success_records["model"], strict=False))

    rows: list[dict[str, Any]] = []
    for _, summary_row in summary.iterrows():
        event_key = str(summary_row["event_key"])
        candidates = truth.loc[truth["event_key"] == event_key].copy()
        selected_truth, match_source = _select_truth_row(summary_row, candidates)

        gemini_tags = set(summary_row["gemini_fall_tags_list"])
        gt_tags = set(selected_truth["gt_fall_tags_list"]) if selected_truth is not None else set()
        union = gemini_tags | gt_tags
        tag_jaccard = float(len(gemini_tags & gt_tags) / len(union)) if union else 1.0

        row: dict[str, Any] = {
            "file": summary_row["file"],
            "event_key": event_key,
            "model": record_models.get(str(summary_row["file"]), pd.NA),
            "match_source": match_source,
            "truth_annotation_count": int(len(candidates.index)),
            "gemini_fall_detected": bool(summary_row["gemini_fall_detected"]),
            "gemini_prefall_location": summary_row["gemini_prefall_location"],
            "gemini_last_furniture": summary_row["gemini_last_furniture"],
            "gemini_fall_time_seconds": _safe_float(summary_row["gemini_fall_time_seconds"]),
            "gemini_fall_time_wallclock": summary_row["gemini_fall_time_wallclock"],
            "gemini_staff_entry_time_seconds": _safe_float(summary_row["gemini_staff_entry_time_seconds"]),
            "gemini_staff_entry_time_wallclock": summary_row["gemini_staff_entry_time_wallclock"],
            "gemini_staff_response_time_seconds": _safe_float(
                summary_row["gemini_staff_response_time_seconds"]
            ),
            "gemini_fall_tags": summary_row["gemini_fall_tags"],
            "gemini_confidence": summary_row["gemini_confidence"],
            "gemini_predicted_offscreen": "offscreen" in gemini_tags,
            "gemini_predicted_no_fall": (not bool(summary_row["gemini_fall_detected"])) or ("no_fall" in gemini_tags),
        }

        if selected_truth is None:
            row.update(
                {
                    "consensus_status": pd.NA,
                    "gt_truth_row_ordinal": pd.NA,
                    "gt_prefall_location": pd.NA,
                    "gt_last_furniture": pd.NA,
                    "gt_fall_time_consensus": pd.NA,
                    "gt_response_time_consensus": pd.NA,
                    "gt_fall_time_seconds": pd.NA,
                    "gt_response_time_seconds": pd.NA,
                    "gt_fall_tags": pd.NA,
                    "location_match": pd.NA,
                    "furniture_match": pd.NA,
                    "tag_exact_match": pd.NA,
                    "tag_jaccard": pd.NA,
                    "fall_time_abs_error_seconds": pd.NA,
                    "response_time_abs_error_seconds": pd.NA,
                    "fall_time_wallclock_abs_error_seconds": pd.NA,
                    "response_time_wallclock_abs_error_seconds": pd.NA,
                }
            )
            rows.append(row)
            continue

        gt_last_furniture = selected_truth["last_furniture"]
        gt_prefall_location = selected_truth["prefall_location"]
        gt_fall_seconds = _safe_float(selected_truth["gt_fall_time_seconds"])
        gt_response_seconds = _safe_float(selected_truth["gt_response_time_seconds"])

        row.update(
            {
                "consensus_status": selected_truth["consensus_status"],
                "gt_truth_row_ordinal": int(selected_truth["truth_row_ordinal"]),
                "gt_prefall_location": gt_prefall_location,
                "gt_last_furniture": gt_last_furniture,
                "gt_fall_time_consensus": selected_truth["fall_time_consensus"],
                "gt_response_time_consensus": selected_truth["response_time_consensus"],
                "gt_fall_time_seconds": gt_fall_seconds,
                "gt_response_time_seconds": gt_response_seconds,
                "gt_fall_tags": selected_truth["fall_tags"],
                "location_match": _exact_match(summary_row["gemini_prefall_location"], gt_prefall_location),
                "furniture_match": _exact_match(summary_row["gemini_last_furniture"], gt_last_furniture),
                "tag_exact_match": gemini_tags == gt_tags,
                "tag_jaccard": tag_jaccard,
                "fall_time_abs_error_seconds": (
                    abs(_safe_float(summary_row["gemini_fall_time_seconds"]) - gt_fall_seconds)
                    if _safe_float(summary_row["gemini_fall_time_seconds"]) is not None and gt_fall_seconds is not None
                    else pd.NA
                ),
                "response_time_abs_error_seconds": (
                    abs(_safe_float(summary_row["gemini_staff_entry_time_seconds"]) - gt_response_seconds)
                    if _safe_float(summary_row["gemini_staff_entry_time_seconds"]) is not None
                    and gt_response_seconds is not None
                    else pd.NA
                ),
                "fall_time_wallclock_abs_error_seconds": _wallclock_error_seconds(
                    summary_row["gemini_fall_time_wallclock"], selected_truth["fall_time_consensus"]
                ),
                "response_time_wallclock_abs_error_seconds": _wallclock_error_seconds(
                    summary_row["gemini_staff_entry_time_wallclock"], selected_truth["response_time_consensus"]
                ),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _truth_status_by_event_key(truth: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event_key, subset in truth.groupby("event_key", dropna=False):
        status = min(
            subset["consensus_status"],
            key=lambda value: _LOCAL_STATUS_PRIORITY.get(str(value), 999),
        )
        rows.append({"event_key": str(event_key), "consensus_status": status})
    return pd.DataFrame(rows)


def build_coverage_table(
    truth: pd.DataFrame,
    comparison: pd.DataFrame,
    failures: pd.DataFrame,
) -> pd.DataFrame:
    truth_status = _truth_status_by_event_key(truth)
    success_keys = set(comparison["event_key"].dropna().astype(str))
    failure_keys: set[str] = set()
    if not failures.empty and "error" in failures.columns:
        failure_keys = set(
            failures.loc[failures["error"].notna(), "event_key"].dropna().astype(str)
        )

    rows: list[dict[str, Any]] = []
    statuses = [*HEADLINE_CONSENSUS_STATUSES, *DIAGNOSTIC_ONLY_STATUSES]
    for status in statuses:
        event_keys = set(
            truth_status.loc[truth_status["consensus_status"] == status, "event_key"].dropna().astype(str)
        )
        rows.append(
            {
                "consensus_status": status,
                "total_gt_event_keys": len(event_keys),
                "successful_event_keys": len(event_keys & success_keys),
                "failed_event_keys": len(event_keys & failure_keys),
                "missing_event_keys": len(event_keys - success_keys),
            }
        )

    all_keys = set(truth_status["event_key"].dropna().astype(str))
    rows.append(
        {
            "consensus_status": "all",
            "total_gt_event_keys": len(all_keys),
            "successful_event_keys": len(all_keys & success_keys),
            "failed_event_keys": len(all_keys & failure_keys),
            "missing_event_keys": len(all_keys - success_keys),
        }
    )
    return pd.DataFrame(rows)


def compute_comparison_metrics(comparison: pd.DataFrame) -> pd.DataFrame:
    included = comparison.loc[comparison["consensus_status"] == CONSENSUS_STATUS_INCLUDED_FALL].copy()
    accepted_nonfall = comparison.loc[
        comparison["consensus_status"] == CONSENSUS_STATUS_ACCEPTED_NONFALL
    ].copy()
    offscreen = comparison.loc[comparison["consensus_status"] == CONSENSUS_STATUS_EXCLUDED_OFFSCREEN].copy()

    metrics: list[dict[str, Any]] = [
        _metric_row("scope", "headline_denominator", len(included.index) + len(accepted_nonfall.index)),
        _metric_row("scope", "included_fall_rows", len(included.index)),
        _metric_row("scope", "accepted_nonfall_rows", len(accepted_nonfall.index)),
        _metric_row("scope", "diagnostic_only_rows", int(len(comparison.index) - len(included.index) - len(accepted_nonfall.index))),
    ]

    if not included.empty:
        location_match = pd.to_numeric(included["location_match"].map({True: 1.0, False: 0.0}), errors="coerce")
        metrics.append(
            _metric_row(
                "included_fall",
                "prefall_location_accuracy",
                float(location_match.dropna().mean()) if not location_match.dropna().empty else pd.NA,
                len(included.index),
            )
        )
        furniture_scored = included.loc[included["gt_last_furniture"].notna()].copy()
        furniture_match = pd.to_numeric(
            furniture_scored["furniture_match"].map({True: 1.0, False: 0.0}),
            errors="coerce",
        )
        metrics.append(
            _metric_row(
                "included_fall",
                "last_furniture_accuracy",
                float(furniture_match.dropna().mean()) if not furniture_match.dropna().empty else pd.NA,
                len(furniture_scored.index),
            )
        )
        fall_errors = pd.to_numeric(included["fall_time_abs_error_seconds"], errors="coerce").dropna()
        if not fall_errors.empty:
            metrics.extend(
                [
                    _metric_row("included_fall", "fall_time_mae_seconds", float(fall_errors.mean()), len(fall_errors.index)),
                    _metric_row(
                        "included_fall",
                        "fall_time_median_abs_error_seconds",
                        float(fall_errors.median()),
                        len(fall_errors.index),
                    ),
                    _metric_row(
                        "included_fall",
                        "fall_time_p90_abs_error_seconds",
                        float(fall_errors.quantile(0.9)),
                        len(fall_errors.index),
                    ),
                ]
            )
            for threshold in TIMING_THRESHOLDS_SECONDS:
                metrics.append(
                    _metric_row(
                        "included_fall",
                        f"fall_time_within_{threshold}s_rate",
                        float((fall_errors <= threshold).mean()),
                        len(fall_errors.index),
                    )
                )
        response_errors = pd.to_numeric(included["response_time_abs_error_seconds"], errors="coerce").dropna()
        if not response_errors.empty:
            metrics.extend(
                [
                    _metric_row(
                        "included_fall",
                        "response_time_mae_seconds",
                        float(response_errors.mean()),
                        len(response_errors.index),
                    ),
                    _metric_row(
                        "included_fall",
                        "response_time_median_abs_error_seconds",
                        float(response_errors.median()),
                        len(response_errors.index),
                    ),
                ]
            )
            for threshold in TIMING_THRESHOLDS_SECONDS:
                metrics.append(
                    _metric_row(
                        "included_fall",
                        f"response_time_within_{threshold}s_rate",
                        float((response_errors <= threshold).mean()),
                        len(response_errors.index),
                    )
                )
        metrics.append(
            _metric_row(
                "included_fall",
                "fall_tag_exact_match_rate",
                float(
                    pd.to_numeric(included["tag_exact_match"].map({True: 1.0, False: 0.0}), errors="coerce")
                    .dropna()
                    .mean()
                ),
                len(included.index),
            )
        )
        metrics.append(
            _metric_row(
                "included_fall",
                "fall_tag_jaccard_mean",
                float(pd.to_numeric(included["tag_jaccard"], errors="coerce").fillna(0.0).mean()),
                len(included.index),
            )
        )

    if not accepted_nonfall.empty:
        false_positive_series = pd.to_numeric(
            accepted_nonfall["gemini_fall_detected"].map({True: 1.0, False: 0.0}),
            errors="coerce",
        ).fillna(0.0)
        false_positive = int(false_positive_series.sum())
        metrics.extend(
            [
                _metric_row("accepted_nonfall", "false_positive_fall_detections", false_positive, len(accepted_nonfall.index)),
                _metric_row(
                    "accepted_nonfall",
                    "false_positive_rate",
                    float(false_positive_series.mean()),
                    len(accepted_nonfall.index),
                ),
            ]
        )

    if not offscreen.empty:
        visible_fall_series = pd.to_numeric(
            offscreen["gemini_fall_detected"].map({True: 1.0, False: 0.0}),
            errors="coerce",
        ).fillna(0.0)
        offscreen_tag_series = pd.to_numeric(
            offscreen["gemini_predicted_offscreen"].map({True: 1.0, False: 0.0}),
            errors="coerce",
        ).fillna(0.0)
        metrics.extend(
            [
                _metric_row(
                    "excluded_offscreen",
                    "predicted_visible_fall_rate",
                    float(visible_fall_series.mean()),
                    len(offscreen.index),
                ),
                _metric_row(
                    "excluded_offscreen",
                    "offscreen_tag_rate",
                    float(offscreen_tag_series.mean()),
                    len(offscreen.index),
                ),
            ]
        )

    return pd.DataFrame(metrics)


def build_failure_table(truth: pd.DataFrame, records: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    truth_status = _truth_status_by_event_key(truth)
    success_keys = set(comparison["event_key"].dropna().astype(str))
    truth_failures = truth_status.loc[~truth_status["event_key"].isin(success_keys)].copy()
    if records.empty:
        truth_failures["file"] = pd.NA
        truth_failures["model"] = pd.NA
        truth_failures["error"] = pd.NA
        return truth_failures.sort_values(["consensus_status", "event_key"], kind="mergesort").reset_index(drop=True)

    error_records = records.loc[records["record_status"] == "error", ["file", "event_key", "model", "error"]].copy()
    merged = truth_failures.merge(error_records, on="event_key", how="left")
    return merged.sort_values(["consensus_status", "event_key"], kind="mergesort").reset_index(drop=True)


def _latest_file(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern), key=lambda path: path.name)
    return matches[-1] if matches else None


def resolve_default_results_jsonl(summary_csv: Path) -> Path | None:
    summary_name = summary_csv.name
    if not summary_name.startswith("summary_") or not summary_name.endswith(".csv"):
        return None
    suffix = summary_name.removeprefix("summary_").removesuffix(".csv")
    candidate = summary_csv.with_name(f"results_{suffix}.jsonl")
    return candidate if candidate.exists() else None


def render_comparison_summary_markdown(
    *,
    summary_csv: Path,
    consensus_csv: Path,
    results_jsonl: Path | None,
    coverage: pd.DataFrame,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    failures: pd.DataFrame,
) -> str:
    metric_map = {
        (row["metric_group"], row["metric"]): row["value"]
        for row in metrics.to_dict(orient="records")
    }
    included = comparison.loc[comparison["consensus_status"] == CONSENSUS_STATUS_INCLUDED_FALL].copy()
    mismatch_table = included.sort_values(
        ["fall_time_abs_error_seconds", "response_time_abs_error_seconds"],
        ascending=[False, False],
        na_position="last",
        kind="mergesort",
    ).head(8)

    lines = [
        "# Gemini Video Summary vs GT Observations",
        "",
        "## Inputs",
        f"- Gemini summary CSV: `{summary_csv}`",
        f"- GT consensus CSV: `{consensus_csv}`",
        f"- Gemini results JSONL: `{results_jsonl}`" if results_jsonl is not None else "- Gemini results JSONL: _not provided_",
        "",
        "## Headline",
        (
            f"- Coverage: {int(coverage.loc[coverage['consensus_status'] == 'all', 'successful_event_keys'].iloc[0])}/"
            f"{int(coverage.loc[coverage['consensus_status'] == 'all', 'total_gt_event_keys'].iloc[0])} GT event keys "
            f"have successful Gemini outputs; {int(coverage.loc[coverage['consensus_status'] == 'all', 'missing_event_keys'].iloc[0])} are still missing."
        ),
        (
            f"- Headline denominator: {int(metric_map.get(('scope', 'headline_denominator'), 0))} rows "
            f"({int(metric_map.get(('scope', 'included_fall_rows'), 0))} included falls, "
            f"{int(metric_map.get(('scope', 'accepted_nonfall_rows'), 0))} accepted non-falls)."
        ),
        (
            f"- Included-fall pre-fall location accuracy: "
            f"{_format_metric_value(metric_map.get(('included_fall', 'prefall_location_accuracy'), pd.NA))}."
        ),
        (
            f"- Included-fall last-furniture accuracy (when GT populated): "
            f"{_format_metric_value(metric_map.get(('included_fall', 'last_furniture_accuracy'), pd.NA))}."
        ),
        (
            f"- Included-fall fall-time MAE / median abs error: "
            f"{_format_metric_value(metric_map.get(('included_fall', 'fall_time_mae_seconds'), pd.NA))}s / "
            f"{_format_metric_value(metric_map.get(('included_fall', 'fall_time_median_abs_error_seconds'), pd.NA))}s."
        ),
        (
            f"- Included-fall response-time MAE / median abs error: "
            f"{_format_metric_value(metric_map.get(('included_fall', 'response_time_mae_seconds'), pd.NA))}s / "
            f"{_format_metric_value(metric_map.get(('included_fall', 'response_time_median_abs_error_seconds'), pd.NA))}s."
        ),
        (
            f"- Included-fall tag exact-match rate / mean Jaccard: "
            f"{_format_metric_value(metric_map.get(('included_fall', 'fall_tag_exact_match_rate'), pd.NA))} / "
            f"{_format_metric_value(metric_map.get(('included_fall', 'fall_tag_jaccard_mean'), pd.NA))}."
        ),
        (
            f"- Accepted non-fall false-positive rate: "
            f"{_format_metric_value(metric_map.get(('accepted_nonfall', 'false_positive_rate'), pd.NA))}."
        ),
        (
            f"- Offscreen diagnostic: visible-fall prediction rate "
            f"{_format_metric_value(metric_map.get(('excluded_offscreen', 'predicted_visible_fall_rate'), pd.NA))}, "
            f"offscreen-tag rate {_format_metric_value(metric_map.get(('excluded_offscreen', 'offscreen_tag_rate'), pd.NA))}."
        ),
        "",
        "## Coverage by GT Status",
        _frame_to_markdown(
            coverage,
            columns=[
                "consensus_status",
                "total_gt_event_keys",
                "successful_event_keys",
                "failed_event_keys",
                "missing_event_keys",
            ],
        ),
        "",
        "## Largest Included-Fall Timing Mismatches",
        _frame_to_markdown(
            mismatch_table,
            columns=[
                "file",
                "gemini_prefall_location",
                "gt_prefall_location",
                "gemini_fall_time_seconds",
                "gt_fall_time_seconds",
                "fall_time_abs_error_seconds",
                "gemini_fall_tags",
                "gt_fall_tags",
            ],
        ),
        "",
        "## Missing / Failed GT Event Keys",
        _frame_to_markdown(
            failures.head(12),
            columns=["event_key", "consensus_status", "file", "model", "error"],
        ),
    ]
    return "\n".join(lines) + "\n"


def compare_gemini_summary_against_truth(
    *,
    summary_csv: Path,
    consensus_csv: Path,
    results_jsonl: Path | None = None,
) -> GeminiGroundTruthArtifacts:
    summary = load_gemini_summary(summary_csv)
    truth = load_truth_annotations(consensus_csv)
    records = load_gemini_records(results_jsonl)

    comparison = build_event_level_comparison(summary, truth, records)
    failures = build_failure_table(truth, records, comparison)
    coverage = build_coverage_table(truth, comparison, failures)
    metrics = compute_comparison_metrics(comparison)

    summary_payload = {
        "summary_csv": str(summary_csv),
        "consensus_csv": str(consensus_csv),
        "results_jsonl": str(results_jsonl) if results_jsonl is not None else None,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "comparison_rows": int(len(comparison.index)),
        "failed_event_keys": int(len(failures.index)),
        "headline_denominator": int(
            metrics.loc[metrics["metric"] == "headline_denominator", "value"].iloc[0]
        ),
    }
    return GeminiGroundTruthArtifacts(
        comparison=comparison,
        coverage=coverage,
        failures=failures,
        metrics=metrics,
        summary=summary_payload,
    )


def write_gemini_comparison_outputs(
    *,
    summary_csv: Path,
    consensus_csv: Path,
    results_jsonl: Path | None = None,
    output_dir: Path,
    report_prefix: str | None = None,
) -> tuple[GeminiGroundTruthArtifacts, dict[str, Path]]:
    artifacts = compare_gemini_summary_against_truth(
        summary_csv=summary_csv,
        consensus_csv=consensus_csv,
        results_jsonl=results_jsonl,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report_prefix or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    comparison_path = output_dir / f"gemini_gt_event_comparison_{prefix}.csv"
    coverage_path = output_dir / f"gemini_gt_coverage_{prefix}.csv"
    failures_path = output_dir / f"gemini_gt_failures_{prefix}.csv"
    metrics_path = output_dir / f"gemini_gt_metrics_{prefix}.csv"
    summary_path = output_dir / f"gemini_gt_summary_{prefix}.md"

    artifacts.comparison.to_csv(comparison_path, index=False)
    artifacts.coverage.to_csv(coverage_path, index=False)
    artifacts.failures.to_csv(failures_path, index=False)
    artifacts.metrics.to_csv(metrics_path, index=False)
    summary_path.write_text(
        render_comparison_summary_markdown(
            summary_csv=summary_csv,
            consensus_csv=consensus_csv,
            results_jsonl=results_jsonl,
            coverage=artifacts.coverage,
            metrics=artifacts.metrics,
            comparison=artifacts.comparison,
            failures=artifacts.failures,
        ),
        encoding="utf-8",
    )

    return artifacts, {
        "comparison_csv": comparison_path,
        "coverage_csv": coverage_path,
        "failures_csv": failures_path,
        "metrics_csv": metrics_path,
        "summary_md": summary_path,
    }


def resolve_default_summary_csv(project_root: Path) -> Path | None:
    return _latest_file(project_root / "outputs" / "gemini_fall_analysis", "summary_*.csv")
