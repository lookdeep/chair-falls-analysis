from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .gemini_eval import (
    CONSENSUS_STATUS_ACCEPTED_NONFALL,
    CONSENSUS_STATUS_EXCLUDED_BAD_OR_NO_VIDEO,
    CONSENSUS_STATUS_EXCLUDED_NO_SIGNAL,
    CONSENSUS_STATUS_EXCLUDED_OFFSCREEN,
    CONSENSUS_STATUS_INCLUDED_FALL,
    GeminiGroundTruthArtifacts,
    compare_gemini_summary_against_truth,
)

PRIMARY_MODELS = (
    "gemini-2.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
)
PRIMARY_VISIBLE_STATUSES = (
    CONSENSUS_STATUS_INCLUDED_FALL,
    CONSENSUS_STATUS_ACCEPTED_NONFALL,
)
DIAGNOSTIC_STATUSES = (
    CONSENSUS_STATUS_EXCLUDED_OFFSCREEN,
    CONSENSUS_STATUS_EXCLUDED_NO_SIGNAL,
    CONSENSUS_STATUS_EXCLUDED_BAD_OR_NO_VIDEO,
)


@dataclass(frozen=True)
class GeminiModelRun:
    model: str
    results_jsonl: Path
    summary_csv: Path
    timestamp: str
    success_rows: int
    error_rows: int


def _timestamp_from_results_path(path: Path) -> str:
    name = path.name
    if name.startswith("results_") and name.endswith(".jsonl"):
        return name.removeprefix("results_").removesuffix(".jsonl")
    raise ValueError(f"Unexpected Gemini results filename: {path}")


def _summarize_results_file(path: Path) -> tuple[str, int, int]:
    model_names: set[str] = set()
    success_rows = 0
    error_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if not token:
                continue
            payload = json.loads(token)
            model_name = str(payload.get("model", "")).strip()
            if model_name:
                model_names.add(model_name)
            if "error" in payload:
                error_rows += 1
            else:
                success_rows += 1
    if len(model_names) != 1:
        raise ValueError(f"Expected exactly one model in {path}, found {sorted(model_names)}")
    return next(iter(model_names)), success_rows, error_rows


def discover_latest_model_runs(
    output_dir: Path,
    *,
    target_models: tuple[str, ...] = PRIMARY_MODELS,
) -> list[GeminiModelRun]:
    latest: dict[str, GeminiModelRun] = {}
    for path in sorted(output_dir.glob("results_*.jsonl")):
        model_name, success_rows, error_rows = _summarize_results_file(path)
        if model_name not in target_models:
            continue
        timestamp = _timestamp_from_results_path(path)
        summary_csv = output_dir / f"summary_{timestamp}.csv"
        if not summary_csv.exists():
            continue
        latest[model_name] = GeminiModelRun(
            model=model_name,
            results_jsonl=path,
            summary_csv=summary_csv,
            timestamp=timestamp,
            success_rows=success_rows,
            error_rows=error_rows,
        )

    missing = [model for model in target_models if model not in latest]
    if missing:
        raise FileNotFoundError(f"Missing completed Gemini runs for models: {missing}")
    return [latest[model] for model in target_models]


def build_model_artifacts(
    runs: list[GeminiModelRun],
    *,
    consensus_csv: Path,
) -> dict[str, GeminiGroundTruthArtifacts]:
    artifacts: dict[str, GeminiGroundTruthArtifacts] = {}
    for run in runs:
        artifacts[run.model] = compare_gemini_summary_against_truth(
            summary_csv=run.summary_csv,
            results_jsonl=run.results_jsonl,
            consensus_csv=consensus_csv,
        )
    return artifacts


def _value_mean(series: pd.Series) -> float | pd.NA:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return pd.NA
    return float(values.mean())


def _bool_rate(series: pd.Series) -> float | pd.NA:
    values = pd.to_numeric(series.map({True: 1.0, False: 0.0}), errors="coerce").dropna()
    if values.empty:
        return pd.NA
    return float(values.mean())


def _metric_row(
    *,
    scope: str,
    model: str,
    metric: str,
    value: Any,
    denominator: int | None = None,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "model": model,
        "metric": metric,
        "value": value,
        "denominator": denominator,
    }


def compute_characterization_metrics(
    comparison: pd.DataFrame,
    *,
    model: str,
    scope: str,
) -> pd.DataFrame:
    visible = comparison.loc[comparison["consensus_status"].isin(PRIMARY_VISIBLE_STATUSES)].copy()
    visible_falls = visible.loc[visible["consensus_status"] == CONSENSUS_STATUS_INCLUDED_FALL].copy()
    visible_nonfalls = visible.loc[
        visible["consensus_status"] == CONSENSUS_STATUS_ACCEPTED_NONFALL
    ].copy()
    detected_visible_falls = visible_falls.loc[visible_falls["gemini_fall_detected"].fillna(False)].copy()

    rows: list[dict[str, Any]] = [
        _metric_row(scope=scope, model=model, metric="rows_total", value=int(len(comparison.index))),
        _metric_row(scope=scope, model=model, metric="visible_primary_rows", value=int(len(visible.index))),
        _metric_row(scope=scope, model=model, metric="visible_fall_rows", value=int(len(visible_falls.index))),
        _metric_row(
            scope=scope,
            model=model,
            metric="visible_nonfall_rows",
            value=int(len(visible_nonfalls.index)),
        ),
    ]

    tp = int(visible_falls["gemini_fall_detected"].fillna(False).sum())
    fn = int(len(visible_falls.index) - tp)
    fp = int(visible_nonfalls["gemini_fall_detected"].fillna(False).sum())
    tn = int(len(visible_nonfalls.index) - fp)
    rows.extend(
        [
            _metric_row(scope=scope, model=model, metric="tp", value=tp, denominator=len(visible_falls.index)),
            _metric_row(scope=scope, model=model, metric="fn", value=fn, denominator=len(visible_falls.index)),
            _metric_row(scope=scope, model=model, metric="fp", value=fp, denominator=len(visible_nonfalls.index)),
            _metric_row(scope=scope, model=model, metric="tn", value=tn, denominator=len(visible_nonfalls.index)),
            _metric_row(
                scope=scope,
                model=model,
                metric="visible_fall_sensitivity",
                value=float(tp / len(visible_falls.index)) if len(visible_falls.index) else pd.NA,
                denominator=len(visible_falls.index),
            ),
            _metric_row(
                scope=scope,
                model=model,
                metric="visible_nonfall_specificity",
                value=float(tn / len(visible_nonfalls.index)) if len(visible_nonfalls.index) else pd.NA,
                denominator=len(visible_nonfalls.index),
            ),
        ]
    )

    rows.extend(
        [
            _metric_row(
                scope=scope,
                model=model,
                metric="location_accuracy_all_visible_falls",
                value=_bool_rate(visible_falls["location_match"]),
                denominator=len(visible_falls.index),
            ),
            _metric_row(
                scope=scope,
                model=model,
                metric="location_accuracy_detected_visible_falls",
                value=_bool_rate(detected_visible_falls["location_match"]),
                denominator=len(detected_visible_falls.index),
            ),
        ]
    )

    visible_furniture = visible_falls.loc[visible_falls["gt_last_furniture"].notna()].copy()
    detected_furniture = detected_visible_falls.loc[
        detected_visible_falls["gt_last_furniture"].notna()
    ].copy()
    rows.extend(
        [
            _metric_row(
                scope=scope,
                model=model,
                metric="last_furniture_accuracy_all_visible_falls",
                value=_bool_rate(visible_furniture["furniture_match"]),
                denominator=len(visible_furniture.index),
            ),
            _metric_row(
                scope=scope,
                model=model,
                metric="last_furniture_accuracy_detected_visible_falls",
                value=_bool_rate(detected_furniture["furniture_match"]),
                denominator=len(detected_furniture.index),
            ),
        ]
    )

    rows.extend(
        [
            _metric_row(
                scope=scope,
                model=model,
                metric="fall_tag_exact_match_rate_detected_visible_falls",
                value=_bool_rate(detected_visible_falls["tag_exact_match"]),
                denominator=len(detected_visible_falls.index),
            ),
            _metric_row(
                scope=scope,
                model=model,
                metric="fall_tag_jaccard_mean_detected_visible_falls",
                value=_value_mean(detected_visible_falls["tag_jaccard"]),
                denominator=len(detected_visible_falls.index),
            ),
            _metric_row(
                scope=scope,
                model=model,
                metric="fall_time_mae_seconds_detected_visible_falls",
                value=_value_mean(detected_visible_falls["fall_time_abs_error_seconds"]),
                denominator=int(
                    pd.to_numeric(detected_visible_falls["fall_time_abs_error_seconds"], errors="coerce")
                    .dropna()
                    .shape[0]
                ),
            ),
            _metric_row(
                scope=scope,
                model=model,
                metric="response_time_mae_seconds_detected_visible_falls",
                value=_value_mean(detected_visible_falls["response_time_abs_error_seconds"]),
                denominator=int(
                    pd.to_numeric(detected_visible_falls["response_time_abs_error_seconds"], errors="coerce")
                    .dropna()
                    .shape[0]
                ),
            ),
        ]
    )
    return pd.DataFrame(rows)


def build_model_metrics(
    artifacts_by_model: dict[str, GeminiGroundTruthArtifacts],
) -> pd.DataFrame:
    success_sets = {
        model: set(artifacts.comparison["event_key"].dropna().astype(str))
        for model, artifacts in artifacts_by_model.items()
    }
    common_event_keys = sorted(set.intersection(*success_sets.values())) if success_sets else []

    frames: list[pd.DataFrame] = []
    for model, artifacts in artifacts_by_model.items():
        frames.append(
            compute_characterization_metrics(
                artifacts.comparison,
                model=model,
                scope="full_success_set",
            )
        )
        overlap = artifacts.comparison.loc[
            artifacts.comparison["event_key"].astype(str).isin(common_event_keys)
        ].copy()
        frames.append(
            compute_characterization_metrics(
                overlap,
                model=model,
                scope="three_way_overlap",
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_coverage_table(
    runs: list[GeminiModelRun],
    artifacts_by_model: dict[str, GeminiGroundTruthArtifacts],
) -> pd.DataFrame:
    success_sets = {
        model: set(artifacts.comparison["event_key"].dropna().astype(str))
        for model, artifacts in artifacts_by_model.items()
    }
    common_event_keys = set.intersection(*success_sets.values()) if success_sets else set()

    rows: list[dict[str, Any]] = []
    for run in runs:
        artifacts = artifacts_by_model[run.model]
        coverage = artifacts.coverage.copy()
        coverage["model"] = run.model
        coverage["results_jsonl"] = str(run.results_jsonl)
        coverage["summary_csv"] = str(run.summary_csv)
        coverage["successful_summary_rows"] = run.success_rows
        coverage["error_rows"] = run.error_rows
        coverage["three_way_overlap_event_keys"] = len(common_event_keys)
        rows.extend(coverage.to_dict(orient="records"))
    return pd.DataFrame(rows)


def build_event_long_table(
    artifacts_by_model: dict[str, GeminiGroundTruthArtifacts],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    success_sets = {
        model: set(artifacts.comparison["event_key"].dropna().astype(str))
        for model, artifacts in artifacts_by_model.items()
    }
    common_event_keys = set.intersection(*success_sets.values()) if success_sets else set()
    for model, artifacts in artifacts_by_model.items():
        frame = artifacts.comparison.copy()
        frame["study_model"] = model
        frame["in_three_way_overlap"] = frame["event_key"].astype(str).isin(common_event_keys)
        frame["visible_primary"] = frame["consensus_status"].isin(PRIMARY_VISIBLE_STATUSES)
        frame["diagnostic_only"] = frame["consensus_status"].isin(DIAGNOSTIC_STATUSES)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_overlap_wide_table(event_long: pd.DataFrame) -> pd.DataFrame:
    overlap = event_long.loc[event_long["in_three_way_overlap"]].copy()
    rows: list[dict[str, Any]] = []
    for event_key, group in overlap.groupby("event_key", dropna=False):
        row: dict[str, Any] = {
            "event_key": event_key,
            "consensus_status": group["consensus_status"].iloc[0],
            "gt_prefall_location": group["gt_prefall_location"].iloc[0],
            "gt_last_furniture": group["gt_last_furniture"].iloc[0],
            "gt_fall_tags": group["gt_fall_tags"].iloc[0],
        }
        for _, item in group.iterrows():
            model = str(item["study_model"])
            row[f"{model}__fall_detected"] = item["gemini_fall_detected"]
            row[f"{model}__prefall_location"] = item["gemini_prefall_location"]
            row[f"{model}__last_furniture"] = item["gemini_last_furniture"]
            row[f"{model}__fall_tags"] = item["gemini_fall_tags"]
            row[f"{model}__location_match"] = item["location_match"]
            row[f"{model}__furniture_match"] = item["furniture_match"]
            row[f"{model}__tag_exact_match"] = item["tag_exact_match"]
            row[f"{model}__tag_jaccard"] = item["tag_jaccard"]
        rows.append(row)
    return pd.DataFrame(rows)


def _iter_scope_frames(event_long: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "full_success_set", event_long.copy()
    yield "three_way_overlap", event_long.loc[event_long["in_three_way_overlap"]].copy()


def build_detection_confusion_table(event_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope_name, scoped_frame in _iter_scope_frames(event_long):
        visible = scoped_frame.loc[scoped_frame["visible_primary"]].copy()
        for model, group in visible.groupby("study_model", dropna=False):
            model_name = str(model)
            falls = group.loc[group["consensus_status"] == CONSENSUS_STATUS_INCLUDED_FALL]
            nonfalls = group.loc[group["consensus_status"] == CONSENSUS_STATUS_ACCEPTED_NONFALL]
            tp = int(falls["gemini_fall_detected"].fillna(False).sum())
            fn = int(len(falls.index) - tp)
            fp = int(nonfalls["gemini_fall_detected"].fillna(False).sum())
            tn = int(len(nonfalls.index) - fp)
            rows.append(
                {
                    "scope": scope_name,
                    "model": model_name,
                    "tp": tp,
                    "fn": fn,
                    "fp": fp,
                    "tn": tn,
                    "sensitivity": float(tp / len(falls.index)) if len(falls.index) else pd.NA,
                    "specificity": float(tn / len(nonfalls.index)) if len(nonfalls.index) else pd.NA,
                }
            )
    return pd.DataFrame(rows)


def build_location_confusion_table(event_long: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for scope_name, scoped_frame in _iter_scope_frames(event_long):
        visible_falls = scoped_frame.loc[
            scoped_frame["visible_primary"]
            & scoped_frame["consensus_status"].eq(CONSENSUS_STATUS_INCLUDED_FALL)
            & scoped_frame["gemini_fall_detected"].fillna(False)
        ].copy()
        if visible_falls.empty:
            continue
        visible_falls["scope"] = scope_name
        visible_falls["predicted_prefall_location"] = (
            visible_falls["gemini_prefall_location"].astype("string").fillna("missing")
        )
        frames.append(visible_falls)
    if not frames:
        return pd.DataFrame(
            columns=["scope", "model", "predicted_prefall_location", "gt_prefall_location", "count"]
        )
    rows = (
        pd.concat(frames, ignore_index=True)
        .groupby(
            ["scope", "study_model", "predicted_prefall_location", "gt_prefall_location"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .rename(columns={"study_model": "model"})
        .sort_values(
            ["scope", "model", "count", "predicted_prefall_location", "gt_prefall_location"],
            ascending=[True, True, False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return rows


def _split_tags(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    token = str(value).strip()
    if not token:
        return []
    return [part.strip() for part in token.split(",") if part.strip()]


def build_false_negative_tag_table(event_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope_name, scoped_frame in _iter_scope_frames(event_long):
        failures = scoped_frame.loc[
            scoped_frame["consensus_status"].eq(CONSENSUS_STATUS_INCLUDED_FALL)
            & ~scoped_frame["gemini_fall_detected"].fillna(False)
        ].copy()
        for _, row in failures.iterrows():
            tags = _split_tags(row["gt_fall_tags"])
            if not tags:
                rows.append(
                    {
                        "scope": scope_name,
                        "model": row["study_model"],
                        "gt_tag": "",
                        "count": 1,
                    }
                )
                continue
            for tag in tags:
                rows.append(
                    {
                        "scope": scope_name,
                        "model": row["study_model"],
                        "gt_tag": tag,
                        "count": 1,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["scope", "model", "gt_tag", "count"])
    return (
        pd.DataFrame(rows)
        .groupby(["scope", "model", "gt_tag"], dropna=False)["count"]
        .sum()
        .reset_index()
        .sort_values(["scope", "model", "count", "gt_tag"], ascending=[True, True, False, True], kind="mergesort")
        .reset_index(drop=True)
    )


def _format_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_None._"
    subset = frame.loc[:, columns].copy()
    headers = [str(column) for column in subset.columns]
    rows = [[_format_metric(value) for value in row] for row in subset.itertuples(index=False, name=None)]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    header_line = "| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |"
    rule_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, rule_line, *body])


def render_study_summary_markdown(
    *,
    runs: list[GeminiModelRun],
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    detection: pd.DataFrame,
    location_confusion: pd.DataFrame,
    false_negative_tags: pd.DataFrame,
) -> str:
    overlap_metrics = metrics.loc[metrics["scope"] == "three_way_overlap"].copy()
    overlap_pivot = overlap_metrics.pivot(index="model", columns="metric", values="value").reset_index()
    overlap_pivot = overlap_pivot.loc[
        :,
        [
            "model",
            "visible_fall_sensitivity",
            "visible_nonfall_specificity",
            "location_accuracy_detected_visible_falls",
            "last_furniture_accuracy_detected_visible_falls",
            "fall_tag_exact_match_rate_detected_visible_falls",
            "fall_tag_jaccard_mean_detected_visible_falls",
            "fall_time_mae_seconds_detected_visible_falls",
        ],
    ].sort_values(
        ["visible_fall_sensitivity", "location_accuracy_detected_visible_falls"],
        ascending=[False, False],
        kind="mergesort",
    )

    full_coverage = coverage.loc[coverage["consensus_status"] == "all", [
        "model",
        "successful_summary_rows",
        "error_rows",
        "three_way_overlap_event_keys",
    ]].drop_duplicates().reset_index(drop=True)

    top_location_confusion = location_confusion.loc[
        location_confusion["scope"] == "three_way_overlap"
    ].head(12)
    top_fn_tags = false_negative_tags.loc[
        false_negative_tags["scope"] == "three_way_overlap"
    ].head(12)

    lines = [
        "# Gemini Model Characterization Study",
        "",
        "## Study Design",
        "- One-off standalone characterization study using Gemini outputs and the v2 consensus GT.",
        "- Primary head-to-head view uses the 3-way successful overlap across all three Gemini models.",
        "- Full-set per-model coverage is reported separately to show availability and failure differences.",
        "- Visible reviewed clips are the headline denominator; offscreen, no-signal, and bad/no-video remain diagnostic strata.",
        "",
        "## Inputs",
    ]
    for run in runs:
        lines.append(
            f"- `{run.model}`: `{run.summary_csv.name}` and `{run.results_jsonl.name}` "
            f"({run.success_rows} successful rows, {run.error_rows} errors)"
        )
    lines.extend(
        [
            "",
            "## Primary 3-Way Overlap Ranking",
            _markdown_table(
                overlap_pivot,
                [
                    "model",
                    "visible_fall_sensitivity",
                    "visible_nonfall_specificity",
                    "location_accuracy_detected_visible_falls",
                    "last_furniture_accuracy_detected_visible_falls",
                    "fall_tag_exact_match_rate_detected_visible_falls",
                    "fall_tag_jaccard_mean_detected_visible_falls",
                    "fall_time_mae_seconds_detected_visible_falls",
                ],
            ),
            "",
            "## Coverage",
            _markdown_table(
                full_coverage,
                ["model", "successful_summary_rows", "error_rows", "three_way_overlap_event_keys"],
            ),
            "",
            "## Three-Way Overlap Detection Counts",
            _markdown_table(
                detection.loc[detection["scope"] == "three_way_overlap"].reset_index(drop=True),
                ["model", "tp", "fn", "fp", "tn", "sensitivity", "specificity"],
            ),
            "",
            "## Leading Location Confusions",
            _markdown_table(
                top_location_confusion,
                ["model", "predicted_prefall_location", "gt_prefall_location", "count"],
            ),
            "",
            "## Leading False-Negative GT Tags",
            _markdown_table(
                top_fn_tags,
                ["model", "gt_tag", "count"],
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_characterization_outputs(
    *,
    output_dir: Path,
    report_prefix: str,
    runs: list[GeminiModelRun],
    artifacts_by_model: dict[str, GeminiGroundTruthArtifacts],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = build_model_metrics(artifacts_by_model)
    coverage = build_coverage_table(runs, artifacts_by_model)
    event_long = build_event_long_table(artifacts_by_model)
    overlap_wide = build_overlap_wide_table(event_long)
    detection = build_detection_confusion_table(event_long)
    location_confusion = build_location_confusion_table(event_long)
    false_negative_tags = build_false_negative_tag_table(event_long)

    summary_md = output_dir / f"gemini_model_characterization_summary_{report_prefix}.md"
    metrics_csv = output_dir / f"gemini_model_characterization_metrics_{report_prefix}.csv"
    coverage_csv = output_dir / f"gemini_model_characterization_coverage_{report_prefix}.csv"
    event_long_csv = output_dir / f"gemini_model_characterization_events_long_{report_prefix}.csv"
    overlap_wide_csv = output_dir / f"gemini_model_characterization_overlap_{report_prefix}.csv"
    detection_csv = output_dir / f"gemini_model_characterization_detection_{report_prefix}.csv"
    location_confusion_csv = output_dir / f"gemini_model_characterization_location_confusion_{report_prefix}.csv"
    false_negative_tags_csv = output_dir / f"gemini_model_characterization_fn_tags_{report_prefix}.csv"

    metrics.to_csv(metrics_csv, index=False)
    coverage.to_csv(coverage_csv, index=False)
    event_long.to_csv(event_long_csv, index=False)
    overlap_wide.to_csv(overlap_wide_csv, index=False)
    detection.to_csv(detection_csv, index=False)
    location_confusion.to_csv(location_confusion_csv, index=False)
    false_negative_tags.to_csv(false_negative_tags_csv, index=False)
    summary_md.write_text(
        render_study_summary_markdown(
            runs=runs,
            metrics=metrics,
            coverage=coverage,
            detection=detection,
            location_confusion=location_confusion,
            false_negative_tags=false_negative_tags,
        ),
        encoding="utf-8",
    )
    return {
        "summary_md": summary_md,
        "metrics_csv": metrics_csv,
        "coverage_csv": coverage_csv,
        "events_long_csv": event_long_csv,
        "overlap_csv": overlap_wide_csv,
        "detection_csv": detection_csv,
        "location_confusion_csv": location_confusion_csv,
        "fn_tags_csv": false_negative_tags_csv,
    }


def default_report_prefix() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
