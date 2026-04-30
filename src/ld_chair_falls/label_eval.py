from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log
from pathlib import Path
from typing import Any

import pandas as pd
import statsmodels.api as sm

from .config import Settings
from .consensus import (
    CONSENSUS_STATUS_EXCLUDED_OFFSCREEN,
    CONSENSUS_STATUS_INCLUDED_FALL,
    assign_sequence_ids,
    included_fall_annotations,
    load_consensus_annotations,
    normalize_consensus_tags,
    parse_event_key,
    time_to_seconds,
)
from .prefall_location import build_panel_prefall_event_windows

LOCATION_CLASSES = ("chair", "bed", "room", "no_patient")
BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S = "primary_4class_departureaware_10s"
BENCHMARK_SPLIT_PRIMARY_4CLASS_V2_RAW_AUDIT = "primary_4class_v2_raw_audit"
BENCHMARK_SPLIT_SECONDARY_INFRAME_LEGACY = "secondary_inframe_legacy"
MODEL_VERSION_LEGACY = "legacy"
MODEL_VERSION_V2 = "v2"
MODEL_VERSION_V3 = "v3"
PREDICTION_REASON_DIRECT_FOCUS_ARGMAX = "direct_focus_argmax"
PREDICTION_REASON_RECENT_VISIBLE_FALLBACK = "recent_visible_fallback"
PREDICTION_REASON_STRONG_NO_PATIENT = "strong_no_patient"
PREDICTION_REASON_ZERO_SIGNAL_DEFAULT = "zero_signal_default"
PREDICTION_REASON_ROOM_RECOVERY_V2 = "room_recovery_v2"
PREDICTION_REASON_NO_DERIVED_MATCH = "no_derived_match"
DEPARTURE_AWARE_PRIMARY_SECONDS = 10.0


@dataclass
class LabelEvalArtifacts:
    sequence_metrics: pd.DataFrame
    confusion_matrix: pd.DataFrame
    probability_quality: pd.DataFrame
    auc_summary: pd.DataFrame
    threshold_sweep: pd.DataFrame
    calibration_curve: pd.DataFrame
    response_timing: pd.DataFrame
    population_stats: pd.DataFrame
    truth_prefall_location: pd.DataFrame
    tag_location_association: pd.DataFrame
    signal_location_profile: pd.DataFrame
    association_summary: pd.DataFrame
    shadow_model_metrics: pd.DataFrame
    shadow_model_predictions: pd.DataFrame
    shadow_model_comparison: pd.DataFrame
    onset_events: pd.DataFrame
    onset_case_crossover: pd.DataFrame
    shadow_feature_importance: pd.DataFrame
    shadow_ablation: pd.DataFrame
    shadow_window_sensitivity: pd.DataFrame
    shadow_cv_fold_metrics: pd.DataFrame
    operating_point: dict[str, Any]
    threshold_checks: dict[str, Any]
    furniture_origin_chain: pd.DataFrame
    post_departure_latency: pd.DataFrame
    tag_origin_chain_crosstab: pd.DataFrame
    benchmark_sequence_metrics: pd.DataFrame
    benchmark_label_metrics: pd.DataFrame
    benchmark_confusion_matrix: pd.DataFrame
    benchmark_probability_quality: pd.DataFrame
    sequence_predictions: pd.DataFrame
    candidate_comparison: pd.DataFrame
    benchmark_status: dict[str, Any]


def _csv_path(settings: Settings, relative_or_abs: Path) -> Path:
    if relative_or_abs.is_absolute():
        return relative_or_abs
    return settings.project_root / relative_or_abs


def _is_case_crossover_control_role(series: pd.Series) -> pd.Series:
    return series.astype("string").str.startswith("control_")


def _empty_artifacts() -> LabelEvalArtifacts:
    return LabelEvalArtifacts(
        sequence_metrics=pd.DataFrame(columns=["metric", "value"]),
        confusion_matrix=pd.DataFrame(columns=["truth_label", "predicted_label", "count"]),
        probability_quality=pd.DataFrame(columns=["metric", "value"]),
        auc_summary=pd.DataFrame(columns=["target_class", "roc_auc", "pr_auc", "support"]),
        threshold_sweep=pd.DataFrame(
            columns=[
                "target_class",
                "threshold",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
                "j_stat",
                "tp",
                "fp",
                "tn",
                "fn",
            ]
        ),
        calibration_curve=pd.DataFrame(
            columns=[
                "target_class",
                "bin_id",
                "bin_lower",
                "bin_upper",
                "avg_confidence",
                "empirical_accuracy",
                "n",
            ]
        ),
        response_timing=pd.DataFrame(columns=["metric", "value"]),
        population_stats=pd.DataFrame(columns=["metric", "value"]),
        truth_prefall_location=pd.DataFrame(columns=["prefall_location", "truth_rows", "pct_of_truth_rows"]),
        tag_location_association=pd.DataFrame(
            columns=[
                "prefall_location",
                "fall_tag",
                "annotations",
                "annotations_for_location",
                "annotations_for_tag",
                "pct_of_location",
                "pct_global",
                "lift_vs_global",
            ]
        ),
        signal_location_profile=pd.DataFrame(
            columns=[
                "truth_label",
                "sequences",
                "accuracy_within_label",
                "mean_prob_chair",
                "mean_prob_bed",
                "mean_prob_room",
                "mean_prob_no_patient",
                "truth_response_detect_rate",
                "derived_response_detect_rate",
                "mean_truth_latency_seconds",
                "mean_derived_latency_seconds",
            ]
        ),
        association_summary=pd.DataFrame(columns=["metric", "value"]),
        shadow_model_metrics=pd.DataFrame(columns=["model", "metric", "value"]),
        shadow_model_predictions=pd.DataFrame(
            columns=[
                "sequence_id",
                "monitor_id",
                "model",
                "repeat_id",
                "fold_id",
                "truth_label",
                "predicted_label",
                "pred_prob_chair",
                "pred_prob_bed",
                "pred_prob_room",
                "pred_prob_no_patient",
                "calibration_status",
                "fallback_reason",
            ]
        ),
        shadow_model_comparison=pd.DataFrame(columns=["metric", "baseline_value", "shadow_value", "delta"]),
        onset_events=pd.DataFrame(
            columns=[
                "fall_event_id",
                "window_role",
                "anchor_ts_utc",
                "sample_presence_ratio",
                "frame_count",
                "eligibility_reason_code",
                "onset_detected",
                "onset_channel",
                "first_signal_offset",
                "latency_to_anchor_seconds",
                "departure_source",
                "destination_label",
                "below_prevalence_threshold",
                "exploratory_status",
            ]
        ),
        onset_case_crossover=pd.DataFrame(
            columns=[
                "fall_event_id",
                "control_role",
                "hazard_onset_detected",
                "control_onset_detected",
                "hazard_latency_to_anchor_seconds",
                "control_latency_to_anchor_seconds",
                "hazard_onset_channel",
                "control_onset_channel",
                "hazard_minus_control_detected",
                "hazard_minus_control_latency_seconds",
                "exclude_from_statistics",
                "exploratory_status",
            ]
        ),
        shadow_feature_importance=pd.DataFrame(
            columns=[
                "model",
                "feature",
                "feature_family",
                "coefficient_abs_mean",
                "coefficient_abs_max",
                "exploratory_status",
            ]
        ),
        shadow_ablation=pd.DataFrame(
            columns=[
                "model",
                "metric",
                "mean",
                "ci_low",
                "ci_high",
                "evaluation_count",
                "feature_family_set",
                "feature_count",
                "calibration_status",
                "skip_reason",
                "exploratory_status",
            ]
        ),
        shadow_window_sensitivity=pd.DataFrame(
            columns=[
                "window_config",
                "metric",
                "mean",
                "ci_low",
                "ci_high",
                "evaluation_count",
                "calibration_status",
                "exploratory_status",
            ]
        ),
        shadow_cv_fold_metrics=pd.DataFrame(
            columns=[
                "model",
                "repeat_id",
                "fold_id",
                "metric",
                "value",
                "test_rows",
                "train_rows",
                "train_class_count",
                "fallback_reason",
                "calibration_status",
            ]
        ),
        operating_point={"objective": "balanced_macro_f1", "selected": [], "macro": {}},
        threshold_checks={
            "enabled": False,
            "evaluated": False,
            "checks": [],
            "overall_pass": None,
        },
        furniture_origin_chain=pd.DataFrame(columns=["furniture_origin_chain", "event_count", "pct_of_events"]),
        post_departure_latency=pd.DataFrame(columns=["last_furniture", "n_events", "latency_seconds_median", "latency_seconds_mean", "latency_seconds_p25", "latency_seconds_p75", "latency_seconds_min", "latency_seconds_max"]),
        tag_origin_chain_crosstab=pd.DataFrame(columns=["furniture_origin_chain", "fall_tag", "count", "pct_within_chain", "pct_global"]),
        benchmark_sequence_metrics=pd.DataFrame(
            columns=["benchmark_split", "model_version", "metric", "value"]
        ),
        benchmark_label_metrics=pd.DataFrame(
            columns=[
                "benchmark_split",
                "model_version",
                "label",
                "support",
                "predicted_count",
                "precision",
                "recall",
                "f1",
                "tp",
                "fp",
                "fn",
            ]
        ),
        benchmark_confusion_matrix=pd.DataFrame(
            columns=["benchmark_split", "model_version", "truth_label", "predicted_label", "count"]
        ),
        benchmark_probability_quality=pd.DataFrame(
            columns=["benchmark_split", "model_version", "metric", "value"]
        ),
        sequence_predictions=pd.DataFrame(
            columns=[
                "benchmark_split",
                "model_version",
                "sequence_id",
                "event_key",
                "monitor_id",
                "truth_label",
                "benchmark_truth_label",
                "benchmark_truth_label_raw",
                "derived_label",
                "derived_prob_chair",
                "derived_prob_bed",
                "derived_prob_room",
                "derived_prob_no_patient",
                "prediction_reason",
                "match_source",
                "has_derived_match",
                "derived_prob_sum",
                "departure_aware_truth_applied",
                "departure_aware_source_furniture",
                "departure_aware_latency_seconds",
                "offscreen_sequence",
                "scored_for_location",
                "sequence_instances",
            ]
        ),
        candidate_comparison=pd.DataFrame(
            columns=[
                "benchmark_split",
                "model_version",
                "scored_sequences",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "log_loss",
                "ece_10_bin",
                "f1_chair",
                "f1_bed",
                "f1_room",
                "f1_no_patient",
            ]
        ),
        benchmark_status={
            "primary_split": BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S,
            "secondary_split": BENCHMARK_SPLIT_SECONDARY_INFRAME_LEGACY,
            "selected_model_version": MODEL_VERSION_V3,
            "checks": [],
        },
    )


def _split_tags(value: str | None) -> list[str]:
    return normalize_consensus_tags(value)


def _parse_event_key(frame: pd.DataFrame) -> pd.DataFrame:
    return parse_event_key(frame)


def _time_to_seconds(value: str | None) -> float | None:
    return time_to_seconds(value)


def _assign_sequence_ids(consensus: pd.DataFrame) -> pd.DataFrame:
    return assign_sequence_ids(consensus)


def _classify_origin_chain(prefall_location: str, last_furniture: str) -> str:
    loc = str(prefall_location).strip().lower()
    furn = str(last_furniture).strip().lower() if pd.notna(last_furniture) else ""
    if loc == "chair":
        return "direct_chair"
    if loc == "bed":
        return "direct_bed"
    if loc in ("room", "no_patient") and furn == "chair":
        return "chair_origin_room"
    if loc in ("room", "no_patient") and furn == "bed":
        return "bed_origin_room"
    if loc in ("room", "no_patient") and furn == "floor":
        return "floor_origin_room"
    if loc in ("room", "no_patient") and furn:
        return "unknown_origin_room"
    if loc in ("room", "no_patient"):
        return "unknown_origin_room"
    return "other"


def _apply_benchmark_truth_policy(
    truth: pd.DataFrame,
    benchmark_split: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    result = truth.copy()
    raw_label = result["prefall_location"].astype("string").str.strip().str.lower()
    raw_label = raw_label.replace({"": pd.NA, "<na>": pd.NA, "nan": pd.NA}).fillna("no_patient")
    result["benchmark_truth_label_raw"] = raw_label
    result["benchmark_truth_label"] = raw_label
    result["departure_aware_truth_applied"] = False
    result["departure_aware_source_furniture"] = pd.NA
    result["departure_aware_latency_seconds"] = pd.NA

    diagnostics = {
        "offscreen_mapped_to_no_patient_rows": 0,
        "departure_aware_relabel_rows": 0,
    }
    if benchmark_split in {
        BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S,
        BENCHMARK_SPLIT_PRIMARY_4CLASS_V2_RAW_AUDIT,
    }:
        offscreen_mask = result["offscreen_flag"].fillna(False)
        result.loc[offscreen_mask, "benchmark_truth_label"] = "no_patient"
        diagnostics["offscreen_mapped_to_no_patient_rows"] = int(offscreen_mask.sum())

    if benchmark_split == BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S:
        last_furniture = result["last_furniture"].astype("string").str.strip().str.lower()
        latency = pd.to_numeric(result["post_departure_latency_seconds"], errors="coerce")
        departure_mask = (
            result["benchmark_truth_label_raw"].eq("room")
            & last_furniture.isin(["chair", "bed"])
            & latency.notna()
            & latency.ge(0.0)
            & latency.lt(DEPARTURE_AWARE_PRIMARY_SECONDS)
        )
        result.loc[departure_mask, "benchmark_truth_label"] = last_furniture.loc[departure_mask]
        result.loc[departure_mask, "departure_aware_truth_applied"] = True
        result.loc[departure_mask, "departure_aware_source_furniture"] = last_furniture.loc[departure_mask]
        result.loc[departure_mask, "departure_aware_latency_seconds"] = latency.loc[departure_mask]
        diagnostics["departure_aware_relabel_rows"] = int(departure_mask.sum())

    return result, diagnostics


def _furniture_origin_chain_summary(truth: pd.DataFrame) -> pd.DataFrame:
    if truth.empty or "furniture_origin_chain" not in truth.columns:
        return pd.DataFrame(columns=["furniture_origin_chain", "event_count", "pct_of_events"])
    counts = truth.groupby("furniture_origin_chain", dropna=False).size().reset_index(name="event_count")
    total = counts["event_count"].sum()
    counts["pct_of_events"] = (counts["event_count"] / max(total, 1) * 100).round(1)
    return counts.sort_values("event_count", ascending=False).reset_index(drop=True)


def _post_departure_latency_table(truth: pd.DataFrame) -> pd.DataFrame:
    cols = ["last_furniture", "n_events", "latency_seconds_median", "latency_seconds_mean",
            "latency_seconds_p25", "latency_seconds_p75", "latency_seconds_min", "latency_seconds_max"]
    if truth.empty or "post_departure_latency_seconds" not in truth.columns:
        return pd.DataFrame(columns=cols)
    valid = truth.loc[
        truth["post_departure_latency_seconds"].notna()
        & truth["last_furniture"].notna()
        & (truth["last_furniture"].astype(str).str.strip() != "")
    ].copy()
    if valid.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for furn, grp in valid.groupby("last_furniture"):
        lat = grp["post_departure_latency_seconds"].dropna()
        if lat.empty:
            continue
        rows.append({
            "last_furniture": furn,
            "n_events": int(len(lat)),
            "latency_seconds_median": round(float(lat.median()), 1),
            "latency_seconds_mean": round(float(lat.mean()), 1),
            "latency_seconds_p25": round(float(lat.quantile(0.25)), 1),
            "latency_seconds_p75": round(float(lat.quantile(0.75)), 1),
            "latency_seconds_min": round(float(lat.min()), 1),
            "latency_seconds_max": round(float(lat.max()), 1),
        })
    return pd.DataFrame(rows, columns=cols)


def _tag_origin_chain_crosstab(truth: pd.DataFrame) -> pd.DataFrame:
    cols = ["furniture_origin_chain", "fall_tag", "count", "pct_within_chain", "pct_global"]
    if truth.empty or not {"furniture_origin_chain", "fall_tags"}.issubset(truth.columns):
        return pd.DataFrame(columns=cols)
    frame = truth[["furniture_origin_chain", "fall_tags"]].copy()
    frame["fall_tag"] = frame["fall_tags"].apply(_split_tags)
    exploded = frame.explode("fall_tag", ignore_index=True)
    exploded["fall_tag"] = exploded["fall_tag"].astype("string").str.strip().str.lower()
    exploded = exploded.loc[exploded["fall_tag"].notna() & (exploded["fall_tag"] != "")].copy()
    if exploded.empty:
        return pd.DataFrame(columns=cols)
    counts = exploded.groupby(["furniture_origin_chain", "fall_tag"], dropna=False).size().reset_index(name="count")
    chain_totals = counts.groupby("furniture_origin_chain")["count"].transform("sum")
    counts["pct_within_chain"] = (counts["count"] / chain_totals.clip(lower=1) * 100).round(1)
    total = counts["count"].sum()
    counts["pct_global"] = (counts["count"] / max(total, 1) * 100).round(1)
    return counts.sort_values(["furniture_origin_chain", "count"], ascending=[True, False]).reset_index(drop=True)


def _prepare_truth_instances(truth: pd.DataFrame, summary: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if truth.empty:
        return truth, {
            **summary,
            "status": "ok",
            "rows": 0,
            "unique_event_keys": 0,
            "unique_monitors": 0,
            "expected_monitor_count": 0,
            "monitor_count_matches_expectation": False,
            "offscreen_rows": 0,
        }

    truth = truth.copy()
    fall_seconds = truth["fall_time_consensus"].apply(_time_to_seconds)
    resp_seconds = truth["response_time_consensus"].apply(_time_to_seconds)
    truth["fall_time_seconds"] = pd.to_numeric(fall_seconds, errors="coerce")
    truth["response_time_seconds"] = pd.to_numeric(resp_seconds, errors="coerce")
    truth["offscreen_flag"] = truth["fall_time_seconds"].isna()

    truth["fall_tags"] = truth["fall_tags"].astype("string").fillna("")
    updated_tags: list[str] = []
    for _, row in truth.iterrows():
        tags = _split_tags(row.get("fall_tags"))
        if bool(row.get("offscreen_flag")) and "offscreen" not in tags:
            tags.append("offscreen")
        updated_tags.append(", ".join(sorted(set(tags))))
    truth["fall_tags"] = pd.Series(updated_tags, index=truth.index, dtype="string")

    truth["last_furniture"] = (
        truth["last_furniture"].astype("string").str.strip().str.lower()
        if "last_furniture" in truth.columns
        else pd.Series(dtype="string", index=truth.index)
    )
    truth["last_furniture"] = truth["last_furniture"].replace({"": pd.NA, "<na>": pd.NA, "nan": pd.NA})

    if "furniture_departure_time" in truth.columns:
        truth["furniture_departure_time_raw"] = truth["furniture_departure_time"].astype("string")
        truth["furniture_departure_seconds"] = truth["furniture_departure_time_raw"].apply(_time_to_seconds)
    else:
        truth["furniture_departure_seconds"] = pd.Series(dtype="float64", index=truth.index)

    truth["post_departure_latency_seconds"] = truth["fall_time_seconds"] - truth["furniture_departure_seconds"]
    neg_mask = truth["post_departure_latency_seconds"] < 0
    truth.loc[neg_mask, "post_departure_latency_seconds"] = (
        truth.loc[neg_mask, "post_departure_latency_seconds"] + 86400.0
    )

    truth, benchmark_diagnostics = _apply_benchmark_truth_policy(
        truth,
        str(summary.get("benchmark_split", BENCHMARK_SPLIT_SECONDARY_INFRAME_LEGACY)),
    )

    truth["furniture_origin_chain"] = truth.apply(
        lambda row: _classify_origin_chain(
            row.get("prefall_location", ""),
            row.get("last_furniture", pd.NA),
        ),
        axis=1,
    )

    truth = _assign_sequence_ids(truth)
    truth["event_instance_id"] = (
        truth["event_key"].astype(str) + "#E" + truth["event_instance_ordinal"].astype(str).str.zfill(2)
    )

    for col in ["monitor_id"]:
        truth[col] = pd.to_numeric(truth[col], errors="coerce").astype("Int64")

    monitor_count = int(truth["monitor_id"].nunique(dropna=True)) if "monitor_id" in truth else 0
    return truth, {
        "status": "ok",
        "rows": int(len(truth.index)),
        "unique_event_keys": int(truth["event_key"].nunique(dropna=True)),
        "unique_monitors": monitor_count,
        "expected_monitor_count": int(summary.get("expected_monitor_count", 0)),
        "monitor_count_matches_expectation": (
            monitor_count == int(summary.get("expected_monitor_count", 0))
            if int(summary.get("expected_monitor_count", 0)) > 0
            else False
        ),
        "offscreen_rows": int(truth["offscreen_flag"].sum()),
        "benchmark_split": summary.get("benchmark_split"),
        "offscreen_mapped_to_no_patient_rows": int(
            benchmark_diagnostics.get(
                "offscreen_mapped_to_no_patient_rows",
                summary.get("offscreen_mapped_to_no_patient_rows", 0),
            )
        ),
        "departure_aware_relabel_rows": int(benchmark_diagnostics.get("departure_aware_relabel_rows", 0)),
    }


def _build_truth_instances(
    settings: Settings,
    *,
    benchmark_split: str = BENCHMARK_SPLIT_SECONDARY_INFRAME_LEGACY,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    consensus_path = _csv_path(settings, settings.fall_labels_consensus_csv_path)
    consensus, summary = load_consensus_annotations(consensus_path)
    if summary.get("status") != "ok":
        return pd.DataFrame(), summary

    truth_summary = {
        **summary,
        "expected_monitor_count": int(settings.label_eval_expected_monitor_count),
        "benchmark_split": benchmark_split,
        "offscreen_mapped_to_no_patient_rows": 0,
        "departure_aware_relabel_rows": 0,
    }
    if benchmark_split == BENCHMARK_SPLIT_SECONDARY_INFRAME_LEGACY:
        truth = _parse_event_key(included_fall_annotations(consensus))
    elif benchmark_split in {
        BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S,
        BENCHMARK_SPLIT_PRIMARY_4CLASS_V2_RAW_AUDIT,
    }:
        truth = consensus.loc[
            consensus["consensus_status"].isin(
                [CONSENSUS_STATUS_INCLUDED_FALL, CONSENSUS_STATUS_EXCLUDED_OFFSCREEN]
            )
        ].copy()
        truth = _parse_event_key(truth)
    else:
        raise ValueError(f"Unsupported benchmark_split: {benchmark_split}")
    return _prepare_truth_instances(truth, truth_summary)


def _match_with_derived(
    truth_instances: pd.DataFrame,
    event_windows: pd.DataFrame,
    *,
    extra_derived_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    if truth_instances.empty or event_windows.empty:
        return pd.DataFrame()

    derived = event_windows.copy()
    derived["monitor_id"] = pd.to_numeric(derived.get("monitor_id"), errors="coerce").astype("Int64")
    fall_ts_local = pd.to_datetime(derived.get("fall_ts_local"), errors="coerce", utc=True)
    fall_ts_utc = pd.to_datetime(derived.get("fall_ts_utc"), errors="coerce", utc=True)
    event_ts = fall_ts_local.fillna(fall_ts_utc)

    derived["date_local"] = event_ts.dt.strftime("%Y-%m-%d")
    derived["minute_local"] = event_ts.dt.strftime("%H:%M")
    derived["event_key"] = (
        derived["monitor_id"].astype("string").fillna("")
        + "|"
        + derived["date_local"].astype("string").fillna("")
        + "|"
        + derived["minute_local"].astype("string").fillna("")
    )

    derived = derived.sort_values(["event_key", "fall_ts_utc"], na_position="last", kind="mergesort").copy()
    derived["event_instance_ordinal"] = derived.groupby("event_key", dropna=False).cumcount() + 1
    fall_event_id = (
        pd.to_numeric(derived.get("fall_event_id"), errors="coerce")
        if "fall_event_id" in derived.columns
        else pd.Series(pd.NA, index=derived.index, dtype="Int64")
    )
    derived["fall_event_id"] = pd.Series(fall_event_id, index=derived.index).astype("Int64")

    for col in [
        "pre_prob_chair",
        "pre_prob_bed",
        "pre_prob_room",
        "pre_prob_no_patient",
        "response_latency_seconds",
    ]:
        derived[col] = pd.to_numeric(derived.get(col), errors="coerce")
    derived["response_detected"] = derived.get("response_detected", False).fillna(False).astype(bool)

    truth_cols = [
        "event_instance_id",
        "event_key",
        "sequence_id",
        "event_instance_ordinal",
        "prefall_location",
        "benchmark_truth_label",
        "benchmark_truth_label_raw",
        "departure_aware_truth_applied",
        "departure_aware_source_furniture",
        "departure_aware_latency_seconds",
        "offscreen_flag",
        "fall_time_seconds",
        "response_time_seconds",
        "monitor_id",
        "date_local",
        "minute_local",
    ]

    derived_cols = [
        "event_key",
        "event_instance_ordinal",
        "fall_event_id",
        "pre_prob_chair",
        "pre_prob_bed",
        "pre_prob_room",
        "pre_prob_no_patient",
        "response_detected",
        "response_latency_seconds",
    ]
    for col in extra_derived_columns:
        if col in derived.columns and col not in derived_cols:
            derived_cols.append(col)

    merged = truth_instances[truth_cols].merge(
        derived[derived_cols],
        on=["event_key", "event_instance_ordinal"],
        how="left",
    )
    merged["match_source"] = pd.NA

    def _has_match(frame: pd.DataFrame) -> pd.Series:
        indicators: list[pd.Series] = []
        for col in [
            "fall_event_id",
            "pre_prob_chair",
            "pre_prob_bed",
            "pre_prob_room",
            "pre_prob_no_patient",
            "response_latency_seconds",
        ]:
            if col in frame.columns:
                indicators.append(frame[col].notna())
        if not indicators:
            return pd.Series(False, index=frame.index)
        matched = indicators[0].copy()
        for indicator in indicators[1:]:
            matched |= indicator
        return matched.fillna(False)

    exact_match_mask = _has_match(merged)
    merged.loc[exact_match_mask, "match_source"] = "exact"

    derived_fill_cols = [
        "fall_event_id",
        "pre_prob_chair",
        "pre_prob_bed",
        "pre_prob_room",
        "pre_prob_no_patient",
        "response_detected",
        "response_latency_seconds",
    ]
    for col in extra_derived_columns:
        if col in merged.columns and col not in derived_fill_cols:
            derived_fill_cols.append(col)

    derived_by_event_key = {
        str(event_key): group.copy()
        for event_key, group in derived.groupby("event_key", dropna=False)
    }
    unmatched_indices = merged.index[~exact_match_mask].tolist()
    for idx in unmatched_indices:
        event_key = str(merged.at[idx, "event_key"])
        candidates = derived_by_event_key.get(event_key)
        if candidates is None or candidates.empty:
            continue
        if len(candidates.index) == 1:
            chosen = candidates.iloc[0]
            match_source = "single_event_key_fallback"
        else:
            truth_ordinal = pd.to_numeric(pd.Series([merged.at[idx, "event_instance_ordinal"]]), errors="coerce").iloc[0]
            ranked = candidates.copy()
            if pd.notna(truth_ordinal):
                ranked["_ordinal_distance"] = (
                    pd.to_numeric(ranked["event_instance_ordinal"], errors="coerce") - float(truth_ordinal)
                ).abs()
            else:
                ranked["_ordinal_distance"] = 9999.0
            chosen = ranked.sort_values(
                ["_ordinal_distance", "event_instance_ordinal"],
                na_position="last",
                kind="mergesort",
            ).iloc[0]
            match_source = "nearest_ordinal_fallback"
        for col in derived_fill_cols:
            if col in chosen.index:
                merged.at[idx, col] = chosen[col]
        merged.at[idx, "match_source"] = match_source

    merged["match_source"] = (
        merged["match_source"]
        .astype("string")
        .fillna("unmatched")
        .replace({"": "unmatched"})
    )

    for col in ["pre_prob_chair", "pre_prob_bed", "pre_prob_room", "pre_prob_no_patient"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    for col in extra_derived_columns:
        if col in merged.columns:
            merged[col] = merged[col].astype("string")

    return merged


def _build_sequence_truth_predictions(
    matched: pd.DataFrame,
    *,
    allow_offscreen_scoring: bool = False,
) -> pd.DataFrame:
    if matched.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for sequence_id, group in matched.groupby("sequence_id", dropna=False):
        group = group.sort_values("event_instance_ordinal", kind="mergesort")

        truth_label_col = "benchmark_truth_label" if "benchmark_truth_label" in group.columns else "prefall_location"
        last_label = str(group.iloc[-1][truth_label_col]).strip().lower()
        if last_label not in LOCATION_CLASSES:
            last_label = "no_patient"

        p_chair = float(pd.to_numeric(group["pre_prob_chair"], errors="coerce").fillna(0.0).sum())
        p_bed = float(pd.to_numeric(group["pre_prob_bed"], errors="coerce").fillna(0.0).sum())
        p_room = float(pd.to_numeric(group["pre_prob_room"], errors="coerce").fillna(0.0).sum())
        p_no_patient = float(pd.to_numeric(group["pre_prob_no_patient"], errors="coerce").fillna(0.0).sum())
        prob_sum = p_chair + p_bed + p_room + p_no_patient
        has_derived_match = False
        if "match_source" in group.columns:
            match_sources = (
                group["match_source"].astype("string").fillna("unmatched").str.strip()
            )
            has_derived_match = bool((match_sources != "unmatched").any())
        if prob_sum > 0:
            p_chair /= prob_sum
            p_bed /= prob_sum
            p_room /= prob_sum
            p_no_patient /= prob_sum

        derived_probs = {
            "chair": p_chair,
            "bed": p_bed,
            "room": p_room,
            "no_patient": p_no_patient,
        }
        if prob_sum > 0:
            derived_label = max(derived_probs, key=lambda key: derived_probs[key])
        else:
            derived_label = "no_patient"

        first_fall = pd.to_numeric(group["fall_time_seconds"], errors="coerce").dropna()
        first_fall_sec = float(first_fall.iloc[0]) if not first_fall.empty else None
        response_truth = pd.to_numeric(group["response_time_seconds"], errors="coerce").dropna()
        response_truth_sec = float(response_truth.iloc[0]) if not response_truth.empty else None

        truth_response_detected = response_truth_sec is not None
        truth_response_latency = None
        if first_fall_sec is not None and response_truth_sec is not None:
            truth_response_latency = response_truth_sec - first_fall_sec
            if truth_response_latency < 0:
                truth_response_latency += 86400.0

        response_detected_series = group["response_detected"].fillna(False).astype(bool)
        derived_detected = bool(response_detected_series.any())
        derived_latency = pd.to_numeric(
            group.loc[response_detected_series, "response_latency_seconds"], errors="coerce"
        ).dropna()
        derived_latency_sec = float(derived_latency.min()) if not derived_latency.empty else None

        any_offscreen = bool(group["offscreen_flag"].fillna(False).any())
        scored_for_location = allow_offscreen_scoring or (not any_offscreen)
        scored_for_response = (not any_offscreen) and (first_fall_sec is not None)
        prediction_reason = pd.NA
        if "prediction_reason" in group.columns:
            reasons = (
                group["prediction_reason"]
                .astype("string")
                .fillna("")
                .str.strip()
            )
            non_empty = reasons.loc[reasons != ""]
            if not non_empty.empty:
                prediction_reason = non_empty.iloc[-1]
        match_source = "unmatched"
        if "match_source" in group.columns:
            non_empty_sources = (
                group["match_source"].astype("string").fillna("unmatched").str.strip()
            )
            non_empty_sources = non_empty_sources.loc[non_empty_sources != ""]
            if not non_empty_sources.empty:
                non_unmatched = non_empty_sources.loc[non_empty_sources != "unmatched"]
                if not non_unmatched.empty:
                    match_source = str(non_unmatched.iloc[-1])
                else:
                    match_source = "unmatched"
        if prob_sum <= 0 and not has_derived_match:
            prediction_reason = PREDICTION_REASON_NO_DERIVED_MATCH
            scored_for_location = False
            scored_for_response = False

        truth_label_raw = pd.NA
        if "benchmark_truth_label_raw" in group.columns:
            raw_values = group["benchmark_truth_label_raw"].astype("string").fillna("").str.strip()
            non_empty = raw_values.loc[raw_values != ""]
            if not non_empty.empty:
                truth_label_raw = non_empty.iloc[-1]
        departure_aware_truth_applied = False
        if "departure_aware_truth_applied" in group.columns:
            departure_aware_truth_applied = bool(group["departure_aware_truth_applied"].fillna(False).astype(bool).any())
        departure_aware_source_furniture = pd.NA
        if "departure_aware_source_furniture" in group.columns:
            source_values = group["departure_aware_source_furniture"].astype("string").fillna("").str.strip()
            non_empty = source_values.loc[source_values != ""]
            if not non_empty.empty:
                departure_aware_source_furniture = non_empty.iloc[-1]
        departure_aware_latency_seconds = pd.NA
        if "departure_aware_latency_seconds" in group.columns:
            latencies = pd.to_numeric(group["departure_aware_latency_seconds"], errors="coerce").dropna()
            if not latencies.empty:
                departure_aware_latency_seconds = float(latencies.iloc[-1])

        rows.append(
            {
                "sequence_id": sequence_id,
                "event_key": group.iloc[0]["event_key"],
                "monitor_id": group.iloc[0]["monitor_id"],
                "truth_label": last_label,
                "benchmark_truth_label": last_label,
                "benchmark_truth_label_raw": truth_label_raw,
                "derived_label": derived_label,
                "derived_prob_chair": p_chair,
                "derived_prob_bed": p_bed,
                "derived_prob_room": p_room,
                "derived_prob_no_patient": p_no_patient,
                "derived_prob_sum": prob_sum,
                "truth_response_detected": truth_response_detected,
                "truth_response_latency_seconds": truth_response_latency,
                "derived_response_detected": derived_detected,
                "derived_response_latency_seconds": derived_latency_sec,
                "offscreen_sequence": any_offscreen,
                "scored_for_location": scored_for_location,
                "scored_for_response": scored_for_response,
                "sequence_instances": int(len(group.index)),
                "prediction_reason": prediction_reason,
                "match_source": match_source,
                "has_derived_match": has_derived_match,
                "departure_aware_truth_applied": departure_aware_truth_applied,
                "departure_aware_source_furniture": departure_aware_source_furniture,
                "departure_aware_latency_seconds": departure_aware_latency_seconds,
            }
        )

    return pd.DataFrame(rows)


def _confusion_matrix_rows(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty or not {"truth_label", "derived_label"}.issubset(scored.columns):
        rows = []
        for truth_label in LOCATION_CLASSES:
            for predicted_label in LOCATION_CLASSES:
                rows.append(
                    {
                        "truth_label": truth_label,
                        "predicted_label": predicted_label,
                        "count": 0,
                    }
                )
        return pd.DataFrame(rows)

    rows: list[dict[str, Any]] = []
    for truth_label in LOCATION_CLASSES:
        for predicted_label in LOCATION_CLASSES:
            count = int(
                (
                    (scored["truth_label"].astype("string") == truth_label)
                    & (scored["derived_label"].astype("string") == predicted_label)
                ).sum()
            )
            rows.append(
                {
                    "truth_label": truth_label,
                    "predicted_label": predicted_label,
                    "count": count,
                }
            )
    return pd.DataFrame(rows)


def _classification_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(columns=["metric", "value"])

    truth = scored["truth_label"].astype("string")
    pred = scored["derived_label"].astype("string")
    total = int(len(scored.index))
    accuracy = float((truth == pred).mean()) if total else 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for label in LOCATION_CLASSES:
        tp = int(((truth == label) & (pred == label)).sum())
        fp = int(((truth != label) & (pred == label)).sum())
        fn = int(((truth == label) & (pred != label)).sum())
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    rows = [
        {"metric": "scored_sequences", "value": float(total)},
        {"metric": "accuracy", "value": round(accuracy, 6)},
        {"metric": "macro_precision", "value": round(float(sum(precisions) / len(precisions)), 6)},
        {"metric": "macro_recall", "value": round(float(sum(recalls) / len(recalls)), 6)},
        {"metric": "macro_f1", "value": round(float(sum(f1s) / len(f1s)), 6)},
    ]
    return pd.DataFrame(rows)


def _label_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "label",
        "support",
        "predicted_count",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
    ]
    if scored.empty:
        return pd.DataFrame(columns=columns)

    truth = scored["truth_label"].astype("string")
    pred = scored["derived_label"].astype("string")
    rows: list[dict[str, Any]] = []
    for label in LOCATION_CLASSES:
        tp = int(((truth == label) & (pred == label)).sum())
        fp = int(((truth != label) & (pred == label)).sum())
        fn = int(((truth == label) & (pred != label)).sum())
        support = int((truth == label).sum())
        predicted_count = int((pred == label).sum())
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
        rows.append(
            {
                "label": label,
                "support": float(support),
                "predicted_count": float(predicted_count),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _multiclass_log_loss(scored: pd.DataFrame) -> float:
    if scored.empty:
        return 0.0

    eps = 1e-15
    truth_idx = {
        "chair": 0,
        "bed": 1,
        "room": 2,
        "no_patient": 3,
    }
    losses: list[float] = []
    for _, row in scored.iterrows():
        truth = str(row.get("truth_label", ""))
        if truth not in truth_idx:
            continue
        probs = {
            "chair": float(row.get("derived_prob_chair", 0.0) or 0.0),
            "bed": float(row.get("derived_prob_bed", 0.0) or 0.0),
            "room": float(row.get("derived_prob_room", 0.0) or 0.0),
            "no_patient": float(row.get("derived_prob_no_patient", 0.0) or 0.0),
        }
        p = max(eps, min(1.0 - eps, probs.get(truth, 0.0)))
        losses.append(-float(log(p)))

    if not losses:
        return 0.0
    return float(sum(losses) / len(losses))


def _multiclass_brier(scored: pd.DataFrame) -> float:
    if scored.empty:
        return 0.0

    total = 0.0
    n = 0
    for _, row in scored.iterrows():
        truth = str(row.get("truth_label", ""))
        truth_vec = {
            "chair": 1.0 if truth == "chair" else 0.0,
            "bed": 1.0 if truth == "bed" else 0.0,
            "room": 1.0 if truth == "room" else 0.0,
            "no_patient": 1.0 if truth == "no_patient" else 0.0,
        }
        pred_vec = {
            "chair": float(row.get("derived_prob_chair", 0.0) or 0.0),
            "bed": float(row.get("derived_prob_bed", 0.0) or 0.0),
            "room": float(row.get("derived_prob_room", 0.0) or 0.0),
            "no_patient": float(row.get("derived_prob_no_patient", 0.0) or 0.0),
        }
        total += sum((pred_vec[label] - truth_vec[label]) ** 2 for label in LOCATION_CLASSES) / float(len(LOCATION_CLASSES))
        n += 1
    return float(total / n) if n else 0.0


def _ece(scored: pd.DataFrame, bins: int = 10) -> float:
    if scored.empty:
        return 0.0

    frame = scored.copy()
    frame["pred_conf"] = frame[["derived_prob_chair", "derived_prob_bed", "derived_prob_room", "derived_prob_no_patient"]].max(axis=1)
    frame["correct"] = (frame["truth_label"].astype("string") == frame["derived_label"].astype("string")).astype(float)

    ece = 0.0
    n = float(len(frame.index))
    for idx in range(bins):
        low = idx / bins
        high = (idx + 1) / bins
        if idx < bins - 1:
            bucket = frame.loc[(frame["pred_conf"] >= low) & (frame["pred_conf"] < high)]
        else:
            bucket = frame.loc[(frame["pred_conf"] >= low) & (frame["pred_conf"] <= high)]
        if bucket.empty:
            continue
        bucket_conf = float(bucket["pred_conf"].mean())
        bucket_acc = float(bucket["correct"].mean())
        ece += abs(bucket_conf - bucket_acc) * (len(bucket.index) / n)
    return float(ece)


def _probability_quality_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(columns=["metric", "value"])

    rows = [
        {"metric": "log_loss", "value": round(_multiclass_log_loss(scored), 6)},
        {"metric": "brier_score_multiclass", "value": round(_multiclass_brier(scored), 6)},
        {"metric": "ece_10_bin", "value": round(_ece(scored, bins=10), 6)},
    ]
    return pd.DataFrame(rows)


def _binary_roc_pr_auc(truth: pd.Series, score: pd.Series) -> tuple[float, float]:
    y = pd.to_numeric(truth, errors="coerce").fillna(0).astype(int)
    s = pd.to_numeric(score, errors="coerce").fillna(0.0).astype(float)
    frame = pd.DataFrame({"y": y, "s": s}).sort_values("s", ascending=False, kind="mergesort").reset_index(drop=True)

    pos = int((frame["y"] == 1).sum())
    neg = int((frame["y"] == 0).sum())
    if pos == 0 or neg == 0:
        return 0.0, 0.0

    tp = 0
    fp = 0
    roc_points: list[tuple[float, float]] = [(0.0, 0.0)]
    pr_points: list[tuple[float, float]] = [(0.0, float(pos / len(frame.index)))]
    idx = 0
    while idx < len(frame.index):
        threshold = float(frame.iloc[idx]["s"])
        while idx < len(frame.index) and float(frame.iloc[idx]["s"]) == threshold:
            if int(frame.iloc[idx]["y"]) == 1:
                tp += 1
            else:
                fp += 1
            idx += 1
        tpr = float(tp / pos)
        fpr = float(fp / neg)
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = tpr
        roc_points.append((fpr, tpr))
        pr_points.append((recall, precision))

    if roc_points[-1] != (1.0, 1.0):
        roc_points.append((1.0, 1.0))

    roc_auc = 0.0
    for i in range(1, len(roc_points)):
        x0, y0 = roc_points[i - 1]
        x1, y1 = roc_points[i]
        roc_auc += (x1 - x0) * ((y0 + y1) / 2.0)

    pr_auc = 0.0
    pr_sorted = sorted(pr_points, key=lambda p: p[0])
    for i in range(1, len(pr_sorted)):
        r0, p0 = pr_sorted[i - 1]
        r1, p1 = pr_sorted[i]
        pr_auc += (r1 - r0) * ((p0 + p1) / 2.0)

    return float(max(0.0, min(1.0, roc_auc))), float(max(0.0, min(1.0, pr_auc)))


def _auc_summary(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(columns=["target_class", "roc_auc", "pr_auc", "support"])

    rows: list[dict[str, Any]] = []
    truth = scored["truth_label"].astype("string")
    for label in LOCATION_CLASSES:
        truth_binary = (truth == label).astype(int)
        score = pd.to_numeric(scored[f"derived_prob_{label}"], errors="coerce").fillna(0.0)
        roc_auc, pr_auc = _binary_roc_pr_auc(truth_binary, score)
        rows.append(
            {
                "target_class": label,
                "roc_auc": round(roc_auc, 6),
                "pr_auc": round(pr_auc, 6),
                "support": float(int(truth_binary.sum())),
            }
        )
    return pd.DataFrame(rows)


def _threshold_row(y: pd.Series, score: pd.Series, threshold: float) -> dict[str, float]:
    pred = score >= threshold
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    tn = int(((y == 0) & (~pred)).sum())
    fn = int(((y == 1) & (~pred)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    tnr = float(tn / (tn + fp)) if (tn + fp) else 0.0
    balanced_accuracy = float((recall + tnr) / 2.0)
    j_stat = float(recall - (1.0 - tnr))
    return {
        "threshold": float(round(threshold, 6)),
        "precision": float(round(precision, 6)),
        "recall": float(round(recall, 6)),
        "f1": float(round(f1, 6)),
        "balanced_accuracy": float(round(balanced_accuracy, 6)),
        "j_stat": float(round(j_stat, 6)),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def _threshold_sweep(scored: pd.DataFrame, sweep_points: int) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(
            columns=[
                "target_class",
                "threshold",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
                "j_stat",
                "tp",
                "fp",
                "tn",
                "fn",
            ]
        )

    points = max(3, int(sweep_points))
    thresholds = [i / float(points - 1) for i in range(points)]
    truth = scored["truth_label"].astype("string")
    rows: list[dict[str, Any]] = []
    for label in LOCATION_CLASSES:
        y = (truth == label).astype(int)
        score = pd.to_numeric(scored[f"derived_prob_{label}"], errors="coerce").fillna(0.0)
        for threshold in thresholds:
            row = _threshold_row(y, score, threshold)
            row["target_class"] = label
            rows.append(row)
    return pd.DataFrame(rows)[
        [
            "target_class",
            "threshold",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "j_stat",
            "tp",
            "fp",
            "tn",
            "fn",
        ]
    ]


def _calibration_curve(scored: pd.DataFrame, bins: int) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(
            columns=[
                "target_class",
                "bin_id",
                "bin_lower",
                "bin_upper",
                "avg_confidence",
                "empirical_accuracy",
                "n",
            ]
        )

    n_bins = max(3, int(bins))
    truth = scored["truth_label"].astype("string")
    rows: list[dict[str, Any]] = []
    for label in LOCATION_CLASSES:
        y = (truth == label).astype(int)
        score = pd.to_numeric(scored[f"derived_prob_{label}"], errors="coerce").fillna(0.0)
        for idx in range(n_bins):
            low = idx / n_bins
            high = (idx + 1) / n_bins
            if idx < n_bins - 1:
                mask = (score >= low) & (score < high)
            else:
                mask = (score >= low) & (score <= high)
            bucket_score = score.loc[mask]
            bucket_truth = y.loc[mask]
            n = int(len(bucket_score.index))
            avg_conf = float(bucket_score.mean()) if n else pd.NA
            emp_acc = float(bucket_truth.mean()) if n else pd.NA
            rows.append(
                {
                    "target_class": label,
                    "bin_id": float(idx),
                    "bin_lower": float(round(low, 6)),
                    "bin_upper": float(round(high, 6)),
                    "avg_confidence": round(avg_conf, 6) if n else pd.NA,
                    "empirical_accuracy": round(emp_acc, 6) if n else pd.NA,
                    "n": float(n),
                }
            )
    return pd.DataFrame(rows)


def _operating_point(settings: Settings, sweep: pd.DataFrame) -> dict[str, Any]:
    objective = settings.label_eval_operating_objective
    if sweep.empty:
        return {"objective": objective, "selected": [], "macro": {}}

    selected: list[dict[str, Any]] = []
    for label in LOCATION_CLASSES:
        candidates = sweep.loc[sweep["target_class"] == label].copy()
        if candidates.empty:
            continue
        candidates["f1_gap"] = (candidates["precision"] - candidates["recall"]).abs()
        candidates = candidates.sort_values(
            ["f1", "balanced_accuracy", "f1_gap", "threshold"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        best = candidates.iloc[0]
        selected.append(
            {
                "target_class": label,
                "threshold": float(best["threshold"]),
                "precision": float(best["precision"]),
                "recall": float(best["recall"]),
                "f1": float(best["f1"]),
                "balanced_accuracy": float(best["balanced_accuracy"]),
                "j_stat": float(best["j_stat"]),
            }
        )

    macro = {}
    if selected:
        keys = ["precision", "recall", "f1", "balanced_accuracy", "j_stat"]
        macro = {
            f"macro_{key}": round(float(sum(float(item[key]) for item in selected) / len(selected)), 6)
            for key in keys
        }
    return {"objective": objective, "selected": selected, "macro": macro}


def _response_metrics(sequences: pd.DataFrame) -> pd.DataFrame:
    if sequences.empty:
        return pd.DataFrame(columns=["metric", "value"])

    scored = sequences.loc[sequences["scored_for_response"]].copy()
    if scored.empty:
        return pd.DataFrame(columns=["metric", "value"])

    truth = scored["truth_response_detected"].fillna(False).astype(bool)
    pred = scored["derived_response_detected"].fillna(False).astype(bool)

    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0

    latency = scored.loc[
        truth & pred
        & pd.to_numeric(scored["truth_response_latency_seconds"], errors="coerce").notna()
        & pd.to_numeric(scored["derived_response_latency_seconds"], errors="coerce").notna()
    ].copy()
    if latency.empty:
        mae = None
        p50 = None
        p90 = None
    else:
        err = (
            pd.to_numeric(latency["derived_response_latency_seconds"], errors="coerce")
            - pd.to_numeric(latency["truth_response_latency_seconds"], errors="coerce")
        ).abs()
        mae = float(err.mean())
        p50 = float(err.quantile(0.50))
        p90 = float(err.quantile(0.90))

    rows = [
        {"metric": "scored_sequences", "value": float(len(scored.index))},
        {"metric": "detection_precision", "value": round(precision, 6)},
        {"metric": "detection_recall", "value": round(recall, 6)},
        {"metric": "detection_f1", "value": round(f1, 6)},
        {"metric": "latency_mae_seconds", "value": round(mae, 6) if mae is not None else pd.NA},
        {"metric": "latency_abs_error_p50_seconds", "value": round(p50, 6) if p50 is not None else pd.NA},
        {"metric": "latency_abs_error_p90_seconds", "value": round(p90, 6) if p90 is not None else pd.NA},
    ]
    return pd.DataFrame(rows)


def _population_stats(truth_instances: pd.DataFrame, sequences: pd.DataFrame, truth_summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"metric": "truth_rows", "value": float(truth_summary.get("rows", 0))},
        {"metric": "truth_unique_event_keys", "value": float(truth_summary.get("unique_event_keys", 0))},
        {"metric": "truth_unique_monitors", "value": float(truth_summary.get("unique_monitors", 0))},
        {
            "metric": "truth_monitor_count_matches_expectation",
            "value": float(1.0 if truth_summary.get("monitor_count_matches_expectation") else 0.0),
        },
        {"metric": "truth_offscreen_rows", "value": float(truth_summary.get("offscreen_rows", 0))},
    ]

    if not sequences.empty:
        rows.extend(
            [
                {"metric": "sequence_count", "value": float(len(sequences.index))},
                {"metric": "sequence_offscreen_count", "value": float(int(sequences["offscreen_sequence"].sum()))},
                {
                    "metric": "sequence_mean_instances",
                    "value": round(float(pd.to_numeric(sequences["sequence_instances"], errors="coerce").mean()), 6),
                },
                {
                    "metric": "sequence_scored_for_location",
                    "value": float(int(sequences["scored_for_location"].sum())),
                },
                {
                    "metric": "sequence_scored_for_response",
                    "value": float(int(sequences["scored_for_response"].sum())),
                },
            ]
        )

    truth_labels = truth_instances["prefall_location"].astype("string").str.lower()
    for label in LOCATION_CLASSES:
        rows.append({"metric": f"truth_label_count_{label}", "value": float(int((truth_labels == label).sum()))})

    return pd.DataFrame(rows)


def _truth_prefall_location_summary(truth_instances: pd.DataFrame) -> pd.DataFrame:
    if truth_instances.empty or "prefall_location" not in truth_instances.columns:
        return pd.DataFrame(columns=["prefall_location", "truth_rows", "pct_of_truth_rows"])

    frame = truth_instances.copy()
    frame["prefall_location"] = frame["prefall_location"].astype("string").str.strip().str.lower()
    counts = (
        frame.groupby("prefall_location", dropna=False)
        .size()
        .rename("truth_rows")
        .reset_index()
    )
    total = float(counts["truth_rows"].sum())
    counts["pct_of_truth_rows"] = (counts["truth_rows"] / total).round(4) if total > 0 else 0.0
    counts = counts.sort_values(["truth_rows", "prefall_location"], ascending=[False, True], kind="mergesort")
    return counts[["prefall_location", "truth_rows", "pct_of_truth_rows"]]


def _tag_location_association(truth_instances: pd.DataFrame, min_tag_count: int = 3) -> pd.DataFrame:
    columns = [
        "prefall_location",
        "fall_tag",
        "annotations",
        "annotations_for_location",
        "annotations_for_tag",
        "pct_of_location",
        "pct_global",
        "lift_vs_global",
    ]
    if truth_instances.empty or not {"prefall_location", "fall_tags"}.issubset(truth_instances.columns):
        return pd.DataFrame(columns=columns)

    frame = truth_instances[["prefall_location", "fall_tags"]].copy()
    frame["prefall_location"] = frame["prefall_location"].astype("string").str.strip().str.lower()
    frame["fall_tag"] = frame["fall_tags"].apply(_split_tags)
    exploded = frame.explode("fall_tag", ignore_index=True)
    exploded["fall_tag"] = exploded["fall_tag"].astype("string").str.strip().str.lower()
    exploded = exploded.loc[exploded["fall_tag"].notna() & (exploded["fall_tag"] != "")].copy()
    if exploded.empty:
        return pd.DataFrame(columns=columns)

    tag_totals = exploded.groupby("fall_tag", dropna=False).size().rename("annotations_for_tag")
    keep_tags = tag_totals.loc[tag_totals >= int(min_tag_count)].index.tolist()
    exploded = exploded.loc[exploded["fall_tag"].isin(keep_tags)].copy()
    if exploded.empty:
        return pd.DataFrame(columns=columns)

    loc_totals = (
        exploded.groupby("prefall_location", dropna=False).size().rename("annotations_for_location").reset_index()
    )
    tag_totals_df = tag_totals.loc[keep_tags].rename_axis("fall_tag").reset_index()

    pairs = (
        exploded.groupby(["prefall_location", "fall_tag"], dropna=False)
        .size()
        .rename("annotations")
        .reset_index()
    )
    pairs = pairs.merge(loc_totals, on="prefall_location", how="left")
    pairs = pairs.merge(tag_totals_df, on="fall_tag", how="left")

    total_annotations = float(len(exploded.index))
    pairs["pct_of_location"] = (
        pairs["annotations"] / pairs["annotations_for_location"].replace(0, pd.NA)
    ).fillna(0.0)
    pairs["pct_global"] = (
        pairs["annotations_for_tag"] / total_annotations
    ) if total_annotations > 0 else 0.0
    pairs["lift_vs_global"] = (
        pairs["pct_of_location"] / pairs["pct_global"].replace(0, pd.NA)
    ).fillna(0.0)

    for col in ["pct_of_location", "pct_global", "lift_vs_global"]:
        pairs[col] = pd.to_numeric(pairs[col], errors="coerce").fillna(0.0).round(4)

    pairs = pairs.sort_values(
        ["annotations", "lift_vs_global", "prefall_location", "fall_tag"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    return pairs[columns]


def _tag_location_cramers_v(tag_assoc: pd.DataFrame) -> float | None:
    if tag_assoc.empty:
        return None
    required = {"prefall_location", "fall_tag", "annotations"}
    if not required.issubset(tag_assoc.columns):
        return None

    matrix = tag_assoc.pivot_table(
        index="prefall_location",
        columns="fall_tag",
        values="annotations",
        aggfunc="sum",
        fill_value=0.0,
    )
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None

    observed = matrix.to_numpy(dtype=float)
    n = float(observed.sum())
    if n <= 1.0:
        return None
    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n
    safe_expected = expected.copy()
    safe_expected[safe_expected == 0] = 1.0
    chi2 = float((((observed - expected) ** 2) / safe_expected).sum())

    r, k = observed.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (((k - 1) * (r - 1)) / (n - 1.0)))
    r_corr = r - (((r - 1) ** 2) / (n - 1.0))
    k_corr = k - (((k - 1) ** 2) / (n - 1.0))
    denom = min(k_corr - 1.0, r_corr - 1.0)
    if denom <= 0:
        return None
    return float((phi2_corr / denom) ** 0.5)


def _signal_location_profile(sequences: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "truth_label",
        "sequences",
        "accuracy_within_label",
        "mean_prob_chair",
        "mean_prob_bed",
        "mean_prob_room",
        "mean_prob_no_patient",
        "truth_response_detect_rate",
        "derived_response_detect_rate",
        "mean_truth_latency_seconds",
        "mean_derived_latency_seconds",
    ]
    if sequences.empty:
        return pd.DataFrame(columns=columns)

    scored_location = sequences.loc[sequences["scored_for_location"]].copy()
    if scored_location.empty:
        return pd.DataFrame(columns=columns)

    scored_location["truth_label"] = scored_location["truth_label"].astype("string").str.lower()
    scored_location["derived_label"] = scored_location["derived_label"].astype("string").str.lower()
    scored_location["correct"] = (scored_location["truth_label"] == scored_location["derived_label"]).astype(float)

    base = (
        scored_location.groupby("truth_label", dropna=False)
        .agg(
            sequences=("truth_label", "size"),
            accuracy_within_label=("correct", "mean"),
            mean_prob_chair=("derived_prob_chair", "mean"),
            mean_prob_bed=("derived_prob_bed", "mean"),
            mean_prob_room=("derived_prob_room", "mean"),
            mean_prob_no_patient=("derived_prob_no_patient", "mean"),
        )
        .reset_index()
    )

    scored_response = sequences.loc[sequences["scored_for_response"]].copy()
    if not scored_response.empty:
        scored_response["truth_label"] = scored_response["truth_label"].astype("string").str.lower()
        response = (
            scored_response.groupby("truth_label", dropna=False)
            .agg(
                truth_response_detect_rate=("truth_response_detected", "mean"),
                derived_response_detect_rate=("derived_response_detected", "mean"),
                mean_truth_latency_seconds=("truth_response_latency_seconds", "mean"),
                mean_derived_latency_seconds=("derived_response_latency_seconds", "mean"),
            )
            .reset_index()
        )
        base = base.merge(response, on="truth_label", how="left")
    else:
        base["truth_response_detect_rate"] = pd.NA
        base["derived_response_detect_rate"] = pd.NA
        base["mean_truth_latency_seconds"] = pd.NA
        base["mean_derived_latency_seconds"] = pd.NA

    for col in [
        "accuracy_within_label",
        "mean_prob_chair",
        "mean_prob_bed",
        "mean_prob_room",
        "mean_prob_no_patient",
        "truth_response_detect_rate",
        "derived_response_detect_rate",
        "mean_truth_latency_seconds",
        "mean_derived_latency_seconds",
    ]:
        base[col] = pd.to_numeric(base[col], errors="coerce").round(4)

    base = base.sort_values(["sequences", "truth_label"], ascending=[False, True], kind="mergesort")
    return base[columns]


def _association_summary(
    truth_prefall_location: pd.DataFrame,
    tag_location_association: pd.DataFrame,
    signal_location_profile: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    truth_rows = float(pd.to_numeric(truth_prefall_location.get("truth_rows"), errors="coerce").fillna(0).sum())
    rows.append({"metric": "truth_prefall_rows", "value": truth_rows})

    if not truth_prefall_location.empty and {"prefall_location", "truth_rows"}.issubset(truth_prefall_location.columns):
        ranked = truth_prefall_location.sort_values(["truth_rows", "prefall_location"], ascending=[False, True], kind="mergesort")
        top_raw = pd.to_numeric(ranked.iloc[0]["truth_rows"], errors="coerce")
        top_count = float(top_raw) if pd.notna(top_raw) else 0.0
        rows.append({"metric": "top_truth_location_count", "value": top_count})
        rows.append({"metric": "top_truth_location_share", "value": round((top_count / truth_rows), 6) if truth_rows > 0 else 0.0})

    tag_total = float(pd.to_numeric(tag_location_association.get("annotations"), errors="coerce").fillna(0).sum())
    rows.append({"metric": "tag_annotations_total", "value": tag_total})
    rows.append(
        {
            "metric": "distinct_tags_retained",
            "value": float(
                int(tag_location_association["fall_tag"].astype("string").nunique(dropna=True))
                if "fall_tag" in tag_location_association.columns
                else 0
            ),
        }
    )

    c_v = _tag_location_cramers_v(tag_location_association)
    rows.append({"metric": "tag_location_cramers_v", "value": round(c_v, 6) if c_v is not None else pd.NA})

    rows.append(
        {
            "metric": "signal_profile_rows",
            "value": float(int(len(signal_location_profile.index))),
        }
    )

    if not signal_location_profile.empty and "accuracy_within_label" in signal_location_profile.columns:
        rows.append(
            {
                "metric": "signal_profile_mean_accuracy",
                "value": round(
                    float(pd.to_numeric(signal_location_profile["accuracy_within_label"], errors="coerce").mean()),
                    6,
                ),
            }
        )

    return pd.DataFrame(rows, columns=["metric", "value"])


def _normalize_probability_dict(probs: dict[str, float]) -> dict[str, float]:
    clean = {
        label: max(0.0, _safe_float(probs.get(label, 0.0)) or 0.0)
        for label in LOCATION_CLASSES
    }
    total = float(sum(clean.values()))
    if total <= 0:
        return {
            "chair": 0.0,
            "bed": 0.0,
            "room": 0.0,
            "no_patient": 1.0,
        }
    return {label: clean[label] / total for label in LOCATION_CLASSES}


def _panel_visible_location_distribution(panel: pd.DataFrame, half_life_seconds: int) -> dict[str, float]:
    visible = panel.loc[panel["frame_has_location_signal"].fillna(False).astype(bool)].copy()
    if visible.empty:
        return {"chair": 0.0, "bed": 0.0, "room": 0.0}

    if int(half_life_seconds) > 0:
        visible["recency_weight"] = 2.0 ** (
            pd.to_numeric(visible["second_offset"], errors="coerce").fillna(-999.0) / float(half_life_seconds)
        )
    else:
        visible["recency_weight"] = 1.0

    distance_totals: dict[str, float] = {}
    for label, col in [
        ("chair", "patient_chair_distance"),
        ("bed", "patient_bed_distance"),
        ("room", "patient_room_distance"),
    ]:
        distance = pd.to_numeric(visible[col], errors="coerce")
        inv = (
            visible["recency_weight"]
            / distance.where(distance > 1e-6, other=1e-6)
        ).where(distance.notna(), other=0.0)
        distance_totals[label] = float(inv.sum())

    distance_sum = sum(distance_totals.values())
    if distance_sum > 0:
        distance_probs = {label: distance_totals[label] / distance_sum for label in ("chair", "bed", "room")}
    else:
        distance_probs = {label: 0.0 for label in ("chair", "bed", "room")}

    dominant = (
        visible["dominant_location_label"]
        .astype("string")
        .fillna("no_patient")
        .str.strip()
        .str.lower()
    )
    dominant_counts = dominant.value_counts(normalize=True)
    dominant_probs = {
        label: float(dominant_counts.get(label, 0.0))
        for label in ("chair", "bed", "room")
    }
    dominant_sum = sum(dominant_probs.values())
    if distance_sum > 0 and dominant_sum > 0:
        combined = {
            label: (0.7 * distance_probs[label]) + (0.3 * dominant_probs[label])
            for label in ("chair", "bed", "room")
        }
    elif distance_sum > 0:
        combined = distance_probs
    else:
        combined = dominant_probs

    total = sum(combined.values())
    if total <= 0:
        return {"chair": 0.0, "bed": 0.0, "room": 0.0}
    return {label: combined[label] / total for label in ("chair", "bed", "room")}


def _v2_panel_summary(settings: Settings, second_level_panel: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "fall_event_id",
        "v2_focus_visible_frames",
        "v2_dropout_visible_frames",
        "v2_last_visible_gap_seconds",
        "v2_focus_prob_chair",
        "v2_focus_prob_bed",
        "v2_focus_prob_room",
        "v2_dropout_prob_chair",
        "v2_dropout_prob_bed",
        "v2_dropout_prob_room",
        "v2_room_support_flag",
    ]
    if second_level_panel.empty:
        return pd.DataFrame(columns=columns)

    required = {
        "fall_event_id",
        "second_offset",
        "frame_has_location_signal",
        "patient_chair_distance",
        "patient_bed_distance",
        "patient_room_distance",
        "dominant_location_label",
    }
    if not required.issubset(second_level_panel.columns):
        return pd.DataFrame(columns=columns)

    panel = second_level_panel.copy()
    panel["fall_event_id"] = pd.to_numeric(panel["fall_event_id"], errors="coerce").astype("Int64")
    panel["second_offset"] = pd.to_numeric(panel["second_offset"], errors="coerce")
    panel["frame_has_location_signal"] = panel["frame_has_location_signal"].fillna(False).astype(bool)
    panel = panel.loc[
        panel["fall_event_id"].notna()
        & panel["second_offset"].notna()
        & (panel["second_offset"] < 0)
        & (panel["second_offset"] >= -int(settings.fall_window_dropout_seconds))
    ].copy()
    if panel.empty:
        return pd.DataFrame(columns=columns)

    for col in ["patient_chair_distance", "patient_bed_distance", "patient_room_distance"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel["dominant_location_label"] = (
        panel["dominant_location_label"].astype("string").fillna("no_patient").str.strip().str.lower()
    )

    rows: list[dict[str, Any]] = []
    for fall_event_id, group in panel.groupby("fall_event_id", dropna=False):
        focus = group.loc[group["second_offset"] >= -int(settings.fall_window_focus_seconds)].copy()
        focus_visible = focus.loc[focus["frame_has_location_signal"]].copy()
        dropout_visible = group.loc[group["frame_has_location_signal"]].copy()
        focus_probs = _panel_visible_location_distribution(
            focus_visible,
            settings.label_eval_v2_recency_half_life_seconds,
        )
        dropout_probs = _panel_visible_location_distribution(
            dropout_visible,
            settings.label_eval_v2_recency_half_life_seconds,
        )
        room_support = bool(
            (focus_visible["dominant_location_label"].eq("room")).any()
            or (dropout_visible["dominant_location_label"].eq("room")).any()
            or focus_probs["room"] > 0
            or dropout_probs["room"] > 0
        )
        last_gap = pd.NA
        if not dropout_visible.empty:
            last_gap = float(abs(float(dropout_visible["second_offset"].max())))

        rows.append(
            {
                "fall_event_id": int(fall_event_id),
                "v2_focus_visible_frames": float(len(focus_visible.index)),
                "v2_dropout_visible_frames": float(len(dropout_visible.index)),
                "v2_last_visible_gap_seconds": last_gap,
                "v2_focus_prob_chair": round(focus_probs["chair"], 6),
                "v2_focus_prob_bed": round(focus_probs["bed"], 6),
                "v2_focus_prob_room": round(focus_probs["room"], 6),
                "v2_dropout_prob_chair": round(dropout_probs["chair"], 6),
                "v2_dropout_prob_bed": round(dropout_probs["bed"], 6),
                "v2_dropout_prob_room": round(dropout_probs["room"], 6),
                "v2_room_support_flag": room_support,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _build_v2_event_windows(
    settings: Settings,
    event_windows: pd.DataFrame,
    second_level_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if event_windows.empty:
        return pd.DataFrame()

    v2 = event_windows.copy()
    for col in [
        "pre_prob_chair",
        "pre_prob_bed",
        "pre_prob_room",
        "pre_prob_no_patient",
        "pre_focus_visible_frames",
        "pre_dropout_visible_frames",
        "pre_state_visible_prob",
        "pre_dropout_last_visible_gap_seconds",
    ]:
        v2[col] = pd.to_numeric(v2.get(col), errors="coerce")

    summary = _v2_panel_summary(settings, second_level_panel if second_level_panel is not None else pd.DataFrame())
    if not summary.empty and "fall_event_id" in v2.columns:
        v2["fall_event_id"] = pd.to_numeric(v2.get("fall_event_id"), errors="coerce").astype("Int64")
        v2 = v2.merge(summary, on="fall_event_id", how="left")
    else:
        for col in summary.columns:
            if col != "fall_event_id":
                v2[col] = pd.NA

    rows: list[dict[str, Any]] = []
    for _, row in v2.iterrows():
        probs = _normalize_probability_dict(
            {
                "chair": row.get("pre_prob_chair", 0.0),
                "bed": row.get("pre_prob_bed", 0.0),
                "room": row.get("pre_prob_room", 0.0),
                "no_patient": row.get("pre_prob_no_patient", 0.0),
            }
        )

        legacy_focus_visible = int(_safe_float(row.get("pre_focus_visible_frames")) or 0.0)
        panel_focus_visible = int(_safe_float(row.get("v2_focus_visible_frames")) or 0.0)
        focus_visible = max(legacy_focus_visible, panel_focus_visible)
        dropout_visible = int(
            max(
                _safe_float(row.get("pre_dropout_visible_frames")) or 0.0,
                _safe_float(row.get("v2_dropout_visible_frames")) or 0.0,
            )
        )
        last_gap = row.get("v2_last_visible_gap_seconds")
        if pd.isna(last_gap):
            last_gap = row.get("pre_dropout_last_visible_gap_seconds")
        last_gap = pd.to_numeric(pd.Series([last_gap]), errors="coerce").iloc[0]

        fallback_probs = _normalize_probability_dict(
            {
                "chair": row.get("v2_dropout_prob_chair", 0.0),
                "bed": row.get("v2_dropout_prob_bed", 0.0),
                "room": row.get("v2_dropout_prob_room", 0.0),
                "no_patient": 0.0,
            }
        )
        visible_fallback = {label: fallback_probs[label] for label in ("chair", "bed", "room")}
        fallback_sum = sum(visible_fallback.values())
        if fallback_sum > 0:
            visible_fallback = {
                label: visible_fallback[label] / fallback_sum
                for label in ("chair", "bed", "room")
            }

        focus_probs = _normalize_probability_dict(
            {
                "chair": row.get("v2_focus_prob_chair", 0.0),
                "bed": row.get("v2_focus_prob_bed", 0.0),
                "room": row.get("v2_focus_prob_room", 0.0),
                "no_patient": 0.0,
            }
        )
        visible_focus = {label: focus_probs[label] for label in ("chair", "bed", "room")}
        focus_sum = sum(visible_focus.values())
        if focus_sum > 0:
            visible_focus = {
                label: visible_focus[label] / focus_sum
                for label in ("chair", "bed", "room")
            }

        reason = PREDICTION_REASON_DIRECT_FOCUS_ARGMAX
        if dropout_visible <= 0 and focus_visible <= 0:
            probs = {"chair": 0.0, "bed": 0.0, "room": 0.0, "no_patient": 1.0}
            reason = PREDICTION_REASON_ZERO_SIGNAL_DEFAULT
        elif panel_focus_visible > 0 and focus_sum > 0:
            max_visible_focus = max(visible_focus.values())
            visible_strength = max(
                float(settings.label_eval_v2_recent_visible_min_strength),
                float(pd.to_numeric(pd.Series([row.get("pre_state_visible_prob")]), errors="coerce").fillna(0.0).iloc[0]),
            )
            visible_strength = max(
                visible_strength,
                min(0.90, 0.20 + (0.75 * max_visible_focus)),
            )
            if (
                max_visible_focus >= 0.70
                and (pd.isna(last_gap) or float(last_gap) <= float(settings.label_eval_v2_recent_visible_gap_seconds))
            ):
                visible_strength = max(visible_strength, 0.75)
            if pd.notna(last_gap) and float(last_gap) > float(settings.label_eval_v2_recent_visible_gap_seconds):
                visible_strength = min(visible_strength, 0.50)
            visible_strength = min(0.90, max(0.35, visible_strength))
            target_no_patient = min(probs["no_patient"], max(0.0, 1.0 - visible_strength))
            if (
                legacy_focus_visible <= 0
                and max_visible_focus >= 0.70
                and (pd.isna(last_gap) or float(last_gap) <= float(settings.label_eval_v2_recent_visible_gap_seconds))
            ):
                target_no_patient = min(target_no_patient, 0.25)
                reason = PREDICTION_REASON_RECENT_VISIBLE_FALLBACK
            remaining = 1.0 - target_no_patient
            probs = {
                "chair": remaining * visible_focus["chair"],
                "bed": remaining * visible_focus["bed"],
                "room": remaining * visible_focus["room"],
                "no_patient": target_no_patient,
            }
        elif (
            legacy_focus_visible <= 0
            and dropout_visible > 0
            and fallback_sum > 0
        ):
            max_visible_fallback = max(visible_fallback.values())
            visible_strength = max(
                float(settings.label_eval_v2_recent_visible_min_strength),
                float(pd.to_numeric(pd.Series([row.get("pre_state_visible_prob")]), errors="coerce").fillna(0.0).iloc[0]),
            )
            visible_strength = max(
                visible_strength,
                min(0.90, 0.20 + (0.75 * max_visible_fallback)),
            )
            if (
                max_visible_fallback >= 0.70
                and (pd.isna(last_gap) or float(last_gap) <= float(settings.label_eval_v2_recent_visible_gap_seconds))
            ):
                visible_strength = max(visible_strength, 0.75)
            if pd.notna(last_gap) and float(last_gap) > float(settings.label_eval_v2_recent_visible_gap_seconds):
                visible_strength = min(visible_strength, 0.50)
            visible_strength = min(0.90, max(0.35, visible_strength))
            target_no_patient = min(probs["no_patient"], max(0.0, 1.0 - visible_strength))
            if (
                max_visible_fallback >= 0.70
                and (pd.isna(last_gap) or float(last_gap) <= float(settings.label_eval_v2_recent_visible_gap_seconds))
            ):
                target_no_patient = min(target_no_patient, 0.25)
            remaining = 1.0 - target_no_patient
            probs = {
                "chair": remaining * visible_fallback["chair"],
                "bed": remaining * visible_fallback["bed"],
                "room": remaining * visible_fallback["room"],
                "no_patient": target_no_patient,
            }
            reason = PREDICTION_REASON_RECENT_VISIBLE_FALLBACK
        elif (
            probs["no_patient"] >= float(settings.label_eval_v2_no_patient_strong_threshold)
            and dropout_visible > 0
            and fallback_sum > 0
            and (pd.isna(last_gap) or float(last_gap) <= float(settings.label_eval_v2_recent_visible_gap_seconds))
        ):
            visible_strength = max(
                float(settings.label_eval_v2_recent_visible_min_strength),
                1.0 - min(probs["no_patient"], 0.45),
            )
            visible_strength = min(0.85, max(0.35, visible_strength))
            target_no_patient = max(0.0, 1.0 - visible_strength)
            if target_no_patient < probs["no_patient"]:
                remaining = 1.0 - target_no_patient
                probs = {
                    "chair": remaining * visible_fallback["chair"],
                    "bed": remaining * visible_fallback["bed"],
                    "room": remaining * visible_fallback["room"],
                    "no_patient": target_no_patient,
                }
                reason = PREDICTION_REASON_RECENT_VISIBLE_FALLBACK

        room_support = bool(_safe_float(row.get("v2_room_support_flag"))) if pd.notna(row.get("v2_room_support_flag")) else False
        if (
            probs["room"] <= 0.0
            and room_support
            and fallback_sum > 0
            and visible_fallback["room"] > 0.0
        ):
            donor = max(("chair", "bed", "no_patient"), key=lambda label: probs[label])
            donor_cap = 0.50 if donor in {"chair", "bed"} else 0.35
            room_floor = min(
                float(settings.label_eval_v2_room_floor_max),
                max(float(settings.label_eval_v2_room_floor_min), float(visible_fallback["room"])),
            )
            transfer = min(room_floor, probs[donor] * donor_cap)
            if transfer > 0:
                probs[donor] -= transfer
                probs["room"] += transfer
                if reason == PREDICTION_REASON_DIRECT_FOCUS_ARGMAX:
                    reason = PREDICTION_REASON_ROOM_RECOVERY_V2

        probs = _normalize_probability_dict(probs)
        if reason == PREDICTION_REASON_DIRECT_FOCUS_ARGMAX and probs["no_patient"] >= float(
            settings.label_eval_v2_no_patient_strong_threshold
        ):
            reason = PREDICTION_REASON_STRONG_NO_PATIENT

        ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        label = ordered[0][0]
        margin = float(ordered[0][1] - ordered[1][1]) if len(ordered) > 1 else float(ordered[0][1])
        rows.append(
            {
                **row.to_dict(),
                "pre_prob_v2_chair": round(probs["chair"], 6),
                "pre_prob_v2_bed": round(probs["bed"], 6),
                "pre_prob_v2_room": round(probs["room"], 6),
                "pre_prob_v2_no_patient": round(probs["no_patient"], 6),
                "prefall_location_v2_label": label,
                "prefall_location_v2_margin": round(margin, 6),
                "prefall_location_v2_reason": reason,
            }
        )

    return pd.DataFrame(rows)


def _normalize_shadow_panel(second_level_panel: pd.DataFrame, default_window_role: str) -> pd.DataFrame:
    if second_level_panel.empty:
        return pd.DataFrame()
    required = {"fall_event_id", "second_offset", "dominant_location_label"}
    if not required.issubset(second_level_panel.columns):
        return pd.DataFrame()

    panel = second_level_panel.copy()
    panel["fall_event_id"] = pd.to_numeric(panel["fall_event_id"], errors="coerce").astype("Int64")
    panel = panel.loc[panel["fall_event_id"].notna()].copy()
    if panel.empty:
        return pd.DataFrame()

    panel["second_offset"] = pd.to_numeric(panel["second_offset"], errors="coerce")
    panel["dominant_location_label"] = (
        panel["dominant_location_label"].astype("string").fillna("no_patient").str.strip().str.lower()
    )
    if "window_role" not in panel.columns:
        panel["window_role"] = default_window_role
    panel["window_role"] = panel["window_role"].astype("string").fillna(default_window_role).str.lower()
    if "anchor_ts_utc" not in panel.columns:
        panel["anchor_ts_utc"] = panel.get("fall_ts_utc", pd.Series(pd.NA, index=panel.index))
    panel["anchor_ts_utc"] = pd.to_datetime(panel["anchor_ts_utc"], errors="coerce", utc=True)
    panel["fall_ts_utc"] = pd.to_datetime(panel.get("fall_ts_utc"), errors="coerce", utc=True)
    if "frame_ts_utc" not in panel.columns:
        panel["frame_ts_utc"] = pd.NaT
    panel["frame_ts_utc"] = pd.to_datetime(panel["frame_ts_utc"], errors="coerce", utc=True)
    if "frame_has_location_signal" not in panel.columns:
        panel["frame_has_location_signal"] = False
    panel["frame_has_location_signal"] = panel["frame_has_location_signal"].fillna(False).astype(bool)
    for column in ["patient_chair_distance", "patient_bed_distance", "patient_room_distance", "nudge_score"]:
        if column not in panel.columns:
            panel[column] = pd.NA
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    for column in [
        "primary_patient_posture_score_sitting",
        "primary_patient_posture_score_standing",
        "primary_patient_posture_score_lying",
    ]:
        if column not in panel.columns:
            panel[column] = pd.NA
        panel[column] = pd.to_numeric(panel[column], errors="coerce").clip(lower=0.0, upper=1.0)
    if "primary_patient_posture_label" not in panel.columns:
        panel["primary_patient_posture_label"] = pd.NA
    panel["primary_patient_posture_label"] = (
        panel["primary_patient_posture_label"].astype("string").fillna("").str.strip().str.lower()
    )
    posture_sum = panel[
        [
            "primary_patient_posture_score_sitting",
            "primary_patient_posture_score_standing",
            "primary_patient_posture_score_lying",
        ]
    ].sum(axis=1, min_count=1)
    has_posture_scores = posture_sum.fillna(0.0) > 0.0
    for column in [
        "primary_patient_posture_score_sitting",
        "primary_patient_posture_score_standing",
        "primary_patient_posture_score_lying",
    ]:
        panel.loc[has_posture_scores, column] = (
            panel.loc[has_posture_scores, column].fillna(0.0) / posture_sum.loc[has_posture_scores]
        )
    panel["frame_has_posture_signal"] = (
        panel["primary_patient_posture_label"].isin(["sitting", "standing", "lying"])
        | has_posture_scores
    )
    if "nudge_state" not in panel.columns:
        panel["nudge_state"] = pd.NA
    panel["nudge_state"] = panel["nudge_state"].astype("string")
    return panel


def _combine_shadow_panels(
    second_level_panel: pd.DataFrame,
    crossover_second_level_panel: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not crossover_second_level_panel.empty:
        frames.append(_normalize_shadow_panel(crossover_second_level_panel, "hazard"))
    if not second_level_panel.empty:
        frames.append(_normalize_shadow_panel(second_level_panel, "hazard"))
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    dedupe_cols = ["fall_event_id", "window_role", "second_offset"]
    if "frame_ts_utc" in combined.columns:
        dedupe_cols.append("frame_ts_utc")
    combined = combined.sort_values(
        ["fall_event_id", "window_role", "second_offset", "frame_ts_utc"],
        kind="mergesort",
        na_position="last",
    ).drop_duplicates(subset=dedupe_cols, keep="first")
    return combined.reset_index(drop=True)


def _window_quality(segment: pd.DataFrame, window_seconds: int, settings: Settings) -> dict[str, Any]:
    offsets = pd.to_numeric(segment["second_offset"], errors="coerce").dropna().astype(int).sort_values()
    observed_seconds = int(offsets.nunique())
    expected_seconds = int(window_seconds)
    presence_ratio = float(observed_seconds / expected_seconds) if expected_seconds > 0 else 0.0
    frame_count = int(len(segment.index))
    gap_exceeded = False
    if len(offsets.index) > 1:
        gap_exceeded = bool(offsets.diff().dropna().gt(int(settings.label_eval_inter_frame_gap_seconds)).any())

    reason = None
    if presence_ratio < float(settings.label_eval_min_sample_presence_ratio):
        reason = "insufficient_sample_presence"
    elif frame_count < int(settings.label_eval_min_frames_per_window):
        reason = "sparse_window"
    elif gap_exceeded:
        reason = "inter_frame_gap"

    return {
        "expected_seconds": expected_seconds,
        "observed_seconds": observed_seconds,
        "sample_presence_ratio": round(presence_ratio, 6),
        "frame_count": frame_count,
        "gap_exceeded": gap_exceeded,
        "window_reason_code": reason,
        "eligible_for_shadow": reason is None,
    }


def _dominant_dwell_max(segment: pd.DataFrame) -> float:
    labels = segment["dominant_location_label"].astype("string").fillna("no_patient")
    if labels.empty:
        return 0.0
    change_points = labels.ne(labels.shift()).cumsum()
    return float(labels.groupby(change_points, dropna=False).size().max())


def _transition_count(segment: pd.DataFrame, source: str, target: str) -> float:
    prev = segment["dominant_location_label"].shift(1)
    curr = segment["dominant_location_label"]
    return float(((prev == source) & (curr == target)).sum())


def _distance_channel_metrics(
    segment: pd.DataFrame,
    column: str,
    settings: Settings,
) -> tuple[dict[str, Any], bool]:
    values = segment.loc[segment[column].notna(), ["second_offset", column]].copy()
    metrics = {
        f"{column}_mean": pd.NA,
        f"{column}_velocity_mean": pd.NA,
        f"{column}_velocity_volatility": pd.NA,
        f"{column}_accel_mean": pd.NA,
    }
    if len(values.index) < 2:
        return metrics, False

    variance = float(pd.to_numeric(values[column], errors="coerce").var(ddof=0))
    if variance == 0.0:
        return metrics, True

    deltas = values.copy()
    deltas["delta_t"] = deltas["second_offset"].diff().abs()
    deltas["delta_v"] = deltas[column].diff()
    deltas = deltas.loc[deltas["delta_t"].notna() & deltas["delta_t"].gt(0)].copy()
    if deltas.empty or deltas["delta_t"].gt(int(settings.label_eval_inter_frame_gap_seconds)).any():
        return metrics, False

    deltas["velocity"] = deltas["delta_v"] / deltas["delta_t"]
    accel = deltas[["second_offset", "velocity"]].copy()
    accel["delta_t"] = accel["second_offset"].diff().abs()
    accel["delta_v"] = accel["velocity"].diff()
    accel = accel.loc[accel["delta_t"].notna() & accel["delta_t"].gt(0)].copy()
    accel["acceleration"] = accel["delta_v"] / accel["delta_t"] if not accel.empty else pd.Series(dtype="float64")

    metrics[f"{column}_mean"] = round(float(pd.to_numeric(values[column], errors="coerce").mean()), 6)
    metrics[f"{column}_velocity_mean"] = round(float(deltas["velocity"].mean()), 6)
    metrics[f"{column}_velocity_volatility"] = round(float(deltas["velocity"].std(ddof=0)), 6)
    metrics[f"{column}_accel_mean"] = (
        round(float(accel["acceleration"].mean()), 6) if not accel.empty else pd.NA
    )
    return metrics, False


def _trajectory_feature_rows(settings: Settings, combined_panel: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "fall_event_id",
        "window_role",
        "window_seconds",
        "expected_seconds",
        "observed_seconds",
        "sample_presence_ratio",
        "frame_count",
        "gap_exceeded",
        "window_reason_code",
        "eligible_for_shadow",
        "visible_ratio",
        "switch_count",
        "chair_share",
        "bed_share",
        "room_share",
        "no_patient_share",
        "location_dwell_max",
        "chair_to_bed_transitions",
        "bed_to_room_transitions",
        "room_to_chair_transitions",
        "bed_to_chair_transitions",
        "posture_observed_ratio",
        "posture_switch_count",
        "posture_sitting_score_mean",
        "posture_standing_score_mean",
        "posture_lying_score_mean",
        "posture_sitting_share",
        "posture_standing_share",
        "posture_lying_share",
        "patient_chair_distance_mean",
        "patient_chair_distance_velocity_mean",
        "patient_chair_distance_velocity_volatility",
        "patient_chair_distance_accel_mean",
        "patient_bed_distance_mean",
        "patient_bed_distance_velocity_mean",
        "patient_bed_distance_velocity_volatility",
        "patient_bed_distance_accel_mean",
        "patient_room_distance_mean",
        "patient_room_distance_velocity_mean",
        "patient_room_distance_velocity_volatility",
        "patient_room_distance_accel_mean",
    ]
    if combined_panel.empty:
        return pd.DataFrame(columns=base_columns)

    rows: list[dict[str, Any]] = []
    window_sizes = sorted({int(value) for value in settings.label_eval_shadow_window_seconds if int(value) > 0})
    for (fall_event_id, window_role), group in combined_panel.groupby(["fall_event_id", "window_role"], dropna=False):
        if pd.isna(fall_event_id):
            continue
        group = group.sort_values("second_offset", kind="mergesort")
        for window_seconds in window_sizes:
            segment = group.loc[(group["second_offset"] >= -window_seconds) & (group["second_offset"] < 0)].copy()
            quality = _window_quality(segment, window_seconds, settings)
            row: dict[str, Any] = {
                "fall_event_id": int(fall_event_id),
                "window_role": str(window_role),
                "window_seconds": int(window_seconds),
                **quality,
                "visible_ratio": pd.NA,
                "switch_count": pd.NA,
                "chair_share": pd.NA,
                "bed_share": pd.NA,
                "room_share": pd.NA,
                "no_patient_share": pd.NA,
                "location_dwell_max": pd.NA,
                "chair_to_bed_transitions": pd.NA,
                "bed_to_room_transitions": pd.NA,
                "room_to_chair_transitions": pd.NA,
                "bed_to_chair_transitions": pd.NA,
                "posture_observed_ratio": pd.NA,
                "posture_switch_count": pd.NA,
                "posture_sitting_score_mean": pd.NA,
                "posture_standing_score_mean": pd.NA,
                "posture_lying_score_mean": pd.NA,
                "posture_sitting_share": pd.NA,
                "posture_standing_share": pd.NA,
                "posture_lying_share": pd.NA,
                "patient_chair_distance_mean": pd.NA,
                "patient_chair_distance_velocity_mean": pd.NA,
                "patient_chair_distance_velocity_volatility": pd.NA,
                "patient_chair_distance_accel_mean": pd.NA,
                "patient_bed_distance_mean": pd.NA,
                "patient_bed_distance_velocity_mean": pd.NA,
                "patient_bed_distance_velocity_volatility": pd.NA,
                "patient_bed_distance_accel_mean": pd.NA,
                "patient_room_distance_mean": pd.NA,
                "patient_room_distance_velocity_mean": pd.NA,
                "patient_room_distance_velocity_volatility": pd.NA,
                "patient_room_distance_accel_mean": pd.NA,
            }
            if segment.empty or row["window_reason_code"] is not None:
                rows.append(row)
                continue

            row["visible_ratio"] = round(float(segment["frame_has_location_signal"].mean()), 6)
            prev = segment["dominant_location_label"].shift(1)
            row["switch_count"] = float(
                ((prev.notna()) & (segment["dominant_location_label"] != prev)).sum()
            )
            for label in LOCATION_CLASSES:
                row[f"{label}_share"] = round(
                    float((segment["dominant_location_label"] == label).mean()),
                    6,
                )
            row["location_dwell_max"] = _dominant_dwell_max(segment)
            row["chair_to_bed_transitions"] = _transition_count(segment, "chair", "bed")
            row["bed_to_room_transitions"] = _transition_count(segment, "bed", "room")
            row["room_to_chair_transitions"] = _transition_count(segment, "room", "chair")
            row["bed_to_chair_transitions"] = _transition_count(segment, "bed", "chair")
            posture_segment = segment.loc[segment["frame_has_posture_signal"].fillna(False)].copy()
            row["posture_observed_ratio"] = round(
                float(segment["frame_has_posture_signal"].fillna(False).astype(float).mean()),
                6,
            )
            if not posture_segment.empty:
                posture_prev = posture_segment["primary_patient_posture_label"].shift(1)
                row["posture_switch_count"] = float(
                    (
                        posture_prev.notna()
                        & posture_segment["primary_patient_posture_label"].notna()
                        & (posture_segment["primary_patient_posture_label"] != posture_prev)
                    ).sum()
                )
                for label in ["sitting", "standing", "lying"]:
                    row[f"posture_{label}_score_mean"] = round(
                        float(
                            pd.to_numeric(
                                posture_segment[f"primary_patient_posture_score_{label}"],
                                errors="coerce",
                            ).mean()
                        ),
                        6,
                    )
                    row[f"posture_{label}_share"] = round(
                        float((posture_segment["primary_patient_posture_label"] == label).mean()),
                        6,
                    )
            else:
                row["posture_switch_count"] = 0.0

            zero_variance = False
            for column in ["patient_chair_distance", "patient_bed_distance", "patient_room_distance"]:
                metrics, channel_zero_variance = _distance_channel_metrics(segment, column, settings)
                row.update(metrics)
                zero_variance = zero_variance or channel_zero_variance
            if zero_variance and row["window_reason_code"] is None:
                row["window_reason_code"] = "zero_variance"
                row["eligible_for_shadow"] = False
            rows.append(row)
    return pd.DataFrame(rows, columns=base_columns)


def _is_active_nudge_state(value: Any) -> bool:
    token = str(value).strip().lower()
    return token not in {"", "<na>", "nan", "none", "inactive", "idle", "off", "unknown"}


def _detect_onset_events(settings: Settings, combined_panel: pd.DataFrame) -> pd.DataFrame:
    columns = _empty_artifacts().onset_events.columns.tolist()
    if combined_panel.empty:
        return pd.DataFrame(columns=columns)

    panel = combined_panel.copy()
    if "frame_has_posture_signal" not in panel.columns:
        panel["frame_has_posture_signal"] = False
    panel["frame_has_posture_signal"] = panel["frame_has_posture_signal"].fillna(False).astype(bool)
    if "primary_patient_posture_label" not in panel.columns:
        panel["primary_patient_posture_label"] = pd.NA
    panel["primary_patient_posture_label"] = (
        panel["primary_patient_posture_label"].astype("string").fillna("").str.strip().str.lower()
    )
    for column in [
        "primary_patient_posture_score_sitting",
        "primary_patient_posture_score_standing",
        "primary_patient_posture_score_lying",
    ]:
        if column not in panel.columns:
            panel[column] = pd.NA
        panel[column] = pd.to_numeric(panel[column], errors="coerce")

    rows: list[dict[str, Any]] = []
    channel_priority = {
        "posture_transition": 0,
        "location_departure": 1,
        "distance_breakout": 2,
        "nudge_escalation": 3,
    }
    baseline_seconds = int(settings.label_eval_onset_baseline_seconds)
    window_seconds = max(int(settings.fall_window_pre_anchor_seconds), baseline_seconds)
    for (fall_event_id, window_role), group in panel.groupby(["fall_event_id", "window_role"], dropna=False):
        if pd.isna(fall_event_id):
            continue
        group = group.loc[(group["second_offset"] >= -window_seconds) & (group["second_offset"] < 0)].copy()
        group = group.sort_values("second_offset", kind="mergesort")
        quality = _window_quality(group, window_seconds, settings)
        anchor_ts = pd.to_datetime(group["anchor_ts_utc"], errors="coerce", utc=True).dropna()
        row: dict[str, Any] = {
            "fall_event_id": int(fall_event_id),
            "window_role": str(window_role),
            "anchor_ts_utc": anchor_ts.iloc[0] if not anchor_ts.empty else pd.NaT,
            "sample_presence_ratio": quality["sample_presence_ratio"],
            "frame_count": quality["frame_count"],
            "eligibility_reason_code": quality["window_reason_code"],
            "onset_detected": False,
            "onset_channel": pd.NA,
            "first_signal_offset": pd.NA,
            "latency_to_anchor_seconds": pd.NA,
            "departure_source": pd.NA,
            "destination_label": pd.NA,
            "below_prevalence_threshold": False,
            "exploratory_status": "exploratory",
        }
        if group.empty or quality["window_reason_code"] is not None:
            rows.append(row)
            continue

        baseline = group.loc[group["second_offset"] < -(window_seconds - baseline_seconds)].copy()
        subsequent = group.loc[group["second_offset"] >= -(window_seconds - baseline_seconds)].copy()
        if baseline.empty or subsequent.empty:
            row["eligibility_reason_code"] = "sparse_window"
            rows.append(row)
            continue

        baseline_label = (
            baseline["dominant_location_label"].mode(dropna=True).iloc[0]
            if baseline["dominant_location_label"].mode(dropna=True).size
            else "no_patient"
        )
        candidates: list[dict[str, Any]] = []

        baseline_posture = baseline.loc[
            baseline["frame_has_posture_signal"].fillna(False).astype(bool)
        ].copy()
        baseline_posture = baseline_posture.loc[
            baseline_posture["primary_patient_posture_label"].astype("string").isin(
                ["sitting", "standing", "lying"]
            )
        ]
        baseline_posture_label = (
            baseline_posture["primary_patient_posture_label"].mode(dropna=True).iloc[0]
            if not baseline_posture.empty
            and baseline_posture["primary_patient_posture_label"].mode(dropna=True).size
            else None
        )
        if baseline_posture_label is not None:
            posture = subsequent.copy()
            posture["posture_label"] = (
                posture["primary_patient_posture_label"].astype("string").fillna("").str.strip().str.lower()
            )
            posture["posture_score_max"] = posture[
                [
                    "primary_patient_posture_score_sitting",
                    "primary_patient_posture_score_standing",
                    "primary_patient_posture_score_lying",
                ]
            ].apply(pd.to_numeric, errors="coerce").max(axis=1)
            posture["posture_candidate"] = (
                posture["frame_has_posture_signal"].fillna(False).astype(bool)
                & posture["posture_label"].isin(["sitting", "standing", "lying"])
                & posture["posture_label"].ne(str(baseline_posture_label))
                & posture["posture_score_max"].ge(float(settings.label_eval_posture_transition_min_score))
            )
            if posture["posture_candidate"].any():
                posture["_candidate_group"] = posture["posture_candidate"].ne(
                    posture["posture_candidate"].shift(fill_value=False)
                ).cumsum()
                candidate_runs = posture.loc[posture["posture_candidate"]].groupby("_candidate_group", dropna=False)
                min_run_length = int(settings.label_eval_posture_transition_min_run_length)
                for _, run in candidate_runs:
                    if len(run.index) >= min_run_length:
                        first = run.iloc[0]
                        candidates.append(
                            {
                                "onset_channel": "posture_transition",
                                "first_signal_offset": float(first["second_offset"]),
                                "departure_source": str(baseline_posture_label),
                                "destination_label": str(first["posture_label"]),
                            }
                        )
                        break

        location_departure = subsequent.loc[
            subsequent["dominant_location_label"].astype("string") != str(baseline_label)
        ]
        if not location_departure.empty:
            first = location_departure.iloc[0]
            candidates.append(
                {
                    "onset_channel": "location_departure",
                    "first_signal_offset": float(first["second_offset"]),
                    "departure_source": str(baseline_label),
                    "destination_label": str(first["dominant_location_label"]),
                }
            )

        breakout_hits: list[tuple[float, str]] = []
        for column in ["patient_chair_distance", "patient_bed_distance", "patient_room_distance"]:
            base_values = pd.to_numeric(baseline[column], errors="coerce").dropna()
            if base_values.empty:
                continue
            threshold = float(base_values.mean()) + (
                float(settings.label_eval_onset_distance_zscore)
                * max(float(base_values.std(ddof=0)), float(settings.label_eval_onset_distance_std_floor_meters))
            )
            hit = subsequent.loc[pd.to_numeric(subsequent[column], errors="coerce") > threshold]
            if not hit.empty:
                breakout_hits.append((float(hit.iloc[0]["second_offset"]), column))
        if breakout_hits:
            breakout_hits.sort(key=lambda item: item[0])
            first_offset, first_column = breakout_hits[0]
            candidates.append(
                {
                    "onset_channel": "distance_breakout",
                    "first_signal_offset": first_offset,
                    "departure_source": first_column.replace("patient_", "").replace("_distance", ""),
                    "destination_label": pd.NA,
                }
            )

        nudge = subsequent.copy()
        nudge["nudge_active"] = nudge["nudge_state"].apply(_is_active_nudge_state)
        nudge["score_above"] = pd.to_numeric(nudge["nudge_score"], errors="coerce").ge(
            float(settings.label_eval_onset_nudge_threshold)
        )
        nudge["active_prev"] = nudge["nudge_active"].shift(1).fillna(False)
        nudge["score_prev"] = nudge["score_above"].shift(1).fillna(False)
        nudge_hit = nudge.loc[
            ((~nudge["score_prev"]) & nudge["score_above"])
            | ((~nudge["active_prev"]) & nudge["nudge_active"])
        ]
        if not nudge_hit.empty:
            first = nudge_hit.iloc[0]
            candidates.append(
                {
                    "onset_channel": "nudge_escalation",
                    "first_signal_offset": float(first["second_offset"]),
                    "departure_source": pd.NA,
                    "destination_label": pd.NA,
                }
            )

        if candidates:
            candidates.sort(
                key=lambda item: (
                    item["first_signal_offset"],
                    channel_priority.get(str(item["onset_channel"]), 99),
                )
            )
            first = candidates[0]
            row["onset_detected"] = True
            row["onset_channel"] = first["onset_channel"]
            row["first_signal_offset"] = first["first_signal_offset"]
            row["latency_to_anchor_seconds"] = abs(float(first["first_signal_offset"]))
            row["departure_source"] = first["departure_source"]
            row["destination_label"] = first["destination_label"]
        rows.append(row)

    result = pd.DataFrame(rows, columns=columns)
    hazard = result.loc[result["window_role"] == "hazard"].copy()
    hazard_events = int(hazard["fall_event_id"].nunique())
    min_events = max(1, int(ceil(hazard_events * float(settings.label_eval_onset_min_prevalence_ratio))))
    active_counts = (
        hazard.loc[hazard["onset_detected"] & hazard["onset_channel"].notna()]
        .groupby("onset_channel")["fall_event_id"]
        .nunique()
        .to_dict()
    )
    result["below_prevalence_threshold"] = result["onset_channel"].map(
        lambda value: bool(pd.notna(value) and active_counts.get(value, 0) < min_events)
    )
    return result


def _onset_case_crossover_comparison(onset_events: pd.DataFrame) -> pd.DataFrame:
    columns = _empty_artifacts().onset_case_crossover.columns.tolist()
    if onset_events.empty:
        return pd.DataFrame(columns=columns)
    hazard = onset_events.loc[onset_events["window_role"] == "hazard"].copy()
    controls = onset_events.loc[_is_case_crossover_control_role(onset_events["window_role"])].copy()
    if hazard.empty or controls.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for _, control in controls.iterrows():
        hazard_row = hazard.loc[hazard["fall_event_id"] == control["fall_event_id"]]
        if hazard_row.empty:
            continue
        hz = hazard_row.iloc[0]
        hz_detect = bool(hz.get("onset_detected", False))
        ctrl_detect = bool(control.get("onset_detected", False))
        hz_latency = pd.to_numeric(hz.get("latency_to_anchor_seconds"), errors="coerce")
        ctrl_latency = pd.to_numeric(control.get("latency_to_anchor_seconds"), errors="coerce")
        delta_latency = pd.NA
        if pd.notna(hz_latency) and pd.notna(ctrl_latency):
            delta_latency = round(float(hz_latency - ctrl_latency), 6)
        rows.append(
            {
                "fall_event_id": int(control["fall_event_id"]),
                "control_role": str(control["window_role"]),
                "hazard_onset_detected": hz_detect,
                "control_onset_detected": ctrl_detect,
                "hazard_latency_to_anchor_seconds": hz_latency if pd.notna(hz_latency) else pd.NA,
                "control_latency_to_anchor_seconds": ctrl_latency if pd.notna(ctrl_latency) else pd.NA,
                "hazard_onset_channel": hz.get("onset_channel", pd.NA),
                "control_onset_channel": control.get("onset_channel", pd.NA),
                "hazard_minus_control_detected": int(hz_detect) - int(ctrl_detect),
                "hazard_minus_control_latency_seconds": delta_latency,
                "exclude_from_statistics": bool(
                    hz.get("below_prevalence_threshold", False)
                    or control.get("below_prevalence_threshold", False)
                ),
                "exploratory_status": "exploratory",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _metric_map(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    out: dict[str, float] = {}
    for _, row in frame.iterrows():
        key = str(row.get("metric", "")).strip()
        value = pd.to_numeric(row.get("value"), errors="coerce")
        if key and pd.notna(value):
            out[key] = float(value)
    return out


def _trajectory_feature_wide(trajectory_rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    if trajectory_rows.empty:
        return pd.DataFrame(columns=["fall_event_id"]), [], []

    hazard_full_features = [
        "visible_ratio",
        "switch_count",
        "chair_share",
        "bed_share",
        "room_share",
        "no_patient_share",
        "location_dwell_max",
        "chair_to_bed_transitions",
        "bed_to_room_transitions",
        "room_to_chair_transitions",
        "bed_to_chair_transitions",
        "patient_chair_distance_mean",
        "patient_chair_distance_velocity_mean",
        "patient_chair_distance_velocity_volatility",
        "patient_chair_distance_accel_mean",
        "patient_bed_distance_mean",
        "patient_bed_distance_velocity_mean",
        "patient_bed_distance_velocity_volatility",
        "patient_bed_distance_accel_mean",
        "patient_room_distance_mean",
        "patient_room_distance_velocity_mean",
        "patient_room_distance_velocity_volatility",
        "patient_room_distance_accel_mean",
    ]
    delta_features = [
        "visible_ratio",
        "switch_count",
        "chair_share",
        "bed_share",
        "room_share",
        "no_patient_share",
        "location_dwell_max",
    ]

    rows: list[dict[str, Any]] = []
    for fall_event_id, event_rows in trajectory_rows.groupby("fall_event_id", dropna=False):
        if pd.isna(fall_event_id):
            continue
        row: dict[str, Any] = {"fall_event_id": int(fall_event_id)}
        controls = event_rows.loc[_is_case_crossover_control_role(event_rows["window_role"])].copy()
        for _, hazard_row in event_rows.loc[event_rows["window_role"] == "hazard"].iterrows():
            window = int(hazard_row["window_seconds"])
            for feature in hazard_full_features:
                row[f"hazard_{window}s__{feature}"] = hazard_row.get(feature, pd.NA)
            window_controls = controls.loc[controls["window_seconds"] == window]
            if not window_controls.empty:
                control_means = window_controls[delta_features].apply(pd.to_numeric, errors="coerce").mean()
                for feature in delta_features:
                    control_value = control_means.get(feature, pd.NA)
                    row[f"control_mean_{window}s__{feature}"] = control_value
                    hazard_value = pd.to_numeric(hazard_row.get(feature), errors="coerce")
                    if pd.notna(hazard_value) and pd.notna(control_value):
                        row[f"hazard_minus_control_mean_{window}s__{feature}"] = round(
                            float(hazard_value - control_value),
                            6,
                        )
                    else:
                        row[f"hazard_minus_control_mean_{window}s__{feature}"] = pd.NA
        rows.append(row)

    wide = pd.DataFrame(rows)
    trajectory_cols = [col for col in wide.columns if col != "fall_event_id"]
    delta_cols = [col for col in trajectory_cols if col.startswith("hazard_minus_control_mean_")]
    return wide, trajectory_cols, delta_cols


def _onset_feature_wide(
    onset_events: pd.DataFrame,
    onset_case_crossover: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    if onset_events.empty:
        return pd.DataFrame(columns=["fall_event_id"]), []

    hazard = onset_events.loc[onset_events["window_role"] == "hazard"].copy()
    if hazard.empty:
        return pd.DataFrame(columns=["fall_event_id"]), []

    rows: list[dict[str, Any]] = []
    for _, row in hazard.iterrows():
        event_id = int(row["fall_event_id"])
        compare = onset_case_crossover.loc[onset_case_crossover["fall_event_id"] == event_id].copy()
        compare["hazard_minus_control_detected"] = pd.to_numeric(
            compare["hazard_minus_control_detected"], errors="coerce"
        )
        compare["hazard_minus_control_latency_seconds"] = pd.to_numeric(
            compare["hazard_minus_control_latency_seconds"], errors="coerce"
        )
        onset_channel = str(row.get("onset_channel")).strip() if pd.notna(row.get("onset_channel")) else ""
        onset_detected = bool(row.get("onset_detected")) if pd.notna(row.get("onset_detected")) else False
        onset_row: dict[str, Any] = {
            "fall_event_id": event_id,
            "hazard_onset_detected": float(onset_detected),
            "hazard_onset_latency_seconds": pd.to_numeric(
                row.get("latency_to_anchor_seconds"),
                errors="coerce",
            ),
            "hazard_onset_below_prevalence_threshold": float(
                bool(row.get("below_prevalence_threshold", False))
            ),
            "hazard_onset_posture_transition": float(onset_channel == "posture_transition"),
            "hazard_onset_location_departure": float(onset_channel == "location_departure"),
            "hazard_onset_distance_breakout": float(onset_channel == "distance_breakout"),
            "hazard_onset_nudge_escalation": float(onset_channel == "nudge_escalation"),
            "hazard_minus_control_onset_detected_mean": (
                round(float(compare["hazard_minus_control_detected"].mean()), 6)
                if not compare.empty
                else pd.NA
            ),
            "hazard_minus_control_onset_latency_seconds_mean": (
                round(float(compare["hazard_minus_control_latency_seconds"].mean()), 6)
                if not compare.empty and compare["hazard_minus_control_latency_seconds"].notna().any()
                else pd.NA
            ),
        }
        rows.append(onset_row)
    wide = pd.DataFrame(rows)
    onset_cols = [col for col in wide.columns if col != "fall_event_id"]
    return wide, onset_cols


def _build_shadow_feature_frame(
    settings: Settings,
    scored_location: pd.DataFrame,
    matched: pd.DataFrame,
    trajectory_rows: pd.DataFrame,
    onset_events: pd.DataFrame,
    onset_case_crossover: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if scored_location.empty or matched.empty:
        return pd.DataFrame(), {}
    if "fall_event_id" not in matched.columns:
        return pd.DataFrame(), {}

    key_frame = matched[["sequence_id", "fall_event_id"]].copy()
    key_frame["fall_event_id"] = pd.to_numeric(key_frame["fall_event_id"], errors="coerce").astype("Int64")
    key_frame = key_frame.dropna(subset=["sequence_id", "fall_event_id"]).drop_duplicates()
    if key_frame.empty:
        return pd.DataFrame(), {}

    trajectory_wide, trajectory_cols, _ = _trajectory_feature_wide(trajectory_rows)
    onset_wide, onset_cols = _onset_feature_wide(onset_events, onset_case_crossover)
    event_features = key_frame.merge(trajectory_wide, on="fall_event_id", how="left")
    event_features = event_features.merge(onset_wide, on="fall_event_id", how="left")
    feature_cols = [col for col in event_features.columns if col not in {"sequence_id", "fall_event_id"}]
    sequence_features = (
        event_features.groupby("sequence_id", dropna=False)[feature_cols]
        .mean(numeric_only=True)
        .reset_index()
        if feature_cols
        else key_frame[["sequence_id"]].drop_duplicates()
    )

    base = scored_location[
        [
            "sequence_id",
            "monitor_id",
            "truth_label",
            "derived_label",
            "derived_prob_chair",
            "derived_prob_bed",
            "derived_prob_room",
            "derived_prob_no_patient",
        ]
    ].copy()
    base = base.rename(
        columns={
            "derived_prob_chair": "baseline_probs__chair",
            "derived_prob_bed": "baseline_probs__bed",
            "derived_prob_room": "baseline_probs__room",
            "derived_prob_no_patient": "baseline_probs__no_patient",
        }
    )
    merged = base.merge(sequence_features, on="sequence_id", how="left")

    posture_cols = [col for col in trajectory_cols + onset_cols if "posture" in col]
    feature_families = {
        "baseline_probs": [
            "baseline_probs__chair",
            "baseline_probs__bed",
            "baseline_probs__room",
            "baseline_probs__no_patient",
        ],
        "trajectory": [col for col in trajectory_cols if col not in posture_cols],
        "posture": posture_cols,
        "onset": [col for col in onset_cols if col not in posture_cols],
    }
    all_features = (
        feature_families["baseline_probs"]
        + feature_families["trajectory"]
        + feature_families["posture"]
        + feature_families["onset"]
    )
    capped = all_features[: int(settings.label_eval_shadow_feature_cap)]
    feature_families["baseline_probs"] = [col for col in feature_families["baseline_probs"] if col in capped]
    feature_families["trajectory"] = [col for col in feature_families["trajectory"] if col in capped]
    feature_families["onset"] = [col for col in feature_families["onset"] if col in capped]
    keep_cols = ["sequence_id", "monitor_id", "truth_label", "derived_label"] + capped
    for column in keep_cols:
        if column not in merged.columns:
            merged[column] = pd.NA
    merged = merged[keep_cols].copy()
    return merged, feature_families


def _build_model_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if not feature_cols:
        return pd.DataFrame(index=train.index), pd.DataFrame(index=test.index), []

    train_x = train[feature_cols].apply(pd.to_numeric, errors="coerce").copy()
    test_x = test[feature_cols].apply(pd.to_numeric, errors="coerce").copy()
    final_cols: list[str] = []
    for column in feature_cols:
        final_cols.append(column)
        if train_x[column].isna().any() or test_x[column].isna().any():
            indicator = f"{column}__missing"
            train_x[indicator] = train_x[column].isna().astype(float)
            test_x[indicator] = test_x[column].isna().astype(float)
            final_cols.append(indicator)
        fill_value = pd.to_numeric(train_x[column], errors="coerce").median()
        if pd.isna(fill_value):
            fill_value = 0.0
        train_x[column] = pd.to_numeric(train_x[column], errors="coerce").fillna(float(fill_value))
        test_x[column] = pd.to_numeric(test_x[column], errors="coerce").fillna(float(fill_value))
    non_constant = [col for col in final_cols if train_x[col].nunique(dropna=True) > 1]
    return train_x[non_constant].copy(), test_x[non_constant].copy(), non_constant


def _fold_metric_rows(
    model_name: str,
    repeat_id: int,
    fold_id: int,
    scored: pd.DataFrame,
    calibration_status: str,
    fallback_reason: str | None,
    train_rows: int,
    train_class_count: int,
) -> list[dict[str, Any]]:
    class_metrics = _classification_metrics(scored)
    prob_metrics = _probability_quality_metrics(scored)
    auc_summary = _auc_summary(scored)
    metric_map = _metric_map(class_metrics)
    metric_map.update(_metric_map(prob_metrics))
    if not auc_summary.empty:
        metric_map["macro_roc_auc"] = float(pd.to_numeric(auc_summary["roc_auc"], errors="coerce").mean())
        metric_map["macro_pr_auc"] = float(pd.to_numeric(auc_summary["pr_auc"], errors="coerce").mean())
    rows: list[dict[str, Any]] = []
    for metric in [
        "scored_sequences",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "log_loss",
        "brier_score_multiclass",
        "ece_10_bin",
        "macro_roc_auc",
        "macro_pr_auc",
    ]:
        value = metric_map.get(metric)
        if value is None:
            continue
        rows.append(
            {
                "model": model_name,
                "repeat_id": int(repeat_id),
                "fold_id": int(fold_id),
                "metric": metric,
                "value": round(float(value), 6),
                "test_rows": int(len(scored.index)),
                "train_rows": int(train_rows),
                "train_class_count": int(train_class_count),
                "fallback_reason": fallback_reason or "",
                "calibration_status": calibration_status,
            }
        )
    return rows


def _baseline_fold_predictions(
    frame: pd.DataFrame,
    assignments: pd.Series,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    for fold_id in sorted(assignments.dropna().astype(int).unique().tolist()):
        test = frame.loc[assignments == fold_id].copy()
        train = frame.loc[assignments != fold_id].copy()
        if test.empty:
            continue
        scored = test[
            [
                "sequence_id",
                "monitor_id",
                "truth_label",
                "derived_label",
                "baseline_probs__chair",
                "baseline_probs__bed",
                "baseline_probs__room",
                "baseline_probs__no_patient",
            ]
        ].rename(
            columns={
                "derived_label": "predicted_label",
                "baseline_probs__chair": "pred_prob_chair",
                "baseline_probs__bed": "pred_prob_bed",
                "baseline_probs__room": "pred_prob_room",
                "baseline_probs__no_patient": "pred_prob_no_patient",
            }
        )
        scored["model"] = model_name
        scored["repeat_id"] = 0
        scored["fold_id"] = int(fold_id)
        scored["calibration_status"] = "production_probabilities"
        scored["fallback_reason"] = ""
        prediction_rows.extend(scored.to_dict("records"))
        fold_metric_rows.extend(
            _fold_metric_rows(
                model_name=model_name,
                repeat_id=0,
                fold_id=int(fold_id),
                scored=scored.rename(
                    columns={
                        "predicted_label": "derived_label",
                        "pred_prob_chair": "derived_prob_chair",
                        "pred_prob_bed": "derived_prob_bed",
                        "pred_prob_room": "derived_prob_room",
                        "pred_prob_no_patient": "derived_prob_no_patient",
                    }
                ),
                calibration_status="production_probabilities",
                fallback_reason=None,
                train_rows=len(train.index),
                train_class_count=int(train["truth_label"].nunique(dropna=True)),
            )
        )
    return pd.DataFrame(prediction_rows), pd.DataFrame(fold_metric_rows)


def _grouped_fold_assignments(
    frame: pd.DataFrame,
    folds: int,
    repeat_id: int,
) -> pd.Series:
    assignments = pd.Series(index=frame.index, dtype="Int64")
    monitor_ids = pd.to_numeric(frame["monitor_id"], errors="coerce")
    unique_monitors = [int(value) for value in sorted(monitor_ids.dropna().astype(int).unique().tolist())]
    if unique_monitors:
        rotation = repeat_id % len(unique_monitors)
        rotated = unique_monitors[rotation:] + unique_monitors[:rotation]
        mapping = {monitor_id: idx % folds for idx, monitor_id in enumerate(rotated)}
        assignments.loc[monitor_ids.notna()] = monitor_ids.loc[monitor_ids.notna()].astype(int).map(mapping).astype("Int64")
    missing_idx = assignments[assignments.isna()].index.tolist()
    for position, idx in enumerate(missing_idx):
        assignments.loc[idx] = int((position + repeat_id) % folds)
    return assignments.astype(int)


def _shadow_cv_predictions(
    settings: Settings,
    feature_frame: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    calibration_status: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_columns = _empty_artifacts().shadow_model_predictions.columns.tolist()
    fold_metric_columns = _empty_artifacts().shadow_cv_fold_metrics.columns.tolist()
    if feature_frame.empty or not feature_cols:
        return pd.DataFrame(columns=prediction_columns), pd.DataFrame(columns=fold_metric_columns)

    frame = feature_frame.copy()
    frame["truth_label"] = frame["truth_label"].astype("string").str.lower()
    frame = frame.loc[frame["truth_label"].isin(LOCATION_CLASSES)].copy()
    if frame.empty:
        return pd.DataFrame(columns=prediction_columns), pd.DataFrame(columns=fold_metric_columns)

    monitor_count = int(pd.to_numeric(frame["monitor_id"], errors="coerce").dropna().nunique())
    folds = max(2, min(int(settings.label_eval_shadow_cv_folds), max(monitor_count, 2), len(frame.index)))
    repeats = max(1, int(settings.label_eval_shadow_cv_repeats))
    global_priors = frame["truth_label"].value_counts(normalize=True).reindex(LOCATION_CLASSES, fill_value=0.0)

    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    for repeat_id in range(repeats):
        assignments = _grouped_fold_assignments(frame, folds, repeat_id)
        for fold_id in sorted(assignments.unique().tolist()):
            test = frame.loc[assignments == fold_id].copy()
            train = frame.loc[assignments != fold_id].copy()
            if test.empty:
                continue

            fallback_reason: str | None = None
            train_classes = [label for label in LOCATION_CLASSES if (train["truth_label"] == label).any()]
            probs = pd.DataFrame(
                {
                    f"pred_prob_{label}": [float(global_priors.get(label, 0.0))] * len(test.index)
                    for label in LOCATION_CLASSES
                },
                index=test.index,
            )

            train_x, test_x, matrix_cols = _build_model_matrix(train, test, feature_cols)
            if len(train.index) == 0 or len(train_classes) < 2:
                fallback_reason = "insufficient_training_classes"
            elif not matrix_cols:
                fallback_reason = "no_nonconstant_features"
            else:
                class_map = {label: idx for idx, label in enumerate(train_classes)}
                y_train = train["truth_label"].map(class_map).astype(int)
                x_train = sm.add_constant(train_x[matrix_cols], has_constant="add")
                x_test = sm.add_constant(test_x[matrix_cols], has_constant="add")
                try:
                    model = sm.MNLogit(y_train, x_train)
                    fit = model.fit_regularized(
                        alpha=float(settings.label_eval_shadow_alpha),
                        disp=False,
                        maxiter=200,
                    )
                    pred_matrix = fit.predict(x_test)
                    pred_matrix = pd.DataFrame(pred_matrix).reset_index(drop=True)
                    for class_idx, label in enumerate(train_classes):
                        series = pred_matrix.get(class_idx, pred_matrix.get(str(class_idx), pd.Series(dtype="float64")))
                        probs.loc[test.index, f"pred_prob_{label}"] = pd.to_numeric(
                            series,
                            errors="coerce",
                        ).fillna(0.0).to_numpy()
                except Exception:
                    fallback_reason = "fit_error"

            if fallback_reason:
                priors = train["truth_label"].value_counts(normalize=True).reindex(LOCATION_CLASSES, fill_value=0.0)
                for label in LOCATION_CLASSES:
                    probs.loc[test.index, f"pred_prob_{label}"] = float(priors.get(label, global_priors.get(label, 0.0)))

            row_sum = probs.sum(axis=1).replace(0.0, 1.0)
            probs = probs.div(row_sum, axis=0)
            label_cols = [f"pred_prob_{label}" for label in LOCATION_CLASSES]
            predicted_label = probs[label_cols].idxmax(axis=1).str.replace("pred_prob_", "", regex=False)
            scored_rows: list[dict[str, Any]] = []
            for idx, row in test.iterrows():
                record = {
                    "sequence_id": row["sequence_id"],
                    "monitor_id": row["monitor_id"],
                    "model": model_name,
                    "repeat_id": int(repeat_id),
                    "fold_id": int(fold_id),
                    "truth_label": row["truth_label"],
                    "predicted_label": predicted_label.loc[idx],
                    "pred_prob_chair": float(probs.loc[idx, "pred_prob_chair"]),
                    "pred_prob_bed": float(probs.loc[idx, "pred_prob_bed"]),
                    "pred_prob_room": float(probs.loc[idx, "pred_prob_room"]),
                    "pred_prob_no_patient": float(probs.loc[idx, "pred_prob_no_patient"]),
                    "calibration_status": calibration_status,
                    "fallback_reason": fallback_reason or "",
                }
                prediction_rows.append(record)
                scored_rows.append(record)
            scored = pd.DataFrame(scored_rows).rename(
                columns={
                    "predicted_label": "derived_label",
                    "pred_prob_chair": "derived_prob_chair",
                    "pred_prob_bed": "derived_prob_bed",
                    "pred_prob_room": "derived_prob_room",
                    "pred_prob_no_patient": "derived_prob_no_patient",
                }
            )
            fold_metric_rows.extend(
                _fold_metric_rows(
                    model_name=model_name,
                    repeat_id=int(repeat_id),
                    fold_id=int(fold_id),
                    scored=scored,
                    calibration_status=calibration_status,
                    fallback_reason=fallback_reason,
                    train_rows=len(train.index),
                    train_class_count=len(train_classes),
                )
            )
    return (
        pd.DataFrame(prediction_rows, columns=prediction_columns),
        pd.DataFrame(fold_metric_rows, columns=fold_metric_columns),
    )


def _summarize_repeated_metrics(
    model_name: str,
    fold_metrics: pd.DataFrame,
    feature_family_set: str,
    feature_count: int,
    calibration_status: str,
    skip_reason: str | None = None,
) -> pd.DataFrame:
    columns = _empty_artifacts().shadow_ablation.columns.tolist()
    if fold_metrics.empty:
        return pd.DataFrame(
            [
                {
                    "model": model_name,
                    "metric": "macro_f1",
                    "mean": pd.NA,
                    "ci_low": pd.NA,
                    "ci_high": pd.NA,
                    "evaluation_count": 0,
                    "feature_family_set": feature_family_set,
                    "feature_count": feature_count,
                    "calibration_status": calibration_status,
                    "skip_reason": skip_reason or "no_evaluations",
                    "exploratory_status": "exploratory",
                }
            ],
            columns=columns,
        )

    rows: list[dict[str, Any]] = []
    for metric, metric_rows in fold_metrics.groupby("metric", dropna=False):
        values = pd.to_numeric(metric_rows["value"], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "model": model_name,
                "metric": str(metric),
                "mean": round(float(values.mean()), 6),
                "ci_low": round(float(values.quantile(0.025)), 6),
                "ci_high": round(float(values.quantile(0.975)), 6),
                "evaluation_count": int(len(values.index)),
                "feature_family_set": feature_family_set,
                "feature_count": int(feature_count),
                "calibration_status": calibration_status,
                "skip_reason": skip_reason or "",
                "exploratory_status": "exploratory",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _summary_metric_value(summary: pd.DataFrame, model_name: str, metric: str) -> float | None:
    if summary.empty:
        return None
    row = summary.loc[(summary["model"] == model_name) & (summary["metric"] == metric), "mean"]
    if row.empty:
        return None
    value = pd.to_numeric(row.iloc[0], errors="coerce")
    return float(value) if pd.notna(value) else None


def _shadow_feature_importance(
    settings: Settings,
    feature_frame: pd.DataFrame,
    feature_cols: list[str],
    feature_family_map: dict[str, str],
    model_name: str,
) -> pd.DataFrame:
    columns = _empty_artifacts().shadow_feature_importance.columns.tolist()
    if feature_frame.empty or not feature_cols:
        return pd.DataFrame(columns=columns)

    frame = feature_frame.copy()
    train_x, _, matrix_cols = _build_model_matrix(frame, frame, feature_cols)
    if not matrix_cols:
        return pd.DataFrame(columns=columns)
    train_classes = [label for label in LOCATION_CLASSES if (frame["truth_label"] == label).any()]
    if len(train_classes) < 2:
        return pd.DataFrame(columns=columns)

    class_map = {label: idx for idx, label in enumerate(train_classes)}
    y = frame["truth_label"].map(class_map).astype(int)
    params = pd.DataFrame()
    try:
        model = sm.MNLogit(y, sm.add_constant(train_x[matrix_cols], has_constant="add"))
        fit = model.fit_regularized(alpha=float(settings.label_eval_shadow_alpha), disp=False, maxiter=200)
        params = pd.DataFrame(fit.params)
    except Exception:
        params = pd.DataFrame()

    rows: list[dict[str, Any]] = []
    if not params.empty:
        params = params.drop(index="const", errors="ignore")
        for feature, series in params.iterrows():
            abs_values = pd.to_numeric(series, errors="coerce").abs().dropna()
            if abs_values.empty:
                continue
            base_feature = feature.replace("__missing", "")
            rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "feature_family": feature_family_map.get(
                        base_feature,
                        "missingness" if feature.endswith("__missing") else "other",
                    ),
                    "coefficient_abs_mean": round(float(abs_values.mean()), 6),
                    "coefficient_abs_max": round(float(abs_values.max()), 6),
                    "exploratory_status": "exploratory",
                }
            )
    else:
        grouped = train_x[matrix_cols].copy()
        grouped["truth_label"] = frame["truth_label"].astype("string")
        class_means = grouped.groupby("truth_label", dropna=False)[matrix_cols].mean(numeric_only=True)
        for feature in matrix_cols:
            means = pd.to_numeric(class_means.get(feature), errors="coerce").dropna()
            if means.empty:
                continue
            score = float(means.std(ddof=0))
            base_feature = feature.replace("__missing", "")
            rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "feature_family": feature_family_map.get(
                        base_feature,
                        "missingness" if feature.endswith("__missing") else "other",
                    ),
                    "coefficient_abs_mean": round(score, 6),
                    "coefficient_abs_max": round(score, 6),
                    "exploratory_status": "exploratory",
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["coefficient_abs_mean", "coefficient_abs_max", "feature"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _shadow_model_outputs(
    settings: Settings,
    scored_location: pd.DataFrame,
    matched: pd.DataFrame,
    second_level_panel: pd.DataFrame,
    crossover_second_level_panel: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    metric_columns = ["model", "metric", "value"]
    comparison_columns = ["metric", "baseline_value", "shadow_value", "delta"]
    prediction_columns = _empty_artifacts().shadow_model_predictions.columns.tolist()
    if scored_location.empty:
        return (
            pd.DataFrame(columns=metric_columns),
            pd.DataFrame(columns=prediction_columns),
            pd.DataFrame(columns=comparison_columns),
            _empty_artifacts().onset_events.copy(),
            _empty_artifacts().onset_case_crossover.copy(),
            _empty_artifacts().shadow_feature_importance.copy(),
            _empty_artifacts().shadow_ablation.copy(),
            _empty_artifacts().shadow_window_sensitivity.copy(),
            _empty_artifacts().shadow_cv_fold_metrics.copy(),
        )

    baseline = scored_location.copy()
    baseline["truth_label"] = baseline["truth_label"].astype("string").str.lower()
    baseline = baseline.loc[baseline["truth_label"].isin(LOCATION_CLASSES)].copy()
    if baseline.empty:
        return (
            pd.DataFrame(columns=metric_columns),
            pd.DataFrame(columns=prediction_columns),
            pd.DataFrame(columns=comparison_columns),
            _empty_artifacts().onset_events.copy(),
            _empty_artifacts().onset_case_crossover.copy(),
            _empty_artifacts().shadow_feature_importance.copy(),
            _empty_artifacts().shadow_ablation.copy(),
            _empty_artifacts().shadow_window_sensitivity.copy(),
            _empty_artifacts().shadow_cv_fold_metrics.copy(),
        )
    baseline["derived_label"] = baseline["derived_label"].astype("string").str.lower()
    for label in LOCATION_CLASSES:
        baseline[f"derived_prob_{label}"] = pd.to_numeric(
            baseline.get(f"derived_prob_{label}"),
            errors="coerce",
        ).fillna(0.0)

    combined_panel = _combine_shadow_panels(second_level_panel, crossover_second_level_panel)
    trajectory_rows = _trajectory_feature_rows(settings, combined_panel)
    onset_events = _detect_onset_events(settings, combined_panel)
    onset_case_crossover = _onset_case_crossover_comparison(onset_events)
    feature_frame, feature_families = _build_shadow_feature_frame(
        settings,
        baseline,
        matched,
        trajectory_rows,
        onset_events,
        onset_case_crossover,
    )
    if feature_frame.empty:
        return (
            pd.DataFrame(columns=metric_columns),
            pd.DataFrame(columns=prediction_columns),
            pd.DataFrame(columns=comparison_columns),
            onset_events,
            onset_case_crossover,
            _empty_artifacts().shadow_feature_importance.copy(),
            _empty_artifacts().shadow_ablation.copy(),
            _empty_artifacts().shadow_window_sensitivity.copy(),
            _empty_artifacts().shadow_cv_fold_metrics.copy(),
        )

    baseline_predictions, baseline_fold_metrics = _baseline_fold_predictions(
        feature_frame,
        _grouped_fold_assignments(
            feature_frame,
            max(2, min(int(settings.label_eval_shadow_cv_folds), len(feature_frame.index))),
            0,
        ),
        "baseline_heuristic",
    )

    ablation_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = [baseline_fold_metrics]
    baseline_cols = feature_families.get("baseline_probs", [])
    trajectory_cols = feature_families.get("trajectory", [])
    posture_cols = feature_families.get("posture", [])
    onset_cols = feature_families.get("onset", [])
    model_specs = [
        ("stacked_baseline_probs", baseline_cols, "baseline_probs"),
        (
            "stacked_baseline_probs_trajectory",
            baseline_cols + trajectory_cols,
            "baseline_probs+trajectory",
        ),
        (
            "stacked_baseline_probs_trajectory_posture",
            baseline_cols + trajectory_cols + posture_cols,
            "baseline_probs+trajectory+posture",
        ),
        (
            "stacked_baseline_probs_trajectory_posture_onset",
            baseline_cols + trajectory_cols + posture_cols + onset_cols,
            "baseline_probs+trajectory+posture+onset",
        ),
        ("signal_only_best", trajectory_cols + posture_cols + onset_cols, "trajectory+posture+onset"),
    ]
    predictions_by_model: dict[str, pd.DataFrame] = {"baseline_heuristic": baseline_predictions}
    feature_family_map = {
        feature: family
        for family, features in feature_families.items()
        for feature in features
    }
    for model_name, feature_cols, family_set in model_specs:
        predictions, fold_metrics = _shadow_cv_predictions(
            settings=settings,
            feature_frame=feature_frame,
            feature_cols=feature_cols,
            model_name=model_name,
            calibration_status="uncalibrated_exploratory",
        )
        predictions_by_model[model_name] = predictions
        fold_frames.append(fold_metrics)
        ablation_frames.append(
            _summarize_repeated_metrics(
                model_name=model_name,
                fold_metrics=fold_metrics,
                feature_family_set=family_set,
                feature_count=len(feature_cols),
                calibration_status="uncalibrated_exploratory",
                skip_reason="no_eligible_features" if not feature_cols else None,
            )
        )

    baseline_summary = _summarize_repeated_metrics(
        model_name="baseline_heuristic",
        fold_metrics=baseline_fold_metrics,
        feature_family_set="production_heuristic",
        feature_count=0,
        calibration_status="production_probabilities",
    )
    shadow_ablation = pd.concat([baseline_summary, *ablation_frames], ignore_index=True)

    stacked_gain = (
        (_summary_metric_value(shadow_ablation, "stacked_baseline_probs_trajectory_posture_onset", "macro_f1") or 0.0)
        - (_summary_metric_value(shadow_ablation, "baseline_heuristic", "macro_f1") or 0.0)
    )
    rf_skip_reason = (
        "eligible_lt_2pp_gain_but_random_forest_not_implemented"
        if stacked_gain < 0.02
        else "deferred_gain_ge_2pp"
    )
    shadow_ablation = pd.concat(
        [
            shadow_ablation,
            _summarize_repeated_metrics(
                model_name="random_forest",
                fold_metrics=pd.DataFrame(),
                feature_family_set="gated_follow_up",
                feature_count=0,
                calibration_status="not_evaluated",
                skip_reason=rf_skip_reason,
            ),
        ],
        ignore_index=True,
    )

    primary_model = "stacked_baseline_probs_trajectory_posture_onset"
    primary_predictions = predictions_by_model.get(primary_model, pd.DataFrame(columns=prediction_columns))
    shadow_metrics_rows: list[dict[str, Any]] = []
    for metric in [
        "accuracy",
        "macro_f1",
        "log_loss",
        "ece_10_bin",
        "macro_roc_auc",
        "macro_pr_auc",
    ]:
        for model_name in ["baseline_heuristic", primary_model]:
            value = _summary_metric_value(shadow_ablation, model_name, metric)
            if value is not None:
                shadow_metrics_rows.append(
                    {"model": model_name, "metric": metric, "value": round(float(value), 6)}
                )
    shadow_metrics = pd.DataFrame(shadow_metrics_rows, columns=metric_columns)

    comparison_rows: list[dict[str, Any]] = []
    for metric in ["accuracy", "macro_f1", "log_loss", "ece_10_bin", "macro_roc_auc", "macro_pr_auc"]:
        baseline_value = _summary_metric_value(shadow_ablation, "baseline_heuristic", metric)
        shadow_value = _summary_metric_value(shadow_ablation, primary_model, metric)
        if baseline_value is None and shadow_value is None:
            continue
        delta = (
            round(float(shadow_value - baseline_value), 6)
            if baseline_value is not None and shadow_value is not None
            else pd.NA
        )
        comparison_rows.append(
            {
                "metric": metric,
                "baseline_value": round(float(baseline_value), 6) if baseline_value is not None else pd.NA,
                "shadow_value": round(float(shadow_value), 6) if shadow_value is not None else pd.NA,
                "delta": delta,
            }
        )
    shadow_comparison = pd.DataFrame(comparison_rows, columns=comparison_columns)

    fold_metrics = pd.concat(fold_frames, ignore_index=True) if fold_frames else _empty_artifacts().shadow_cv_fold_metrics.copy()
    feature_importance = _shadow_feature_importance(
        settings=settings,
        feature_frame=feature_frame,
        feature_cols=baseline_cols + trajectory_cols + posture_cols + onset_cols,
        feature_family_map=feature_family_map,
        model_name=primary_model,
    )

    window_frames: list[pd.DataFrame] = []
    for window_config, window_tokens in [
        ("60s", ["60s__"]),
        ("300s", ["300s__"]),
        ("60s_300s", ["60s__", "300s__"]),
    ]:
        selected_trajectory = [
            col for col in trajectory_cols if any(token in col for token in window_tokens)
        ]
        selected_posture = [
            col for col in posture_cols if any(token in col for token in window_tokens)
        ]
        feature_cols = baseline_cols + selected_trajectory + selected_posture + onset_cols
        _, window_fold_metrics = _shadow_cv_predictions(
            settings=settings,
            feature_frame=feature_frame,
            feature_cols=feature_cols,
            model_name=f"window_sensitivity_{window_config}",
            calibration_status="uncalibrated_exploratory",
        )
        summary = _summarize_repeated_metrics(
            model_name=f"window_sensitivity_{window_config}",
            fold_metrics=window_fold_metrics,
            feature_family_set=f"baseline_probs+trajectory[{window_config}]+onset",
            feature_count=len(feature_cols),
            calibration_status="uncalibrated_exploratory",
            skip_reason="no_eligible_features" if not feature_cols else None,
        )
        if summary.empty:
            continue
        summary = summary.rename(columns={"model": "window_config"})[
            [
                "window_config",
                "metric",
                "mean",
                "ci_low",
                "ci_high",
                "evaluation_count",
                "calibration_status",
                "exploratory_status",
            ]
        ]
        summary["window_config"] = window_config
        window_frames.append(summary)
    shadow_window_sensitivity = (
        pd.concat(window_frames, ignore_index=True)
        if window_frames
        else _empty_artifacts().shadow_window_sensitivity.copy()
    )

    return (
        shadow_metrics,
        primary_predictions if not primary_predictions.empty else pd.DataFrame(columns=prediction_columns),
        shadow_comparison,
        onset_events,
        onset_case_crossover,
        feature_importance,
        shadow_ablation,
        shadow_window_sensitivity,
        fold_metrics,
    )


def _metrics_value(frame: pd.DataFrame, name: str) -> float | None:
    if frame.empty:
        return None
    row = frame.loc[frame["metric"] == name, "value"]
    if row.empty:
        return None
    value = pd.to_numeric(row.iloc[0], errors="coerce")
    if pd.isna(value):
        return None
    return float(value)


def _annotate_benchmark_frame(
    frame: pd.DataFrame,
    benchmark_split: str,
    model_version: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    annotated = frame.copy()
    annotated.insert(0, "model_version", model_version)
    annotated.insert(0, "benchmark_split", benchmark_split)
    return annotated


def _sequence_predictions_table(
    sequence_frame: pd.DataFrame,
    benchmark_split: str,
    model_version: str,
) -> pd.DataFrame:
    columns = _empty_artifacts().sequence_predictions.columns.tolist()
    if sequence_frame.empty:
        return pd.DataFrame(columns=columns)

    subset = sequence_frame.copy()
    if "prediction_reason" not in subset.columns:
        subset["prediction_reason"] = pd.NA
    for column, default in [
        ("benchmark_truth_label", subset.get("truth_label", pd.Series(pd.NA, index=subset.index))),
        ("benchmark_truth_label_raw", pd.NA),
        ("match_source", "unmatched"),
        ("has_derived_match", False),
        ("derived_prob_sum", 0.0),
        ("departure_aware_truth_applied", False),
        ("departure_aware_source_furniture", pd.NA),
        ("departure_aware_latency_seconds", pd.NA),
    ]:
        if column not in subset.columns:
            subset[column] = default
    subset = subset[
        [
            "sequence_id",
            "event_key",
            "monitor_id",
            "truth_label",
            "benchmark_truth_label",
            "benchmark_truth_label_raw",
            "derived_label",
            "derived_prob_chair",
            "derived_prob_bed",
            "derived_prob_room",
            "derived_prob_no_patient",
            "prediction_reason",
            "match_source",
            "has_derived_match",
            "derived_prob_sum",
            "departure_aware_truth_applied",
            "departure_aware_source_furniture",
            "departure_aware_latency_seconds",
            "offscreen_sequence",
            "scored_for_location",
            "sequence_instances",
        ]
    ].copy()
    subset.insert(0, "model_version", model_version)
    subset.insert(0, "benchmark_split", benchmark_split)
    return subset[columns]


def _candidate_comparison_row(
    benchmark_split: str,
    model_version: str,
    class_metrics: pd.DataFrame,
    prob_metrics: pd.DataFrame,
    label_metrics: pd.DataFrame,
) -> dict[str, Any]:
    row = {
        "benchmark_split": benchmark_split,
        "model_version": model_version,
        "scored_sequences": _metrics_value(class_metrics, "scored_sequences"),
        "accuracy": _metrics_value(class_metrics, "accuracy"),
        "macro_precision": _metrics_value(class_metrics, "macro_precision"),
        "macro_recall": _metrics_value(class_metrics, "macro_recall"),
        "macro_f1": _metrics_value(class_metrics, "macro_f1"),
        "log_loss": _metrics_value(prob_metrics, "log_loss"),
        "ece_10_bin": _metrics_value(prob_metrics, "ece_10_bin"),
        "f1_chair": None,
        "f1_bed": None,
        "f1_room": None,
        "f1_no_patient": None,
    }
    if not label_metrics.empty:
        for label in LOCATION_CLASSES:
            f1 = label_metrics.loc[label_metrics["label"] == label, "f1"]
            row[f"f1_{label}"] = (
                float(pd.to_numeric(f1.iloc[0], errors="coerce"))
                if not f1.empty and pd.notna(pd.to_numeric(f1.iloc[0], errors="coerce"))
                else None
            )
    return row


def _benchmark_status(
    settings: Settings,
    candidate_comparison: pd.DataFrame,
) -> dict[str, Any]:
    status = {
        "primary_split": BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S,
        "secondary_split": BENCHMARK_SPLIT_SECONDARY_INFRAME_LEGACY,
        "selected_model_version": MODEL_VERSION_V3,
        "checks": [],
        "selection_checks": [],
    }
    if candidate_comparison.empty:
        return status

    primary = candidate_comparison.loc[
        candidate_comparison["benchmark_split"] == BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S
    ].copy()
    if primary.empty:
        return status

    legacy = primary.loc[primary["model_version"] == MODEL_VERSION_LEGACY]
    v2 = primary.loc[primary["model_version"] == MODEL_VERSION_V2]
    v3 = primary.loc[primary["model_version"] == MODEL_VERSION_V3]

    def _candidate_payload(
        model_version: str,
        model_frame: pd.DataFrame,
        *,
        selection_baseline: pd.DataFrame,
    ) -> dict[str, Any] | None:
        if model_frame.empty or legacy.empty:
            return None
        model_row = model_frame.iloc[0]
        legacy_row = legacy.iloc[0]
        legacy_bed_f1 = _safe_float(legacy_row.get("f1_bed"))
        bed_f1_drop = (
            round(max(0.0, legacy_bed_f1 - _safe_float(model_row.get("f1_bed"))), 6)
            if legacy_bed_f1 is not None and _safe_float(model_row.get("f1_bed")) is not None
            else None
        )
        checks = [
            {
                "name": "macro_f1",
                "operator": ">=",
                "threshold": float(settings.label_eval_v2_target_macro_f1),
                "actual": _safe_float(model_row.get("macro_f1")),
            },
            {
                "name": "chair_f1",
                "operator": ">=",
                "threshold": float(settings.label_eval_v2_target_chair_f1),
                "actual": _safe_float(model_row.get("f1_chair")),
            },
            {
                "name": "room_f1",
                "operator": ">=",
                "threshold": float(settings.label_eval_v2_target_room_f1),
                "actual": _safe_float(model_row.get("f1_room")),
            },
            {
                "name": "no_patient_f1_non_decreasing",
                "operator": ">=",
                "threshold": _safe_float(legacy_row.get("f1_no_patient")),
                "actual": _safe_float(model_row.get("f1_no_patient")),
            },
            {
                "name": "bed_f1_drop",
                "operator": "<=",
                "threshold": float(settings.label_eval_v2_target_bed_f1_drop),
                "actual": bed_f1_drop,
            },
        ]
        for check in checks:
            actual = check["actual"]
            threshold = check["threshold"]
            if actual is None or threshold is None:
                check["pass"] = False
            elif check["operator"] == ">=":
                check["pass"] = bool(actual >= float(threshold))
            else:
                check["pass"] = bool(actual <= float(threshold))

        baseline_row = selection_baseline.iloc[0] if not selection_baseline.empty else legacy_row
        selection_checks = [
            {
                "name": "macro_f1_beats_baseline",
                "operator": ">",
                "threshold": _safe_float(baseline_row.get("macro_f1")),
                "actual": _safe_float(model_row.get("macro_f1")),
            },
            {
                "name": "room_f1_non_decreasing",
                "operator": ">=",
                "threshold": _safe_float(baseline_row.get("f1_room")),
                "actual": _safe_float(model_row.get("f1_room")),
            },
            {
                "name": "no_patient_f1_non_decreasing",
                "operator": ">=",
                "threshold": _safe_float(baseline_row.get("f1_no_patient")),
                "actual": _safe_float(model_row.get("f1_no_patient")),
            },
            {
                "name": "bed_f1_drop",
                "operator": "<=",
                "threshold": float(settings.label_eval_v2_target_bed_f1_drop),
                "actual": bed_f1_drop,
            },
        ]
        for check in selection_checks:
            actual = check["actual"]
            threshold = check["threshold"]
            if actual is None or threshold is None:
                check["pass"] = False
            elif check["operator"] == ">":
                check["pass"] = bool(actual > float(threshold))
            elif check["operator"] == ">=":
                check["pass"] = bool(actual >= float(threshold))
            else:
                check["pass"] = bool(actual <= float(threshold))

        return {
            "model_version": model_version,
            "checks": checks,
            "overall_pass": all(bool(check["pass"]) for check in checks) if checks else False,
            "selection_checks": selection_checks,
            "selection_pass": all(bool(check["pass"]) for check in selection_checks) if selection_checks else False,
        }

    candidate_payloads: list[dict[str, Any]] = []
    preferred_payload = None
    if not v3.empty:
        preferred_payload = _candidate_payload(MODEL_VERSION_V3, v3, selection_baseline=v2 if not v2.empty else legacy)
        if preferred_payload is not None:
            candidate_payloads.append(preferred_payload)
    if not v2.empty:
        payload_v2 = _candidate_payload(MODEL_VERSION_V2, v2, selection_baseline=legacy)
        if payload_v2 is not None:
            candidate_payloads.append(payload_v2)
            if preferred_payload is None:
                preferred_payload = payload_v2

    if preferred_payload is None:
        status["selected_model_version"] = MODEL_VERSION_LEGACY
        return status

    status["checks"] = preferred_payload["checks"]
    status["overall_pass"] = preferred_payload["overall_pass"]
    status["selection_checks"] = preferred_payload["selection_checks"]
    status["selection_pass"] = preferred_payload["selection_pass"]
    for payload in candidate_payloads:
        if payload["selection_pass"]:
            status["selected_model_version"] = payload["model_version"]
            status["checks"] = payload["checks"]
            status["overall_pass"] = payload["overall_pass"]
            status["selection_checks"] = payload["selection_checks"]
            status["selection_pass"] = payload["selection_pass"]
            break
    else:
        status["selected_model_version"] = MODEL_VERSION_LEGACY
    return status


def _safe_float(value: Any) -> float | None:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted):
        return None
    return float(converted)


def _threshold_checks(settings: Settings, class_metrics: pd.DataFrame, prob_metrics: pd.DataFrame, resp_metrics: pd.DataFrame) -> dict[str, Any]:
    enabled = bool(settings.label_eval_thresholds_enabled)
    evaluated = enabled and settings.label_eval_mode in {"threshold_only", "dual"}
    checks: list[dict[str, Any]] = []

    if evaluated:
        macro_f1 = _metrics_value(class_metrics, "macro_f1")
        ece = _metrics_value(prob_metrics, "ece_10_bin")
        det_f1 = _metrics_value(resp_metrics, "detection_f1")
        latency_mae = _metrics_value(resp_metrics, "latency_mae_seconds")

        checks = [
            {
                "name": "macro_f1",
                "operator": ">=",
                "threshold": float(settings.label_eval_threshold_macro_f1),
                "actual": macro_f1,
                "pass": bool(macro_f1 is not None and macro_f1 >= settings.label_eval_threshold_macro_f1),
            },
            {
                "name": "ece_10_bin",
                "operator": "<=",
                "threshold": float(settings.label_eval_threshold_ece),
                "actual": ece,
                "pass": bool(ece is not None and ece <= settings.label_eval_threshold_ece),
            },
            {
                "name": "response_detection_f1",
                "operator": ">=",
                "threshold": float(settings.label_eval_threshold_response_detection_f1),
                "actual": det_f1,
                "pass": bool(
                    det_f1 is not None and det_f1 >= settings.label_eval_threshold_response_detection_f1
                ),
            },
            {
                "name": "response_latency_mae_seconds",
                "operator": "<=",
                "threshold": float(settings.label_eval_threshold_response_latency_mae_seconds),
                "actual": latency_mae,
                "pass": bool(
                    latency_mae is not None
                    and latency_mae <= settings.label_eval_threshold_response_latency_mae_seconds
                ),
            },
        ]

    overall_pass = None
    if checks:
        overall_pass = all(bool(check["pass"]) for check in checks)

    return {
        "enabled": enabled,
        "mode": settings.label_eval_mode,
        "evaluated": evaluated,
        "checks": checks,
        "overall_pass": overall_pass,
    }


def evaluate_livestream_derivations_against_truth(
    settings: Settings,
    event_windows: pd.DataFrame,
    second_level_panel: pd.DataFrame | None = None,
    crossover_second_level_panel: pd.DataFrame | None = None,
) -> LabelEvalArtifacts:
    truth_instances, truth_summary = _build_truth_instances(
        settings,
        benchmark_split=BENCHMARK_SPLIT_SECONDARY_INFRAME_LEGACY,
    )
    primary_truth_instances, _ = _build_truth_instances(
        settings,
        benchmark_split=BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S,
    )
    primary_truth_raw_audit_instances, _ = _build_truth_instances(
        settings,
        benchmark_split=BENCHMARK_SPLIT_PRIMARY_4CLASS_V2_RAW_AUDIT,
    )
    if truth_instances.empty and primary_truth_instances.empty and primary_truth_raw_audit_instances.empty:
        artifacts = _empty_artifacts()
        if truth_summary.get("status") == "ok":
            artifacts.population_stats = _population_stats(truth_instances, pd.DataFrame(), truth_summary)
        else:
            artifacts.population_stats = pd.DataFrame(
                [
                    {
                        "metric": "status",
                        "value": truth_summary.get("status", "missing_truth"),
                    }
                ]
            )
        return artifacts

    panel = second_level_panel if second_level_panel is not None else pd.DataFrame()
    crossover_panel = crossover_second_level_panel if crossover_second_level_panel is not None else pd.DataFrame()
    derived_event_windows = build_panel_prefall_event_windows(settings, event_windows, panel)

    matched = _match_with_derived(
        truth_instances,
        derived_event_windows,
        extra_derived_columns=("prefall_location_reason",),
    )
    if "prefall_location_reason" in matched.columns:
        matched["prediction_reason"] = matched["prefall_location_reason"]
    sequence_frame = _build_sequence_truth_predictions(matched)

    scored_location = sequence_frame.loc[sequence_frame["scored_for_location"]].copy() if not sequence_frame.empty else pd.DataFrame()
    class_metrics = _classification_metrics(scored_location)
    confusion = _confusion_matrix_rows(scored_location)
    prob_metrics = _probability_quality_metrics(scored_location)
    auc_summary = _auc_summary(scored_location)
    threshold_sweep = _threshold_sweep(scored_location, settings.label_eval_sweep_points)
    calibration_curve = _calibration_curve(scored_location, settings.label_eval_calibration_bins)
    operating_point = _operating_point(settings, threshold_sweep)
    response_metrics = _response_metrics(sequence_frame)
    pop_stats = _population_stats(truth_instances, sequence_frame, truth_summary)
    truth_prefall_location = _truth_prefall_location_summary(truth_instances)
    tag_location_association = _tag_location_association(truth_instances)
    signal_location_profile = _signal_location_profile(sequence_frame)
    association_summary = _association_summary(
        truth_prefall_location,
        tag_location_association,
        signal_location_profile,
    )
    (
        shadow_metrics,
        shadow_predictions,
        shadow_comparison,
        onset_events,
        onset_case_crossover,
        shadow_feature_importance,
        shadow_ablation,
        shadow_window_sensitivity,
        shadow_cv_fold_metrics,
    ) = _shadow_model_outputs(
        settings,
        scored_location,
        matched,
        panel,
        crossover_panel,
    )
    threshold_checks = _threshold_checks(settings, class_metrics, prob_metrics, response_metrics)

    benchmark_sequence_metric_frames: list[pd.DataFrame] = []
    benchmark_label_metric_frames: list[pd.DataFrame] = []
    benchmark_confusion_frames: list[pd.DataFrame] = []
    benchmark_probability_frames: list[pd.DataFrame] = []
    sequence_prediction_frames: list[pd.DataFrame] = []
    candidate_rows: list[dict[str, Any]] = []

    v2_eval_windows = derived_event_windows.copy()
    for target, source in [
        ("pre_prob_chair", "pre_prob_v2_chair"),
        ("pre_prob_bed", "pre_prob_v2_bed"),
        ("pre_prob_room", "pre_prob_v2_room"),
        ("pre_prob_no_patient", "pre_prob_v2_no_patient"),
    ]:
        if source in v2_eval_windows.columns:
            v2_eval_windows[target] = v2_eval_windows[source]
    if "prefall_location_v2_reason" in v2_eval_windows.columns:
        v2_eval_windows["prediction_reason"] = v2_eval_windows["prefall_location_v2_reason"]

    v3_eval_windows = derived_event_windows.copy()
    for target, source in [
        ("pre_prob_chair", "pre_prob_v3_chair"),
        ("pre_prob_bed", "pre_prob_v3_bed"),
        ("pre_prob_room", "pre_prob_v3_room"),
        ("pre_prob_no_patient", "pre_prob_v3_no_patient"),
    ]:
        if source in v3_eval_windows.columns:
            v3_eval_windows[target] = v3_eval_windows[source]
    if "prefall_location_v3_reason" in v3_eval_windows.columns:
        v3_eval_windows["prediction_reason"] = v3_eval_windows["prefall_location_v3_reason"]

    benchmark_specs = [
        (
            BENCHMARK_SPLIT_SECONDARY_INFRAME_LEGACY,
            truth_instances,
            False,
        ),
        (
            BENCHMARK_SPLIT_PRIMARY_4CLASS_DEPARTURE_AWARE_10S,
            primary_truth_instances,
            True,
        ),
        (
            BENCHMARK_SPLIT_PRIMARY_4CLASS_V2_RAW_AUDIT,
            primary_truth_raw_audit_instances,
            True,
        ),
    ]
    model_specs = [
        (MODEL_VERSION_LEGACY, event_windows, ()),
        (MODEL_VERSION_V2, v2_eval_windows, ("prediction_reason",)),
        (MODEL_VERSION_V3, v3_eval_windows, ("prediction_reason",)),
    ]
    for benchmark_split, benchmark_truth, allow_offscreen_scoring in benchmark_specs:
        if benchmark_truth.empty:
            continue
        for model_version, benchmark_windows, extra_cols in model_specs:
            matched_benchmark = _match_with_derived(
                benchmark_truth,
                benchmark_windows,
                extra_derived_columns=extra_cols,
            )
            sequence_benchmark = _build_sequence_truth_predictions(
                matched_benchmark,
                allow_offscreen_scoring=allow_offscreen_scoring,
            )
            scored_benchmark = (
                sequence_benchmark.loc[sequence_benchmark["scored_for_location"]].copy()
                if not sequence_benchmark.empty
                else pd.DataFrame()
            )
            class_metrics_benchmark = _classification_metrics(scored_benchmark)
            label_metrics_benchmark = _label_metrics(scored_benchmark)
            confusion_benchmark = _confusion_matrix_rows(scored_benchmark)
            prob_metrics_benchmark = _probability_quality_metrics(scored_benchmark)

            benchmark_sequence_metric_frames.append(
                _annotate_benchmark_frame(class_metrics_benchmark, benchmark_split, model_version)
            )
            benchmark_label_metric_frames.append(
                _annotate_benchmark_frame(label_metrics_benchmark, benchmark_split, model_version)
            )
            benchmark_confusion_frames.append(
                _annotate_benchmark_frame(confusion_benchmark, benchmark_split, model_version)
            )
            benchmark_probability_frames.append(
                _annotate_benchmark_frame(prob_metrics_benchmark, benchmark_split, model_version)
            )
            sequence_prediction_frames.append(
                _sequence_predictions_table(sequence_benchmark, benchmark_split, model_version)
            )
            candidate_rows.append(
                _candidate_comparison_row(
                    benchmark_split,
                    model_version,
                    class_metrics_benchmark,
                    prob_metrics_benchmark,
                    label_metrics_benchmark,
                )
            )

    benchmark_sequence_metrics = (
        pd.concat(benchmark_sequence_metric_frames, ignore_index=True)
        if benchmark_sequence_metric_frames
        else _empty_artifacts().benchmark_sequence_metrics.copy()
    )
    benchmark_label_metrics = (
        pd.concat(benchmark_label_metric_frames, ignore_index=True)
        if benchmark_label_metric_frames
        else _empty_artifacts().benchmark_label_metrics.copy()
    )
    benchmark_confusion_matrix = (
        pd.concat(benchmark_confusion_frames, ignore_index=True)
        if benchmark_confusion_frames
        else _empty_artifacts().benchmark_confusion_matrix.copy()
    )
    benchmark_probability_quality = (
        pd.concat(benchmark_probability_frames, ignore_index=True)
        if benchmark_probability_frames
        else _empty_artifacts().benchmark_probability_quality.copy()
    )
    sequence_predictions = (
        pd.concat(sequence_prediction_frames, ignore_index=True)
        if sequence_prediction_frames
        else _empty_artifacts().sequence_predictions.copy()
    )
    candidate_comparison = pd.DataFrame(
        candidate_rows,
        columns=_empty_artifacts().candidate_comparison.columns.tolist(),
    )
    benchmark_status = _benchmark_status(settings, candidate_comparison)

    furniture_origin_chain_df = _furniture_origin_chain_summary(truth_instances)
    post_departure_latency_df = _post_departure_latency_table(truth_instances)
    tag_origin_chain_crosstab_df = _tag_origin_chain_crosstab(truth_instances)

    return LabelEvalArtifacts(
        sequence_metrics=class_metrics,
        confusion_matrix=confusion,
        probability_quality=prob_metrics,
        auc_summary=auc_summary,
        threshold_sweep=threshold_sweep,
        calibration_curve=calibration_curve,
        response_timing=response_metrics,
        population_stats=pop_stats,
        truth_prefall_location=truth_prefall_location,
        tag_location_association=tag_location_association,
        signal_location_profile=signal_location_profile,
        association_summary=association_summary,
        shadow_model_metrics=shadow_metrics,
        shadow_model_predictions=shadow_predictions,
        shadow_model_comparison=shadow_comparison,
        onset_events=onset_events,
        onset_case_crossover=onset_case_crossover,
        shadow_feature_importance=shadow_feature_importance,
        shadow_ablation=shadow_ablation,
        shadow_window_sensitivity=shadow_window_sensitivity,
        shadow_cv_fold_metrics=shadow_cv_fold_metrics,
        operating_point=operating_point,
        threshold_checks=threshold_checks,
        furniture_origin_chain=furniture_origin_chain_df,
        post_departure_latency=post_departure_latency_df,
        tag_origin_chain_crosstab=tag_origin_chain_crosstab_df,
        benchmark_sequence_metrics=benchmark_sequence_metrics,
        benchmark_label_metrics=benchmark_label_metrics,
        benchmark_confusion_matrix=benchmark_confusion_matrix,
        benchmark_probability_quality=benchmark_probability_quality,
        sequence_predictions=sequence_predictions,
        candidate_comparison=candidate_comparison,
        benchmark_status=benchmark_status,
    )
