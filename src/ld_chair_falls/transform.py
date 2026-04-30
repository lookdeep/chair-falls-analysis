from __future__ import annotations

from typing import Any

import pandas as pd

from .cohort_map import load_and_validate_cohort_map, persist_cohort_map
from .config import Settings
from .dayparts import classify_hour
from .extract import load_raw_tables
from .utils import ensure_dir, write_json, write_yaml

FALL_BASE_COLUMNS = [
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
]

HOURLY_BASE_COLUMNS = [
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
]

DEFAULT_MIN_OBSERVED_HOURS = 4
MIN_COVERAGE_RATIO = 0.95
MIN_VALID_PCT_RATIO = 0.60


def _parse_utc_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _parse_local_wall_timestamp(series: pd.Series) -> pd.Series:
    return _parse_utc_timestamp(series).dt.tz_localize(None)


def _utc_to_local_wall_timestamp(series: pd.Series, timezone: str) -> pd.Series:
    return _parse_utc_timestamp(series).dt.tz_convert(timezone).dt.tz_localize(None)


def _normalize_hour_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None).dt.floor("h")


def _unique_monitor_hour_patients(hour_keys: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["hospital_id", "division_id", "monitor_id", "hour_ts"]
    optional_cols = [column for column in ("patient_id", "cohort_type", "daypart") if column in hour_keys.columns]
    available_cols = base_cols + optional_cols
    frame = hour_keys[available_cols].copy()
    if "patient_id" not in frame.columns:
        return pd.DataFrame(columns=base_cols + ["patient_id"])
    frame = frame.dropna(subset=["hour_ts", "patient_id"]).drop_duplicates()
    if frame.empty:
        return pd.DataFrame(columns=base_cols + ["patient_id"])
    aggregations: dict[str, tuple[str, str]] = {
        "patient_id": ("patient_id", "first"),
        "patient_id_nunique": ("patient_id", "nunique"),
    }
    if "cohort_type" in frame.columns:
        aggregations["cohort_type"] = ("cohort_type", "first")
    if "daypart" in frame.columns:
        aggregations["daypart"] = ("daypart", "first")
    grouped = frame.groupby(base_cols, dropna=False).agg(**aggregations).reset_index()
    grouped = grouped.loc[grouped["patient_id_nunique"] == 1].copy()
    return grouped.drop(columns=["patient_id_nunique"])


def _link_events_on_hour(
    events: pd.DataFrame,
    hour_keys: pd.DataFrame,
    hour_series: pd.Series,
) -> pd.DataFrame:
    if events.empty or hour_keys.empty:
        return pd.DataFrame(columns=events.columns.tolist() + [column for column in hour_keys.columns if column not in events.columns])

    working_hour_keys = hour_keys.copy()
    working_hour_keys["hour_ts"] = _normalize_hour_timestamp(working_hour_keys["hour_ts"])
    working_events = events.copy()
    normalized_hour_series = _normalize_hour_timestamp(hour_series)

    full_keys = ["hospital_id", "division_id", "monitor_id", "patient_id", "cohort_type", "hour_ts"]
    event_hour = working_events.assign(hour_ts=normalized_hour_series).copy()

    known = event_hour.loc[event_hour["patient_id"].notna()].merge(
        working_hour_keys,
        on=full_keys,
        how="inner",
    )

    missing = event_hour.loc[event_hour["patient_id"].isna()].copy()
    if missing.empty:
        return known

    monitor_hour = _unique_monitor_hour_patients(hour_keys)
    if monitor_hour.empty:
        return known
    monitor_hour["hour_ts"] = _normalize_hour_timestamp(monitor_hour["hour_ts"])

    recovered = missing.merge(
        monitor_hour,
        on=["hospital_id", "division_id", "monitor_id", "hour_ts"],
        how="inner",
        suffixes=("", "_resolved"),
    )
    if recovered.empty:
        return known

    recovered["patient_id"] = recovered["patient_id_resolved"]
    if "cohort_type_resolved" in recovered.columns:
        recovered["cohort_type"] = recovered["cohort_type_resolved"]
    recovered = recovered.drop(
        columns=[
            column
            for column in ("patient_id_resolved", "cohort_type_resolved", "daypart_resolved")
            if column in recovered.columns
        ]
    )
    return pd.concat([known, recovered], ignore_index=True)


def _coerce_unit_key_types(df: pd.DataFrame, *, include_cohort: bool = False) -> pd.DataFrame:
    result = df.copy()
    numeric_cols = ["hospital_id", "division_id", "monitor_id", "patient_id"]
    for column in numeric_cols:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    if include_cohort and "cohort_type" in result.columns:
        result["cohort_type"] = result["cohort_type"].astype("string")
    return result


def _ensure_columns(df: pd.DataFrame, columns: list[str], numeric: set[str] | None = None) -> pd.DataFrame:
    result = df.copy()
    numeric = numeric or set()
    for column in columns:
        if column not in result.columns:
            result[column] = 0.0 if column in numeric else pd.NA
    return result


def _derive_daypart(hour_series: pd.Series) -> pd.Series:
    return hour_series.apply(classify_hour)


def build_fall_events_clean(
    raw_falls: pd.DataFrame,
    *,
    hospital_timezone: str = "America/Chicago",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = _ensure_columns(raw_falls, FALL_BASE_COLUMNS)

    ts_utc = _parse_utc_timestamp(df["timestamp"])
    ts_local = _parse_local_wall_timestamp(df["timestamp_local"])
    fallback_local_from_utc = _utc_to_local_wall_timestamp(df["timestamp"], hospital_timezone)
    df["timestamp"] = ts_utc
    df["timestamp_local"] = ts_local
    df["event_ts_utc"] = ts_utc
    df["event_ts_local"] = ts_local.fillna(fallback_local_from_utc)
    df["event_hour_ts"] = df["event_ts_local"].dt.floor("h")
    df["event_local_timezone_source"] = "timestamp_local"
    df.loc[ts_local.isna() & fallback_local_from_utc.notna(), "event_local_timezone_source"] = "hospital_timezone"
    df.loc[df["event_ts_local"].isna(), "event_local_timezone_source"] = "missing"

    before = int(len(df.index))
    dedupe_cols = [
        "hospital_id",
        "division_id",
        "patient_id",
        "monitor_id",
        "event_ts_local",
    ]
    df = df.drop_duplicates(subset=dedupe_cols, keep="first").copy()
    after = int(len(df.index))

    injury_cols = [column for column in df.columns if "injury" in column.lower()]
    if not injury_cols:
        df["injury_label"] = pd.NA
        injury_cols = ["injury_label"]

    df["injury_fields_present"] = df[injury_cols].notna().any(axis=1)
    df["injury_fields_complete"] = df[injury_cols].notna().all(axis=1)
    df["lineage_source"] = "bq_falls_directory"

    metrics = {
        "fall_events_before_dedupe": before,
        "fall_events_after_dedupe": after,
        "fall_events_removed_duplicates": before - after,
        "injury_columns_detected": injury_cols,
        "injury_complete_rows": int(df["injury_fields_complete"].sum()),
    }
    return df, metrics


def build_hourly_location(raw_hourly: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_columns(
        raw_hourly,
        HOURLY_BASE_COLUMNS,
        numeric={
            "pct_chair",
            "pct_bed",
            "pct_ambulatory",
            "pct_not_located",
            "num_alarms",
            "num_nudges",
            "num_announcements",
        },
    )

    # hour_ts is sourced from hour_bin_local and treated as local wall-clock time.
    df["hour_ts"] = _parse_local_wall_timestamp(df["hour_ts"]).dt.floor("h")
    for column in ["pct_chair", "pct_bed", "pct_ambulatory", "pct_not_located"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    for column in ["num_alarms", "num_nudges", "num_announcements"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype("int64")

    pct_sum = df["pct_chair"] + df["pct_bed"] + df["pct_ambulatory"] + df["pct_not_located"]
    df["pct_sum"] = pct_sum
    df["pct_bounds_violation"] = (
        (df["pct_chair"] < 0)
        | (df["pct_bed"] < 0)
        | (df["pct_ambulatory"] < 0)
        | (df["pct_not_located"] < 0)
        | (df["pct_chair"] > 1)
        | (df["pct_bed"] > 1)
        | (df["pct_ambulatory"] > 1)
        | (df["pct_not_located"] > 1)
    )
    df["pct_sum_violation"] = (pct_sum - 1.0).abs() > 0.02
    df["row_valid_pct"] = ~(df["pct_bounds_violation"] | df["pct_sum_violation"])

    df["daypart"] = _derive_daypart(df["hour_ts"].dt.hour)
    df["day_of_week"] = df["hour_ts"].dt.day_name().fillna("unknown")
    df["calendar_month"] = df["hour_ts"].dt.strftime("%Y-%m").fillna("unknown")
    df["calendar_quarter"] = (
        df["hour_ts"].dt.strftime("%Y")
        + "Q"
        + (((df["hour_ts"].dt.month - 1) // 3) + 1).astype("Int64").astype("string")
    ).fillna("unknown")

    return df


def attach_cohort_labels(
    hourly_df: pd.DataFrame,
    fall_events_df: pd.DataFrame,
    cohort_map_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    map_df = cohort_map_df.drop_duplicates(subset=["monitor_id"], keep="first").copy()

    hourly = hourly_df.merge(map_df, on="monitor_id", how="left")
    events = fall_events_df.merge(map_df, on="monitor_id", how="left")

    hourly["cohort_mapped"] = hourly["cohort_type"].notna()
    events["cohort_mapped"] = events["cohort_type"].notna()

    monitors_hourly = set(hourly.loc[~hourly["cohort_mapped"], "monitor_id"].dropna().tolist())
    monitors_events = set(events.loc[~events["cohort_mapped"], "monitor_id"].dropna().tolist())
    unmapped = sorted(monitors_hourly.union(monitors_events))

    hourly_rows = int(len(hourly.index))
    event_rows = int(len(events.index))
    monitor_uniqueness_violations: list[int] = []
    scope_cols = [col for col in ("hospital_id", "division_id") if col in hourly.columns]
    if scope_cols and "monitor_id" in hourly.columns:
        scope_cardinality = (
            hourly.dropna(subset=["monitor_id"])
            .groupby("monitor_id", dropna=True)[scope_cols]
            .nunique(dropna=True)
        )
        violated = scope_cardinality[(scope_cardinality > 1).any(axis=1)]
        monitor_uniqueness_violations = sorted(int(value) for value in violated.index.tolist())

    report = {
        "hourly_total_rows": hourly_rows,
        "hourly_mapped_rows": int(hourly["cohort_mapped"].sum()) if hourly_rows else 0,
        "hourly_mapped_rate": float(hourly["cohort_mapped"].mean()) if hourly_rows else 1.0,
        "events_total_rows": event_rows,
        "events_mapped_rows": int(events["cohort_mapped"].sum()) if event_rows else 0,
        "events_mapped_rate": float(events["cohort_mapped"].mean()) if event_rows else 1.0,
        "unmapped_monitor_ids": unmapped,
        "monitor_id_uniqueness_violations": monitor_uniqueness_violations,
    }
    return hourly, events, report


def build_eligibility(
    hourly_df: pd.DataFrame,
    *,
    min_observed_hours: int = DEFAULT_MIN_OBSERVED_HOURS,
) -> pd.DataFrame:
    group_cols = ["hospital_id", "division_id", "monitor_id", "patient_id", "cohort_type"]
    if hourly_df.empty:
        return pd.DataFrame(
            columns=group_cols
            + [
                "observed_hours",
                "min_observed_hours_threshold",
                "expected_hours",
                "coverage_ratio",
                "complete_hourly_coverage",
                "duration_requirement_met",
                "valid_pct_ratio",
                "all_rows_valid_pct",
                "eligible",
                "exclusion_reason",
            ]
        )

    rows = []
    grouped = hourly_df.groupby(group_cols, dropna=False)
    for group_key, group in grouped:
        hours = group["hour_ts"].dropna().drop_duplicates().sort_values()
        observed = int(len(hours.index))
        if observed == 0:
            expected = 0
            coverage_ratio = 0.0
        else:
            span = hours.iloc[-1] - hours.iloc[0]
            expected = int(span.total_seconds() // 3600) + 1
            coverage_ratio = float(observed / expected) if expected else 0.0

        complete = coverage_ratio >= MIN_COVERAGE_RATIO
        duration_ok = observed >= min_observed_hours
        valid_pct_ratio = (
            float(pd.to_numeric(group["row_valid_pct"], errors="coerce").fillna(False).mean())
            if "row_valid_pct" in group
            else 1.0
        )
        valid_pct = valid_pct_ratio >= MIN_VALID_PCT_RATIO
        cohort_mapped = bool(group["cohort_type"].notna().all())

        eligible = complete and duration_ok and valid_pct and cohort_mapped
        reasons = []
        if not complete:
            reasons.append("incomplete_hourly_coverage")
        if not duration_ok:
            reasons.append("duration_lt_min_hours")
        if not valid_pct:
            reasons.append("invalid_pct_row")
        if not cohort_mapped:
            reasons.append("unmapped_cohort")

        row = {column: value for column, value in zip(group_cols, group_key, strict=False)}
        row.update(
            {
                "observed_hours": observed,
                "min_observed_hours_threshold": min_observed_hours,
                "expected_hours": expected,
                "coverage_ratio": round(coverage_ratio, 6),
                "complete_hourly_coverage": complete,
                "duration_requirement_met": duration_ok,
                "valid_pct_ratio": round(valid_pct_ratio, 6),
                "all_rows_valid_pct": valid_pct,
                "eligible": eligible,
                "exclusion_reason": "|".join(reasons),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def build_analysis_base(
    hourly_df: pd.DataFrame,
    events_df: pd.DataFrame,
    eligibility_df: pd.DataFrame,
) -> pd.DataFrame:
    unit_cols = ["hospital_id", "division_id", "monitor_id", "patient_id", "cohort_type"]
    if hourly_df.empty:
        return pd.DataFrame(columns=HOURLY_BASE_COLUMNS + ["cohort_type", "falls_in_hour", "fall_event_flag"])

    hourly_df = _coerce_unit_key_types(hourly_df, include_cohort=True)
    hourly_df["hour_ts"] = _normalize_hour_timestamp(hourly_df["hour_ts"])
    eligible_units = _coerce_unit_key_types(
        eligibility_df.loc[eligibility_df["eligible"], unit_cols],
        include_cohort=True,
    )
    if eligible_units.empty:
        base = hourly_df.iloc[0:0].copy()
        base["falls_in_hour"] = pd.Series(dtype="int64")
        base["fall_event_flag"] = pd.Series(dtype="bool")
        return base

    base = hourly_df.merge(eligible_units, on=unit_cols, how="inner")

    if events_df.empty:
        base["falls_in_hour"] = 0
        base["fall_event_flag"] = False
        return base

    events = _coerce_unit_key_types(events_df, include_cohort=True)
    event_hour_local = pd.to_datetime(events.get("event_ts_local"), errors="coerce").dt.floor("h")
    event_hour_utc = pd.to_datetime(events.get("event_ts_utc"), errors="coerce", utc=True).dt.tz_localize(None).dt.floor("h")
    base_hour_keys = (
        base[["hospital_id", "division_id", "monitor_id", "patient_id", "cohort_type", "hour_ts"]]
        .dropna(subset=["hour_ts"])
        .drop_duplicates()
    )
    local_linkable = _link_events_on_hour(events, base_hour_keys, event_hour_local)
    utc_linkable = _link_events_on_hour(events, base_hour_keys, event_hour_utc)
    if len(utc_linkable.index) > len(local_linkable.index):
        events["hour_ts"] = event_hour_utc
        events = utc_linkable.copy()
    else:
        events["hour_ts"] = event_hour_local
        events = local_linkable.copy()
    event_link_cols = ["hospital_id", "division_id", "monitor_id", "patient_id", "cohort_type", "hour_ts"]
    fall_counts = (
        events.dropna(subset=["hour_ts"])
        .groupby(event_link_cols, dropna=False)
        .size()
        .rename("falls_in_hour")
        .reset_index()
    )

    base = base.merge(fall_counts, on=event_link_cols, how="left")
    base["falls_in_hour"] = base["falls_in_hour"].fillna(0).astype(int)
    base["fall_event_flag"] = base["falls_in_hour"] > 0
    return base


def run_transform_pipeline(settings: Settings) -> dict[str, Any]:
    raw_tables = load_raw_tables(settings)
    for table_name in ("fall_events_source", "hourly_location_aggregation", "key_dimensions"):
        table = raw_tables.get(table_name, pd.DataFrame())
        if "hospital_id" in table.columns:
            raw_tables[table_name] = table.loc[table["hospital_id"] == settings.study_hospital_id].copy()

    cohort_map_df, cohort_report = load_and_validate_cohort_map(settings, raw_tables=raw_tables)
    ensure_dir(settings.paths.staged_run_dir)
    ensure_dir(settings.paths.qa_dir)

    cohort_map_path, cohort_report_path, cohort_meta = persist_cohort_map(
        settings,
        cohort_map_df,
        cohort_report,
    )

    fall_clean, fall_metrics = build_fall_events_clean(
        raw_tables["fall_events_source"],
        hospital_timezone=settings.hospital_timezone,
    )
    hourly = build_hourly_location(raw_tables["hourly_location_aggregation"])
    hourly_labeled, events_labeled, mapping_report = attach_cohort_labels(hourly, fall_clean, cohort_map_df)
    eligibility = build_eligibility(hourly_labeled, min_observed_hours=settings.min_observed_hours)
    analysis_base = build_analysis_base(hourly_labeled, events_labeled, eligibility)

    staged_dir = settings.paths.staged_run_dir
    fall_path = staged_dir / "prep.fall_events_clean_v1.parquet"
    hourly_path = staged_dir / "prep.patient_hour_location_v1.parquet"
    hourly_labeled_path = staged_dir / "prep.patient_hour_location_with_cohort_v1.parquet"
    events_labeled_path = staged_dir / "prep.fall_events_with_cohort_v1.parquet"
    eligibility_path = staged_dir / "prep.patient_hour_eligibility_v1.parquet"
    analysis_base_path = staged_dir / "prep.patient_hour_analysis_base_v1.parquet"

    fall_clean.to_parquet(fall_path, index=False)
    hourly.to_parquet(hourly_path, index=False)
    hourly_labeled.to_parquet(hourly_labeled_path, index=False)
    events_labeled.to_parquet(events_labeled_path, index=False)
    eligibility.to_parquet(eligibility_path, index=False)
    analysis_base.to_parquet(analysis_base_path, index=False)

    unmapped_path = settings.paths.qa_dir / f"unmapped_monitors_{settings.run_id}.csv"
    pd.DataFrame({"monitor_id": mapping_report["unmapped_monitor_ids"]}).to_csv(unmapped_path, index=False)

    cohort_status = str(cohort_report.get("status", "missing"))
    cohort_fallback_used = cohort_status != "valid"
    cohort_warning_codes: list[str] = []
    if cohort_fallback_used:
        cohort_warning_codes.append(f"cohort_map_status_{cohort_status}")
    if mapping_report["unmapped_monitor_ids"]:
        cohort_warning_codes.append("unmapped_monitors_present")
    if mapping_report["monitor_id_uniqueness_violations"]:
        cohort_warning_codes.append("monitor_id_uniqueness_violations_detected")

    transform_metrics = {
        "run_id": settings.run_id,
        "hospital_timezone": settings.hospital_timezone,
        "cohort_validation_status": cohort_status,
        "cohort_fallback_used": cohort_fallback_used,
        "cohort_warning_codes": cohort_warning_codes,
        "eligibility_policy": {
            "min_observed_hours": settings.min_observed_hours,
            "min_coverage_ratio": MIN_COVERAGE_RATIO,
            "min_valid_pct_ratio": MIN_VALID_PCT_RATIO,
        },
        "fall_metrics": fall_metrics,
        "mapping_report": mapping_report,
        "row_counts": {
            "cohort_map": int(len(cohort_map_df.index)),
            "fall_events_clean": int(len(fall_clean.index)),
            "hourly_location": int(len(hourly.index)),
            "hourly_location_with_cohort": int(len(hourly_labeled.index)),
            "patient_hour_eligibility": int(len(eligibility.index)),
            "patient_hour_analysis_base": int(len(analysis_base.index)),
        },
        "paths": {
            "cohort_map": str(cohort_map_path),
            "cohort_report": str(cohort_report_path),
            "fall_events_clean": str(fall_path),
            "hourly_location": str(hourly_path),
            "hourly_labeled": str(hourly_labeled_path),
            "events_labeled": str(events_labeled_path),
            "eligibility": str(eligibility_path),
            "analysis_base": str(analysis_base_path),
            "unmapped_monitors": str(unmapped_path),
        },
    }
    transform_metrics.update(cohort_meta)

    metrics_path = settings.paths.qa_dir / f"transform_metrics_{settings.run_id}.json"
    write_json(metrics_path, transform_metrics)

    manifest_path = settings.paths.manifests_dir / f"transform_manifest_{settings.run_id}.yaml"
    transform_metrics["manifest_path"] = str(manifest_path)
    write_yaml(manifest_path, transform_metrics)
    return transform_metrics
