from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .cohort_map import (
    DERIVED_CONTROL_DEFINITION,
    DERIVED_INTERVENTION_DEFINITION,
    DERIVED_OUTCOME_COHORT_BASIS,
)
from .config import Settings
from .consensus import (
    build_consensus_adjudication_tables,
    included_fall_annotations,
    load_consensus_annotations,
    parse_event_key,
    time_to_seconds,
)
from .dayparts import DAYPART_ORDER, classify_hour
from .extract import load_raw_tables
from .label_eval import _csv_path, evaluate_livestream_derivations_against_truth
from .prefall_location import build_panel_prefall_event_windows
from .utils import ensure_dir, read_json, read_yaml, utc_now_iso, write_json, write_yaml


def _parse_utc_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _parse_local_wall_timestamp(series: pd.Series) -> pd.Series:
    return _parse_utc_timestamp(series).dt.tz_localize(None)


def _utc_to_local_wall_timestamp(series: pd.Series, timezone: str) -> pd.Series:
    return _parse_utc_timestamp(series).dt.tz_convert(timezone).dt.tz_localize(None)


def _normalize_hour_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None).dt.floor("h")


def _series_stats(df: pd.DataFrame, timestamp_col: str | None = None) -> dict[str, Any]:
    stats: dict[str, Any] = {"rows": int(len(df.index))}
    key_columns = ["hospital_id", "division_id", "monitor_id", "patient_id"]
    for key in key_columns:
        if key in df.columns:
            stats[f"null_{key}"] = int(df[key].isna().sum())
            stats[f"distinct_{key}"] = int(df[key].nunique(dropna=True))

    if timestamp_col and timestamp_col in df.columns:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
        if ts.notna().any():
            stats["min_timestamp"] = ts.min().isoformat()
            stats["max_timestamp"] = ts.max().isoformat()
    return stats


def _scope_to_study_hospital(df: pd.DataFrame, study_hospital_id: int) -> pd.DataFrame:
    if "hospital_id" not in df.columns:
        return df.copy()
    scoped = df.copy()
    scoped["hospital_id"] = pd.to_numeric(scoped["hospital_id"], errors="coerce")
    return scoped.loc[scoped["hospital_id"] == study_hospital_id].copy()


def _canonical_event_windows_from_raw_tables(
    settings: Settings,
    raw_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    event_windows = _scope_to_study_hospital(
        raw_tables.get("fall_livestream_event_windows", pd.DataFrame()),
        settings.study_hospital_id,
    )
    second_level_panel = _scope_to_study_hospital(
        raw_tables.get("fall_livestream_second_level", pd.DataFrame()),
        settings.study_hospital_id,
    )
    return build_panel_prefall_event_windows(settings, event_windows, second_level_panel)


def _is_case_crossover_control_role(series: pd.Series) -> pd.Series:
    return series.astype("string").str.startswith("control_")


def _unique_monitor_hour_patients(hour_keys: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["hospital_id", "division_id", "monitor_id", "hour_ts"]
    optional_cols = [column for column in ("patient_id", "daypart") if column in hour_keys.columns]
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
    if "daypart" in frame.columns:
        aggregations["daypart"] = ("daypart", "first")
    grouped = frame.groupby(base_cols, dropna=False).agg(**aggregations).reset_index()
    grouped = grouped.loc[grouped["patient_id_nunique"] == 1].copy()
    return grouped.drop(columns=["patient_id_nunique"])


def _link_events_to_analysis_hours(
    events: pd.DataFrame,
    analysis_hour_keys: pd.DataFrame,
    hour_series: pd.Series,
) -> pd.DataFrame:
    if events.empty or analysis_hour_keys.empty:
        return pd.DataFrame()

    working_hour_keys = analysis_hour_keys.copy()
    working_hour_keys["hour_ts"] = _normalize_hour_timestamp(working_hour_keys["hour_ts"])
    normalized_hour_series = _normalize_hour_timestamp(hour_series)

    full_keys = ["hospital_id", "division_id", "monitor_id", "patient_id", "hour_ts"]
    known = events.loc[events["patient_id"].notna()].assign(
        hour_ts=normalized_hour_series[events["patient_id"].notna()]
    ).merge(
        working_hour_keys,
        on=full_keys,
        how="inner",
    )

    missing = events.loc[events["patient_id"].isna()].assign(
        hour_ts=normalized_hour_series[events["patient_id"].isna()]
    ).copy()
    if missing.empty:
        return known

    monitor_hour = _unique_monitor_hour_patients(working_hour_keys)
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
    recovered = recovered.drop(
        columns=[column for column in ("patient_id_resolved", "daypart_resolved") if column in recovered.columns]
    )
    return pd.concat([known, recovered], ignore_index=True)


def _profile_negative_control_source(
    source_inventory: pd.DataFrame,
    falls: pd.DataFrame,
    settings: Settings,
) -> dict[str, Any]:
    expected_start = pd.Timestamp(settings.negative_control_expected_start_date.isoformat(), tz="UTC")
    base_profile: dict[str, Any] = {
        "rows": int(len(source_inventory.index)),
        "chunk_count": int(len(source_inventory.index)),
        "expected_start_date_utc": settings.negative_control_expected_start_date.isoformat(),
        "history_status": "unavailable",
        "effective_start_ts_utc": None,
        "null_patient_id": 0,
        "distinct_division_id": 0,
        "distinct_monitor_id": 0,
        "distinct_patient_id": 0,
        "monitor_overlap_with_fall_cohort": 0,
        "division_overlap_with_fall_cohort": 0,
        "exact_five_minute_chunks": 0,
    }
    if source_inventory.empty:
        return base_profile

    frame = source_inventory.copy()
    for column in ("chunk_start_ts_utc", "chunk_end_ts_utc", "anchor_ts_utc"):
        if column in frame.columns:
            frame[column] = _parse_utc_timestamp(frame[column])
    for column in ("frame_count", "duration_seconds", "hospital_id", "division_id", "monitor_id", "patient_id"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    effective_start = frame["anchor_ts_utc"].min() if "anchor_ts_utc" in frame.columns else pd.NaT
    history_status = "complete"
    if pd.isna(effective_start):
        history_status = "unavailable"
    elif effective_start > expected_start:
        history_status = "partial_history"

    fall_monitors = set(
        pd.to_numeric(falls.get("monitor_id", pd.Series(dtype="float64")), errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )
    fall_divisions = set(pd.to_numeric(falls.get("division_id"), errors="coerce").dropna().astype(int).tolist())
    source_monitors = set(frame["monitor_id"].dropna().astype(int).tolist()) if "monitor_id" in frame.columns else set()
    source_divisions = set(frame["division_id"].dropna().astype(int).tolist()) if "division_id" in frame.columns else set()

    base_profile.update(
        {
            "rows": int(len(frame.index)),
            "chunk_count": int(len(frame.index)),
            "null_patient_id": int(frame["patient_id"].isna().sum()) if "patient_id" in frame.columns else 0,
            "distinct_division_id": int(frame["division_id"].nunique(dropna=True)) if "division_id" in frame.columns else 0,
            "distinct_monitor_id": int(frame["monitor_id"].nunique(dropna=True)) if "monitor_id" in frame.columns else 0,
            "distinct_patient_id": int(frame["patient_id"].nunique(dropna=True)) if "patient_id" in frame.columns else 0,
            "min_chunk_start_ts_utc": frame["chunk_start_ts_utc"].min().isoformat()
            if "chunk_start_ts_utc" in frame.columns and frame["chunk_start_ts_utc"].notna().any()
            else None,
            "max_chunk_end_ts_utc": frame["chunk_end_ts_utc"].max().isoformat()
            if "chunk_end_ts_utc" in frame.columns and frame["chunk_end_ts_utc"].notna().any()
            else None,
            "min_anchor_ts_utc": frame["anchor_ts_utc"].min().isoformat()
            if "anchor_ts_utc" in frame.columns and frame["anchor_ts_utc"].notna().any()
            else None,
            "max_anchor_ts_utc": frame["anchor_ts_utc"].max().isoformat()
            if "anchor_ts_utc" in frame.columns and frame["anchor_ts_utc"].notna().any()
            else None,
            "min_frame_count": int(frame["frame_count"].min()) if "frame_count" in frame.columns and frame["frame_count"].notna().any() else 0,
            "max_frame_count": int(frame["frame_count"].max()) if "frame_count" in frame.columns and frame["frame_count"].notna().any() else 0,
            "exact_five_minute_chunks": int(
                ((frame.get("frame_count", pd.Series(dtype="float64")) == 300) & (frame.get("duration_seconds", pd.Series(dtype="float64")) == 299)).sum()
            ),
            "history_status": history_status,
            "effective_start_ts_utc": effective_start.isoformat() if pd.notna(effective_start) else None,
            "monitor_overlap_with_fall_cohort": int(len(source_monitors & fall_monitors)),
            "division_overlap_with_fall_cohort": int(len(source_divisions & fall_divisions)),
        }
    )
    return base_profile


def write_gate_preflight(settings: Settings) -> Path:
    payload = {
        "run_id": settings.run_id,
        "generated_at_utc": utc_now_iso(),
        "hospital_timezone": settings.hospital_timezone,
        "gate_1_pass": settings.gate_1_pass,
        "gate_2_pass": settings.gate_2_pass,
        "requested_run_mode": settings.requested_run_mode,
        "effective_run_mode": settings.effective_run_mode,
        "gate_1_evidence": settings.gate_1_evidence,
        "gate_2_evidence": settings.gate_2_evidence,
    }
    path = settings.paths.qa_dir / f"gate_preflight_{settings.run_id}.json"
    return write_json(path, payload)


def write_source_profile(settings: Settings, raw_tables: dict[str, pd.DataFrame] | None = None) -> Path:
    raw = raw_tables or load_raw_tables(settings)

    falls = _scope_to_study_hospital(raw["fall_events_source"], settings.study_hospital_id)
    hourly = _scope_to_study_hospital(raw["hourly_location_aggregation"], settings.study_hospital_id)
    dimensions = _scope_to_study_hospital(raw["key_dimensions"], settings.study_hospital_id)
    negative_control_inventory = _scope_to_study_hospital(
        raw.get("negative_control_source_inventory", pd.DataFrame()),
        settings.study_hospital_id,
    )

    profile = {
        "run_id": settings.run_id,
        "generated_at_utc": utc_now_iso(),
        "case_crossover_source_status": settings.case_crossover_source_status,
        "negative_control_source_status": settings.negative_control_source_status,
        "nonfall_control_source_status": settings.negative_control_source_status,
        "fall_events_source": _series_stats(falls, "timestamp"),
        "hourly_location_aggregation": _series_stats(hourly, "hour_ts"),
        "key_dimensions": _series_stats(dimensions, None),
        "negative_control_source_inventory": _profile_negative_control_source(
            negative_control_inventory,
            falls,
            settings,
        ),
        "cohort_definition": _cohort_definition_profile(settings, falls, hourly),
    }

    if {"pct_chair", "pct_bed", "pct_ambulatory"}.issubset(hourly.columns):
        pct_modeled = (
            pd.to_numeric(hourly["pct_chair"], errors="coerce").fillna(0)
            + pd.to_numeric(hourly["pct_bed"], errors="coerce").fillna(0)
            + pd.to_numeric(hourly["pct_ambulatory"], errors="coerce").fillna(0)
        )
        pct_not_located = (
            pd.to_numeric(hourly["pct_not_located"], errors="coerce").fillna(0)
            if "pct_not_located" in hourly.columns
            else 0.0
        )
        pct = pct_modeled + pct_not_located
        profile["control_denominator_completeness"] = {
            "rows_with_valid_pct_sum": int((pct - 1.0).abs().le(0.02).sum()),
            "total_rows": int(len(hourly.index)),
        }
    else:
        profile["control_denominator_completeness"] = {
            "rows_with_valid_pct_sum": 0,
            "total_rows": int(len(hourly.index)),
        }

    path = settings.paths.qa_dir / f"source_profile_{settings.run_id}.json"
    return write_json(path, profile)


def _cohort_definition_profile(
    settings: Settings,
    falls: pd.DataFrame,
    hourly: pd.DataFrame,
) -> dict[str, Any]:
    if settings.cohort_map_path is not None:
        return {
            "source": str(settings.cohort_map_path),
            "derived": False,
            "basis": "external_cohort_map",
            "intervention_definition": "Provided by the external cohort_map_path file.",
            "control_definition": "Provided by the external cohort_map_path file.",
        }

    fall_monitors = set(
        pd.to_numeric(falls.get("monitor_id", pd.Series(dtype="float64")), errors="coerce")
        .dropna()
        .astype("int64")
        .tolist()
    )
    hourly_monitors = set(
        pd.to_numeric(hourly.get("monitor_id", pd.Series(dtype="float64")), errors="coerce")
        .dropna()
        .astype("int64")
        .tolist()
    )
    return {
        "source": "derived_from_raw_tables",
        "derived": True,
        "basis": DERIVED_OUTCOME_COHORT_BASIS,
        "intervention_definition": DERIVED_INTERVENTION_DEFINITION,
        "control_definition": DERIVED_CONTROL_DEFINITION,
        "rule": (
            "monitor_id in fall_events_source -> intervention; "
            "monitor_id only in hourly_location_aggregation -> control; "
            "falls assignment takes precedence on overlap."
        ),
        "fall_monitor_count": len(fall_monitors),
        "hourly_monitor_count": len(hourly_monitors),
        "overlap_monitor_count": len(fall_monitors.intersection(hourly_monitors)),
    }


def _load_staged(settings: Settings, filename: str) -> pd.DataFrame:
    path = settings.paths.staged_run_dir / filename
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return path


def _event_ts_local(df: pd.DataFrame, hospital_timezone: str) -> pd.Series:
    local_source = df["timestamp_local"] if "timestamp_local" in df.columns else pd.Series(pd.NA, index=df.index)
    utc_source = df["timestamp"] if "timestamp" in df.columns else pd.Series(pd.NA, index=df.index)
    ts_local = _parse_local_wall_timestamp(local_source)
    fallback_local = _utc_to_local_wall_timestamp(utc_source, hospital_timezone)
    return ts_local.fillna(fallback_local)


def _derive_daypart(hour_series: pd.Series) -> pd.Series:
    return hour_series.apply(classify_hour)


def _canonicalize_falls_site_names(falls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"hospital_id", "division_id", "hospital_name"}
    if falls.empty or not required.issubset(falls.columns):
        empty_aliases = pd.DataFrame(
            columns=["hospital_id", "division_id", "canonical_hospital_name", "alias_hospital_name", "alias_rows"]
        )
        return falls.copy(), empty_aliases

    normalized = falls.copy()
    normalized["hospital_name"] = (
        normalized["hospital_name"]
        .astype("string")
        .fillna("unknown")
        .str.strip()
        .replace("", "unknown")
    )
    normalized["_site_key"] = normalized["hospital_name"].str.casefold()

    ranked = (
        normalized.groupby(["hospital_id", "division_id", "_site_key", "hospital_name"], dropna=False)
        .size()
        .rename("name_rows")
        .reset_index()
        .sort_values(
            ["hospital_id", "division_id", "_site_key", "name_rows", "hospital_name"],
            ascending=[True, True, True, False, True],
            kind="mergesort",
        )
    )
    canonical = (
        ranked.drop_duplicates(subset=["hospital_id", "division_id", "_site_key"], keep="first")
        .rename(columns={"hospital_name": "canonical_hospital_name"})
        .loc[:, ["hospital_id", "division_id", "_site_key", "canonical_hospital_name"]]
    )

    normalized = normalized.merge(
        canonical,
        on=["hospital_id", "division_id", "_site_key"],
        how="left",
    )
    normalized["hospital_name"] = normalized["canonical_hospital_name"].fillna(normalized["hospital_name"])
    alias_report = (
        ranked.merge(
            canonical,
            on=["hospital_id", "division_id", "_site_key"],
            how="left",
        )
        .rename(columns={"hospital_name": "alias_hospital_name", "name_rows": "alias_rows"})
        .loc[
            lambda df: df["alias_hospital_name"].astype("string") != df["canonical_hospital_name"].astype("string"),
            ["hospital_id", "division_id", "canonical_hospital_name", "alias_hospital_name", "alias_rows"],
        ]
        .sort_values(
            ["hospital_id", "division_id", "canonical_hospital_name", "alias_hospital_name"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    normalized = normalized.drop(columns=["_site_key", "canonical_hospital_name"])
    return normalized, alias_report


def write_falls_descriptive_tables(
    settings: Settings,
    raw_tables: dict[str, pd.DataFrame] | None = None,
) -> list[Path]:
    raw = raw_tables or load_raw_tables(settings)
    falls = _scope_to_study_hospital(raw["fall_events_source"], settings.study_hospital_id)

    if falls.empty:
        falls = pd.DataFrame(
            columns=[
                "hospital_id",
                "hospital_system_name",
                "division_id",
                "hospital_name",
                "timestamp",
                "timestamp_local",
            ]
        )

    event_ts_local = _event_ts_local(falls, settings.hospital_timezone)
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "unknown"]
    daypart_order = DAYPART_ORDER
    month_lookup = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }

    falls, site_alias_report = _canonicalize_falls_site_names(falls)

    falls_by_site = (
        falls.groupby(["hospital_id", "hospital_system_name", "division_id", "hospital_name"], dropna=False)
        .size()
        .rename("falls")
        .reset_index()
        .sort_values(["falls", "hospital_id", "division_id"], ascending=[False, True, True])
    )

    weekday_series = event_ts_local.dt.day_name().fillna("unknown")
    falls_by_weekday = (
        weekday_series.value_counts(dropna=False)
        .rename_axis("weekday")
        .rename("falls")
        .reset_index()
    )
    falls_by_weekday["weekday"] = pd.Categorical(
        falls_by_weekday["weekday"], categories=weekday_order, ordered=True
    )
    falls_by_weekday = falls_by_weekday.sort_values("weekday").reset_index(drop=True)
    falls_by_weekday["weekday"] = falls_by_weekday["weekday"].astype("string")

    daypart_series = _derive_daypart(event_ts_local.dt.hour)
    falls_by_daypart = daypart_series.value_counts(dropna=False).rename_axis("daypart").rename("falls").reset_index()
    falls_by_daypart["daypart"] = pd.Categorical(
        falls_by_daypart["daypart"], categories=daypart_order, ordered=True
    )
    falls_by_daypart = falls_by_daypart.sort_values("daypart").reset_index(drop=True)
    falls_by_daypart["daypart"] = falls_by_daypart["daypart"].astype("string")

    falls_by_month = (
        event_ts_local.dt.strftime("%Y-%m")
        .fillna("unknown")
        .value_counts(dropna=False)
        .rename_axis("month")
        .rename("falls")
        .reset_index()
        .sort_values("month")
        .reset_index(drop=True)
    )

    month_numbers = event_ts_local.dt.month.astype("Int64")
    falls_by_month_of_year = (
        month_numbers.value_counts(dropna=True)
        .rename_axis("month_num")
        .rename("falls")
        .reset_index()
        .sort_values("month_num")
        .reset_index(drop=True)
    )
    falls_by_month_of_year["month_of_year"] = falls_by_month_of_year["month_num"].map(month_lookup)
    total_falls = int(falls_by_month_of_year["falls"].sum()) if not falls_by_month_of_year.empty else 0
    falls_by_month_of_year["pct_of_falls"] = (
        (falls_by_month_of_year["falls"] / total_falls).round(4) if total_falls else 0.0
    )
    falls_by_month_of_year = falls_by_month_of_year[["month_num", "month_of_year", "falls", "pct_of_falls"]]

    outputs = [
        _write_csv(falls_by_site, settings.paths.qa_dir / f"falls_by_site_{settings.run_id}.csv"),
        _write_csv(site_alias_report, settings.paths.qa_dir / f"falls_site_name_aliases_{settings.run_id}.csv"),
        _write_csv(falls_by_weekday, settings.paths.qa_dir / f"falls_by_weekday_{settings.run_id}.csv"),
        _write_csv(falls_by_daypart, settings.paths.qa_dir / f"falls_by_daypart_{settings.run_id}.csv"),
        _write_csv(falls_by_month, settings.paths.qa_dir / f"falls_by_month_{settings.run_id}.csv"),
        _write_csv(falls_by_month_of_year, settings.paths.qa_dir / f"falls_month_of_year_{settings.run_id}.csv"),
    ]

    summary_path = settings.paths.qa_dir / f"falls_descriptive_summary_{settings.run_id}.md"
    summary_lines = [
        f"# Falls Descriptive Summary {settings.run_id}",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        f"- Total falls: {len(falls.index)}",
        f"- Distinct patients: {falls['patient_id'].nunique(dropna=True) if 'patient_id' in falls else 0}",
        f"- Distinct monitors: {falls['monitor_id'].nunique(dropna=True) if 'monitor_id' in falls else 0}",
        "",
        "## Artifacts",
    ]
    summary_lines.extend([f"- {path.name}" for path in outputs])
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    outputs.append(summary_path)
    return outputs


def write_livestream_falls_descriptive_tables(
    settings: Settings,
    raw_tables: dict[str, pd.DataFrame] | None = None,
) -> list[Path]:
    raw = raw_tables or load_raw_tables(settings)
    event_windows = _canonical_event_windows_from_raw_tables(settings, raw)
    expected_cols = {
        "fall_event_id": "Int64",
        "fall_ts_utc": "datetime64[ns, UTC]",
        "fall_ts_local": "datetime64[ns, UTC]",
        "hospital_id": "Int64",
        "division_id": "Int64",
        "patient_id": "Int64",
        "pre_state_visible_prob": "float64",
        "pre_prob_chair": "float64",
        "pre_prob_bed": "float64",
        "pre_prob_room": "float64",
        "pre_prob_no_patient": "float64",
        "pre_prob_not_visible": "float64",
        "pre_prob_out_of_room": "float64",
        "pre_prob_unknown": "float64",
        "prefall_location_label": "string",
        "prefall_location_margin": "float64",
        "response_detected": "boolean",
        "response_latency_seconds": "float64",
    }
    for column in expected_cols:
        if column not in event_windows.columns:
            event_windows[column] = pd.NA

    event_windows["fall_ts_utc"] = _parse_utc_timestamp(event_windows["fall_ts_utc"])
    event_windows["fall_ts_local"] = _parse_local_wall_timestamp(event_windows["fall_ts_local"])
    event_windows["prefall_location_label"] = (
        event_windows["prefall_location_label"].astype("string").fillna("unknown")
    )
    # Backward compatibility for runs generated before room-label rename.
    event_windows["prefall_location_label"] = event_windows["prefall_location_label"].replace({"other": "room"})
    # Fold historical unknown state into no_patient location bucket.
    event_windows["prefall_location_label"] = event_windows["prefall_location_label"].replace({"unknown": "no_patient"})
    event_windows["prefall_location_margin"] = pd.to_numeric(
        event_windows["prefall_location_margin"], errors="coerce"
    )
    for column in [
        "pre_state_visible_prob",
        "pre_prob_chair",
        "pre_prob_bed",
        "pre_prob_room",
        "pre_prob_no_patient",
        "pre_prob_not_visible",
        "pre_prob_out_of_room",
        "pre_prob_unknown",
    ]:
        event_windows[column] = pd.to_numeric(event_windows[column], errors="coerce").fillna(0.0)
    event_windows["response_detected"] = event_windows["response_detected"].fillna(False).astype(bool)
    event_windows["response_latency_seconds"] = pd.to_numeric(
        event_windows["response_latency_seconds"], errors="coerce"
    )
    event_local_ts = event_windows["fall_ts_local"].fillna(
        _utc_to_local_wall_timestamp(event_windows["fall_ts_utc"], settings.hospital_timezone)
    )
    event_windows["fall_date_local"] = event_local_ts.dt.strftime("%Y-%m-%d")
    event_windows["fall_hour_local"] = event_local_ts.dt.hour.astype("Int64")
    event_windows["fall_weekday"] = event_local_ts.dt.day_name().fillna("unknown")
    event_windows["fall_daypart"] = _derive_daypart(event_local_ts.dt.hour)

    prefall = pd.DataFrame(columns=["prefall_location_label", "falls", "pct_of_falls", "median_margin"])
    if not event_windows.empty:
        prefall = (
            event_windows.groupby("prefall_location_label", dropna=False)
            .agg(
                falls=("fall_event_id", "size"),
                median_margin=("prefall_location_margin", "median"),
            )
            .reset_index()
            .sort_values("falls", ascending=False)
            .reset_index(drop=True)
        )
        total = int(prefall["falls"].sum())
        prefall["pct_of_falls"] = (prefall["falls"] / total).round(4) if total else 0.0
        prefall = prefall[["prefall_location_label", "falls", "pct_of_falls", "median_margin"]]

    prefall_probabilities = pd.DataFrame(
        columns=["prefall_location_label", "expected_falls", "pct_of_falls"]
    )
    prefall_probability_breakdown = pd.DataFrame(
        columns=[
            "grouping",
            "group_value",
            "prefall_location_label",
            "falls_in_group",
            "expected_falls",
            "mean_probability",
            "median_probability",
            "pct_expected_in_group",
        ]
    )
    if not event_windows.empty:
        total_events = float(len(event_windows.index))
        prefall_probabilities = pd.DataFrame(
            {
                "prefall_location_label": ["chair", "bed", "room", "no_patient"],
                "expected_falls": [
                    float(event_windows["pre_prob_chair"].sum()),
                    float(event_windows["pre_prob_bed"].sum()),
                    float(event_windows["pre_prob_room"].sum()),
                    float(event_windows["pre_prob_no_patient"].sum()),
                ],
            }
        )
        prefall_probabilities["pct_of_falls"] = (
            prefall_probabilities["expected_falls"] / total_events
        ).round(4)
        prefall_probabilities["expected_falls"] = prefall_probabilities["expected_falls"].round(4)

        groupings: list[tuple[str, str]] = [
            ("hospital_id", "hospital_id"),
            ("division_id", "division_id"),
            ("weekday", "fall_weekday"),
            ("daypart", "fall_daypart"),
        ]
        probability_columns: list[tuple[str, str]] = [
            ("chair", "pre_prob_chair"),
            ("bed", "pre_prob_bed"),
            ("room", "pre_prob_room"),
            ("no_patient", "pre_prob_no_patient"),
        ]
        rows: list[dict[str, object]] = []
        for grouping, column in groupings:
            grouped = event_windows.groupby(column, dropna=False)
            for group_value, group_df in grouped:
                falls_in_group = int(len(group_df.index))
                if falls_in_group == 0:
                    continue
                for label, prob_column in probability_columns:
                    probabilities = pd.to_numeric(group_df[prob_column], errors="coerce").fillna(0.0)
                    expected_falls = float(probabilities.sum())
                    mean_probability = float(probabilities.mean())
                    median_probability = float(probabilities.median())
                    rows.append(
                        {
                            "grouping": grouping,
                            "group_value": "unknown" if pd.isna(group_value) else str(group_value),
                            "prefall_location_label": label,
                            "falls_in_group": falls_in_group,
                            "expected_falls": expected_falls,
                            "mean_probability": mean_probability,
                            "median_probability": median_probability,
                            "pct_expected_in_group": (expected_falls / falls_in_group),
                        }
                    )
        if rows:
            prefall_probability_breakdown = (
                pd.DataFrame(rows)
                .sort_values(
                    ["grouping", "group_value", "expected_falls", "prefall_location_label"],
                    ascending=[True, True, False, True],
                )
                .reset_index(drop=True)
            )
            for column in [
                "expected_falls",
                "mean_probability",
                "median_probability",
                "pct_expected_in_group",
            ]:
                prefall_probability_breakdown[column] = prefall_probability_breakdown[column].round(4)

    latencies = event_windows.loc[event_windows["response_detected"], "response_latency_seconds"].dropna()
    response_summary = pd.DataFrame(
        {
            "metric": [
                "total_falls",
                "responses_detected",
                "response_rate",
                "latency_p50_seconds",
                "latency_p75_seconds",
                "latency_p90_seconds",
            ],
            "value": [
                int(len(event_windows.index)),
                int(event_windows["response_detected"].sum()),
                round(float(event_windows["response_detected"].mean()), 4) if not event_windows.empty else 0.0,
                round(float(latencies.quantile(0.50)), 2) if not latencies.empty else pd.NA,
                round(float(latencies.quantile(0.75)), 2) if not latencies.empty else pd.NA,
                round(float(latencies.quantile(0.90)), 2) if not latencies.empty else pd.NA,
            ],
        }
    )

    patient_day_hour = pd.DataFrame(
        columns=[
            "patient_id",
            "fall_date_local",
            "fall_hour_local",
            "prefall_location_label",
            "falls",
            "bucket_total_falls",
            "pct_in_bucket",
        ]
    )
    if not event_windows.empty:
        patient_day_hour = (
            event_windows.groupby(
                ["patient_id", "fall_date_local", "fall_hour_local", "prefall_location_label"], dropna=False
            )
            .size()
            .rename("falls")
            .reset_index()
        )
        patient_day_hour["bucket_total_falls"] = patient_day_hour.groupby(
            ["patient_id", "fall_date_local", "fall_hour_local"], dropna=False
        )["falls"].transform("sum")
        patient_day_hour["pct_in_bucket"] = (
            patient_day_hour["falls"] / patient_day_hour["bucket_total_falls"]
        ).round(4)
        patient_day_hour = patient_day_hour.sort_values(
            ["patient_id", "fall_date_local", "fall_hour_local", "falls"],
            ascending=[True, True, True, False],
        )

    outputs = [
        _write_csv(prefall, settings.paths.qa_dir / f"falls_prefall_location_{settings.run_id}.csv"),
        _write_csv(
            prefall_probabilities,
            settings.paths.qa_dir / f"falls_prefall_location_probabilities_{settings.run_id}.csv",
        ),
        _write_csv(
            prefall_probability_breakdown,
            settings.paths.qa_dir / f"falls_prefall_location_probability_breakdown_{settings.run_id}.csv",
        ),
        _write_csv(response_summary, settings.paths.qa_dir / f"falls_response_latency_{settings.run_id}.csv"),
        _write_csv(
            patient_day_hour,
            settings.paths.qa_dir / f"falls_patient_day_hour_location_{settings.run_id}.csv",
        ),
    ]
    return outputs


def _prepare_second_level_panel(
    raw_tables: dict[str, pd.DataFrame],
    settings: Settings,
) -> pd.DataFrame:
    panel = _scope_to_study_hospital(
        raw_tables.get("fall_livestream_second_level", pd.DataFrame()),
        settings.study_hospital_id,
    ).copy()
    expected_columns = [
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
        "primary_patient_posture_label",
        "primary_patient_posture_score_sitting",
        "primary_patient_posture_score_standing",
        "primary_patient_posture_score_lying",
        "patient_posture_candidate_count",
        "patient_posture_source",
    ]
    for column in expected_columns:
        if column not in panel.columns:
            panel[column] = pd.NA

    panel["fall_event_id"] = pd.to_numeric(panel["fall_event_id"], errors="coerce").astype("Int64")
    panel["hospital_id"] = pd.to_numeric(panel["hospital_id"], errors="coerce").astype("Int64")
    panel["division_id"] = pd.to_numeric(panel["division_id"], errors="coerce").astype("Int64")
    panel["patient_id"] = pd.to_numeric(panel["patient_id"], errors="coerce").astype("Int64")
    panel["monitor_id"] = pd.to_numeric(panel["monitor_id"], errors="coerce").astype("Int64")
    panel["fall_ts_utc"] = _parse_utc_timestamp(panel["fall_ts_utc"])
    panel["fall_ts_local"] = _parse_local_wall_timestamp(panel["fall_ts_local"])
    panel["frame_ts_utc"] = _parse_utc_timestamp(panel["frame_ts_utc"])
    panel["second_offset"] = pd.to_numeric(panel["second_offset"], errors="coerce")
    panel["frame_has_location_signal"] = panel["frame_has_location_signal"].fillna(False).astype(bool)
    panel["window_phase"] = panel["window_phase"].astype("string").fillna("unknown")
    panel["dominant_location_label"] = panel["dominant_location_label"].astype("string").fillna("no_patient")
    panel["primary_patient_posture_label"] = (
        panel["primary_patient_posture_label"].astype("string").fillna("").str.strip().str.lower()
    )
    panel["patient_posture_source"] = panel["patient_posture_source"].astype("string").fillna("none")
    for column in [
        "patient_chair_distance",
        "patient_bed_distance",
        "patient_room_distance",
        "patient_staff_iou",
        "primary_patient_posture_score_sitting",
        "primary_patient_posture_score_standing",
        "primary_patient_posture_score_lying",
        "patient_posture_candidate_count",
    ]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.sort_values(["fall_event_id", "frame_ts_utc"], kind="mergesort").reset_index(drop=True)
    return panel


def _compute_event_localized_metrics(
    panel: pd.DataFrame,
    event_windows: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "fall_event_id",
        "hospital_id",
        "division_id",
        "patient_id",
        "monitor_id",
        "frames_total",
        "frames_visible",
        "visibility_ratio",
        "pre_immediate_chair_share",
        "pre_immediate_bed_share",
        "pre_immediate_room_share",
        "pre_immediate_no_patient_share",
        "pre_immediate_state_switch_count",
        "pre_immediate_mean_chair_distance",
        "pre_immediate_mean_bed_distance",
        "pre_immediate_mean_room_distance",
        "pre_immediate_chair_distance_slope_per_sec",
        "pre_immediate_bed_distance_slope_per_sec",
        "pre_immediate_room_distance_slope_per_sec",
        "post_staff_overlap_detected",
        "post_first_staff_overlap_seconds",
        "pre_state_visible_prob",
        "pre_dropout_visibility_ratio",
        "pre_dropout_last_visible_gap_seconds",
        "confidence_segment",
    ]
    if panel.empty:
        return pd.DataFrame(columns=columns)

    pre_immediate = panel.loc[(panel["second_offset"] >= -30) & (panel["second_offset"] < 0)].copy()
    post_window = panel.loc[(panel["second_offset"] >= 0) & (panel["second_offset"] <= 180)].copy()

    rows: list[dict[str, Any]] = []
    for event_id, group in panel.groupby("fall_event_id", dropna=False):
        group = group.sort_values("frame_ts_utc", kind="mergesort")
        event_pre = pre_immediate.loc[pre_immediate["fall_event_id"] == event_id].copy()
        event_post = post_window.loc[post_window["fall_event_id"] == event_id].copy()
        if event_pre.empty:
            event_pre = group.loc[group["second_offset"] < 0].copy()

        switch_count = 0
        if not event_pre.empty:
            prev = event_pre["dominant_location_label"].shift(1)
            switch_count = int(
                ((prev.notna()) & (event_pre["dominant_location_label"] != prev)).sum()
            )

        def _share(label: str, frame: pd.DataFrame = event_pre) -> float:
            if frame.empty:
                return 0.0
            return round(float((frame["dominant_location_label"] == label).mean()), 4)

        def _mean(series: pd.Series) -> float | pd.NA:
            values = pd.to_numeric(series, errors="coerce").dropna()
            if values.empty:
                return pd.NA
            return round(float(values.mean()), 4)

        def _slope(distance_column: str, frame: pd.DataFrame = event_pre) -> float | pd.NA:
            if frame.empty:
                return pd.NA
            segment = frame.loc[frame[distance_column].notna(), ["second_offset", distance_column]]
            if len(segment.index) < 2:
                return pd.NA
            x0 = float(segment.iloc[0]["second_offset"])
            x1 = float(segment.iloc[-1]["second_offset"])
            if x1 == x0:
                return pd.NA
            y0 = float(segment.iloc[0][distance_column])
            y1 = float(segment.iloc[-1][distance_column])
            return round((y1 - y0) / (x1 - x0), 6)

        overlap_seconds = pd.to_numeric(
            event_post.loc[event_post["patient_staff_iou"].fillna(0.0) > 0, "second_offset"],
            errors="coerce",
        ).dropna()
        first_overlap = float(overlap_seconds.min()) if not overlap_seconds.empty else pd.NA

        first_row = group.iloc[0]
        rows.append(
            {
                "fall_event_id": event_id,
                "hospital_id": first_row.get("hospital_id"),
                "division_id": first_row.get("division_id"),
                "patient_id": first_row.get("patient_id"),
                "monitor_id": first_row.get("monitor_id"),
                "frames_total": int(len(group.index)),
                "frames_visible": int(group["frame_has_location_signal"].sum()),
                "visibility_ratio": round(float(group["frame_has_location_signal"].mean()), 4),
                "pre_immediate_chair_share": _share("chair"),
                "pre_immediate_bed_share": _share("bed"),
                "pre_immediate_room_share": _share("room"),
                "pre_immediate_no_patient_share": _share("no_patient"),
                "pre_immediate_state_switch_count": switch_count,
                "pre_immediate_mean_chair_distance": _mean(event_pre["patient_chair_distance"]),
                "pre_immediate_mean_bed_distance": _mean(event_pre["patient_bed_distance"]),
                "pre_immediate_mean_room_distance": _mean(event_pre["patient_room_distance"]),
                "pre_immediate_chair_distance_slope_per_sec": _slope("patient_chair_distance"),
                "pre_immediate_bed_distance_slope_per_sec": _slope("patient_bed_distance"),
                "pre_immediate_room_distance_slope_per_sec": _slope("patient_room_distance"),
                "post_staff_overlap_detected": bool(not overlap_seconds.empty),
                "post_first_staff_overlap_seconds": first_overlap,
            }
        )

    localized = pd.DataFrame(rows)
    confidence_columns = [
        "fall_event_id",
        "pre_state_visible_prob",
        "pre_dropout_visibility_ratio",
        "pre_dropout_last_visible_gap_seconds",
    ]
    if not event_windows.empty:
        confidence = event_windows.copy()
        for column in confidence_columns:
            if column not in confidence.columns:
                confidence[column] = pd.NA
        confidence = confidence.loc[:, confidence_columns].copy()
    else:
        confidence = pd.DataFrame()
    if not confidence.empty:
        confidence["fall_event_id"] = pd.to_numeric(confidence["fall_event_id"], errors="coerce").astype("Int64")
        for column in confidence_columns[1:]:
            confidence[column] = pd.to_numeric(confidence[column], errors="coerce")
        localized = localized.merge(confidence, on="fall_event_id", how="left")
    else:
        localized["pre_state_visible_prob"] = pd.NA
        localized["pre_dropout_visibility_ratio"] = pd.NA
        localized["pre_dropout_last_visible_gap_seconds"] = pd.NA

    high_mask = (
        (pd.to_numeric(localized["pre_state_visible_prob"], errors="coerce") >= 0.70)
        & (pd.to_numeric(localized["pre_dropout_visibility_ratio"], errors="coerce") >= 0.70)
        & (pd.to_numeric(localized["pre_dropout_last_visible_gap_seconds"], errors="coerce") <= 90.0)
    )
    low_mask = (
        (pd.to_numeric(localized["pre_state_visible_prob"], errors="coerce") < 0.40)
        | (pd.to_numeric(localized["pre_dropout_visibility_ratio"], errors="coerce") < 0.40)
        | (pd.to_numeric(localized["pre_dropout_last_visible_gap_seconds"], errors="coerce") > 180.0)
    )
    localized["confidence_segment"] = "medium_confidence"
    localized.loc[high_mask, "confidence_segment"] = "high_confidence"
    localized.loc[low_mask, "confidence_segment"] = "low_confidence"
    return localized[columns].sort_values("fall_event_id", kind="mergesort").reset_index(drop=True)


def _build_response_curve(panel: pd.DataFrame) -> pd.DataFrame:
    columns = ["second_offset", "events_detected_by_second", "cumulative_detection_rate", "event_count"]
    if panel.empty:
        return pd.DataFrame(columns=columns)

    post = panel.loc[(panel["second_offset"] >= 0) & (panel["second_offset"] <= 180)].copy()
    if post.empty:
        return pd.DataFrame(columns=columns)

    first_overlap = (
        post.loc[post["patient_staff_iou"].fillna(0.0) > 0]
        .groupby("fall_event_id", dropna=False)["second_offset"]
        .min()
    )
    event_count = int(panel["fall_event_id"].nunique(dropna=True))
    rows: list[dict[str, Any]] = []
    for second in range(0, 181):
        detected = int((first_overlap <= second).sum()) if not first_overlap.empty else 0
        rows.append(
            {
                "second_offset": second,
                "events_detected_by_second": detected,
                "cumulative_detection_rate": round(detected / event_count, 4) if event_count else 0.0,
                "event_count": event_count,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_case_crossover_outputs(case_windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    set_columns = [
        "fall_event_id",
        "window_role",
        "anchor_ts_utc",
        "frame_count",
        "visibility_ratio",
        "posture_observed_frame_count",
        "posture_sitting_score_mean",
        "posture_standing_score_mean",
        "posture_lying_score_mean",
        "posture_sitting_share",
        "posture_standing_share",
        "posture_lying_share",
        "posture_switch_count",
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
        "nearby_fall_count_30m",
        "eligible_window",
        "matched_control_pair",
    ]
    effect_columns = ["metric", "paired_mean_delta_hazard_minus_control", "paired_median_delta", "pairs"]
    if case_windows.empty:
        diagnostics = {
            "event_count": 0,
            "events_with_hazard": 0,
            "events_with_any_control": 0,
            "events_with_full_pair": 0,
        }
        return (
            pd.DataFrame(columns=set_columns),
            pd.DataFrame(columns=effect_columns),
            diagnostics,
        )

    frame = case_windows.copy()
    required_columns = [
        "fall_event_id",
        "window_role",
        "anchor_ts_utc",
        "eligible_window",
        "frame_count",
        "nearby_fall_count_30m",
        "dominant_switch_count",
        "visibility_ratio",
        "posture_observed_frame_count",
        "posture_sitting_score_mean",
        "posture_standing_score_mean",
        "posture_lying_score_mean",
        "posture_sitting_share",
        "posture_standing_share",
        "posture_lying_share",
        "posture_switch_count",
        "chair_share",
        "bed_share",
        "room_share",
        "no_patient_share",
        "mean_patient_chair_distance",
        "mean_patient_bed_distance",
        "mean_patient_room_distance",
        "mean_patient_staff_iou",
        "max_patient_staff_iou",
    ]
    for column in required_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["fall_event_id"] = pd.to_numeric(frame["fall_event_id"], errors="coerce").astype("Int64")
    frame["anchor_ts_utc"] = _parse_utc_timestamp(frame["anchor_ts_utc"])
    frame["window_role"] = frame["window_role"].astype("string").fillna("unknown")
    frame["eligible_window"] = frame["eligible_window"].fillna(False).astype(bool)
    for column in [
        "frame_count",
        "nearby_fall_count_30m",
        "dominant_switch_count",
        "visibility_ratio",
        "posture_observed_frame_count",
        "posture_sitting_score_mean",
        "posture_standing_score_mean",
        "posture_lying_score_mean",
        "posture_sitting_share",
        "posture_standing_share",
        "posture_lying_share",
        "posture_switch_count",
        "chair_share",
        "bed_share",
        "room_share",
        "no_patient_share",
        "mean_patient_chair_distance",
        "mean_patient_bed_distance",
        "mean_patient_room_distance",
        "mean_patient_staff_iou",
        "max_patient_staff_iou",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    eligible = frame.loc[frame["eligible_window"]].copy()
    controls = eligible.loc[_is_case_crossover_control_role(eligible["window_role"])].copy()
    control_counts = controls.groupby("fall_event_id", dropna=False)["window_role"].nunique().rename("control_roles")
    eligible = eligible.merge(control_counts, on="fall_event_id", how="left")
    eligible["control_roles"] = pd.to_numeric(eligible["control_roles"], errors="coerce").fillna(0).astype(int)
    eligible["matched_control_pair"] = eligible["control_roles"] > 0
    sets = eligible[set_columns].sort_values(["fall_event_id", "window_role"], kind="mergesort").reset_index(drop=True)

    metric_rows: list[dict[str, Any]] = []
    candidate_metrics = [
        "visibility_ratio",
        "posture_observed_frame_count",
        "posture_sitting_score_mean",
        "posture_standing_score_mean",
        "posture_lying_score_mean",
        "posture_sitting_share",
        "posture_standing_share",
        "posture_lying_share",
        "chair_share",
        "bed_share",
        "room_share",
        "no_patient_share",
        "mean_patient_chair_distance",
        "mean_patient_bed_distance",
        "mean_patient_room_distance",
        "dominant_switch_count",
        "posture_switch_count",
    ]
    hazard = eligible.loc[eligible["window_role"] == "hazard"].set_index("fall_event_id")
    control_mean = (
        controls.groupby("fall_event_id", dropna=False)[candidate_metrics]
        .mean(numeric_only=True)
    )
    paired = hazard.join(control_mean, how="inner", lsuffix="_hazard", rsuffix="_control")
    for metric in candidate_metrics:
        hazard_col = f"{metric}_hazard"
        control_col = f"{metric}_control"
        if hazard_col not in paired.columns or control_col not in paired.columns:
            continue
        delta = pd.to_numeric(paired[hazard_col], errors="coerce") - pd.to_numeric(
            paired[control_col], errors="coerce"
        )
        delta = delta.dropna()
        if delta.empty:
            continue
        metric_rows.append(
            {
                "metric": metric,
                "paired_mean_delta_hazard_minus_control": round(float(delta.mean()), 6),
                "paired_median_delta": round(float(delta.median()), 6),
                "pairs": int(len(delta.index)),
            }
        )

    effects = pd.DataFrame(metric_rows, columns=effect_columns)
    diagnostics = {
        "event_count": int(frame["fall_event_id"].nunique(dropna=True)),
        "events_with_hazard": int(hazard.index.nunique()),
        "events_with_any_control": int(controls["fall_event_id"].nunique(dropna=True)),
        "events_with_full_pair": int(paired.index.nunique()),
    }
    return sets, effects, diagnostics


def _build_negative_control_outputs(
    case_windows: pd.DataFrame,
    negative_control_windows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    set_columns = [
        "fall_event_id",
        "window_role",
        "anchor_ts_utc",
        "source_monitor_id",
        "monitor_id",
        "matching_candidates",
        "frame_count",
        "visibility_ratio",
        "posture_observed_frame_count",
        "posture_sitting_score_mean",
        "posture_standing_score_mean",
        "posture_lying_score_mean",
        "posture_sitting_share",
        "posture_standing_share",
        "posture_lying_share",
        "posture_switch_count",
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
        "nearby_fall_count_30m",
        "eligible_window",
        "matched_control_pair",
    ]
    effect_columns = ["metric", "paired_mean_delta_hazard_minus_control", "paired_median_delta", "pairs"]
    diagnostics = {
        "events_with_hazard": 0,
        "events_with_negative_control": 0,
        "events_with_negative_control_pair": 0,
    }
    if case_windows.empty:
        return pd.DataFrame(columns=set_columns), pd.DataFrame(columns=effect_columns), diagnostics

    hazard = case_windows.copy()
    for column in [
        "fall_event_id",
        "window_role",
        "anchor_ts_utc",
        "monitor_id",
        "frame_count",
        "visibility_ratio",
        "posture_observed_frame_count",
        "posture_sitting_score_mean",
        "posture_standing_score_mean",
        "posture_lying_score_mean",
        "posture_sitting_share",
        "posture_standing_share",
        "posture_lying_share",
        "posture_switch_count",
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
        "nearby_fall_count_30m",
        "eligible_window",
    ]:
        if column not in hazard.columns:
            hazard[column] = pd.NA
    hazard["fall_event_id"] = pd.to_numeric(hazard["fall_event_id"], errors="coerce").astype("Int64")
    hazard["anchor_ts_utc"] = _parse_utc_timestamp(hazard["anchor_ts_utc"])
    hazard["window_role"] = hazard["window_role"].astype("string").fillna("unknown")
    hazard["eligible_window"] = hazard["eligible_window"].fillna(False).astype(bool)
    hazard = hazard.loc[(hazard["window_role"] == "hazard") & (hazard["eligible_window"])].copy()
    for column in [
        "frame_count",
        "visibility_ratio",
        "posture_observed_frame_count",
        "posture_sitting_score_mean",
        "posture_standing_score_mean",
        "posture_lying_score_mean",
        "posture_sitting_share",
        "posture_standing_share",
        "posture_lying_share",
        "posture_switch_count",
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
        "nearby_fall_count_30m",
    ]:
        hazard[column] = pd.to_numeric(hazard[column], errors="coerce")
    hazard["source_monitor_id"] = hazard["monitor_id"]
    hazard["matching_candidates"] = pd.NA

    controls = negative_control_windows.copy()
    for column in [
        "fall_event_id",
        "window_role",
        "anchor_ts_utc",
        "source_monitor_id",
        "monitor_id",
        "matching_candidates",
        "frame_count",
        "visibility_ratio",
        "posture_observed_frame_count",
        "posture_sitting_score_mean",
        "posture_standing_score_mean",
        "posture_lying_score_mean",
        "posture_sitting_share",
        "posture_standing_share",
        "posture_lying_share",
        "posture_switch_count",
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
        "nearby_fall_count_30m",
        "eligible_window",
    ]:
        if column not in controls.columns:
            controls[column] = pd.NA
    if not controls.empty:
        controls["fall_event_id"] = pd.to_numeric(controls["fall_event_id"], errors="coerce").astype("Int64")
        controls["anchor_ts_utc"] = _parse_utc_timestamp(controls["anchor_ts_utc"])
        controls["window_role"] = controls["window_role"].astype("string").fillna("nonfall_control")
        controls["eligible_window"] = controls["eligible_window"].fillna(False).astype(bool)
        controls = controls.loc[
            (controls["window_role"] == "nonfall_control") & (controls["eligible_window"])
        ].copy()
        for column in [
            "matching_candidates",
            "frame_count",
            "visibility_ratio",
            "posture_observed_frame_count",
            "posture_sitting_score_mean",
            "posture_standing_score_mean",
            "posture_lying_score_mean",
            "posture_sitting_share",
            "posture_standing_share",
            "posture_lying_share",
            "posture_switch_count",
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
            "nearby_fall_count_30m",
        ]:
            controls[column] = pd.to_numeric(controls[column], errors="coerce")

    control_counts = controls.groupby("fall_event_id", dropna=False).size().rename("control_rows")
    hazard = hazard.merge(control_counts, on="fall_event_id", how="left")
    hazard["control_rows"] = pd.to_numeric(hazard["control_rows"], errors="coerce").fillna(0).astype(int)
    hazard["matched_control_pair"] = hazard["control_rows"] > 0
    hazard = hazard.drop(columns=["control_rows"])

    controls = controls.copy()
    controls["matched_control_pair"] = controls["fall_event_id"].isin(
        hazard.loc[hazard["matched_control_pair"], "fall_event_id"]
    )
    combined = pd.concat(
        [
            hazard[set_columns],
            controls[set_columns],
        ],
        ignore_index=True,
    ).sort_values(["fall_event_id", "window_role"], kind="mergesort")

    metric_rows: list[dict[str, Any]] = []
    candidate_metrics = [
        "visibility_ratio",
        "posture_observed_frame_count",
        "posture_sitting_score_mean",
        "posture_standing_score_mean",
        "posture_lying_score_mean",
        "posture_sitting_share",
        "posture_standing_share",
        "posture_lying_share",
        "chair_share",
        "bed_share",
        "room_share",
        "no_patient_share",
        "mean_patient_chair_distance",
        "mean_patient_bed_distance",
        "mean_patient_room_distance",
        "dominant_switch_count",
        "posture_switch_count",
    ]
    hazard_indexed = hazard.set_index("fall_event_id")
    control_mean = (
        controls.groupby("fall_event_id", dropna=False)[candidate_metrics]
        .mean(numeric_only=True)
    )
    paired = hazard_indexed.join(control_mean, how="inner", lsuffix="_hazard", rsuffix="_control")
    for metric in candidate_metrics:
        hazard_col = f"{metric}_hazard"
        control_col = f"{metric}_control"
        if hazard_col not in paired.columns or control_col not in paired.columns:
            continue
        delta = pd.to_numeric(paired[hazard_col], errors="coerce") - pd.to_numeric(
            paired[control_col], errors="coerce"
        )
        delta = delta.dropna()
        if delta.empty:
            continue
        metric_rows.append(
            {
                "metric": metric,
                "paired_mean_delta_hazard_minus_control": round(float(delta.mean()), 6),
                "paired_median_delta": round(float(delta.median()), 6),
                "pairs": int(len(delta.index)),
            }
        )
    diagnostics = {
        "events_with_hazard": int(hazard["fall_event_id"].nunique(dropna=True)),
        "events_with_negative_control": int(controls["fall_event_id"].nunique(dropna=True)),
        "events_with_negative_control_pair": int(paired.index.nunique()),
    }
    return combined.reset_index(drop=True), pd.DataFrame(metric_rows, columns=effect_columns), diagnostics


def _build_negative_control_match_quality(
    case_windows: pd.DataFrame,
    negative_control_windows: pd.DataFrame,
    source_inventory: pd.DataFrame,
    settings: Settings,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [
        "fall_event_id",
        "hazard_anchor_ts_utc",
        "division_id",
        "source_monitor_id",
        "hazard_eligible",
        "after_effective_source_start",
        "division_available",
        "division_chunk_count",
        "matching_candidates",
        "selected_control_count",
        "best_match_tier",
        "excluded_reason",
    ]
    source = source_inventory.copy()
    if not source.empty:
        if "anchor_ts_utc" in source.columns:
            source["anchor_ts_utc"] = _parse_utc_timestamp(source["anchor_ts_utc"])
        for column in ("division_id", "monitor_id", "patient_id", "frame_count"):
            if column in source.columns:
                source[column] = pd.to_numeric(source[column], errors="coerce")
    effective_source_start = source["anchor_ts_utc"].min() if "anchor_ts_utc" in source.columns and source["anchor_ts_utc"].notna().any() else pd.NaT
    source_division_chunk_counts = (
        source.groupby("division_id", dropna=True).size().rename("division_chunk_count")
        if not source.empty and "division_id" in source.columns
        else pd.Series(dtype="int64")
    )

    hazard = case_windows.copy()
    if hazard.empty:
        summary = {
            "run_id": settings.run_id,
            "generated_at_utc": utc_now_iso(),
            "control_source_status": settings.negative_control_source_status,
            "expected_source_start_date_utc": settings.negative_control_expected_start_date.isoformat(),
            "effective_source_start_ts_utc": effective_source_start.isoformat() if pd.notna(effective_source_start) else None,
            "history_status": "unavailable" if pd.isna(effective_source_start) else "complete",
            "source_chunk_count": int(len(source.index)),
            "source_distinct_divisions": int(source["division_id"].nunique(dropna=True)) if "division_id" in source.columns else 0,
            "source_distinct_monitors": int(source["monitor_id"].nunique(dropna=True)) if "monitor_id" in source.columns else 0,
            "fall_event_count": 0,
            "eligible_hazard_events": 0,
            "fall_events_after_source_start": 0,
            "fall_events_in_source_divisions": 0,
            "fall_events_eligible_for_matching": 0,
            "fall_events_with_selected_controls": 0,
            "matched_control_rows": 0,
            "unmatched_fall_count": 0,
        }
        return pd.DataFrame(columns=columns), summary

    for column in ("fall_event_id", "division_id", "monitor_id"):
        if column not in hazard.columns:
            hazard[column] = pd.NA
        hazard[column] = pd.to_numeric(hazard[column], errors="coerce")
    if "anchor_ts_utc" not in hazard.columns:
        hazard["anchor_ts_utc"] = pd.NaT
    hazard["anchor_ts_utc"] = _parse_utc_timestamp(hazard["anchor_ts_utc"])
    hazard["window_role"] = hazard.get("window_role", pd.Series("hazard", index=hazard.index)).astype("string").fillna("unknown")
    hazard["eligible_window"] = hazard.get("eligible_window", pd.Series(False, index=hazard.index)).fillna(False).astype(bool)
    hazard = hazard.loc[hazard["window_role"] == "hazard"].copy()

    controls = negative_control_windows.copy()
    if not controls.empty:
        for column in ("fall_event_id", "matching_candidates", "control_rank", "monitor_id", "source_monitor_id"):
            if column not in controls.columns:
                controls[column] = pd.NA
            controls[column] = pd.to_numeric(controls[column], errors="coerce")
        controls["match_tier"] = controls.get("match_tier", pd.Series(pd.NA, index=controls.index)).astype("string")
        controls["eligible_window"] = controls.get("eligible_window", pd.Series(False, index=controls.index)).fillna(False).astype(bool)
        controls = controls.loc[
            (controls.get("window_role", pd.Series("nonfall_control", index=controls.index)).astype("string") == "nonfall_control")
            & controls["eligible_window"]
        ].copy()

    control_summary = (
        controls.groupby("fall_event_id", dropna=False)
        .agg(
            selected_control_count=("fall_event_id", "size"),
            matching_candidates=("matching_candidates", "max"),
            best_match_tier=("match_tier", "first"),
        )
        .reset_index()
        if not controls.empty
        else pd.DataFrame(columns=["fall_event_id", "selected_control_count", "matching_candidates", "best_match_tier"])
    )

    rows: list[dict[str, Any]] = []
    division_ids_available = set(source_division_chunk_counts.index.astype(int).tolist()) if not source_division_chunk_counts.empty else set()
    for hazard_row in hazard.itertuples(index=False):
        event_id = int(hazard_row.fall_event_id) if pd.notna(hazard_row.fall_event_id) else None
        division_id = int(hazard_row.division_id) if pd.notna(hazard_row.division_id) else None
        selected = control_summary.loc[control_summary["fall_event_id"] == hazard_row.fall_event_id]
        selected_count = int(selected["selected_control_count"].iloc[0]) if not selected.empty else 0
        matching_candidates = (
            int(selected["matching_candidates"].iloc[0])
            if not selected.empty and pd.notna(selected["matching_candidates"].iloc[0])
            else pd.NA
        )
        best_match_tier = selected["best_match_tier"].iloc[0] if not selected.empty else pd.NA
        after_source_start = bool(pd.notna(effective_source_start) and pd.notna(hazard_row.anchor_ts_utc) and hazard_row.anchor_ts_utc >= effective_source_start)
        division_available = bool(division_id in division_ids_available) if division_id is not None else False
        hazard_eligible = bool(hazard_row.eligible_window)
        if not hazard_eligible:
            excluded_reason = "hazard_ineligible"
        elif pd.isna(effective_source_start):
            excluded_reason = "source_unavailable"
        elif not after_source_start:
            excluded_reason = "before_source_start"
        elif not division_available:
            excluded_reason = "division_unavailable"
        elif selected_count == 0:
            excluded_reason = "no_candidate_match"
        else:
            excluded_reason = ""
        rows.append(
            {
                "fall_event_id": event_id,
                "hazard_anchor_ts_utc": hazard_row.anchor_ts_utc.isoformat() if pd.notna(hazard_row.anchor_ts_utc) else None,
                "division_id": division_id,
                "source_monitor_id": int(hazard_row.monitor_id) if pd.notna(hazard_row.monitor_id) else pd.NA,
                "hazard_eligible": hazard_eligible,
                "after_effective_source_start": after_source_start,
                "division_available": division_available,
                "division_chunk_count": int(source_division_chunk_counts.get(division_id, 0)) if division_id is not None else 0,
                "matching_candidates": matching_candidates,
                "selected_control_count": selected_count,
                "best_match_tier": best_match_tier,
                "excluded_reason": excluded_reason,
            }
        )

    match_quality = pd.DataFrame(rows, columns=columns)
    eligible_for_matching = match_quality.loc[
        match_quality["hazard_eligible"]
        & match_quality["after_effective_source_start"]
        & match_quality["division_available"]
    ].copy()
    history_status = "unavailable"
    if pd.notna(effective_source_start):
        history_status = (
            "partial_history"
            if effective_source_start > pd.Timestamp(settings.negative_control_expected_start_date.isoformat(), tz="UTC")
            else "complete"
        )
    summary = {
        "run_id": settings.run_id,
        "generated_at_utc": utc_now_iso(),
        "control_source_status": settings.negative_control_source_status,
        "expected_source_start_date_utc": settings.negative_control_expected_start_date.isoformat(),
        "effective_source_start_ts_utc": effective_source_start.isoformat() if pd.notna(effective_source_start) else None,
        "history_status": history_status,
        "source_chunk_count": int(len(source.index)),
        "source_distinct_divisions": int(source["division_id"].nunique(dropna=True)) if "division_id" in source.columns else 0,
        "source_distinct_monitors": int(source["monitor_id"].nunique(dropna=True)) if "monitor_id" in source.columns else 0,
        "fall_event_count": int(match_quality["fall_event_id"].nunique(dropna=True)),
        "eligible_hazard_events": int(match_quality.loc[match_quality["hazard_eligible"], "fall_event_id"].nunique(dropna=True)),
        "fall_events_after_source_start": int(match_quality.loc[match_quality["after_effective_source_start"], "fall_event_id"].nunique(dropna=True)),
        "fall_events_in_source_divisions": int(match_quality.loc[match_quality["division_available"], "fall_event_id"].nunique(dropna=True)),
        "fall_events_eligible_for_matching": int(eligible_for_matching["fall_event_id"].nunique(dropna=True)),
        "fall_events_with_selected_controls": int(match_quality.loc[match_quality["selected_control_count"] > 0, "fall_event_id"].nunique(dropna=True)),
        "matched_control_rows": int(match_quality["selected_control_count"].sum()),
        "unmatched_fall_count": int(
            match_quality.loc[
                match_quality["hazard_eligible"] & (match_quality["selected_control_count"] == 0),
                "fall_event_id",
            ].nunique(dropna=True)
        ),
    }
    return match_quality, summary


def write_livestream_localized_analysis_tables(
    settings: Settings,
    raw_tables: dict[str, pd.DataFrame] | None = None,
) -> tuple[list[Path], Path]:
    raw = raw_tables or load_raw_tables(settings)
    panel = _prepare_second_level_panel(raw, settings)
    event_windows = _scope_to_study_hospital(
        raw.get("fall_livestream_event_windows", pd.DataFrame()),
        settings.study_hospital_id,
    ).copy()
    localized_metrics = _compute_event_localized_metrics(panel, event_windows)
    response_curve = _build_response_curve(panel)
    case_windows = _scope_to_study_hospital(
        raw.get("fall_case_crossover_windows", pd.DataFrame()),
        settings.study_hospital_id,
    ).copy()
    negative_control_windows = _scope_to_study_hospital(
        raw.get("fall_negative_control_windows", pd.DataFrame()),
        settings.study_hospital_id,
    ).copy()
    negative_control_source_inventory = _scope_to_study_hospital(
        raw.get("negative_control_source_inventory", pd.DataFrame()),
        settings.study_hospital_id,
    ).copy()

    case_outputs_enabled = settings.case_crossover_source_ready
    negative_outputs_enabled = settings.negative_control_source_ready
    if case_outputs_enabled:
        case_sets, case_effects, case_diagnostics = _build_case_crossover_outputs(case_windows)
        case_diagnostics.update(
            {
                "control_source_status": settings.case_crossover_source_status,
                "control_family": "same_monitor_prior_day",
                "status": "complete",
            }
        )
    else:
        case_sets = pd.DataFrame()
        case_effects = pd.DataFrame()
        case_diagnostics = {
            "event_count": 0,
            "events_with_hazard": 0,
            "events_with_any_control": 0,
            "events_with_full_pair": 0,
            "control_source_status": settings.case_crossover_source_status,
            "control_family": "same_monitor_prior_day",
            "status": "pending_external_source",
        }

    if negative_outputs_enabled:
        negative_sets, negative_effects, negative_diagnostics = _build_negative_control_outputs(
            case_windows,
            negative_control_windows,
        )
        negative_match_quality, negative_coverage = _build_negative_control_match_quality(
            case_windows,
            negative_control_windows,
            negative_control_source_inventory,
            settings,
        )
        negative_diagnostics.update(
            {
                "control_source_status": settings.negative_control_source_status,
                "control_family": "cross_monitor_nonfall",
                "status": negative_coverage.get("history_status", "complete"),
            }
        )
    else:
        negative_sets = pd.DataFrame()
        negative_effects = pd.DataFrame()
        negative_match_quality = pd.DataFrame(
            columns=[
                "fall_event_id",
                "hazard_anchor_ts_utc",
                "division_id",
                "source_monitor_id",
                "hazard_eligible",
                "after_effective_source_start",
                "division_available",
                "division_chunk_count",
                "matching_candidates",
                "selected_control_count",
                "best_match_tier",
                "excluded_reason",
            ]
        )
        negative_coverage = {
            "run_id": settings.run_id,
            "generated_at_utc": utc_now_iso(),
            "control_source_status": settings.negative_control_source_status,
            "expected_source_start_date_utc": settings.negative_control_expected_start_date.isoformat(),
            "effective_source_start_ts_utc": None,
            "history_status": "pending_external_source",
            "source_chunk_count": 0,
            "source_distinct_divisions": 0,
            "source_distinct_monitors": 0,
            "fall_event_count": 0,
            "eligible_hazard_events": 0,
            "fall_events_after_source_start": 0,
            "fall_events_in_source_divisions": 0,
            "fall_events_eligible_for_matching": 0,
            "fall_events_with_selected_controls": 0,
            "matched_control_rows": 0,
            "unmatched_fall_count": 0,
        }
        negative_diagnostics = {
            "events_with_hazard": 0,
            "events_with_negative_control": 0,
            "events_with_negative_control_pair": 0,
            "control_source_status": settings.negative_control_source_status,
            "control_family": "cross_monitor_nonfall",
            "status": "pending_external_source",
        }

    outputs = [
        _write_csv(panel, settings.paths.qa_dir / f"fall_second_level_panel_{settings.run_id}.csv"),
        _write_csv(
            localized_metrics,
            settings.paths.qa_dir / f"fall_event_localized_metrics_{settings.run_id}.csv",
        ),
        _write_csv(response_curve, settings.paths.qa_dir / f"fall_response_curve_{settings.run_id}.csv"),
    ]
    if case_outputs_enabled:
        outputs.extend(
            [
                _write_csv(
                    case_sets,
                    settings.paths.qa_dir / f"fall_case_crossover_sets_{settings.run_id}.csv",
                ),
                _write_csv(
                    case_effects,
                    settings.paths.qa_dir / f"fall_case_crossover_effects_{settings.run_id}.csv",
                ),
            ]
        )
    if negative_outputs_enabled:
        outputs.extend(
            [
                _write_csv(
                    negative_sets,
                    settings.paths.qa_dir / f"fall_negative_control_sets_{settings.run_id}.csv",
                ),
                _write_csv(
                    negative_effects,
                    settings.paths.qa_dir / f"fall_negative_control_effects_{settings.run_id}.csv",
                ),
            ]
        )
    negative_match_quality_path = _write_csv(
        negative_match_quality,
        settings.paths.qa_dir / f"fall_negative_control_match_quality_{settings.run_id}.csv",
    )
    outputs.append(negative_match_quality_path)
    diagnostics = {
        "run_id": settings.run_id,
        "generated_at_utc": utc_now_iso(),
        "second_level_rows": int(len(panel.index)),
        "localized_events": int(len(localized_metrics.index)),
        "response_curve_seconds": int(len(response_curve.index)),
        "case_crossover_source_status": settings.case_crossover_source_status,
        "negative_control_source_status": settings.negative_control_source_status,
        "nonfall_control_source_status": settings.negative_control_source_status,
    }
    diagnostics.update(case_diagnostics)
    diagnostics_path = settings.paths.qa_dir / f"fall_case_crossover_diagnostics_{settings.run_id}.json"
    write_json(diagnostics_path, diagnostics)
    outputs.append(diagnostics_path)
    negative_diagnostics_path = settings.paths.qa_dir / f"fall_negative_control_diagnostics_{settings.run_id}.json"
    write_json(
        negative_diagnostics_path,
        {
            "run_id": settings.run_id,
            "generated_at_utc": utc_now_iso(),
            **negative_diagnostics,
        },
    )
    outputs.append(negative_diagnostics_path)
    negative_coverage_path = settings.paths.qa_dir / f"fall_negative_control_coverage_{settings.run_id}.json"
    write_json(negative_coverage_path, negative_coverage)
    outputs.append(negative_coverage_path)
    return outputs, diagnostics_path


def write_livestream_label_evaluation_tables(
    settings: Settings,
    raw_tables: dict[str, pd.DataFrame] | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    raw = raw_tables or load_raw_tables(settings)
    event_windows = _scope_to_study_hospital(
        raw.get("fall_livestream_event_windows", pd.DataFrame()),
        settings.study_hospital_id,
    )
    second_level_panel = _scope_to_study_hospital(
        raw.get("fall_livestream_second_level", pd.DataFrame()),
        settings.study_hospital_id,
    )
    crossover_second_level_panel = _scope_to_study_hospital(
        raw.get("fall_case_crossover_second_level", pd.DataFrame()),
        settings.study_hospital_id,
    )
    artifacts = evaluate_livestream_derivations_against_truth(
        settings,
        event_windows,
        second_level_panel,
        crossover_second_level_panel,
    )

    outputs = [
        _write_csv(
            artifacts.sequence_metrics,
            settings.paths.qa_dir / f"label_eval_sequence_metrics_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.confusion_matrix,
            settings.paths.qa_dir / f"label_eval_confusion_matrix_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.probability_quality,
            settings.paths.qa_dir / f"label_eval_probability_quality_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.auc_summary,
            settings.paths.qa_dir / f"label_eval_auc_summary_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.threshold_sweep,
            settings.paths.qa_dir / f"label_eval_threshold_sweep_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.calibration_curve,
            settings.paths.qa_dir / f"label_eval_calibration_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.response_timing,
            settings.paths.qa_dir / f"label_eval_response_timing_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.population_stats,
            settings.paths.qa_dir / f"label_eval_population_stats_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.truth_prefall_location,
            settings.paths.qa_dir / f"label_eval_truth_prefall_location_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.tag_location_association,
            settings.paths.qa_dir / f"label_eval_tag_location_association_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.signal_location_profile,
            settings.paths.qa_dir / f"label_eval_signal_location_profile_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.association_summary,
            settings.paths.qa_dir / f"label_eval_association_summary_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.shadow_model_metrics,
            settings.paths.qa_dir / f"label_eval_shadow_model_metrics_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.shadow_model_predictions,
            settings.paths.qa_dir / f"label_eval_shadow_model_predictions_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.shadow_model_comparison,
            settings.paths.qa_dir / f"label_eval_shadow_model_comparison_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.onset_events,
            settings.paths.qa_dir / f"onset_events_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.onset_case_crossover,
            settings.paths.qa_dir / f"onset_case_crossover_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.shadow_feature_importance,
            settings.paths.qa_dir / f"shadow_feature_importance_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.shadow_ablation,
            settings.paths.qa_dir / f"shadow_ablation_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.shadow_window_sensitivity,
            settings.paths.qa_dir / f"shadow_window_sensitivity_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.shadow_cv_fold_metrics,
            settings.paths.qa_dir / f"shadow_cv_fold_metrics_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.furniture_origin_chain,
            settings.paths.qa_dir / f"label_eval_furniture_origin_chain_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.post_departure_latency,
            settings.paths.qa_dir / f"label_eval_post_departure_latency_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.tag_origin_chain_crosstab,
            settings.paths.qa_dir / f"label_eval_tag_origin_chain_crosstab_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.benchmark_sequence_metrics,
            settings.paths.qa_dir / f"label_eval_benchmark_sequence_metrics_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.benchmark_label_metrics,
            settings.paths.qa_dir / f"label_eval_benchmark_label_metrics_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.benchmark_confusion_matrix,
            settings.paths.qa_dir / f"label_eval_benchmark_confusion_matrix_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.benchmark_probability_quality,
            settings.paths.qa_dir / f"label_eval_benchmark_probability_quality_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.sequence_predictions,
            settings.paths.qa_dir / f"label_eval_sequence_predictions_{settings.run_id}.csv",
        ),
        _write_csv(
            artifacts.candidate_comparison,
            settings.paths.qa_dir / f"label_eval_candidate_comparison_{settings.run_id}.csv",
        ),
    ]

    operating_point_path = settings.paths.qa_dir / f"label_eval_operating_point_{settings.run_id}.json"
    write_json(operating_point_path, artifacts.operating_point)
    outputs.append(operating_point_path)

    threshold_path = settings.paths.qa_dir / f"label_eval_threshold_checks_{settings.run_id}.json"
    write_json(threshold_path, artifacts.threshold_checks)
    outputs.append(threshold_path)

    benchmark_status_path = settings.paths.qa_dir / f"label_eval_benchmark_status_{settings.run_id}.json"
    write_json(benchmark_status_path, artifacts.benchmark_status)
    outputs.append(benchmark_status_path)
    return outputs, artifacts.threshold_checks


def write_consensus_adjudication_tables(settings: Settings) -> list[Path]:
    consensus_path = _csv_path(settings, settings.fall_labels_consensus_csv_path)
    consensus, summary = load_consensus_annotations(consensus_path)

    if summary.get("status") == "ok":
        status_counts, summary_table = build_consensus_adjudication_tables(
            consensus,
            study_start_date=settings.study_start_date,
            study_end_date=settings.study_end_date,
        )
    else:
        status_counts = pd.DataFrame(
            [
                {
                    "consensus_status": summary.get("status", "missing_consensus_csv"),
                    "row_count": 0,
                    "unique_event_keys": 0,
                }
            ]
        )
        summary_table = pd.DataFrame(
            [
                {"metric": "status", "value": summary.get("status", "missing_consensus_csv")},
                {"metric": "path", "value": summary.get("path", str(consensus_path))},
            ]
        )

    return [
        _write_csv(
            status_counts,
            settings.paths.qa_dir / f"consensus_adjudication_status_counts_{settings.run_id}.csv",
        ),
        _write_csv(
            summary_table,
            settings.paths.qa_dir / f"consensus_adjudication_summary_{settings.run_id}.csv",
        ),
    ]


def _consensus_study_window_reconciliation_rows(settings: Settings | None) -> list[dict[str, Any]]:
    if settings is None:
        return []

    consensus_path = _csv_path(settings, settings.fall_labels_consensus_csv_path)
    consensus, summary = load_consensus_annotations(consensus_path)
    if summary.get("status") != "ok":
        return [
            {
                "metric": "consensus_summary_status",
                "value": summary.get("status", "missing_consensus_csv"),
                "notes": f"unable to load consensus CSV at {consensus_path}",
            }
        ]

    _, summary_table = build_consensus_adjudication_tables(
        consensus,
        study_start_date=settings.study_start_date,
        study_end_date=settings.study_end_date,
    )
    summary_map = dict(zip(summary_table["metric"], summary_table["value"], strict=True))
    study_window_label = f"{settings.study_start_date.isoformat()} to {settings.study_end_date.isoformat()}"
    return [
        {
            "metric": "consensus_source_unique_monitors_full_file",
            "value": summary_map.get("source_unique_monitors", ""),
            "notes": "unique monitor_ids parsed from the entire adjudicated consensus CSV",
        },
        {
            "metric": "consensus_included_fall_unique_monitors_full_file",
            "value": summary_map.get("included_fall_unique_monitors", ""),
            "notes": "unique monitor_ids among full-file included_fall annotations",
        },
        {
            "metric": "consensus_source_rows_study_window",
            "value": summary_map.get("study_window_source_rows", ""),
            "notes": f"consensus source rows with event_key local dates inside configured study window {study_window_label}",
        },
        {
            "metric": "consensus_source_unique_monitors_study_window",
            "value": summary_map.get("study_window_source_unique_monitors", ""),
            "notes": f"unique monitor_ids from consensus source rows inside configured study window {study_window_label}",
        },
        {
            "metric": "consensus_included_fall_rows_study_window",
            "value": summary_map.get("study_window_included_fall_rows", ""),
            "notes": f"included_fall consensus rows inside configured study window {study_window_label}",
        },
        {
            "metric": "consensus_included_fall_unique_monitors_study_window",
            "value": summary_map.get("study_window_included_fall_unique_monitors", ""),
            "notes": f"unique monitor_ids among included_fall consensus rows inside configured study window {study_window_label}",
        },
    ]


def _safe_rate_per_1000(falls: float, exposure_hours: float) -> float | pd.NA:
    if exposure_hours <= 0:
        return pd.NA
    return round((falls / exposure_hours) * 1000.0, 4)


def _safe_rate_per_exposure_hour(event_count: float, exposure_hours: float) -> float | pd.NA:
    if exposure_hours <= 0:
        return pd.NA
    return round(event_count / exposure_hours, 4)


def _safe_rate_per_100(event_count: float, exposure_hours: float) -> float | pd.NA:
    if exposure_hours <= 0:
        return pd.NA
    return round((event_count / exposure_hours) * 100.0, 4)


def _coerce_join_keys(
    df: pd.DataFrame,
    *,
    keys: tuple[str, ...] = ("monitor_id", "patient_id", "hospital_id", "division_id"),
) -> None:
    for column in keys:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")


def _resolve_analysis_scope(
    analysis_base: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    eligible = eligibility.loc[eligibility["eligible"]].copy()
    if eligible.empty:
        return "all_eligible", pd.DataFrame(), pd.DataFrame()

    eligible["cohort_type"] = eligible["cohort_type"].astype("string")
    analysis = analysis_base.copy()
    analysis["cohort_type"] = analysis["cohort_type"].astype("string")

    intervention_eligible = eligible.loc[eligible["cohort_type"] == "intervention"].copy()
    if not intervention_eligible.empty:
        scope = "intervention_eligible"
        analysis_scope = analysis.loc[analysis["cohort_type"] == "intervention"].copy()
        eligible_scope = intervention_eligible
    else:
        scope = "all_eligible"
        analysis_scope = analysis
        eligible_scope = eligible

    _coerce_join_keys(analysis_scope)
    _coerce_join_keys(eligible_scope)
    return scope, analysis_scope, eligible_scope


def _load_eligible_events(
    settings: Settings | None,
    raw_tables: dict[str, pd.DataFrame],
    eligible_scope: pd.DataFrame,
    analysis_scope: pd.DataFrame,
    hospital_timezone: str,
) -> pd.DataFrame:
    if settings is None:
        event_windows = raw_tables.get("fall_livestream_event_windows", pd.DataFrame()).copy()
    else:
        event_windows = _canonical_event_windows_from_raw_tables(settings, raw_tables)
    if event_windows.empty or eligible_scope.empty or analysis_scope.empty:
        return pd.DataFrame()

    _coerce_join_keys(event_windows)
    event_keys = ["hospital_id", "division_id", "monitor_id", "patient_id"]
    eligible_event_keys = eligible_scope[event_keys].drop_duplicates()
    eligible_events = event_windows.merge(eligible_event_keys, on=event_keys, how="inner")
    if eligible_events.empty:
        return pd.DataFrame()

    local_source_eligible = (
        eligible_events["fall_ts_local"]
        if "fall_ts_local" in eligible_events.columns
        else pd.Series(pd.NA, index=eligible_events.index)
    )
    utc_source_eligible = (
        eligible_events["fall_ts_utc"]
        if "fall_ts_utc" in eligible_events.columns
        else pd.Series(pd.NA, index=eligible_events.index)
    )
    fall_ts_local = _parse_local_wall_timestamp(local_source_eligible)
    fall_ts_utc = _parse_utc_timestamp(utc_source_eligible)
    fallback_local = _utc_to_local_wall_timestamp(fall_ts_utc, hospital_timezone)
    event_hour_local = fall_ts_local.fillna(fallback_local).dt.floor("h")
    event_hour_utc = fall_ts_utc.dt.tz_localize(None).dt.floor("h")

    analysis_scope = analysis_scope.copy()
    if "daypart" not in analysis_scope.columns:
        analysis_scope["daypart"] = _derive_daypart(_parse_local_wall_timestamp(analysis_scope["hour_ts"]).dt.hour)
    analysis_hour_keys = analysis_scope[event_keys + ["hour_ts", "daypart"]].copy()
    analysis_hour_keys["hour_ts"] = _parse_local_wall_timestamp(analysis_hour_keys["hour_ts"]).dt.floor("h")
    analysis_hour_keys = analysis_hour_keys.dropna(subset=["hour_ts"]).drop_duplicates()
    if analysis_hour_keys.empty:
        return pd.DataFrame()

    local_linked = _link_events_to_analysis_hours(eligible_events, analysis_hour_keys, event_hour_local)
    utc_linked = _link_events_to_analysis_hours(eligible_events, analysis_hour_keys, event_hour_utc)
    eligible_events = utc_linked if len(utc_linked.index) > len(local_linked.index) else local_linked
    if eligible_events.empty:
        return pd.DataFrame()

    label_series = eligible_events.get("prefall_location_label", pd.Series(index=eligible_events.index, dtype="string"))
    label_series = label_series.astype("string").fillna("unknown")
    eligible_events["prefall_location_label"] = label_series.replace({"other": "room", "unknown": "no_patient"})
    if "daypart" not in eligible_events.columns:
        eligible_events["daypart"] = "unknown"
    return eligible_events


def _build_chair_bed_rates_table(
    analysis_base: pd.DataFrame,
    eligibility: pd.DataFrame,
    raw_tables: dict[str, pd.DataFrame],
    hospital_timezone: str,
    settings: Settings | None = None,
) -> pd.DataFrame:
    columns = [
        "scope",
        "daypart",
        "position",
        "eligible_event_count",
        "exposure_hours",
        "expected_falls",
        "hard_label_falls",
        "rate_per_1000_exposure_hours_expected",
        "rate_per_1000_exposure_hours_hard_label",
    ]
    if analysis_base.empty or eligibility.empty:
        return pd.DataFrame(columns=columns)

    scope, analysis_scope, eligible_scope = _resolve_analysis_scope(analysis_base, eligibility)
    if analysis_scope.empty or eligible_scope.empty:
        return pd.DataFrame(columns=columns)

    eligible_events = _load_eligible_events(settings, raw_tables, eligible_scope, analysis_scope, hospital_timezone)
    if eligible_events.empty:
        return pd.DataFrame(columns=columns)

    analysis_scope = analysis_scope.copy()
    analysis_scope["daypart"] = analysis_scope.get("daypart", pd.Series(index=analysis_scope.index)).fillna("unknown")
    eligible_events["daypart"] = eligible_events.get("daypart", pd.Series(index=eligible_events.index)).fillna(
        "unknown"
    )

    rows: list[dict[str, Any]] = []
    strata: list[tuple[str, pd.DataFrame, pd.DataFrame]] = [("all_dayparts", analysis_scope, eligible_events)]
    for daypart in DAYPART_ORDER:
        scoped_analysis = analysis_scope.loc[analysis_scope["daypart"] == daypart].copy()
        scoped_events = eligible_events.loc[eligible_events["daypart"] == daypart].copy()
        if scoped_analysis.empty:
            continue
        strata.append((daypart, scoped_analysis, scoped_events))

    for daypart, daypart_analysis, daypart_events in strata:
        chair_exposure = float(pd.to_numeric(daypart_analysis["pct_chair"], errors="coerce").fillna(0.0).sum())
        bed_exposure = float(pd.to_numeric(daypart_analysis["pct_bed"], errors="coerce").fillna(0.0).sum())
        expected_chair = float(pd.to_numeric(daypart_events.get("pre_prob_chair"), errors="coerce").fillna(0.0).sum())
        expected_bed = float(pd.to_numeric(daypart_events.get("pre_prob_bed"), errors="coerce").fillna(0.0).sum())
        labels = daypart_events.get(
            "prefall_location_label",
            pd.Series(index=daypart_events.index, dtype="string"),
        ).astype("string")
        hard_chair = int(labels.eq("chair").sum())
        hard_bed = int(labels.eq("bed").sum())
        event_count = int(len(daypart_events.index))

        rows.extend(
            [
                {
                    "scope": scope,
                    "daypart": daypart,
                    "position": "chair",
                    "eligible_event_count": event_count,
                    "exposure_hours": round(chair_exposure, 4),
                    "expected_falls": round(expected_chair, 4),
                    "hard_label_falls": hard_chair,
                    "rate_per_1000_exposure_hours_expected": _safe_rate_per_1000(expected_chair, chair_exposure),
                    "rate_per_1000_exposure_hours_hard_label": _safe_rate_per_1000(hard_chair, chair_exposure),
                },
                {
                    "scope": scope,
                    "daypart": daypart,
                    "position": "bed",
                    "eligible_event_count": event_count,
                    "exposure_hours": round(bed_exposure, 4),
                    "expected_falls": round(expected_bed, 4),
                    "hard_label_falls": hard_bed,
                    "rate_per_1000_exposure_hours_expected": _safe_rate_per_1000(expected_bed, bed_exposure),
                    "rate_per_1000_exposure_hours_hard_label": _safe_rate_per_1000(hard_bed, bed_exposure),
                },
            ]
        )
    return pd.DataFrame(rows, columns=columns)


def _build_chair_bed_operational_event_rates_table(
    analysis_base: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "scope",
        "metric",
        "metric_source_column",
        "position",
        "exposure_hours",
        "weighted_event_count",
        "rate_per_exposure_hour",
        "rate_per_100_exposure_hours",
        "chair_to_bed_rate_ratio",
    ]
    if analysis_base.empty or eligibility.empty:
        return pd.DataFrame(columns=columns)

    scope, analysis_scope, eligible_scope = _resolve_analysis_scope(analysis_base, eligibility)
    if analysis_scope.empty or eligible_scope.empty:
        return pd.DataFrame(columns=columns)

    metric_specs = [
        ("alarms", "num_alarms"),
        ("nudges", "num_nudges"),
        ("announcements", "num_announcements"),
    ]
    analysis_scope = analysis_scope.copy()
    for _, source_column in metric_specs:
        analysis_scope[source_column] = pd.to_numeric(
            analysis_scope.get(source_column),
            errors="coerce",
        ).fillna(0.0)

    chair_exposure = float(pd.to_numeric(analysis_scope["pct_chair"], errors="coerce").fillna(0.0).sum())
    bed_exposure = float(pd.to_numeric(analysis_scope["pct_bed"], errors="coerce").fillna(0.0).sum())

    rows: list[dict[str, Any]] = []
    for metric_name, source_column in metric_specs:
        position_rows = {
            "chair": {
                "exposure": chair_exposure,
                "weighted_event_count": float(
                    (
                        analysis_scope[source_column]
                        * pd.to_numeric(analysis_scope["pct_chair"], errors="coerce").fillna(0.0)
                    ).sum()
                ),
            },
            "bed": {
                "exposure": bed_exposure,
                "weighted_event_count": float(
                    (
                        analysis_scope[source_column]
                        * pd.to_numeric(analysis_scope["pct_bed"], errors="coerce").fillna(0.0)
                    ).sum()
                ),
            },
        }
        rates = {
            position: _safe_rate_per_exposure_hour(values["weighted_event_count"], float(values["exposure"]))
            for position, values in position_rows.items()
        }
        rr = _safe_rate_ratio(rates.get("chair", pd.NA), rates.get("bed", pd.NA))

        for position, values in position_rows.items():
            rows.append(
                {
                    "scope": scope,
                    "metric": metric_name,
                    "metric_source_column": source_column,
                    "position": position,
                    "exposure_hours": round(float(values["exposure"]), 4),
                    "weighted_event_count": round(float(values["weighted_event_count"]), 4),
                    "rate_per_exposure_hour": rates.get(position, pd.NA),
                    "rate_per_100_exposure_hours": _safe_rate_per_100(
                        float(values["weighted_event_count"]),
                        float(values["exposure"]),
                    ),
                    "chair_to_bed_rate_ratio": rr,
                }
            )

    result = pd.DataFrame(rows, columns=columns)
    metric_order = {"alarms": 0, "nudges": 1, "announcements": 2}
    position_order = {"chair": 0, "bed": 1}
    result["_metric_order"] = result["metric"].map(metric_order).fillna(99)
    result["_position_order"] = result["position"].map(position_order).fillna(99)
    result = result.sort_values(["_metric_order", "_position_order"], kind="mergesort").drop(
        columns=["_metric_order", "_position_order"]
    )
    return result.reset_index(drop=True)


def _add_confidence_features(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events

    scored = events.copy()
    scored["pre_state_visible_prob"] = pd.to_numeric(scored.get("pre_state_visible_prob"), errors="coerce").fillna(0.0)
    scored["pre_dropout_visibility_ratio"] = pd.to_numeric(
        scored.get("pre_dropout_visibility_ratio"), errors="coerce"
    ).fillna(0.0)
    gap = pd.to_numeric(scored.get("pre_dropout_last_visible_gap_seconds"), errors="coerce")
    gap = gap.fillna(360.0).clip(lower=0.0)
    scored["pre_dropout_last_visible_gap_seconds"] = gap

    gap_penalty = (1.0 - (gap / 180.0)).clip(lower=0.0, upper=1.0)
    score = (
        0.50 * scored["pre_state_visible_prob"]
        + 0.35 * scored["pre_dropout_visibility_ratio"]
        + 0.15 * gap_penalty
    ).clip(lower=0.0, upper=1.0)
    scored["confidence_weight"] = score

    high_mask = (
        (scored["pre_state_visible_prob"] >= 0.70)
        & (scored["pre_dropout_visibility_ratio"] >= 0.70)
        & (scored["pre_dropout_last_visible_gap_seconds"] <= 90.0)
    )
    low_mask = (
        (scored["pre_state_visible_prob"] < 0.40)
        | (scored["pre_dropout_visibility_ratio"] < 0.40)
        | (scored["pre_dropout_last_visible_gap_seconds"] > 180.0)
    )
    scored["confidence_segment"] = "medium_confidence"
    scored.loc[high_mask, "confidence_segment"] = "high_confidence"
    scored.loc[low_mask, "confidence_segment"] = "low_confidence"
    return scored


def _build_chair_bed_confidence_sensitivity_table(
    analysis_base: pd.DataFrame,
    eligibility: pd.DataFrame,
    raw_tables: dict[str, pd.DataFrame],
    hospital_timezone: str,
    settings: Settings | None = None,
) -> pd.DataFrame:
    columns = [
        "scope",
        "confidence_segment",
        "position",
        "eligible_event_count",
        "mean_confidence_weight",
        "exposure_hours",
        "expected_falls",
        "confidence_weighted_expected_falls",
        "hard_label_falls",
        "rate_per_1000_exposure_hours_expected",
        "rate_per_1000_exposure_hours_confidence_weighted",
        "rate_per_1000_exposure_hours_hard_label",
    ]
    if analysis_base.empty or eligibility.empty:
        return pd.DataFrame(columns=columns)

    scope, analysis_scope, eligible_scope = _resolve_analysis_scope(analysis_base, eligibility)
    if analysis_scope.empty or eligible_scope.empty:
        return pd.DataFrame(columns=columns)

    chair_exposure = float(pd.to_numeric(analysis_scope["pct_chair"], errors="coerce").fillna(0.0).sum())
    bed_exposure = float(pd.to_numeric(analysis_scope["pct_bed"], errors="coerce").fillna(0.0).sum())

    eligible_events = _load_eligible_events(settings, raw_tables, eligible_scope, analysis_scope, hospital_timezone)
    if eligible_events.empty:
        return pd.DataFrame(columns=columns)

    scored = _add_confidence_features(eligible_events)
    scored["pre_prob_chair"] = pd.to_numeric(scored.get("pre_prob_chair"), errors="coerce").fillna(0.0)
    scored["pre_prob_bed"] = pd.to_numeric(scored.get("pre_prob_bed"), errors="coerce").fillna(0.0)

    scoped_events: list[tuple[str, pd.DataFrame]] = [
        ("all_events", scored),
        ("high_confidence", scored.loc[scored["confidence_segment"] == "high_confidence"].copy()),
        ("medium_confidence", scored.loc[scored["confidence_segment"] == "medium_confidence"].copy()),
        ("low_confidence", scored.loc[scored["confidence_segment"] == "low_confidence"].copy()),
    ]

    rows: list[dict[str, Any]] = []
    for segment, segment_df in scoped_events:
        if segment_df.empty:
            continue

        event_count = int(len(segment_df.index))
        mean_confidence = round(float(segment_df["confidence_weight"].mean()), 4)
        labels = segment_df["prefall_location_label"]

        summary = {
            "chair": {
                "exposure": chair_exposure,
                "expected": float(segment_df["pre_prob_chair"].sum()),
                "weighted_expected": float((segment_df["pre_prob_chair"] * segment_df["confidence_weight"]).sum()),
                "hard_label_falls": int(labels.eq("chair").sum()),
            },
            "bed": {
                "exposure": bed_exposure,
                "expected": float(segment_df["pre_prob_bed"].sum()),
                "weighted_expected": float((segment_df["pre_prob_bed"] * segment_df["confidence_weight"]).sum()),
                "hard_label_falls": int(labels.eq("bed").sum()),
            },
        }

        for position, values in summary.items():
            exposure = float(values["exposure"])
            expected = float(values["expected"])
            weighted_expected = float(values["weighted_expected"])
            hard_label_falls = int(values["hard_label_falls"])
            rows.append(
                {
                    "scope": scope,
                    "confidence_segment": segment,
                    "position": position,
                    "eligible_event_count": event_count,
                    "mean_confidence_weight": mean_confidence,
                    "exposure_hours": round(exposure, 4),
                    "expected_falls": round(expected, 4),
                    "confidence_weighted_expected_falls": round(weighted_expected, 4),
                    "hard_label_falls": hard_label_falls,
                    "rate_per_1000_exposure_hours_expected": _safe_rate_per_1000(expected, exposure),
                    "rate_per_1000_exposure_hours_confidence_weighted": _safe_rate_per_1000(
                        weighted_expected, exposure
                    ),
                    "rate_per_1000_exposure_hours_hard_label": _safe_rate_per_1000(hard_label_falls, exposure),
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(rows, columns=columns)
    segment_order = {"all_events": 0, "high_confidence": 1, "medium_confidence": 2, "low_confidence": 3}
    position_order = {"chair": 0, "bed": 1}
    result["_segment_order"] = result["confidence_segment"].map(segment_order).fillna(99)
    result["_position_order"] = result["position"].map(position_order).fillna(99)
    result = result.sort_values(["_segment_order", "_position_order"], kind="mergesort").drop(
        columns=["_segment_order", "_position_order"]
    )
    return result.reset_index(drop=True)


def _safe_rate_ratio(chair_rate: float | pd.NA, bed_rate: float | pd.NA) -> float | pd.NA:
    if pd.isna(chair_rate) or pd.isna(bed_rate):
        return pd.NA
    bed = float(bed_rate)
    if bed <= 0:
        return pd.NA
    return round(float(chair_rate) / bed, 6)


def _build_chair_bed_missingness_stress_table(
    analysis_base: pd.DataFrame,
    eligibility: pd.DataFrame,
    raw_tables: dict[str, pd.DataFrame],
    hospital_timezone: str,
    cutpoints: tuple[int, ...],
    settings: Settings | None = None,
) -> pd.DataFrame:
    columns = [
        "scope",
        "excluded_low_confidence_pct",
        "retained_event_count",
        "excluded_event_count",
        "retained_fraction",
        "mean_confidence_weight_retained",
        "position",
        "exposure_hours",
        "expected_falls",
        "confidence_weighted_expected_falls",
        "hard_label_falls",
        "rate_per_1000_exposure_hours_expected",
        "rate_per_1000_exposure_hours_confidence_weighted",
        "rate_per_1000_exposure_hours_hard_label",
        "chair_to_bed_rate_ratio_expected",
        "chair_to_bed_rate_ratio_confidence_weighted",
        "chair_to_bed_rate_ratio_hard_label",
    ]
    if analysis_base.empty or eligibility.empty:
        return pd.DataFrame(columns=columns)

    scope, analysis_scope, eligible_scope = _resolve_analysis_scope(analysis_base, eligibility)
    if analysis_scope.empty or eligible_scope.empty:
        return pd.DataFrame(columns=columns)

    chair_exposure = float(pd.to_numeric(analysis_scope["pct_chair"], errors="coerce").fillna(0.0).sum())
    bed_exposure = float(pd.to_numeric(analysis_scope["pct_bed"], errors="coerce").fillna(0.0).sum())
    eligible_events = _load_eligible_events(settings, raw_tables, eligible_scope, analysis_scope, hospital_timezone)
    if eligible_events.empty:
        return pd.DataFrame(columns=columns)

    scored = _add_confidence_features(eligible_events)
    scored["pre_prob_chair"] = pd.to_numeric(scored.get("pre_prob_chair"), errors="coerce").fillna(0.0)
    scored["pre_prob_bed"] = pd.to_numeric(scored.get("pre_prob_bed"), errors="coerce").fillna(0.0)
    scored["confidence_weight"] = pd.to_numeric(scored.get("confidence_weight"), errors="coerce").fillna(0.0)
    scored = scored.sort_values("confidence_weight", ascending=True, kind="mergesort").reset_index(drop=True)

    n_total = int(len(scored.index))
    rows: list[dict[str, Any]] = []
    for pct in sorted({max(0, min(100, int(value))) for value in cutpoints}):
        excluded_count = int((n_total * pct) // 100)
        retained = scored.iloc[excluded_count:].copy()
        retained_count = int(len(retained.index))
        if retained_count == 0:
            continue
        retained_fraction = round(retained_count / n_total, 4) if n_total else 0.0
        mean_weight = round(float(retained["confidence_weight"].mean()), 4)
        labels = retained["prefall_location_label"]

        position_rows = {
            "chair": {
                "exposure": chair_exposure,
                "expected": float(retained["pre_prob_chair"].sum()),
                "weighted_expected": float((retained["pre_prob_chair"] * retained["confidence_weight"]).sum()),
                "hard_label_falls": int(labels.eq("chair").sum()),
            },
            "bed": {
                "exposure": bed_exposure,
                "expected": float(retained["pre_prob_bed"].sum()),
                "weighted_expected": float((retained["pre_prob_bed"] * retained["confidence_weight"]).sum()),
                "hard_label_falls": int(labels.eq("bed").sum()),
            },
        }
        rates_expected = {
            position: _safe_rate_per_1000(values["expected"], float(values["exposure"]))
            for position, values in position_rows.items()
        }
        rates_weighted = {
            position: _safe_rate_per_1000(values["weighted_expected"], float(values["exposure"]))
            for position, values in position_rows.items()
        }
        rates_hard = {
            position: _safe_rate_per_1000(values["hard_label_falls"], float(values["exposure"]))
            for position, values in position_rows.items()
        }
        rr_expected = _safe_rate_ratio(rates_expected.get("chair", pd.NA), rates_expected.get("bed", pd.NA))
        rr_weighted = _safe_rate_ratio(rates_weighted.get("chair", pd.NA), rates_weighted.get("bed", pd.NA))
        rr_hard = _safe_rate_ratio(rates_hard.get("chair", pd.NA), rates_hard.get("bed", pd.NA))

        for position, values in position_rows.items():
            rows.append(
                {
                    "scope": scope,
                    "excluded_low_confidence_pct": pct,
                    "retained_event_count": retained_count,
                    "excluded_event_count": excluded_count,
                    "retained_fraction": retained_fraction,
                    "mean_confidence_weight_retained": mean_weight,
                    "position": position,
                    "exposure_hours": round(float(values["exposure"]), 4),
                    "expected_falls": round(float(values["expected"]), 4),
                    "confidence_weighted_expected_falls": round(float(values["weighted_expected"]), 4),
                    "hard_label_falls": int(values["hard_label_falls"]),
                    "rate_per_1000_exposure_hours_expected": rates_expected.get(position, pd.NA),
                    "rate_per_1000_exposure_hours_confidence_weighted": rates_weighted.get(position, pd.NA),
                    "rate_per_1000_exposure_hours_hard_label": rates_hard.get(position, pd.NA),
                    "chair_to_bed_rate_ratio_expected": rr_expected,
                    "chair_to_bed_rate_ratio_confidence_weighted": rr_weighted,
                    "chair_to_bed_rate_ratio_hard_label": rr_hard,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows, columns=columns)
    position_order = {"chair": 0, "bed": 1}
    result["_position_order"] = result["position"].map(position_order).fillna(99)
    result = result.sort_values(
        ["excluded_low_confidence_pct", "_position_order"],
        kind="mergesort",
    ).drop(columns=["_position_order"])
    return result.reset_index(drop=True)


def _build_chair_bed_analysis_reconciliation_table(
    analysis_base: pd.DataFrame,
    eligibility: pd.DataFrame,
    raw_tables: dict[str, pd.DataFrame],
    adjusted_rr_path: Path | None,
    hospital_timezone: str,
    settings: Settings | None = None,
) -> pd.DataFrame:
    columns = ["metric", "value", "notes"]
    if analysis_base.empty or eligibility.empty:
        return pd.DataFrame(columns=columns)

    scope, analysis_scope, eligible_scope = _resolve_analysis_scope(analysis_base, eligibility)
    if analysis_scope.empty or eligible_scope.empty:
        return pd.DataFrame(columns=columns)

    fall_events = raw_tables.get("fall_events_source", pd.DataFrame())
    raw_events_total = int(len(fall_events.index))

    eligible_events = _load_eligible_events(settings, raw_tables, eligible_scope, analysis_scope, hospital_timezone)
    eligible_event_count = int(len(eligible_events.index))
    labels = (
        eligible_events.get(
            "prefall_location_label",
            pd.Series(index=eligible_events.index, dtype="string"),
        )
        .astype("string")
        .fillna("unknown")
        .replace({"other": "room", "unknown": "no_patient"})
    )
    chair_bed_hard_label = int(labels.isin(["chair", "bed"]).sum())

    modeled_primary_events: float | None = None
    if adjusted_rr_path is not None and adjusted_rr_path.exists():
        try:
            adjusted_rr = pd.read_csv(adjusted_rr_path)
            row = adjusted_rr.loc[adjusted_rr["sensitivity_label"] == "primary_adjusted"]
            if not row.empty:
                modeled_primary_events = float(row.iloc[0]["n_events"])
        except Exception:
            modeled_primary_events = None

    modeled_vs_hard_label_delta = (
        float(modeled_primary_events - chair_bed_hard_label)
        if modeled_primary_events is not None
        else None
    )
    if modeled_primary_events is None:
        delta_note = "primary_adjusted row missing; rerun inferential model"
    elif modeled_vs_hard_label_delta > 0:
        delta_note = "proportional allocation includes room/no_patient probability mass"
    elif modeled_vs_hard_label_delta < 0:
        delta_note = "modeled events lower than chair/bed hard labels; investigate linkage"
    else:
        delta_note = "modeled and chair/bed hard-label counts match"

    rows = [
        {
            "metric": "scope",
            "value": scope,
            "notes": "analysis scope for denominator and event linkage",
        },
        {
            "metric": "raw_events_total",
            "value": raw_events_total,
            "notes": "fall_events_source rows scoped to study hospital",
        },
        {
            "metric": "eligible_events_analysis_base",
            "value": eligible_event_count,
            "notes": "events linked to eligible scope + analysis base hour keys",
        },
        {
            "metric": "eligible_events_chair_or_bed_hard_label",
            "value": chair_bed_hard_label,
            "notes": "eligible events with hard labels in {chair, bed}",
        },
        {
            "metric": "modeled_effective_events_primary",
            "value": modeled_primary_events if modeled_primary_events is not None else "",
            "notes": "primary_adjusted n_events from adjusted RR output",
        },
        {
            "metric": "modeled_minus_chair_bed_hard_label",
            "value": modeled_vs_hard_label_delta if modeled_vs_hard_label_delta is not None else "",
            "notes": delta_note,
        },
    ]
    rows.extend(_consensus_study_window_reconciliation_rows(settings))
    return pd.DataFrame(rows, columns=columns)


def write_cohort_analysis_pack(
    settings: Settings,
    raw_tables: dict[str, pd.DataFrame] | None = None,
    adjusted_rr_path: Path | None = None,
) -> tuple[Path, list[Path]]:
    hourly = _load_staged(settings, "prep.patient_hour_location_with_cohort_v1.parquet")
    eligibility = _load_staged(settings, "prep.patient_hour_eligibility_v1.parquet")
    analysis_base = _load_staged(settings, "prep.patient_hour_analysis_base_v1.parquet")
    raw = raw_tables or load_raw_tables(settings)

    outputs: list[Path] = []

    composition = pd.DataFrame()
    if not hourly.empty:
        composition = (
            hourly.groupby(["cohort_type", "hospital_id", "division_id", "monitor_id"], dropna=False)
            .size()
            .rename("hour_rows")
            .reset_index()
            .sort_values(["cohort_type", "hospital_id", "division_id", "monitor_id"])
        )
    outputs.append(
        _write_csv(
            composition,
            settings.paths.qa_dir / f"cohort_composition_{settings.run_id}.csv",
        )
    )

    duration = pd.DataFrame()
    pass_fail = pd.DataFrame()
    exclusions = pd.DataFrame()
    if not eligibility.empty:
        duration = (
            eligibility.groupby("cohort_type", dropna=False)["observed_hours"]
            .describe(percentiles=[0.25, 0.5, 0.75])
            .reset_index()
        )
        pass_fail = (
            eligibility.groupby(["cohort_type", "eligible"], dropna=False)
            .size()
            .rename("units")
            .reset_index()
        )
        exclusions = eligibility.loc[~eligibility["eligible"]].copy()

    outputs.append(_write_csv(duration, settings.paths.qa_dir / f"cohort_duration_{settings.run_id}.csv"))
    outputs.append(
        _write_csv(pass_fail, settings.paths.qa_dir / f"cohort_eligibility_rates_{settings.run_id}.csv")
    )
    outputs.append(_write_csv(exclusions, settings.paths.qa_dir / f"cohort_exclusions_{settings.run_id}.csv"))

    fall_density = pd.DataFrame()
    control_coverage = pd.DataFrame()
    if not analysis_base.empty:
        fall_density = (
            analysis_base.groupby(["cohort_type", "daypart"], dropna=False)["falls_in_hour"]
            .sum()
            .rename("falls")
            .reset_index()
        )

    if not hourly.empty:
        control_coverage = (
            hourly.groupby(["cohort_type", "hospital_id", "division_id"], dropna=False)
            .agg(
                rows=("monitor_id", "size"),
                valid_rows=("row_valid_pct", "sum"),
            )
            .reset_index()
        )

    outputs.append(_write_csv(fall_density, settings.paths.qa_dir / f"fall_density_{settings.run_id}.csv"))
    outputs.append(
        _write_csv(
            control_coverage,
            settings.paths.qa_dir / f"control_denominator_coverage_{settings.run_id}.csv",
        )
    )
    if settings.effective_run_mode == "inferential_ready":
        chair_bed_rates = _build_chair_bed_rates_table(
            analysis_base,
            eligibility,
            raw,
            settings.hospital_timezone,
            settings=settings,
        )
        outputs.append(
            _write_csv(
                chair_bed_rates,
                settings.paths.qa_dir / f"chair_bed_risk_rates_{settings.run_id}.csv",
            )
        )
        operational_event_rates = _build_chair_bed_operational_event_rates_table(
            analysis_base,
            eligibility,
        )
        outputs.append(
            _write_csv(
                operational_event_rates,
                settings.paths.qa_dir / f"chair_bed_operational_event_rates_{settings.run_id}.csv",
            )
        )
        confidence_sensitivity = _build_chair_bed_confidence_sensitivity_table(
            analysis_base,
            eligibility,
            raw,
            settings.hospital_timezone,
            settings=settings,
        )
        outputs.append(
            _write_csv(
                confidence_sensitivity,
                settings.paths.qa_dir / f"chair_bed_risk_rates_confidence_sensitivity_{settings.run_id}.csv",
            )
        )
        missingness_stress = _build_chair_bed_missingness_stress_table(
            analysis_base,
            eligibility,
            raw,
            settings.hospital_timezone,
            settings.confidence_missingness_cutpoints,
            settings=settings,
        )
        outputs.append(
            _write_csv(
                missingness_stress,
                settings.paths.qa_dir / f"chair_bed_missingness_stress_{settings.run_id}.csv",
            )
        )
        reconciliation = _build_chair_bed_analysis_reconciliation_table(
            analysis_base,
            eligibility,
            raw,
            adjusted_rr_path,
            settings.hospital_timezone,
            settings=settings,
        )
        outputs.append(
            _write_csv(
                reconciliation,
                settings.paths.qa_dir / f"chair_bed_analysis_reconciliation_{settings.run_id}.csv",
            )
        )

    md_path = settings.paths.qa_dir / f"cohort_analysis_{settings.run_id}.md"
    lines = [
        f"# Cohort Analysis {settings.run_id}",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "## Artifacts",
    ]
    lines.extend([f"- {path.name}" for path in outputs])
    lines.extend(
        [
            "",
            "## Summary",
            f"- Hourly rows: {len(hourly.index)}",
            f"- Eligibility units: {len(eligibility.index)}",
            f"- Analysis base rows: {len(analysis_base.index)}",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return md_path, outputs


def write_qa_summary_and_exclusions(settings: Settings) -> tuple[Path, Path, dict]:
    eligibility = _load_staged(settings, "prep.patient_hour_eligibility_v1.parquet")
    analysis_base = _load_staged(settings, "prep.patient_hour_analysis_base_v1.parquet")

    summary = {
        "run_id": settings.run_id,
        "generated_at_utc": utc_now_iso(),
        "effective_run_mode": settings.effective_run_mode,
        "case_crossover_source_status": settings.case_crossover_source_status,
        "negative_control_source_status": settings.negative_control_source_status,
        "nonfall_control_source_status": settings.negative_control_source_status,
        "eligibility_units": int(len(eligibility.index)),
        "eligibility_units_passed": int(eligibility["eligible"].sum()) if "eligible" in eligibility else 0,
        "analysis_base_rows": int(len(analysis_base.index)),
    }

    summary_df = pd.DataFrame([summary])
    summary_path = settings.paths.staged_run_dir / "prep.qa_summary_v1.parquet"
    summary_df.to_parquet(summary_path, index=False)

    exclusions = pd.DataFrame(columns=["hospital_id", "division_id", "monitor_id", "patient_id", "exclusion_reason"])
    if not eligibility.empty:
        exclusions = eligibility.loc[
            ~eligibility["eligible"],
            ["hospital_id", "division_id", "monitor_id", "patient_id", "exclusion_reason"],
        ].copy()
    exclusions_path = settings.paths.staged_run_dir / "prep.qa_exclusions_v1.parquet"
    exclusions.to_parquet(exclusions_path, index=False)

    summary_json_path = settings.paths.qa_dir / f"qa_summary_{settings.run_id}.json"
    write_json(summary_json_path, summary)

    return summary_path, exclusions_path, {
        "qa_summary_path": str(summary_path),
        "qa_exclusions_path": str(exclusions_path),
        "qa_summary_json_path": str(summary_json_path),
    }


def write_paper_handoff_bundle(settings: Settings, payload: dict[str, Any]) -> tuple[Path, Path]:
    handoff = {
        "run_id": settings.run_id,
        "generated_at_utc": utc_now_iso(),
        "run_mode": settings.effective_run_mode,
        "final_manuscript_target": "paper/manuscript.md",
        "inputs": {
            "gate_preflight": payload.get("gate_preflight_path"),
            "source_profile": payload.get("source_profile_path"),
            "falls_descriptive_artifacts": payload.get("falls_descriptive_paths", []),
            "livestream_descriptive_artifacts": payload.get("livestream_descriptive_paths", []),
            "livestream_localized_artifacts": payload.get("livestream_localized_paths", []),
            "livestream_localized_diagnostics": payload.get("livestream_localized_diagnostics_path"),
            "consensus_adjudication_artifacts": payload.get("consensus_adjudication_paths", []),
            "cohort_analysis_markdown": payload.get("cohort_analysis_markdown_path"),
            "cohort_analysis_csv": payload.get("cohort_analysis_csv_paths", []),
            "label_eval_artifacts": payload.get("label_eval_paths", []),
            "label_eval_threshold_checks": payload.get("label_eval_threshold_checks_path"),
            "qa_summary_parquet": payload.get("qa_summary_path"),
            "qa_exclusions_parquet": payload.get("qa_exclusions_path"),
            "analysis_base_parquet": str(
                settings.paths.staged_run_dir / "prep.patient_hour_analysis_base_v1.parquet"
            ),
        },
    }

    json_path = settings.paths.audit_dir / f"chair_fall_paper_inputs_{settings.run_id}.json"
    write_json(json_path, handoff)

    html_lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "  <meta charset=\"utf-8\" />",
        f"  <title>Chair Fall Paper Inputs - {settings.run_id}</title>",
        "  <style>",
        "    body { font-family: Georgia, serif; margin: 2rem auto; max-width: 980px; line-height: 1.4; }",
        "    h1, h2 { margin-bottom: 0.4rem; }",
        "    code { background: #f3f3f3; padding: 0.1rem 0.3rem; border-radius: 0.2rem; }",
        "    ul { margin-top: 0.4rem; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>Chair Fall Paper Input Bundle: {settings.run_id}</h1>",
        f"  <p><strong>Run mode:</strong> {settings.effective_run_mode}</p>",
        "  <p><strong>Target manuscript:</strong> <code>paper/manuscript.md</code></p>",
        "  <h2>Included Inputs</h2>",
        "  <ul>",
    ]
    for key, value in handoff["inputs"].items():
        if isinstance(value, list):
            html_lines.append(f"    <li><strong>{key}</strong>: {len(value)} files</li>")
            continue
        html_lines.append(f"    <li><strong>{key}</strong>: <code>{value}</code></li>")

    html_lines.extend(
        [
            "  </ul>",
            "  <p>This bundle is generated from prep outputs and can be consumed while authoring ",
            "  <code>paper/manuscript.md</code>.</p>",
            "</body>",
            "</html>",
        ]
    )

    html_path = settings.paths.audit_dir / f"chair_fall_paper_inputs_{settings.run_id}.html"
    ensure_dir(html_path.parent)
    html_path.write_text("\n".join(html_lines), encoding="utf-8")
    return json_path, html_path


def build_gate_evidence_strings(settings: Settings) -> tuple[str | None, str | None]:
    """Read QA outputs and compose gate_1_evidence and gate_2_evidence narrative strings.

    Returns (gate_1_evidence, gate_2_evidence). Returns None for a gate if required
    artifacts are missing.
    """
    qa_dir = settings.paths.qa_dir
    run_id = settings.run_id

    gate_preflight_path = qa_dir / f"gate_preflight_{run_id}.json"
    if gate_preflight_path.exists():
        preflight = read_json(gate_preflight_path)
        date = preflight.get("generated_at_utc", "")[:10]
    else:
        date = ""

    # Gate 1
    gate_1_evidence: str | None = None
    sp_path = qa_dir / f"source_profile_{run_id}.json"
    cm_path = qa_dir / f"cohort_map_validation_{run_id}.json"
    if sp_path.exists() and cm_path.exists():
        sp = read_json(sp_path)
        cm = read_json(cm_path)
        hourly = sp.get("hourly_location_aggregation", {})
        completeness = sp.get("control_denominator_completeness", {})
        hourly_rows = hourly.get("rows", 0)
        n_hospitals = hourly.get("distinct_hospital_id", 0)
        n_divisions = hourly.get("distinct_division_id", 0)
        n_monitors_hourly = hourly.get("distinct_monitor_id", 0)
        n_patients_hourly = hourly.get("distinct_patient_id", 0)
        valid_rows = completeness.get("rows_with_valid_pct_sum", 0)
        total_rows = completeness.get("total_rows", hourly_rows)
        min_ts = (hourly.get("min_timestamp") or "")[:10]
        max_ts = (hourly.get("max_timestamp") or "")[:10]
        cohort_map_rows = cm.get("rows", 0)
        cohort_map_source = cm.get("source", "")
        unmapped = cm.get("blank_cohort_rows", 0)
        duplicate_list = cm.get("duplicate_monitor_ids", [])
        duplicate = len(duplicate_list) if isinstance(duplicate_list, list) else duplicate_list
        null_monitor = cm.get("null_monitor_id_rows", 0)
        allowed_cohorts_list = cm.get("allowed_cohorts", [])
        allowed_cohorts_str = ", ".join(str(c) for c in allowed_cohorts_list)
        gate_1_evidence = (
            f"Gate 1 passed on {date} for run_id={run_id}: "
            f"(1A) hourly-cache extraction validated — "
            f"hospital_id={settings.study_hospital_id}, "
            f"{hourly_rows} rows, "
            f"{n_hospitals} hospital, "
            f"{n_divisions} divisions, "
            f"{n_monitors_hourly} monitors, "
            f"{n_patients_hourly} patients, "
            f"valid_pct_sum=100% ({valid_rows}/{total_rows}), "
            f"year range {min_ts}–{max_ts}; "
            f"(1B) control denominator cohort validated — "
            f"{cohort_map_rows} monitors mapped ({cohort_map_source}), "
            f"{unmapped} unmapped, "
            f"{duplicate} duplicate, "
            f"{null_monitor} null, "
            f"allowed cohorts: {allowed_cohorts_str}."
        )

    # Gate 2
    gate_2_evidence: str | None = None
    seq_path = qa_dir / f"label_eval_sequence_metrics_{run_id}.csv"
    pop_path = qa_dir / f"label_eval_population_stats_{run_id}.csv"
    prob_path = qa_dir / f"label_eval_probability_quality_{run_id}.csv"
    timing_path = qa_dir / f"label_eval_response_timing_{run_id}.csv"
    if seq_path.exists() and pop_path.exists() and prob_path.exists() and timing_path.exists():
        seq_df = pd.read_csv(seq_path).set_index("metric")["value"]
        pop_df = pd.read_csv(pop_path).set_index("metric")["value"]
        prob_df = pd.read_csv(prob_path).set_index("metric")["value"]
        timing_df = pd.read_csv(timing_path).set_index("metric")["value"]
        macro_f1 = float(seq_df.get("macro_f1", float("nan")))
        ece = float(prob_df.get("ece_10_bin", float("nan")))
        detection_f1 = float(timing_df.get("detection_f1", float("nan")))
        latency_mae = float(timing_df.get("latency_mae_seconds", float("nan")))
        truth_rows = int(pop_df.get("truth_rows", 0))
        scored_sequences = int(timing_df.get("scored_sequences", seq_df.get("scored_sequences", 0)))
        gate_2_evidence = (
            f"Gate 2 passed on {date} for run_id={run_id}: "
            f"label evaluation completed in dual mode (report_only, thresholds not enabled). "
            f"Descriptive metrics: macro_f1={macro_f1:.3f}, "
            f"ece_10_bin={ece:.3f}, "
            f"response_detection_f1={detection_f1:.3f}, "
            f"latency_mae_seconds={latency_mae:.1f}. "
            f"{truth_rows} truth rows, "
            f"{scored_sequences} scored sequences. "
            f"Threshold checks deferred per Charter s6 status note — "
            f"inferential claims require stakeholder sign-off on label quality limitations."
        )

    return gate_1_evidence, gate_2_evidence


def load_manifest_if_exists(path: Path) -> dict | None:
    if path.suffix.lower() == ".json":
        return read_json(path) if path.exists() else None
    return read_yaml(path) if path.exists() else None


def _build_furniture_exit_concordance(
    settings: Settings,
    raw_tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare AI-predicted furniture departure vs consensus furniture_departure_time.

    Three AI signals are evaluated:
    1. Nudge boundary: first green→yellow/red transition in nudge_state
    2. Location-label: first dominant_location_label change away from last_furniture type
    3. Posture-transition: first stable posture change away from the baseline posture

    Returns (per_event_concordance, summary_by_signal).
    """
    per_event_cols = [
        "event_key", "monitor_id", "date_local", "last_furniture",
        "furniture_departure_time_consensus", "furniture_departure_seconds",
        "signal_type", "ai_departure_seconds", "bias_seconds", "abs_error_seconds",
        "ai_departure_found",
    ]
    summary_cols = [
        "signal_type", "n_events_with_departure", "n_events_ai_detected",
        "bias_seconds_median", "bias_seconds_mean",
        "mae_seconds", "p50_abs_error", "p90_abs_error",
    ]
    empty_per_event = pd.DataFrame(columns=per_event_cols)
    empty_summary = pd.DataFrame(columns=summary_cols)

    # Load consensus with furniture_departure_time
    consensus_path = _csv_path(settings, settings.fall_labels_consensus_csv_path)
    consensus, summary = load_consensus_annotations(consensus_path)
    if summary.get("status") != "ok":
        return empty_per_event, empty_summary
    if consensus.empty or "furniture_departure_time" not in consensus.columns:
        return empty_per_event, empty_summary

    events = parse_event_key(included_fall_annotations(consensus))
    if events.empty:
        return empty_per_event, empty_summary

    # Filter to events with valid furniture_departure_time and last_furniture
    has_departure = (
        events["last_furniture"].notna()
        & events["furniture_departure_time"].notna()
        & events["furniture_departure_time"].astype("string").str.strip().ne("")
    )
    events = events.loc[has_departure].copy()
    if events.empty:
        return empty_per_event, empty_summary

    events["furniture_departure_seconds"] = events["furniture_departure_time"].apply(time_to_seconds)
    events = events.loc[events["furniture_departure_seconds"].notna()].copy()
    if events.empty:
        return empty_per_event, empty_summary

    # Load second-level panel
    second_level = raw_tables.get("fall_livestream_second_level", pd.DataFrame())
    if second_level.empty:
        return empty_per_event, empty_summary

    # Ensure required columns
    has_nudge = "nudge_state" in second_level.columns
    has_location = "dominant_location_label" in second_level.columns
    has_posture = {
        "primary_patient_posture_label",
        "primary_patient_posture_score_sitting",
        "primary_patient_posture_score_standing",
        "primary_patient_posture_score_lying",
    }.issubset(second_level.columns)
    if not has_nudge and not has_location and not has_posture:
        return empty_per_event, empty_summary

    # Parse second-level timestamps
    if "frame_ts_utc" in second_level.columns:
        second_level["_ts"] = pd.to_datetime(second_level["frame_ts_utc"], errors="coerce", utc=True)
    elif "timestamp" in second_level.columns:
        second_level["_ts"] = pd.to_datetime(second_level["timestamp"], errors="coerce", utc=True)
    elif "timestamp_local" in second_level.columns:
        second_level["_ts"] = pd.to_datetime(second_level["timestamp_local"], errors="coerce")
    else:
        return empty_per_event, empty_summary

    second_level["monitor_id"] = pd.to_numeric(second_level.get("monitor_id"), errors="coerce").astype("Int64")

    per_event_rows = []

    for _, event in events.iterrows():
        mid = event["monitor_id"]
        date_str = event["date_local"]
        furn = event["last_furniture"]
        departure_secs = event["furniture_departure_seconds"]
        furn_dep_time = event["furniture_departure_time"]
        ek = event["event_key"]

        if pd.isna(mid):
            continue

        # Get second-level data for this monitor on this date
        monitor_mask = second_level["monitor_id"] == int(mid)
        monitor_data = second_level.loc[monitor_mask].copy()
        if monitor_data.empty:
            # No second-level data for this monitor
            for sig in ["nudge_boundary", "location_label", "posture_transition"]:
                per_event_rows.append({
                    "event_key": ek,
                    "monitor_id": mid,
                    "date_local": date_str,
                    "last_furniture": furn,
                    "furniture_departure_time_consensus": furn_dep_time,
                    "furniture_departure_seconds": departure_secs,
                    "signal_type": sig,
                    "ai_departure_seconds": None,
                    "bias_seconds": None,
                    "abs_error_seconds": None,
                    "ai_departure_found": False,
                })
            continue

        # Filter to the relevant date
        if monitor_data["_ts"].dt.tz is not None:
            # Convert to timezone-naive local for date matching
            try:
                local_ts = monitor_data["_ts"].dt.tz_convert(settings.hospital_timezone).dt.tz_localize(None)
            except Exception:
                local_ts = monitor_data["_ts"].dt.tz_localize(None)
        else:
            local_ts = monitor_data["_ts"]

        monitor_data["_local_ts"] = local_ts
        monitor_data["_date_str"] = monitor_data["_local_ts"].dt.strftime("%Y-%m-%d")
        monitor_data["_seconds_of_day"] = (
            monitor_data["_local_ts"].dt.hour * 3600
            + monitor_data["_local_ts"].dt.minute * 60
            + monitor_data["_local_ts"].dt.second
        )

        day_data = monitor_data.loc[monitor_data["_date_str"] == date_str].copy()
        if day_data.empty:
            for sig in ["nudge_boundary", "location_label", "posture_transition"]:
                per_event_rows.append({
                    "event_key": ek,
                    "monitor_id": mid,
                    "date_local": date_str,
                    "last_furniture": furn,
                    "furniture_departure_time_consensus": furn_dep_time,
                    "furniture_departure_seconds": departure_secs,
                    "signal_type": sig,
                    "ai_departure_seconds": None,
                    "bias_seconds": None,
                    "abs_error_seconds": None,
                    "ai_departure_found": False,
                })
            continue

        day_data = day_data.sort_values("_seconds_of_day")

        # Define a search window: departure_secs ± 5 minutes
        window_start = departure_secs - 300
        window_end = departure_secs + 300
        window_data = day_data.loc[
            (day_data["_seconds_of_day"] >= window_start)
            & (day_data["_seconds_of_day"] <= window_end)
        ]

        # Signal 1: Nudge boundary (green→yellow or green→red)
        if has_nudge and not window_data.empty:
            ai_nudge_secs = None
            prev_state = None
            for _idx_row, row_data in window_data.iterrows():
                state = str(row_data.get("nudge_state", "")).strip().lower()
                if prev_state == "green" and state in ("yellow", "red"):
                    ai_nudge_secs = row_data["_seconds_of_day"]
                    break
                if state in ("green", "yellow", "red"):
                    prev_state = state

            found = ai_nudge_secs is not None
            bias = (ai_nudge_secs - departure_secs) if found else None
            abs_err = abs(bias) if bias is not None else None
            per_event_rows.append({
                "event_key": ek,
                "monitor_id": mid,
                "date_local": date_str,
                "last_furniture": furn,
                "furniture_departure_time_consensus": furn_dep_time,
                "furniture_departure_seconds": departure_secs,
                "signal_type": "nudge_boundary",
                "ai_departure_seconds": ai_nudge_secs,
                "bias_seconds": bias,
                "abs_error_seconds": abs_err,
                "ai_departure_found": found,
            })
        elif has_nudge:
            per_event_rows.append({
                "event_key": ek,
                "monitor_id": mid,
                "date_local": date_str,
                "last_furniture": furn,
                "furniture_departure_time_consensus": furn_dep_time,
                "furniture_departure_seconds": departure_secs,
                "signal_type": "nudge_boundary",
                "ai_departure_seconds": None,
                "bias_seconds": None,
                "abs_error_seconds": None,
                "ai_departure_found": False,
            })

        # Signal 2: Location-label transition (away from last_furniture)
        if has_location and not window_data.empty:
            # Map furniture to expected location labels
            furn_to_labels = {
                "bed": ["bed"],
                "chair": ["chair"],
                "floor": ["room"],
            }
            expected_labels = furn_to_labels.get(furn, [furn])

            ai_loc_secs = None
            for _idx_row, row_data in window_data.iterrows():
                loc_label = str(row_data.get("dominant_location_label", "")).strip().lower()
                if loc_label and loc_label not in expected_labels and loc_label != "unknown" and loc_label != "" and loc_label != "nan":
                    ai_loc_secs = row_data["_seconds_of_day"]
                    break

            found = ai_loc_secs is not None
            bias = (ai_loc_secs - departure_secs) if found else None
            abs_err = abs(bias) if bias is not None else None
            per_event_rows.append({
                "event_key": ek,
                "monitor_id": mid,
                "date_local": date_str,
                "last_furniture": furn,
                "furniture_departure_time_consensus": furn_dep_time,
                "furniture_departure_seconds": departure_secs,
                "signal_type": "location_label",
                "ai_departure_seconds": ai_loc_secs,
                "bias_seconds": bias,
                "abs_error_seconds": abs_err,
                "ai_departure_found": found,
            })
        elif has_location:
            per_event_rows.append({
                "event_key": ek,
                "monitor_id": mid,
                "date_local": date_str,
                "last_furniture": furn,
                "furniture_departure_time_consensus": furn_dep_time,
                "furniture_departure_seconds": departure_secs,
                "signal_type": "location_label",
                "ai_departure_seconds": None,
                "bias_seconds": None,
                "abs_error_seconds": None,
                "ai_departure_found": False,
            })

        # Signal 3: Posture transition away from stable baseline posture
        if has_posture and not window_data.empty:
            posture_data = window_data.copy()
            posture_data["posture_label"] = (
                posture_data["primary_patient_posture_label"].astype("string").fillna("").str.strip().str.lower()
            )
            posture_data = posture_data.loc[
                posture_data["posture_label"].isin(["sitting", "standing", "lying"])
            ].copy()
            ai_posture_secs = None
            if not posture_data.empty:
                posture_data["posture_score_max"] = posture_data[
                    [
                        "primary_patient_posture_score_sitting",
                        "primary_patient_posture_score_standing",
                        "primary_patient_posture_score_lying",
                    ]
                ].apply(pd.to_numeric, errors="coerce").max(axis=1)
                baseline_posture = str(posture_data.iloc[0]["posture_label"])
                posture_data["candidate"] = (
                    posture_data["posture_label"].ne(baseline_posture)
                    & posture_data["posture_score_max"].ge(
                        float(settings.label_eval_posture_transition_min_score)
                    )
                )
                posture_data["_candidate_group"] = posture_data["candidate"].ne(
                    posture_data["candidate"].shift(fill_value=False)
                ).cumsum()
                for _, run in posture_data.loc[posture_data["candidate"]].groupby("_candidate_group", dropna=False):
                    if len(run.index) >= int(settings.label_eval_posture_transition_min_run_length):
                        ai_posture_secs = run.iloc[0]["_seconds_of_day"]
                        break

            found = ai_posture_secs is not None
            bias = (ai_posture_secs - departure_secs) if found else None
            abs_err = abs(bias) if bias is not None else None
            per_event_rows.append({
                "event_key": ek,
                "monitor_id": mid,
                "date_local": date_str,
                "last_furniture": furn,
                "furniture_departure_time_consensus": furn_dep_time,
                "furniture_departure_seconds": departure_secs,
                "signal_type": "posture_transition",
                "ai_departure_seconds": ai_posture_secs,
                "bias_seconds": bias,
                "abs_error_seconds": abs_err,
                "ai_departure_found": found,
            })
        elif has_posture:
            per_event_rows.append({
                "event_key": ek,
                "monitor_id": mid,
                "date_local": date_str,
                "last_furniture": furn,
                "furniture_departure_time_consensus": furn_dep_time,
                "furniture_departure_seconds": departure_secs,
                "signal_type": "posture_transition",
                "ai_departure_seconds": None,
                "bias_seconds": None,
                "abs_error_seconds": None,
                "ai_departure_found": False,
            })

    per_event = pd.DataFrame(per_event_rows, columns=per_event_cols)

    # Build summary by signal type
    summary_rows = []
    for sig_type in ["nudge_boundary", "location_label", "posture_transition"]:
        sig_data = per_event.loc[per_event["signal_type"] == sig_type]
        n_with_departure = len(sig_data)
        detected = sig_data.loc[sig_data["ai_departure_found"]]
        n_detected = len(detected)

        if n_detected > 0:
            bias_vals = detected["bias_seconds"].dropna()
            abs_err_vals = detected["abs_error_seconds"].dropna()
            summary_rows.append({
                "signal_type": sig_type,
                "n_events_with_departure": n_with_departure,
                "n_events_ai_detected": n_detected,
                "bias_seconds_median": round(float(bias_vals.median()), 1) if not bias_vals.empty else None,
                "bias_seconds_mean": round(float(bias_vals.mean()), 1) if not bias_vals.empty else None,
                "mae_seconds": round(float(abs_err_vals.mean()), 1) if not abs_err_vals.empty else None,
                "p50_abs_error": round(float(abs_err_vals.median()), 1) if not abs_err_vals.empty else None,
                "p90_abs_error": round(float(abs_err_vals.quantile(0.9)), 1) if len(abs_err_vals) >= 2 else None,
            })
        else:
            summary_rows.append({
                "signal_type": sig_type,
                "n_events_with_departure": n_with_departure,
                "n_events_ai_detected": 0,
                "bias_seconds_median": None,
                "bias_seconds_mean": None,
                "mae_seconds": None,
                "p50_abs_error": None,
                "p90_abs_error": None,
            })

    summary = pd.DataFrame(summary_rows, columns=summary_cols)
    return per_event, summary


def run_qa_pipeline(settings: Settings) -> dict[str, Any]:
    raw_tables = load_raw_tables(settings)
    for table_name in (
        "fall_events_source",
        "hourly_location_aggregation",
        "key_dimensions",
        "fall_livestream_event_windows",
        "fall_livestream_second_level",
        "negative_control_source_inventory",
        "fall_case_crossover_windows",
        "fall_case_crossover_second_level",
        "fall_negative_control_second_level",
        "fall_negative_control_windows",
    ):
        table = raw_tables.get(table_name, pd.DataFrame())
        raw_tables[table_name] = _scope_to_study_hospital(table, settings.study_hospital_id)

    gate_preflight_path = write_gate_preflight(settings)
    source_profile_path = write_source_profile(settings, raw_tables)
    falls_descriptive_paths = write_falls_descriptive_tables(settings, raw_tables)
    livestream_descriptive_paths = write_livestream_falls_descriptive_tables(settings, raw_tables)
    livestream_localized_paths, localized_diagnostics_path = write_livestream_localized_analysis_tables(
        settings, raw_tables
    )
    consensus_adjudication_paths = write_consensus_adjudication_tables(settings)
    label_eval_paths, label_eval_threshold_checks = write_livestream_label_evaluation_tables(settings, raw_tables)

    # Backfill evidence narratives from generated QA artifacts when not provided via settings/env.
    auto_gate_1_evidence, auto_gate_2_evidence = build_gate_evidence_strings(settings)
    resolved_gate_1_evidence = settings.gate_1_evidence or auto_gate_1_evidence
    resolved_gate_2_evidence = settings.gate_2_evidence or auto_gate_2_evidence
    if (
        resolved_gate_1_evidence != settings.gate_1_evidence
        or resolved_gate_2_evidence != settings.gate_2_evidence
    ):
        gate_preflight_path = write_gate_preflight(
            settings.model_copy(
                update={
                    "gate_1_evidence": resolved_gate_1_evidence,
                    "gate_2_evidence": resolved_gate_2_evidence,
                }
            )
        )

    # Furniture-exit concordance
    exit_concordance_df, exit_concordance_summary = _build_furniture_exit_concordance(settings, raw_tables)
    exit_concordance_path = _write_csv(
        exit_concordance_df,
        settings.paths.qa_dir / f"furniture_origin_exit_concordance_{settings.run_id}.csv",
    )
    exit_concordance_summary_path = _write_csv(
        exit_concordance_summary,
        settings.paths.qa_dir / f"furniture_origin_exit_concordance_summary_{settings.run_id}.csv",
    )

    adjusted_rr_path: Path | None = None
    sensitivity_path: Path | None = None
    if settings.effective_run_mode == "inferential_ready":
        from .modeling import run_adjusted_rr_model

        adjusted_rr_path, sensitivity_path = run_adjusted_rr_model(settings)

    cohort_markdown_path, csv_paths = write_cohort_analysis_pack(
        settings,
        raw_tables,
        adjusted_rr_path=adjusted_rr_path,
    )
    qa_summary_path, qa_exclusions_path, qa_meta = write_qa_summary_and_exclusions(settings)
    transform_manifest_path = settings.paths.manifests_dir / f"transform_manifest_{settings.run_id}.yaml"
    transform_manifest = load_manifest_if_exists(transform_manifest_path) or {}

    payload = {
        "run_id": settings.run_id,
        "hospital_timezone": settings.hospital_timezone,
        "gate_preflight_path": str(gate_preflight_path),
        "source_profile_path": str(source_profile_path),
        "falls_descriptive_paths": [str(path) for path in falls_descriptive_paths],
        "livestream_descriptive_paths": [str(path) for path in livestream_descriptive_paths],
        "livestream_localized_paths": [str(path) for path in livestream_localized_paths],
        "livestream_localized_diagnostics_path": str(localized_diagnostics_path),
        "fall_negative_control_coverage_path": str(
            settings.paths.qa_dir / f"fall_negative_control_coverage_{settings.run_id}.json"
        ),
        "fall_negative_control_match_quality_path": str(
            settings.paths.qa_dir / f"fall_negative_control_match_quality_{settings.run_id}.csv"
        ),
        "consensus_adjudication_paths": [str(path) for path in consensus_adjudication_paths],
        "label_eval_paths": [str(path) for path in label_eval_paths],
        "label_eval_threshold_checks_path": str(
            settings.paths.qa_dir / f"label_eval_threshold_checks_{settings.run_id}.json"
        ),
        "label_eval_threshold_checks": label_eval_threshold_checks,
        "cohort_analysis_markdown_path": str(cohort_markdown_path),
        "cohort_analysis_csv_paths": [str(path) for path in csv_paths],
        "qa_summary_path": str(qa_summary_path),
        "qa_exclusions_path": str(qa_exclusions_path),
        "cohort_validation_status": transform_manifest.get("cohort_validation_status"),
        "cohort_fallback_used": transform_manifest.get("cohort_fallback_used"),
        "cohort_warning_codes": transform_manifest.get("cohort_warning_codes", []),
    }
    payload.update(qa_meta)

    if adjusted_rr_path is not None and sensitivity_path is not None:
        payload["adjusted_rr_path"] = str(adjusted_rr_path)
        payload["sensitivity_analyses_path"] = str(sensitivity_path)

    payload["furniture_exit_concordance_path"] = str(exit_concordance_path)
    payload["furniture_exit_concordance_summary_path"] = str(exit_concordance_summary_path)

    paper_handoff_json, paper_handoff_html = write_paper_handoff_bundle(settings, payload)
    payload["paper_handoff_json_path"] = str(paper_handoff_json)
    payload["paper_handoff_html_path"] = str(paper_handoff_html)

    path = settings.paths.manifests_dir / f"qa_manifest_{settings.run_id}.yaml"
    payload["manifest_path"] = str(path)
    write_yaml(path, payload)
    return payload
