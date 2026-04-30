from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Settings
from .utils import sha256_file, write_json

REQUIRED_COLUMNS = ("monitor_id", "cohort_type")
DERIVED_OUTCOME_COHORT_BASIS = "outcome_defined_monitor_membership"
DERIVED_INTERVENTION_DEFINITION = (
    "monitor_id present in fall_events_source within the extracted run scope"
)
DERIVED_CONTROL_DEFINITION = (
    "monitor_id present in hourly_location_aggregation without a linked fall_events_source row "
    "within the extracted run scope"
)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: column.strip().lower() for column in df.columns}
    result = df.rename(columns=renamed).copy()
    for column in REQUIRED_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result["cohort_type"] = result["cohort_type"].astype("string").str.strip().str.lower()
    return result


def _load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _monitor_id_set(df: pd.DataFrame) -> set[int]:
    if "monitor_id" not in df.columns:
        return set()
    series = pd.to_numeric(df["monitor_id"], errors="coerce").dropna()
    if series.empty:
        return set()
    return {int(value) for value in series.tolist()}


def _validate_and_clean(
    settings: Settings,
    cohort_map: pd.DataFrame,
    *,
    source: str,
    derived: bool,
    message: str | None = None,
    derivation_meta: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    cohort_map = _normalize(cohort_map)
    cohort_map = cohort_map[list(REQUIRED_COLUMNS)].copy()

    null_monitor = int(pd.to_numeric(cohort_map["monitor_id"], errors="coerce").isna().sum())
    blank_cohort = int(cohort_map["cohort_type"].fillna("").str.strip().eq("").sum())
    cohort_map["monitor_id"] = pd.to_numeric(cohort_map["monitor_id"], errors="coerce")

    duplicates = (
        cohort_map.dropna(subset=["monitor_id"])
        .groupby("monitor_id", dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    duplicate_monitor_ids = duplicates.loc[duplicates["n"] > 1, "monitor_id"].tolist()

    allowed = set(settings.allowed_cohorts)
    invalid_values = sorted(
        set(cohort_map["cohort_type"].dropna().tolist())
        - allowed
    )

    is_valid = (
        null_monitor == 0
        and blank_cohort == 0
        and not duplicate_monitor_ids
        and not invalid_values
    )

    cleaned = cohort_map.dropna(subset=["monitor_id", "cohort_type"]).copy()
    cleaned = cleaned[~cleaned["cohort_type"].eq("")].copy()
    cleaned = cleaned.drop_duplicates(subset=["monitor_id"], keep="first")
    cleaned["monitor_id"] = cleaned["monitor_id"].astype("int64")

    report = {
        "status": "valid" if is_valid else "invalid",
        "source": source,
        "derived": derived,
        "rows": int(len(cohort_map.index)),
        "clean_rows": int(len(cleaned.index)),
        "null_monitor_id_rows": null_monitor,
        "blank_cohort_rows": blank_cohort,
        "duplicate_monitor_ids": duplicate_monitor_ids,
        "invalid_cohort_values": invalid_values,
        "allowed_cohorts": list(settings.allowed_cohorts),
    }
    if message:
        report["message"] = message
    if derivation_meta:
        report["derivation"] = derivation_meta

    return cleaned, report


def _derive_cohort_map_from_raw_tables(raw_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    falls = raw_tables.get("fall_events_source", pd.DataFrame())
    hourly = raw_tables.get("hourly_location_aggregation", pd.DataFrame())

    falls_monitors = _monitor_id_set(falls)
    hourly_monitors = _monitor_id_set(hourly)

    all_monitors = sorted(falls_monitors.union(hourly_monitors))
    if not all_monitors:
        return (
            pd.DataFrame(columns=list(REQUIRED_COLUMNS)),
            {
                "message": "No monitor_id values available in raw fall/hourly extracts for derivation.",
            },
        )

    rows = []
    for monitor_id in all_monitors:
        # Falls monitor list defines the "event" cohort and takes precedence on overlap.
        cohort_type = "intervention" if monitor_id in falls_monitors else "control"
        rows.append({"monitor_id": monitor_id, "cohort_type": cohort_type})

    derivation = {
        "cohort_basis": DERIVED_OUTCOME_COHORT_BASIS,
        "intervention_definition": DERIVED_INTERVENTION_DEFINITION,
        "control_definition": DERIVED_CONTROL_DEFINITION,
        "rule": (
            "monitor_id in fall_events_source -> intervention; "
            "monitor_id only in hourly_location_aggregation -> control; "
            "falls assignment takes precedence on overlap."
        ),
        "fall_monitor_count": len(falls_monitors),
        "hourly_monitor_count": len(hourly_monitors),
        "overlap_monitor_count": len(falls_monitors.intersection(hourly_monitors)),
    }
    return pd.DataFrame(rows), derivation


def _apply_manual_intervention_overrides(
    cohort_map: pd.DataFrame,
    forced_monitor_ids: tuple[int, ...],
) -> tuple[pd.DataFrame, list[int]]:
    if not forced_monitor_ids:
        return cohort_map, []

    result = cohort_map.copy()
    existing = set(result["monitor_id"].tolist()) if not result.empty else set()
    applied = []
    for monitor_id in sorted(set(forced_monitor_ids)):
        if monitor_id in existing:
            result.loc[result["monitor_id"] == monitor_id, "cohort_type"] = "intervention"
        else:
            result = pd.concat(
                [result, pd.DataFrame([{"monitor_id": monitor_id, "cohort_type": "intervention"}])],
                ignore_index=True,
            )
        applied.append(monitor_id)
    return result, applied


def load_and_validate_cohort_map(
    settings: Settings,
    raw_tables: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict]:
    if settings.cohort_map_path is None:
        if raw_tables is None:
            empty = pd.DataFrame(columns=list(REQUIRED_COLUMNS))
            report = {
                "status": "missing",
                "message": "No cohort map path provided; raw tables unavailable for derivation.",
                "rows": 0,
            }
            return empty, report

        derived_map, derivation_meta = _derive_cohort_map_from_raw_tables(raw_tables)
        if derived_map.empty:
            empty = pd.DataFrame(columns=list(REQUIRED_COLUMNS))
            report = {
                "status": "missing",
                "message": derivation_meta.get(
                    "message", "No cohort map path provided; derived cohort map is empty."
                ),
                "rows": 0,
            }
            return empty, report

        derived_map, forced_overrides = _apply_manual_intervention_overrides(
            derived_map,
            settings.manual_intervention_monitor_ids,
        )
        if forced_overrides:
            derivation_meta["manual_intervention_monitor_ids"] = forced_overrides

        return _validate_and_clean(
            settings,
            derived_map,
            source="derived_from_raw_tables",
            derived=True,
            message=(
                "Derived cohort map from extracted fall and hourly monitor_id sets."
                if not forced_overrides
                else (
                    "Derived cohort map from extracted fall/hourly monitor_id sets with manual "
                    "intervention monitor overrides."
                )
            ),
            derivation_meta=derivation_meta,
        )

    source_path = settings.cohort_map_path
    if not source_path.is_absolute():
        source_path = settings.project_root / source_path
    if not source_path.exists():
        empty = pd.DataFrame(columns=list(REQUIRED_COLUMNS))
        report = {
            "status": "missing",
            "message": f"Cohort map file not found: {source_path}",
            "rows": 0,
        }
        return empty, report

    raw = _load(source_path)
    return _validate_and_clean(
        settings,
        raw,
        source=str(source_path),
        derived=False,
    )


def persist_cohort_map(
    settings: Settings,
    cohort_map: pd.DataFrame,
    report: dict,
) -> tuple[Path, Path, dict]:
    staged_path = settings.paths.staged_run_dir / "prep.monitor_cohort_map_v1.parquet"
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_map.to_parquet(staged_path, index=False)

    report_path = settings.paths.qa_dir / f"cohort_map_validation_{settings.run_id}.json"
    write_json(report_path, report)

    metadata = {
        "cohort_map_path": str(staged_path),
        "cohort_map_sha256": sha256_file(staged_path),
        "validation_report_path": str(report_path),
        "cohort_definition": {
            "source": report.get("source"),
            "derived": bool(report.get("derived", False)),
            "basis": report.get("derivation", {}).get("cohort_basis"),
            "intervention_definition": report.get("derivation", {}).get(
                "intervention_definition"
            ),
            "control_definition": report.get("derivation", {}).get("control_definition"),
            "rule": report.get("derivation", {}).get("rule"),
        },
    }
    return staged_path, report_path, metadata
