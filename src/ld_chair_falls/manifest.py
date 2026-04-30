from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Settings
from .qa import load_manifest_if_exists
from .utils import sha256_file, utc_now_iso, write_yaml


def _path_if_exists(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _file_hash_records(paths: list[Path]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        records.append({"path": str(path), "sha256": sha256_file(path)})
    return records


def write_run_manifest(settings: Settings) -> Path:
    extract_manifest_path = settings.paths.manifests_dir / f"extract_manifest_{settings.run_id}.yaml"
    transform_manifest_path = settings.paths.manifests_dir / f"transform_manifest_{settings.run_id}.yaml"
    qa_manifest_path = settings.paths.manifests_dir / f"qa_manifest_{settings.run_id}.yaml"
    gate_preflight_path = settings.paths.qa_dir / f"gate_preflight_{settings.run_id}.json"

    extract_manifest = load_manifest_if_exists(extract_manifest_path) or {}
    transform_manifest = load_manifest_if_exists(transform_manifest_path) or {}
    qa_manifest = load_manifest_if_exists(qa_manifest_path) or {}
    gate_preflight = load_manifest_if_exists(gate_preflight_path) or {}

    raw_files = list(settings.paths.raw_run_dir.glob("*.parquet"))
    staged_files = list(settings.paths.staged_run_dir.glob("*.parquet"))

    query_identifiers = []
    query_log_path = extract_manifest.get("query_log_path")
    if query_log_path:
        qpath = Path(query_log_path)
        if not qpath.is_absolute():
            qpath = settings.project_root / qpath
        log = load_manifest_if_exists(qpath) or []
        query_identifiers = [
            {
                "name": entry.get("name"),
                "sql_hash": entry.get("sql_hash"),
            }
            for entry in log
        ]

    manifest: dict[str, Any] = {
        "run_id": settings.run_id,
        "generated_at_utc": utc_now_iso(),
        "run_mode": settings.effective_run_mode,
        "requested_run_mode": settings.requested_run_mode,
        "final_publication_artifact_target": "paper/manuscript.md",
        "gate_preflight": gate_preflight,
        "input_tables": {
            "study_hospital_id": settings.study_hospital_id,
            "project": settings.google_cloud_project,
            "dataset": settings.bq_dataset,
            "fall_events_table": settings.fall_events_table,
            "live_stream_derivatives_table": settings.live_stream_derivatives_table,
            "negative_control_derivatives_table": settings.negative_control_derivatives_table,
            "hourly_location_sql": str(settings.hourly_location_sql_path),
            "key_dimensions_sql": str(settings.key_dimensions_sql_path),
            "fall_livestream_event_windows_sql": str(settings.fall_livestream_event_windows_sql_path),
            "negative_control_source_inventory_sql": str(settings.negative_control_source_inventory_sql_path),
            "fall_case_crossover_second_level_sql": str(settings.fall_case_crossover_second_level_sql_path),
            "fall_negative_control_windows_sql": str(settings.fall_negative_control_windows_sql_path),
            "fall_negative_control_second_level_sql": str(settings.fall_negative_control_second_level_sql_path),
            "fall_labels_consensus_csv": str(settings.fall_labels_consensus_csv_path),
            "fall_labels_raw_logs_csv": str(settings.fall_labels_raw_logs_csv_path),
            "fall_labels_rubric_csv": str(settings.fall_labels_rubric_csv_path),
            "case_crossover_source_status": settings.case_crossover_source_status,
            "negative_control_source_status": settings.negative_control_source_status,
        },
        "extract_manifest_path": _path_if_exists(extract_manifest_path),
        "transform_manifest_path": _path_if_exists(transform_manifest_path),
        "qa_manifest_path": _path_if_exists(qa_manifest_path),
        "raw_snapshot": {
            "path": str(settings.paths.raw_run_dir),
            "file_hashes": _file_hash_records(raw_files),
        },
        "staged_outputs": {
            "path": str(settings.paths.staged_run_dir),
            "file_hashes": _file_hash_records(staged_files),
        },
        "cohort_map": {
            "version": "prep.monitor_cohort_map_v1",
            "path": transform_manifest.get("paths", {}).get("cohort_map"),
            "sha256": transform_manifest.get("cohort_map_sha256"),
        },
        "cohort_definition": transform_manifest.get("cohort_definition", {}),
        "control_denominator_query_identifiers": query_identifiers,
        "row_count_reconciliation": transform_manifest.get("row_counts", {}),
        "eligibility_policy": transform_manifest.get("eligibility_policy", {}),
        "hospital_timezone": transform_manifest.get("hospital_timezone", settings.hospital_timezone),
        "cohort_validation_status": transform_manifest.get("cohort_validation_status"),
        "cohort_fallback_used": transform_manifest.get("cohort_fallback_used"),
        "cohort_warning_codes": transform_manifest.get("cohort_warning_codes", []),
        "evidence_refs": {
            "gate_preflight_path": _path_if_exists(gate_preflight_path),
            "source_profile_path": qa_manifest.get("source_profile_path"),
            "cohort_analysis_markdown_path": qa_manifest.get("cohort_analysis_markdown_path"),
            "label_eval_paths": qa_manifest.get("label_eval_paths"),
            "label_eval_threshold_checks_path": qa_manifest.get("label_eval_threshold_checks_path"),
            "fall_negative_control_coverage_path": qa_manifest.get("fall_negative_control_coverage_path"),
            "fall_negative_control_match_quality_path": qa_manifest.get("fall_negative_control_match_quality_path"),
            "paper_handoff_json_path": qa_manifest.get("paper_handoff_json_path"),
            "paper_handoff_html_path": qa_manifest.get("paper_handoff_html_path"),
        },
        "script_versions": {
            "extract_manifest": extract_manifest.get("manifest_path"),
            "transform_manifest": transform_manifest.get("manifest_path"),
            "qa_manifest": qa_manifest.get("manifest_path"),
        },
    }

    output = settings.paths.manifests_dir / f"run_manifest_{settings.run_id}.yaml"
    return write_yaml(output, manifest)
