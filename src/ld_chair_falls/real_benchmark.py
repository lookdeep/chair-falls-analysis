from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings
from .manifest import write_run_manifest
from .paper_defaults import CURRENT_MANUSCRIPT_RUN_ID
from .qa import run_qa_pipeline
from .transform import run_transform_pipeline
from .utils import ensure_dir, read_json, sha256_file, utc_now_iso, write_json

REAL_BENCHMARK_FIXTURE_ENV = "REAL_BENCHMARK_FIXTURE_DIR"
DEFAULT_REAL_BENCHMARK_FIXTURE_VERSION = "frozen_v1"
DEFAULT_REAL_BENCHMARK_SOURCE_RUN_ID = CURRENT_MANUSCRIPT_RUN_ID
PRIMARY_BENCHMARK_SPLIT = "primary_4class_departureaware_10s"
COHORT_MAP_FILENAME = "prep.monitor_cohort_map_v1.parquet"
DEFAULT_CONSENSUS_FILENAME = "falls-observations-v3-consensus.csv"
LEGACY_CONSENSUS_FILENAMES = (
    "falls-observations-v3 - consensus.csv",
    "falls-observations-v2 - consensus.csv",
)
DEFAULT_MUST_PASS_SEQUENCE_IDS = (
    "1456|2023-05-04|23:07#S01",
    "5051|2024-09-12|08:01#S01",
)
DEFAULT_WATCH_ONLY_SEQUENCE_IDS = ("2536|2023-10-29|12:50#S01",)


@dataclass(frozen=True)
class FrozenRealBenchmarkReplayResult:
    fixture_dir: Path
    project_root: Path
    run_id: str
    manifest_path: Path
    benchmark_status_path: Path
    candidate_comparison_path: Path
    label_metrics_path: Path
    sequence_predictions_path: Path
    validation_summary_path: Path


def default_real_benchmark_fixture_dir(
    project_root: Path,
    fixture_version: str = DEFAULT_REAL_BENCHMARK_FIXTURE_VERSION,
) -> Path:
    return project_root / "data" / "fixtures" / "real_benchmark" / fixture_version


def resolve_real_benchmark_fixture_dir(
    project_root: Path,
    fixture_dir: str | Path | None = None,
    fixture_version: str = DEFAULT_REAL_BENCHMARK_FIXTURE_VERSION,
) -> Path:
    if fixture_dir is not None:
        return Path(fixture_dir).expanduser().resolve()
    env_value = os.getenv(REAL_BENCHMARK_FIXTURE_ENV)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return default_real_benchmark_fixture_dir(project_root, fixture_version).resolve()


def load_real_benchmark_fixture_manifest(fixture_dir: Path) -> dict[str, Any]:
    manifest_path = fixture_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Real benchmark fixture manifest not found: {manifest_path}")
    return read_json(manifest_path)


def verify_real_benchmark_fixture(fixture_dir: Path) -> dict[str, Any]:
    fixture_dir = fixture_dir.resolve()
    manifest = load_real_benchmark_fixture_manifest(fixture_dir)
    _verify_hash_records(fixture_dir, manifest.get("raw_snapshot", []))
    _verify_hash_records(fixture_dir, manifest.get("supporting_inputs", []))
    return manifest


def _resolve_source_consensus_path(
    source_project_root: Path,
    source_consensus_path: Path | None = None,
) -> Path:
    if source_consensus_path is not None:
        if source_consensus_path.is_absolute():
            return source_consensus_path
        return (source_project_root / source_consensus_path).resolve()

    search_roots = (
        source_project_root / "data" / "public",
        source_project_root / "docs",
    )
    for root in search_roots:
        for filename in (DEFAULT_CONSENSUS_FILENAME, *LEGACY_CONSENSUS_FILENAMES):
            candidate = (root / filename).resolve()
            if candidate.exists():
                return candidate

    return (source_project_root / "data" / "public" / DEFAULT_CONSENSUS_FILENAME).resolve()


def _resolve_fixture_consensus_path(fixture_dir: Path) -> Path:
    for filename in (DEFAULT_CONSENSUS_FILENAME, *LEGACY_CONSENSUS_FILENAMES):
        candidate = fixture_dir / "truth" / filename
        if candidate.exists():
            return candidate
    return fixture_dir / "truth" / DEFAULT_CONSENSUS_FILENAME


def freeze_real_benchmark_fixture(
    source_project_root: Path,
    fixture_dir: Path,
    *,
    source_run_id: str = DEFAULT_REAL_BENCHMARK_SOURCE_RUN_ID,
    source_consensus_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    source_project_root = source_project_root.resolve()
    fixture_dir = fixture_dir.resolve()

    source_raw_dir = source_project_root / "data" / "raw" / source_run_id
    source_staged_dir = source_project_root / "data" / "staged" / source_run_id
    source_qa_dir = source_project_root / "outputs" / "qa"
    source_consensus_path = _resolve_source_consensus_path(
        source_project_root,
        source_consensus_path,
    )
    source_cohort_map_path = source_staged_dir / COHORT_MAP_FILENAME

    required_paths = [
        source_raw_dir,
        source_staged_dir,
        source_qa_dir / f"label_eval_benchmark_status_{source_run_id}.json",
        source_qa_dir / f"label_eval_candidate_comparison_{source_run_id}.csv",
        source_qa_dir / f"label_eval_benchmark_label_metrics_{source_run_id}.csv",
        source_qa_dir / f"label_eval_sequence_predictions_{source_run_id}.csv",
        source_consensus_path,
        source_cohort_map_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot freeze real benchmark fixture; missing source artifacts:\n" + "\n".join(missing)
        )

    if fixture_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Real benchmark fixture already exists: {fixture_dir}")
        shutil.rmtree(fixture_dir)

    raw_fixture_dir = ensure_dir(fixture_dir / "raw")
    supporting_dir = ensure_dir(fixture_dir / "supporting_inputs")
    truth_dir = ensure_dir(fixture_dir / "truth")

    for source_path in sorted(source_raw_dir.glob("*.parquet")):
        shutil.copy2(source_path, raw_fixture_dir / source_path.name)

    fixture_consensus_path = truth_dir / source_consensus_path.name
    shutil.copy2(source_cohort_map_path, supporting_dir / source_cohort_map_path.name)
    shutil.copy2(source_consensus_path, fixture_consensus_path)

    benchmark_status = read_json(source_qa_dir / f"label_eval_benchmark_status_{source_run_id}.json")
    candidate_comparison = pd.read_csv(
        source_qa_dir / f"label_eval_candidate_comparison_{source_run_id}.csv"
    )
    label_metrics = pd.read_csv(source_qa_dir / f"label_eval_benchmark_label_metrics_{source_run_id}.csv")
    sequence_predictions = pd.read_csv(
        source_qa_dir / f"label_eval_sequence_predictions_{source_run_id}.csv"
    )

    primary_v2 = _single_candidate_row(
        candidate_comparison,
        benchmark_split=PRIMARY_BENCHMARK_SPLIT,
        model_version="v2",
    )
    primary_v3 = _single_candidate_row(
        candidate_comparison,
        benchmark_split=PRIMARY_BENCHMARK_SPLIT,
        model_version="v3",
    )
    primary_v3_label_metrics = _label_metric_map(
        label_metrics,
        benchmark_split=PRIMARY_BENCHMARK_SPLIT,
        model_version="v3",
    )
    primary_v3_sequences = _sequence_subset(
        sequence_predictions,
        benchmark_split=PRIMARY_BENCHMARK_SPLIT,
        model_version="v3",
    )

    must_pass_sequences = _sequence_contract_rows(
        primary_v3_sequences,
        DEFAULT_MUST_PASS_SEQUENCE_IDS,
    )
    watch_only_sequences = _sequence_contract_rows(
        primary_v3_sequences,
        DEFAULT_WATCH_ONLY_SEQUENCE_IDS,
    )
    unmatched_unscored_ids = sorted(
        primary_v3_sequences.loc[~primary_v3_sequences["has_derived_match"], "sequence_id"].astype(str).tolist()
    )

    manifest = {
        "fixture_version": fixture_dir.name,
        "frozen_at_utc": utc_now_iso(),
        "source_project_root": str(source_project_root),
        "source_run_id": source_run_id,
        "raw_snapshot": _hash_records(raw_fixture_dir, fixture_dir),
        "supporting_inputs": _hash_records(
            [supporting_dir / COHORT_MAP_FILENAME, fixture_consensus_path],
            fixture_dir,
        ),
        "contract": {
            "primary_split": PRIMARY_BENCHMARK_SPLIT,
            "selected_model_version": benchmark_status["selected_model_version"],
            "overall_pass": bool(benchmark_status["overall_pass"]),
            "scored_sequences": int(round(float(primary_v3["scored_sequences"]))),
            "label_supports": {
                label: int(round(float(metric["support"])))
                for label, metric in primary_v3_label_metrics.items()
            },
            "metric_floors": {
                "macro_f1": float(primary_v3["macro_f1"]),
                "room_f1": float(primary_v3["f1_room"]),
                "no_patient_f1": float(primary_v3["f1_no_patient"]),
            },
            "selection_reference": {
                "v2_macro_f1": float(primary_v2["macro_f1"]),
                "v2_room_f1": float(primary_v2["f1_room"]),
                "v2_no_patient_f1": float(primary_v2["f1_no_patient"]),
                "v2_bed_f1": float(primary_v2["f1_bed"]),
            },
            "must_pass_sequences": must_pass_sequences,
            "watch_only_sequences": watch_only_sequences,
            "unmatched_unscored_sequence_ids": unmatched_unscored_ids,
        },
    }
    return write_json(fixture_dir / "manifest.json", manifest)


def replay_real_benchmark_fixture(
    project_root: Path,
    fixture_dir: Path,
    *,
    run_id: str,
    overwrite: bool = False,
) -> FrozenRealBenchmarkReplayResult:
    project_root = project_root.resolve()
    fixture_dir = fixture_dir.resolve()
    verify_real_benchmark_fixture(fixture_dir)

    raw_target_dir = project_root / "data" / "raw" / run_id
    staged_target_dir = project_root / "data" / "staged" / run_id
    qa_dir = project_root / "outputs" / "qa"
    manifests_dir = project_root / "outputs" / "manifests"
    audit_dir = project_root / "outputs" / "audit"

    if raw_target_dir.exists() and any(raw_target_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Replay raw target already exists: {raw_target_dir}")
        shutil.rmtree(raw_target_dir)
    if staged_target_dir.exists() and any(staged_target_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Replay staged target already exists: {staged_target_dir}")
        shutil.rmtree(staged_target_dir)

    ensure_dir(raw_target_dir)
    ensure_dir(staged_target_dir)
    ensure_dir(qa_dir)
    ensure_dir(manifests_dir)
    ensure_dir(audit_dir)

    for source_path in sorted((fixture_dir / "raw").glob("*.parquet")):
        _link_or_copy(source_path, raw_target_dir / source_path.name)

    cohort_map_path = fixture_dir / "supporting_inputs" / COHORT_MAP_FILENAME
    consensus_path = _resolve_fixture_consensus_path(fixture_dir)

    settings = Settings(
        project_root=project_root,
        run_id=run_id,
        cohort_map_path=cohort_map_path,
        fall_labels_consensus_csv_path=consensus_path,
        gate_1_pass=True,
        gate_2_pass=True,
        dry_run=False,
    )
    run_transform_pipeline(settings)
    run_qa_pipeline(settings)
    manifest_path = write_run_manifest(settings)

    return FrozenRealBenchmarkReplayResult(
        fixture_dir=fixture_dir,
        project_root=project_root,
        run_id=run_id,
        manifest_path=manifest_path,
        benchmark_status_path=qa_dir / f"label_eval_benchmark_status_{run_id}.json",
        candidate_comparison_path=qa_dir / f"label_eval_candidate_comparison_{run_id}.csv",
        label_metrics_path=qa_dir / f"label_eval_benchmark_label_metrics_{run_id}.csv",
        sequence_predictions_path=qa_dir / f"label_eval_sequence_predictions_{run_id}.csv",
        validation_summary_path=qa_dir / f"real_benchmark_validation_{run_id}.json",
    )


def validate_real_benchmark_replay(
    replay: FrozenRealBenchmarkReplayResult,
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    manifest = verify_real_benchmark_fixture(replay.fixture_dir)
    contract = manifest["contract"]

    benchmark_status = read_json(replay.benchmark_status_path)
    candidate_comparison = pd.read_csv(replay.candidate_comparison_path)
    label_metrics = pd.read_csv(replay.label_metrics_path)
    sequence_predictions = pd.read_csv(replay.sequence_predictions_path)

    primary_split = contract["primary_split"]
    primary_v2 = _single_candidate_row(
        candidate_comparison,
        benchmark_split=primary_split,
        model_version="v2",
    )
    primary_v3 = _single_candidate_row(
        candidate_comparison,
        benchmark_split=primary_split,
        model_version="v3",
    )
    primary_v3_label_metrics = _label_metric_map(
        label_metrics,
        benchmark_split=primary_split,
        model_version="v3",
    )
    primary_v3_sequences = _sequence_subset(
        sequence_predictions,
        benchmark_split=primary_split,
        model_version="v3",
    )

    checks: list[dict[str, Any]] = []
    checks.append(
        _eq_check(
            "selected_model_version",
            benchmark_status["selected_model_version"],
            contract["selected_model_version"],
        )
    )
    checks.append(
        _eq_check(
            "overall_pass",
            bool(benchmark_status["overall_pass"]),
            bool(contract["overall_pass"]),
        )
    )
    checks.append(
        _eq_check(
            "primary_scored_sequences",
            int(round(float(primary_v3["scored_sequences"]))),
            int(contract["scored_sequences"]),
        )
    )

    for label, expected_support in contract["label_supports"].items():
        metric = primary_v3_label_metrics.get(label)
        actual_support = int(round(float(metric["support"]))) if metric is not None else None
        checks.append(_eq_check(f"support_{label}", actual_support, int(expected_support)))

    metric_floors = contract["metric_floors"]
    checks.append(_gte_check("macro_f1_floor", float(primary_v3["macro_f1"]), metric_floors["macro_f1"], tolerance))
    checks.append(_gte_check("room_f1_floor", float(primary_v3["f1_room"]), metric_floors["room_f1"], tolerance))
    checks.append(
        _gte_check(
            "no_patient_f1_floor",
            float(primary_v3["f1_no_patient"]),
            metric_floors["no_patient_f1"],
            tolerance,
        )
    )
    checks.append(
        _gt_check(
            "v3_macro_f1_beats_v2",
            float(primary_v3["macro_f1"]),
            float(primary_v2["macro_f1"]),
            tolerance,
        )
    )
    checks.append(
        _gte_check(
            "v3_room_f1_non_decreasing",
            float(primary_v3["f1_room"]),
            float(primary_v2["f1_room"]),
            tolerance,
        )
    )
    checks.append(
        _gte_check(
            "v3_no_patient_f1_non_decreasing",
            float(primary_v3["f1_no_patient"]),
            float(primary_v2["f1_no_patient"]),
            tolerance,
        )
    )
    checks.append(
        _gte_check(
            "v3_bed_f1_non_decreasing",
            float(primary_v3["f1_bed"]),
            float(primary_v2["f1_bed"]),
            tolerance,
        )
    )

    must_pass_results: list[dict[str, Any]] = []
    for sentinel in contract["must_pass_sequences"]:
        row = _single_sequence_row(primary_v3_sequences, sentinel["sequence_id"])
        check = _eq_check(
            f"must_pass_{sentinel['sequence_id']}",
            row["derived_label"],
            sentinel["expected_label"],
        )
        checks.append(check)
        must_pass_results.append(
            {
                "sequence_id": sentinel["sequence_id"],
                "truth_label": row["truth_label"],
                "actual_label": row["derived_label"],
                "expected_label": sentinel["expected_label"],
                "prediction_reason": row["prediction_reason"],
                "pass": check["pass"],
            }
        )

    unmatched_results: list[dict[str, Any]] = []
    for sequence_id in contract["unmatched_unscored_sequence_ids"]:
        row = _single_sequence_row(primary_v3_sequences, sequence_id)
        has_derived_match = bool(row["has_derived_match"])
        scored_for_location = bool(row["scored_for_location"])
        checks.append(_eq_check(f"unmatched_{sequence_id}_has_derived_match", has_derived_match, False))
        checks.append(_eq_check(f"unmatched_{sequence_id}_scored_for_location", scored_for_location, False))
        unmatched_results.append(
            {
                "sequence_id": sequence_id,
                "derived_label": row["derived_label"],
                "has_derived_match": has_derived_match,
                "scored_for_location": scored_for_location,
            }
        )

    watch_only_results: list[dict[str, Any]] = []
    for sentinel in contract["watch_only_sequences"]:
        row = _single_sequence_row(primary_v3_sequences, sentinel["sequence_id"])
        watch_only_results.append(
            {
                "sequence_id": sentinel["sequence_id"],
                "truth_label": row["truth_label"],
                "actual_label": row["derived_label"],
                "prediction_reason": row["prediction_reason"],
                "expected_current_label": sentinel["expected_label"],
            }
        )

    summary = {
        "fixture_dir": str(replay.fixture_dir),
        "run_id": replay.run_id,
        "primary_split": primary_split,
        "passed": all(check["pass"] for check in checks),
        "checks": checks,
        "primary_metrics": {
            "v2": {
                "macro_f1": float(primary_v2["macro_f1"]),
                "f1_bed": float(primary_v2["f1_bed"]),
                "f1_room": float(primary_v2["f1_room"]),
                "f1_no_patient": float(primary_v2["f1_no_patient"]),
                "accuracy": float(primary_v2["accuracy"]),
            },
            "v3": {
                "macro_f1": float(primary_v3["macro_f1"]),
                "f1_bed": float(primary_v3["f1_bed"]),
                "f1_room": float(primary_v3["f1_room"]),
                "f1_no_patient": float(primary_v3["f1_no_patient"]),
                "accuracy": float(primary_v3["accuracy"]),
            },
        },
        "must_pass_results": must_pass_results,
        "unmatched_results": unmatched_results,
        "watch_only_results": watch_only_results,
        "source_run_id": manifest["source_run_id"],
    }
    write_json(replay.validation_summary_path, summary)
    return summary


def _hash_records(paths: Path | list[Path], root: Path) -> list[dict[str, Any]]:
    if isinstance(paths, Path):
        paths = sorted(paths.glob("*"))

    records: list[dict[str, Any]] = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        records.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "rows": _row_count(path),
            }
        )
    return records


def _verify_hash_records(root: Path, records: list[dict[str, Any]]) -> None:
    for record in records:
        path = root / record["path"]
        if not path.exists():
            raise FileNotFoundError(f"Frozen real benchmark file missing: {path}")
        actual_hash = sha256_file(path)
        expected_hash = record["sha256"]
        if actual_hash != expected_hash:
            raise ValueError(
                f"Frozen real benchmark file hash mismatch for {path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )


def _row_count(path: Path) -> int:
    if path.suffix == ".parquet":
        return int(len(pd.read_parquet(path)))
    if path.suffix == ".csv":
        return int(len(pd.read_csv(path)))
    raise ValueError(f"Unsupported frozen real benchmark file type: {path}")


def _link_or_copy(source_path: Path, target_path: Path) -> None:
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    try:
        target_path.symlink_to(source_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def _single_candidate_row(
    candidate_comparison: pd.DataFrame,
    *,
    benchmark_split: str,
    model_version: str,
) -> pd.Series:
    subset = candidate_comparison.loc[
        (candidate_comparison["benchmark_split"] == benchmark_split)
        & (candidate_comparison["model_version"] == model_version)
    ]
    if len(subset.index) != 1:
        raise ValueError(
            f"Expected exactly one candidate row for split={benchmark_split} "
            f"model_version={model_version}, found {len(subset.index)}"
        )
    return subset.iloc[0]


def _label_metric_map(
    label_metrics: pd.DataFrame,
    *,
    benchmark_split: str,
    model_version: str,
) -> dict[str, pd.Series]:
    subset = label_metrics.loc[
        (label_metrics["benchmark_split"] == benchmark_split)
        & (label_metrics["model_version"] == model_version)
    ]
    return {str(row["label"]): row for _, row in subset.iterrows()}


def _sequence_subset(
    sequence_predictions: pd.DataFrame,
    *,
    benchmark_split: str,
    model_version: str,
) -> pd.DataFrame:
    return sequence_predictions.loc[
        (sequence_predictions["benchmark_split"] == benchmark_split)
        & (sequence_predictions["model_version"] == model_version)
    ].copy()


def _sequence_contract_rows(
    sequence_predictions: pd.DataFrame,
    sequence_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for sequence_id in sequence_ids:
        row = _single_sequence_row(sequence_predictions, sequence_id)
        contracts.append(
            {
                "sequence_id": sequence_id,
                "truth_label": row["truth_label"],
                "expected_label": row["derived_label"],
                "prediction_reason": row["prediction_reason"],
            }
        )
    return contracts


def _single_sequence_row(sequence_predictions: pd.DataFrame, sequence_id: str) -> pd.Series:
    subset = sequence_predictions.loc[sequence_predictions["sequence_id"] == sequence_id]
    if len(subset.index) != 1:
        raise ValueError(f"Expected exactly one sequence row for {sequence_id}, found {len(subset.index)}")
    return subset.iloc[0]


def _eq_check(name: str, actual: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "operator": "==",
        "actual": actual,
        "expected": expected,
        "pass": actual == expected,
    }


def _gte_check(name: str, actual: float, expected: float, tolerance: float) -> dict[str, Any]:
    return {
        "name": name,
        "operator": ">=",
        "actual": actual,
        "expected": expected,
        "pass": actual + tolerance >= expected,
    }


def _gt_check(name: str, actual: float, expected: float, tolerance: float) -> dict[str, Any]:
    return {
        "name": name,
        "operator": ">",
        "actual": actual,
        "expected": expected,
        "pass": actual > expected + tolerance,
    }
