# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.gemini_characterization import (
    PRIMARY_MODELS,
    build_detection_confusion_table,
    build_event_long_table,
    build_false_negative_tag_table,
    build_location_confusion_table,
    build_model_artifacts,
    build_model_metrics,
    build_overlap_wide_table,
    compute_characterization_metrics,
    discover_latest_model_runs,
)
from ld_chair_falls.gemini_eval import (
    CONSENSUS_STATUS_ACCEPTED_NONFALL,
    CONSENSUS_STATUS_EXCLUDED_OFFSCREEN,
    CONSENSUS_STATUS_INCLUDED_FALL,
)


def _write_results_jsonl(path: Path, *, model: str, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps({"model": model, **row}) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_discover_latest_model_runs_picks_latest_per_target_model(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    for index, model in enumerate(PRIMARY_MODELS, start=1):
        older_results = output_dir / f"results_2026031{index}T000000Z.jsonl"
        newer_results = output_dir / f"results_2026031{index}T010000Z.jsonl"
        older_summary = output_dir / f"summary_2026031{index}T000000Z.csv"
        newer_summary = output_dir / f"summary_2026031{index}T010000Z.csv"
        _write_results_jsonl(
            older_results,
            model=model,
            rows=[{"file": f"{100+index}_2025-01-01_10-0{index}.mp4", "result": {"ok": True}}],
        )
        _write_results_jsonl(
            newer_results,
            model=model,
            rows=[
                {"file": f"{200+index}_2025-01-01_10-0{index}.mp4", "result": {"ok": True}},
                {"file": f"{300+index}_2025-01-01_10-1{index}.mp4", "error": "timeout"},
            ],
        )
        _write_summary_csv(
            older_summary,
            [{"file": f"{100+index}_2025-01-01_10-0{index}.mp4", "fall_detected": True, "fall_tags": "slip"}],
        )
        _write_summary_csv(
            newer_summary,
            [{"file": f"{200+index}_2025-01-01_10-0{index}.mp4", "fall_detected": True, "fall_tags": "slip"}],
        )

    runs = discover_latest_model_runs(output_dir)

    assert [run.model for run in runs] == list(PRIMARY_MODELS)
    assert all(run.timestamp.endswith("010000Z") for run in runs)
    assert all(run.success_rows == 1 for run in runs)
    assert all(run.error_rows == 1 for run in runs)


def test_discover_latest_model_runs_accepts_model_suffixed_batch_filenames(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    for model in PRIMARY_MODELS:
        suffix = model.replace(".", "-")
        timestamp = f"20260313T010000Z_{suffix}"
        results_path = output_dir / f"results_{timestamp}.jsonl"
        summary_path = output_dir / f"summary_{timestamp}.csv"
        _write_results_jsonl(
            results_path,
            model=model,
            rows=[{"file": "200_2025-01-01_10-00.mp4", "result": {"ok": True}}],
        )
        _write_summary_csv(
            summary_path,
            [{"file": "200_2025-01-01_10-00.mp4", "fall_detected": True, "fall_tags": "slip"}],
        )

    runs = discover_latest_model_runs(output_dir)

    assert [run.model for run in runs] == list(PRIMARY_MODELS)
    assert all("_gemini-" in run.timestamp for run in runs)


def test_compute_characterization_metrics_separates_detection_and_attribute_denominators() -> None:
    frame = pd.DataFrame(
        [
            {
                "consensus_status": CONSENSUS_STATUS_INCLUDED_FALL,
                "gemini_fall_detected": True,
                "location_match": True,
                "furniture_match": True,
                "gt_last_furniture": "bed",
                "tag_exact_match": True,
                "tag_jaccard": 1.0,
                "fall_time_abs_error_seconds": 5.0,
                "response_time_abs_error_seconds": 7.0,
            },
            {
                "consensus_status": CONSENSUS_STATUS_INCLUDED_FALL,
                "gemini_fall_detected": False,
                "location_match": False,
                "furniture_match": False,
                "gt_last_furniture": "chair",
                "tag_exact_match": False,
                "tag_jaccard": 0.0,
                "fall_time_abs_error_seconds": pd.NA,
                "response_time_abs_error_seconds": pd.NA,
            },
            {
                "consensus_status": CONSENSUS_STATUS_ACCEPTED_NONFALL,
                "gemini_fall_detected": False,
                "location_match": pd.NA,
                "furniture_match": pd.NA,
                "gt_last_furniture": pd.NA,
                "tag_exact_match": pd.NA,
                "tag_jaccard": pd.NA,
                "fall_time_abs_error_seconds": pd.NA,
                "response_time_abs_error_seconds": pd.NA,
            },
            {
                "consensus_status": CONSENSUS_STATUS_EXCLUDED_OFFSCREEN,
                "gemini_fall_detected": True,
                "location_match": True,
                "furniture_match": pd.NA,
                "gt_last_furniture": pd.NA,
                "tag_exact_match": False,
                "tag_jaccard": 0.1,
                "fall_time_abs_error_seconds": 100.0,
                "response_time_abs_error_seconds": 120.0,
            },
        ]
    )

    metrics = compute_characterization_metrics(frame, model="gemini-2.5-flash", scope="three_way_overlap")
    metric_map = {
        row["metric"]: (row["value"], row["denominator"])
        for row in metrics.to_dict(orient="records")
    }

    assert metric_map["visible_fall_sensitivity"] == (0.5, 2)
    assert metric_map["visible_nonfall_specificity"] == (1.0, 1)
    assert metric_map["location_accuracy_all_visible_falls"] == (0.5, 2)
    assert metric_map["location_accuracy_detected_visible_falls"] == (1.0, 1)
    assert metric_map["last_furniture_accuracy_all_visible_falls"] == (0.5, 2)
    assert metric_map["last_furniture_accuracy_detected_visible_falls"] == (1.0, 1)


def test_build_overlap_wide_table_uses_only_common_success_keys(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    consensus_csv = tmp_path / "consensus.csv"
    pd.DataFrame(
        [
            {
                "event_key": "100|2025-01-01|10:00",
                "last_furniture": "bed",
                "furniture_departure_time": "10:00:05",
                "prefall_location": "room",
                "fall_time_consensus": "10:00:10",
                "response_time_consensus": "10:00:20",
                "fall_tags": "slip",
            },
            {
                "event_key": "101|2025-01-01|10:01",
                "last_furniture": "",
                "furniture_departure_time": "",
                "prefall_location": "bed",
                "fall_time_consensus": "",
                "response_time_consensus": "",
                "fall_tags": "no_fall",
            },
            {
                "event_key": "102|2025-01-01|10:02",
                "last_furniture": "chair",
                "furniture_departure_time": "10:02:05",
                "prefall_location": "room",
                "fall_time_consensus": "10:02:10",
                "response_time_consensus": "10:02:25",
                "fall_tags": "trip",
            },
        ]
    ).to_csv(consensus_csv, index=False)

    shared_rows = [
        {
            "file": "100_2025-01-01_10-00.mp4",
            "fall_detected": True,
            "prefall_location": "room",
            "last_furniture": "bed",
            "fall_time_seconds": 10,
            "fall_time_wallclock": "10:00:10",
            "staff_entry_time_seconds": 20,
            "staff_entry_time_wallclock": "10:00:20",
            "staff_response_time_seconds": 10,
            "fall_tags": "slip",
            "confidence": "high",
        },
        {
            "file": "101_2025-01-01_10-01.mp4",
            "fall_detected": False,
            "prefall_location": "",
            "last_furniture": "",
            "fall_time_seconds": "",
            "fall_time_wallclock": "",
            "staff_entry_time_seconds": "",
            "staff_entry_time_wallclock": "",
            "staff_response_time_seconds": "",
            "fall_tags": "no_fall",
            "confidence": "high",
        },
    ]
    exclusive_row = {
        "file": "102_2025-01-01_10-02.mp4",
        "fall_detected": True,
        "prefall_location": "room",
        "last_furniture": "chair",
        "fall_time_seconds": 10,
        "fall_time_wallclock": "10:02:10",
        "staff_entry_time_seconds": 25,
        "staff_entry_time_wallclock": "10:02:25",
        "staff_response_time_seconds": 15,
        "fall_tags": "trip",
        "confidence": "high",
    }

    for idx, model in enumerate(PRIMARY_MODELS):
        timestamp = f"20260313T0{idx}0000Z"
        summary_csv = output_dir / f"summary_{timestamp}.csv"
        results_jsonl = output_dir / f"results_{timestamp}.jsonl"
        rows = list(shared_rows)
        if idx == 0:
            rows.append(exclusive_row)
        _write_summary_csv(summary_csv, rows)
        _write_results_jsonl(
            results_jsonl,
            model=model,
            rows=[{"file": row["file"], "result": {"ok": True}} for row in rows],
        )

    runs = discover_latest_model_runs(output_dir)
    artifacts = build_model_artifacts(runs, consensus_csv=consensus_csv)
    metrics = build_model_metrics(artifacts)
    event_long = build_event_long_table(artifacts)
    overlap = build_overlap_wide_table(event_long)

    overlap_rows = metrics.loc[
        (metrics["scope"] == "three_way_overlap") & (metrics["metric"] == "rows_total"), "value"
    ]
    assert set(overlap_rows.tolist()) == {2}
    assert overlap["event_key"].tolist() == ["100|2025-01-01|10:00", "101|2025-01-01|10:01"]


def test_auxiliary_tables_materialize_true_full_success_set_scope() -> None:
    event_long = pd.DataFrame(
        [
            {
                "study_model": "gemini-2.5-flash",
                "event_key": "1",
                "in_three_way_overlap": True,
                "visible_primary": True,
                "consensus_status": CONSENSUS_STATUS_INCLUDED_FALL,
                "gemini_fall_detected": True,
                "gemini_prefall_location": "room",
                "gt_prefall_location": "room",
                "gt_fall_tags": "slip,slow",
            },
            {
                "study_model": "gemini-2.5-flash",
                "event_key": "2",
                "in_three_way_overlap": False,
                "visible_primary": True,
                "consensus_status": CONSENSUS_STATUS_INCLUDED_FALL,
                "gemini_fall_detected": False,
                "gemini_prefall_location": pd.NA,
                "gt_prefall_location": "bed",
                "gt_fall_tags": "slow",
            },
            {
                "study_model": "gemini-2.5-flash",
                "event_key": "3",
                "in_three_way_overlap": False,
                "visible_primary": True,
                "consensus_status": CONSENSUS_STATUS_ACCEPTED_NONFALL,
                "gemini_fall_detected": False,
                "gemini_prefall_location": pd.NA,
                "gt_prefall_location": pd.NA,
                "gt_fall_tags": "",
            },
        ]
    )

    detection = build_detection_confusion_table(event_long)
    location = build_location_confusion_table(event_long)
    false_negative_tags = build_false_negative_tag_table(event_long)

    full_detection = detection.loc[detection["scope"] == "full_success_set"].iloc[0].to_dict()
    overlap_detection = detection.loc[detection["scope"] == "three_way_overlap"].iloc[0].to_dict()

    assert full_detection["tp"] == 1
    assert full_detection["fn"] == 1
    assert full_detection["tn"] == 1
    assert full_detection["fp"] == 0
    assert overlap_detection["tp"] == 1
    assert overlap_detection["fn"] == 0

    assert location.loc[location["scope"] == "full_success_set", "count"].tolist() == [1]
    assert location.loc[location["scope"] == "three_way_overlap", "count"].tolist() == [1]

    full_fn_rows = false_negative_tags.loc[false_negative_tags["scope"] == "full_success_set"]
    overlap_fn_rows = false_negative_tags.loc[false_negative_tags["scope"] == "three_way_overlap"]
    assert full_fn_rows.to_dict(orient="records") == [
        {
            "scope": "full_success_set",
            "model": "gemini-2.5-flash",
            "gt_tag": "slow",
            "count": 1,
        }
    ]
    assert overlap_fn_rows.empty
