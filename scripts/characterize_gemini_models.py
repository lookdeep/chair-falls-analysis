#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.gemini_characterization import (  # noqa: E402
    PRIMARY_MODELS,
    build_model_artifacts,
    default_report_prefix,
    discover_latest_model_runs,
    write_characterization_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-off characterization comparing multiple Gemini fall-summary models against GT."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for characterization artifacts. "
            "Defaults to outputs/gemini_fall_analysis/model_characterization_<timestamp>/."
        ),
    )
    parser.add_argument(
        "--gemini-output-dir",
        default=str(PROJECT_ROOT / "outputs" / "gemini_fall_analysis"),
        help="Directory containing Gemini summary/results files (default: outputs/gemini_fall_analysis).",
    )
    parser.add_argument(
        "--consensus-csv",
        default=str(PROJECT_ROOT / "data" / "public" / "falls-observations-v3-consensus.csv"),
        help="Path to GT consensus CSV (default: data/public/falls-observations-v3-consensus.csv).",
    )
    parser.add_argument(
        "--report-prefix",
        default=None,
        help="Optional deterministic suffix for generated artifact names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report_prefix = args.report_prefix or default_report_prefix()
    gemini_output_dir = Path(args.gemini_output_dir).resolve()
    consensus_csv = Path(args.consensus_csv).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (gemini_output_dir / f"model_characterization_{report_prefix}")
    )

    runs = discover_latest_model_runs(gemini_output_dir, target_models=PRIMARY_MODELS)
    artifacts_by_model = build_model_artifacts(runs, consensus_csv=consensus_csv)
    paths = write_characterization_outputs(
        output_dir=output_dir,
        report_prefix=report_prefix,
        runs=runs,
        artifacts_by_model=artifacts_by_model,
    )

    print(f"consensus_csv={consensus_csv}")
    print(f"gemini_output_dir={gemini_output_dir}")
    print(f"output_dir={output_dir}")
    for run in runs:
        print(
            f"model={run.model} summary_csv={run.summary_csv} results_jsonl={run.results_jsonl} "
            f"success_rows={run.success_rows} error_rows={run.error_rows}"
        )
    for key, value in paths.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
