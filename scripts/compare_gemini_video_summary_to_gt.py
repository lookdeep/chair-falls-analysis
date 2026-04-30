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

from ld_chair_falls.gemini_eval import (  # noqa: E402
    resolve_default_results_jsonl,
    resolve_default_summary_csv,
    write_gemini_comparison_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Gemini video-summary outputs against consensus GT observations."
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Path to Gemini summary CSV. Defaults to the latest outputs/gemini_fall_analysis/summary_*.csv.",
    )
    parser.add_argument(
        "--results-jsonl",
        default=None,
        help="Path to Gemini JSONL results log. Defaults to the matching results_<timestamp>.jsonl when available.",
    )
    parser.add_argument(
        "--consensus-csv",
        default=str(PROJECT_ROOT / "data" / "public" / "falls-observations-v3-consensus.csv"),
        help="Path to GT consensus CSV (default: data/public/falls-observations-v3-consensus.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "gemini_fall_analysis"),
        help="Directory for comparison artifacts (default: outputs/gemini_fall_analysis).",
    )
    parser.add_argument(
        "--report-prefix",
        default=None,
        help="Optional file-name suffix for generated artifacts. Defaults to the summary timestamp when possible.",
    )
    return parser.parse_args()


def _default_report_prefix(summary_csv: Path) -> str | None:
    name = summary_csv.name
    if name.startswith("summary_") and name.endswith(".csv"):
        return name.removeprefix("summary_").removesuffix(".csv")
    return None


def main() -> None:
    args = parse_args()

    summary_csv = Path(args.summary_csv).resolve() if args.summary_csv else resolve_default_summary_csv(PROJECT_ROOT)
    if summary_csv is None:
        raise FileNotFoundError("No Gemini summary CSV found. Provide --summary-csv explicitly.")

    results_jsonl = Path(args.results_jsonl).resolve() if args.results_jsonl else resolve_default_results_jsonl(summary_csv)
    consensus_csv = Path(args.consensus_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_prefix = args.report_prefix or _default_report_prefix(summary_csv)

    artifacts, paths = write_gemini_comparison_outputs(
        summary_csv=summary_csv,
        consensus_csv=consensus_csv,
        results_jsonl=results_jsonl,
        output_dir=output_dir,
        report_prefix=report_prefix,
    )

    metric_map = {
        (row["metric_group"], row["metric"]): row["value"]
        for row in artifacts.metrics.to_dict(orient="records")
    }
    coverage_row = artifacts.coverage.loc[artifacts.coverage["consensus_status"] == "all"].iloc[0]

    print(f"summary_csv={summary_csv}")
    print(f"results_jsonl={results_jsonl}" if results_jsonl is not None else "results_jsonl=<none>")
    print(f"consensus_csv={consensus_csv}")
    print(
        "coverage="
        f"{int(coverage_row['successful_event_keys'])}/{int(coverage_row['total_gt_event_keys'])}"
    )
    print(f"headline_denominator={int(metric_map.get(('scope', 'headline_denominator'), 0))}")
    print(
        "included_fall_location_accuracy="
        f"{metric_map.get(('included_fall', 'prefall_location_accuracy'), 'NA')}"
    )
    print(
        "included_fall_fall_time_mae_seconds="
        f"{metric_map.get(('included_fall', 'fall_time_mae_seconds'), 'NA')}"
    )
    print(
        "accepted_nonfall_false_positive_rate="
        f"{metric_map.get(('accepted_nonfall', 'false_positive_rate'), 'NA')}"
    )
    for key, path in paths.items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()
