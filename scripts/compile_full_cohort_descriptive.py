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

from ld_chair_falls.report_descriptive import (  # noqa: E402
    resolve_full_cohort_run_id,
    write_full_cohort_hourly_descriptive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile outputs/full_cohort_hourly_descriptive.html from cohort/hourly QA artifacts."
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit run id (for example chair_bed_fix_20260304_exec).",
    )
    parser.add_argument(
        "--output",
        default="outputs/full_cohort_hourly_descriptive.html",
        help="Output HTML path (default: outputs/full_cohort_hourly_descriptive.html).",
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Project root directory. Defaults to repository project root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    run_id = resolve_full_cohort_run_id(project_root, args.run_id)
    output_path = write_full_cohort_hourly_descriptive(project_root, run_id, output_filename=args.output)
    print(f"run_id={run_id}")
    print(f"output_html={output_path}")


if __name__ == "__main__":
    main()
