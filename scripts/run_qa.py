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

from ld_chair_falls.config import load_settings
from ld_chair_falls.manifest import write_run_manifest
from ld_chair_falls.qa import run_qa_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chair-falls QA/reporting stage.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--cohort-map", default=None)
    parser.add_argument("--gate-1-pass", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gate-2-pass", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(
        run_id=args.run_id,
        dry_run=args.dry_run,
        cohort_map_path=args.cohort_map,
        gate_1_pass=args.gate_1_pass,
        gate_2_pass=args.gate_2_pass,
    )

    qa_manifest = run_qa_pipeline(settings)
    run_manifest = write_run_manifest(settings)

    print(f"run_id={settings.run_id}")
    print(f"qa_manifest={qa_manifest['manifest_path']}")
    print(f"run_manifest={run_manifest}")


if __name__ == "__main__":
    main()
