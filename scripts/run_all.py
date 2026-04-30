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
from ld_chair_falls.extract import load_raw_tables, run_extract_pipeline
from ld_chair_falls.manifest import write_run_manifest
from ld_chair_falls.qa import run_qa_pipeline, write_gate_preflight, write_source_profile
from ld_chair_falls.transform import run_transform_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full chair-falls prep initialization pipeline.")
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

    gate_path = write_gate_preflight(settings)
    extract_manifest = run_extract_pipeline(settings)
    raw_tables = load_raw_tables(settings)
    source_profile_path = write_source_profile(settings, raw_tables)
    transform_manifest = run_transform_pipeline(settings)
    qa_manifest = run_qa_pipeline(settings)
    run_manifest_path = write_run_manifest(settings)

    print(f"run_id={settings.run_id}")
    print(f"effective_run_mode={settings.effective_run_mode}")
    print(f"gate_preflight={gate_path}")
    print(f"extract_manifest={extract_manifest['manifest_path']}")
    print(f"source_profile={source_profile_path}")
    print(f"transform_manifest={transform_manifest['manifest_path']}")
    print(f"qa_manifest={qa_manifest['manifest_path']}")
    print(f"run_manifest={run_manifest_path}")


if __name__ == "__main__":
    main()
