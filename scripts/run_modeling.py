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
from ld_chair_falls.modeling import run_adjusted_rr_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chair-falls adjusted RR model.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--gate-1-pass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gate-2-pass", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(
        run_id=args.run_id,
        dry_run=False,
        gate_1_pass=args.gate_1_pass,
        gate_2_pass=args.gate_2_pass,
    )

    adjusted_rr_path, sensitivity_path = run_adjusted_rr_model(settings)

    print(f"run_id={settings.run_id}")
    print(f"adjusted_rr_path={adjusted_rr_path}")
    print(f"sensitivity_analyses_path={sensitivity_path}")


if __name__ == "__main__":
    main()
