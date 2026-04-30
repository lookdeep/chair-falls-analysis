from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.real_benchmark import (  # noqa: E402
    DEFAULT_REAL_BENCHMARK_FIXTURE_VERSION,
    DEFAULT_REAL_BENCHMARK_SOURCE_RUN_ID,
    freeze_real_benchmark_fixture,
    replay_real_benchmark_fixture,
    resolve_real_benchmark_fixture_dir,
    validate_real_benchmark_replay,
)


def _default_replay_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"real_benchmark_replay_{timestamp}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze and replay the frozen real-data benchmark.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze", help="Freeze the source run into a local fixture.")
    freeze_parser.add_argument(
        "--source-run-id",
        default=DEFAULT_REAL_BENCHMARK_SOURCE_RUN_ID,
        help="Run id to freeze from the source project root.",
    )
    freeze_parser.add_argument(
        "--fixture-version",
        default=DEFAULT_REAL_BENCHMARK_FIXTURE_VERSION,
        help="Fixture version directory name under data/fixtures/real_benchmark.",
    )
    freeze_parser.add_argument(
        "--fixture-dir",
        help="Explicit fixture directory override.",
    )
    freeze_parser.add_argument(
        "--source-project-root",
        default=str(PROJECT_ROOT),
        help="Project root containing the source raw/staged/qa artifacts.",
    )
    freeze_parser.add_argument(
        "--source-consensus-path",
        help="Optional absolute or source-project-relative override for the consensus CSV path.",
    )
    freeze_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing fixture directory if it already exists.",
    )

    replay_parser = subparsers.add_parser("replay", help="Replay the frozen benchmark locally.")
    replay_parser.add_argument(
        "--fixture-version",
        default=DEFAULT_REAL_BENCHMARK_FIXTURE_VERSION,
        help="Fixture version directory name under data/fixtures/real_benchmark.",
    )
    replay_parser.add_argument(
        "--fixture-dir",
        help="Explicit fixture directory override.",
    )
    replay_parser.add_argument(
        "--run-id",
        default=_default_replay_run_id(),
        help="Run id for the replayed local benchmark.",
    )
    replay_parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Project root that should receive the replay outputs.",
    )
    replay_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace any existing raw/staged replay directories for the requested run id.",
    )

    args = parser.parse_args()

    if args.command == "freeze":
        fixture_dir = resolve_real_benchmark_fixture_dir(
            PROJECT_ROOT,
            fixture_dir=args.fixture_dir,
            fixture_version=args.fixture_version,
        )
        manifest_path = freeze_real_benchmark_fixture(
            Path(args.source_project_root),
            fixture_dir,
            source_run_id=args.source_run_id,
            source_consensus_path=Path(args.source_consensus_path) if args.source_consensus_path else None,
            overwrite=args.overwrite,
        )
        print(json.dumps({"fixture_dir": str(fixture_dir), "manifest_path": str(manifest_path)}, indent=2))
        return 0

    fixture_dir = resolve_real_benchmark_fixture_dir(
        PROJECT_ROOT,
        fixture_dir=args.fixture_dir,
        fixture_version=args.fixture_version,
    )
    replay = replay_real_benchmark_fixture(
        Path(args.project_root),
        fixture_dir,
        run_id=args.run_id,
        overwrite=args.overwrite,
    )
    summary = validate_real_benchmark_replay(replay)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
