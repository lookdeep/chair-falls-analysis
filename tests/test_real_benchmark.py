# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.real_benchmark import (  # noqa: E402
    DEFAULT_REAL_BENCHMARK_FIXTURE_VERSION,
    replay_real_benchmark_fixture,
    resolve_real_benchmark_fixture_dir,
    validate_real_benchmark_replay,
)


@pytest.mark.real_benchmark
def test_frozen_real_benchmark_contract(tmp_path: Path) -> None:
    fixture_dir = resolve_real_benchmark_fixture_dir(
        PROJECT_ROOT,
        fixture_version=DEFAULT_REAL_BENCHMARK_FIXTURE_VERSION,
    )
    if not fixture_dir.exists():
        pytest.skip(f"Frozen real benchmark fixture not found: {fixture_dir}")

    replay = replay_real_benchmark_fixture(
        tmp_path,
        fixture_dir,
        run_id=f"real_benchmark_pytest_{uuid4().hex[:8]}",
    )
    validation = validate_real_benchmark_replay(replay)

    assert validation["passed"], validation
    assert replay.validation_summary_path.exists()
