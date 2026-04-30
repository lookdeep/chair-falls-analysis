# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.config import load_settings


@pytest.mark.smoke
def test_load_settings_smoke() -> None:
    settings = load_settings(run_id="smoke_run", dry_run=True)
    assert settings.run_id == "smoke_run"
    assert settings.effective_run_mode in {"inferential_ready", "descriptive_only"}
    assert settings.case_crossover_source_status == "ready"
    assert settings.case_crossover_source_ready is True
    assert settings.negative_control_source_status == "ready"
    assert settings.negative_control_source_ready is True


@pytest.mark.smoke
def test_paths_smoke() -> None:
    settings = load_settings(run_id="smoke_paths", dry_run=True)
    assert settings.paths.raw_run_dir.name == "smoke_paths"
