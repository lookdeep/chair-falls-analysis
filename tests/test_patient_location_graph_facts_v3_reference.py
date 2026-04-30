# ruff: noqa: E402

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.config import Settings
from ld_chair_falls.prefall_location import build_panel_prefall_event_windows

REFERENCE_PATH = PROJECT_ROOT / "sql" / "03_handoff" / "patient_location_graph_facts_v3_reference.py"
EXAMPLE_PATH = PROJECT_ROOT / "sql" / "03_handoff" / "patient_location_graph_facts_v3_reference_example.json"

FLOAT_OUTPUT_KEYS = {
    "pre_prob_v2_chair",
    "pre_prob_v2_bed",
    "pre_prob_v2_room",
    "pre_prob_v2_no_patient",
    "pre_prob_v3_chair",
    "pre_prob_v3_bed",
    "pre_prob_v3_room",
    "pre_prob_v3_no_patient",
    "prefall_location_v3_departure_confidence",
}

REFERENCE_SMOKE_KEYS = {
    "pre_prob_v2_chair",
    "pre_prob_v2_bed",
    "pre_prob_v2_room",
    "pre_prob_v2_no_patient",
    "pre_prob_v3_chair",
    "pre_prob_v3_bed",
    "pre_prob_v3_room",
    "pre_prob_v3_no_patient",
    "prefall_location_v2_label",
    "prefall_location_v2_reason",
    "prefall_location_v3_label",
    "prefall_location_v3_reason",
    "prefall_location_v3_signal_regime",
    "prefall_location_reason",
}

REFERENCE_FLOAT_TOLERANCE = 2e-6


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(project_root=tmp_path, run_id="graph_facts_v3_reference_test", dry_run=True)


def _load_examples() -> list[dict[str, object]]:
    payload = json.loads(EXAMPLE_PATH.read_text())
    return payload["scenarios"]


def _load_reference_module():
    spec = importlib.util.spec_from_file_location("patient_location_graph_facts_v3_reference", REFERENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reference_example_matches_canonical_v3_outputs(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    for scenario in _load_examples():
        event_windows = pd.DataFrame([scenario["event_window"]])
        second_level = pd.DataFrame(scenario["second_level_rows"])
        derived = build_panel_prefall_event_windows(settings, event_windows, second_level)
        row = derived.iloc[0]
        expected = scenario["expected_output"]

        for key, value in expected.items():
            actual = row[key]
            if key in FLOAT_OUTPUT_KEYS:
                assert float(actual) == pytest.approx(float(value), abs=1e-6)
            else:
                assert actual == value


def test_reference_defaults_match_settings_defaults(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    reference_module = _load_reference_module()

    for key, value in reference_module.DEFAULTS.items():
        actual = getattr(settings, key)
        if isinstance(value, float):
            assert actual == pytest.approx(value, abs=1e-12)
        else:
            assert actual == value


def test_reference_python_smoke_matches_example_core_outputs() -> None:
    reference_module = _load_reference_module()

    for scenario in _load_examples():
        summary = reference_module.summarize_panel_history(scenario["second_level_rows"])
        derived = reference_module.derive_prefall_location_v3(scenario["event_window"], summary)
        expected = scenario["expected_output"]

        for key in REFERENCE_SMOKE_KEYS:
            if key in FLOAT_OUTPUT_KEYS:
                assert float(derived[key]) == pytest.approx(
                    float(expected[key]),
                    abs=REFERENCE_FLOAT_TOLERANCE,
                )
            else:
                assert derived[key] == expected[key]
