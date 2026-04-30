from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_gemini_fall_analysis.py"


def _load_script_module():
    original_modules = {name: sys.modules.get(name) for name in ("dotenv", "google", "google.genai")}

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = object
    fake_genai.types = types.SimpleNamespace(GenerateContentConfig=object)

    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai

    sys.modules["dotenv"] = fake_dotenv
    sys.modules["google"] = fake_google
    sys.modules["google.genai"] = fake_genai

    try:
        spec = importlib.util.spec_from_file_location("test_run_gemini_fall_analysis_module", SCRIPT_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_system_instructions_allow_inferred_falls_from_on_ground_observation() -> None:
    module = _load_script_module()

    assert "later clearly seen on the ground" in module.SYSTEM_INSTRUCTIONS
    assert "leave `fall_time_seconds` null" in module.SYSTEM_INSTRUCTIONS
    assert "set `fall_detected` to true" in module.SYSTEM_INSTRUCTIONS


def test_normalize_response_payload_clears_response_time_without_visible_fall_time() -> None:
    module = _load_script_module()

    parsed = {
        "fall_event": {
            "fall_detected": True,
            "prefall_location": "room",
            "last_furniture": "bed",
            "fall_time_seconds": None,
            "staff_entry_time_seconds": 22,
            "fall_tags": ["collapse"],
        },
        "staff_response_time_seconds": 999,
        "confidence": "medium",
        "report": "Fall inferred because the patient is later seen on the ground.",
    }

    normalized = module.normalize_response_payload(parsed)

    assert normalized["fall_event"]["fall_time_seconds"] is None
    assert normalized["fall_event"]["staff_entry_time_seconds"] == 22
    assert normalized["staff_response_time_seconds"] is None
