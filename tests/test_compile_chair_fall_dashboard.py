# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import compile_chair_fall_dashboard as dashboard
import pandas as pd


def test_build_manuscript_renders_public_release_dashboard(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    derived_dir = project_root / "data" / "public" / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    run_id = "my_run_20260312T000000Z"

    adjusted_rr = pd.DataFrame(
        [
            {
                "sensitivity_label": "primary_adjusted",
                "rr": 2.3507,
                "ci_lower": 0.8734,
                "ci_upper": 6.3272,
                "p_value": 0.0907,
                "n_events": 40,
                "model_notes": "estimable",
            },
            {
                "sensitivity_label": "primary_adjusted_hc3",
                "rr": 2.3507,
                "ci_lower": 0.9301,
                "ci_upper": 5.9411,
                "p_value": 0.0709,
                "n_events": 40,
                "model_notes": "HC3 SE",
            },
            {
                "sensitivity_label": "primary_adjusted_clustered",
                "rr": 2.3507,
                "ci_lower": 1.8899,
                "ci_upper": 2.9211,
                "p_value": "<0.0001",
                "n_events": 40,
                "model_notes": "9 intervention divisions; not a primary estimate",
            },
            {
                "sensitivity_label": "position_certain_only",
                "rr": None,
                "ci_lower": None,
                "ci_upper": None,
                "p_value": None,
                "n_events": 38,
                "model_notes": "insufficient_data",
            },
            {
                "sensitivity_label": "furniture_origin_reclassified",
                "rr": 2.3507,
                "ci_lower": 0.8734,
                "ci_upper": 6.3272,
                "p_value": 0.0907,
                "n_events": 40,
                "model_notes": "furniture-origin sensitivity",
            },
        ]
    )
    adjusted_rr.to_csv(derived_dir / f"chair_bed_adjusted_rr_{run_id}.csv", index=False)

    thresholds = pd.DataFrame(
        [
            {
                "threshold_hours": 12,
                "rr": 2.39,
                "ci_lower": 0.88,
                "ci_upper": 6.47,
                "p_value": 0.0859,
                "n_events": 38,
                "model_notes": "estimable",
            },
            {
                "threshold_hours": 24,
                "rr": 2.43,
                "ci_lower": 0.89,
                "ci_upper": 6.63,
                "p_value": 0.0831,
                "n_events": 36,
                "model_notes": "estimable",
            },
            {
                "threshold_hours": 48,
                "rr": 2.42,
                "ci_lower": 0.84,
                "ci_upper": 6.98,
                "p_value": 0.1008,
                "n_events": 30,
                "model_notes": "estimable",
            },
        ]
    )
    thresholds.to_csv(derived_dir / f"chair_bed_threshold_sensitivity_{run_id}.csv", index=False)

    output_path = project_root / "docs" / "chair_fall_dashboard.html"
    result = dashboard.build_manuscript(project_root, run_id, output_path)
    html = result.read_text(encoding="utf-8")

    assert result == output_path
    assert run_id in html
    assert "Chair Falls Risk Dashboard - my_run_20260312T000000Z" in html
    assert "<h1>Chair Falls Risk Dashboard</h1>" in html
    assert "Reference manuscript:</strong> Exposure-Normalized Bed and Chair Fall Rates via Continuous AI Monitoring" in html
    assert "Zack Drumm" in html
    assert "paper/manuscript.md" in html
    assert "data/public/" in html
    assert "Table 1. Cohort summary statistics" in html
    assert "Table 2. Unadjusted fall rates by patient position" in html
    assert "Table 3. Adjusted rate ratios and sensitivity analyses" in html
    assert "Primary (HC3)" in html
    assert "Exploratory clustered SE (division)" in html
    assert "Eligibility threshold &gt;= 12 h" in html
    assert "17.8" in html
    assert "4.3" in html
    assert "6 of 7" in html
    assert "52.7%" in html
    assert "Target artifact:</strong>" not in html
    assert "Missingness Stress" not in html
    assert "Negative-Control Anchor Check" not in html
    assert "Expected fall rate is higher for chair by 13.45" not in html
    assert "paper/arxiv-upload-20260323T203330.zip" not in html
