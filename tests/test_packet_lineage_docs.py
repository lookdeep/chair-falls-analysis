# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.paper_defaults import CURRENT_MANUSCRIPT_RUN_ID

LEGACY_OUTCOME_RUN_ID = "consensus_v3_outcome_20260324T014635Z"


def test_manuscript_sources_distinguish_v2_context_from_hard_labels() -> None:
    required_phrase = (
        "The legacy v2 broader-observation raw consensus table (85 annotations: room 30, bed 26, "
        "chair 14, blank 11, no_patient 4)"
    )
    distinction_phrase = (
        "is retained for annotation context only; it is not used as benchmark truth and is not "
        "reported as hard-label output."
    )
    hard_label_phrase = (
        "Within the broader monitoring feed (2022-2026, n=91 deduplicated events), the hard-label "
        "position distribution was: bed (n=48, 52.7%), no_patient (n=23, 25.3%), room (n=12, "
        "13.2%), and chair (n=8, 8.8%)."
    )

    text = (PROJECT_ROOT / "paper/manuscript.md").read_text(encoding="utf-8")
    assert required_phrase in text
    assert distinction_phrase in text
    assert hard_label_phrase in text


def test_public_release_docs_reference_canonical_v3_refresh_bundle() -> None:
    for relative_path in (
        Path("docs/artifact_map.md"),
        Path("docs/reproduce_this_paper.md"),
        Path("docs/deidentification_checklist.md"),
        Path("docs/ethics_irb_dua_record.md"),
    ):
        text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert CURRENT_MANUSCRIPT_RUN_ID in text
        assert "data/public/" in text
        assert LEGACY_OUTCOME_RUN_ID not in text
    for relative_path in (
        Path("docs/artifact_map.md"),
        Path("docs/reproduce_this_paper.md"),
    ):
        text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert "chair_bed_operational_event_rates_" in text

    strobe_text = (PROJECT_ROOT / "docs/strobe_flow_diagram.md").read_text(encoding="utf-8")
    assert CURRENT_MANUSCRIPT_RUN_ID in strobe_text
    assert "data/public/falls-observations-v3-consensus.csv" in strobe_text
