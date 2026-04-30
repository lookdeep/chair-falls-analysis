# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import reconcile_peer_edit_docx


def test_normalize_markdown_for_diff_normalizes_images_and_punctuation() -> None:
    text = (
        "“Quoted” text with em dash — and nbsp\u00a0space.\n\n"
        "![Figure 1. Example](media/image1.png){ width=95% }\n"
    )

    normalized = reconcile_peer_edit_docx.normalize_markdown_for_diff(text)

    assert '"Quoted" text with em dash - and nbsp space.' in normalized
    assert "![Figure 1. Example]" in normalized
    assert "media/image1.png" not in normalized


def test_build_section_map_disambiguates_duplicate_subheadings() -> None:
    text = (
        "# Methods\n\n"
        "## Sensitivity Analyses\n\n"
        "Methods text.\n\n"
        "# Results\n\n"
        "## Sensitivity Analyses\n\n"
        "Results text.\n"
    )

    sections = reconcile_peer_edit_docx.build_section_map(text)

    assert "Methods > Sensitivity Analyses" in sections
    assert "Results > Sensitivity Analyses" in sections
    assert sections["Methods > Sensitivity Analyses"].endswith("Methods text.\n")
    assert sections["Results > Sensitivity Analyses"].endswith("Results text.\n")


def test_build_section_map_infers_bold_export_headings() -> None:
    text = (
        "**ABSTRACT**\n\n"
        "Abstract text.\n\n"
        "**METHODS**\n\n"
        "**Study Design and Setting**\n\n"
        "Methods text.\n\n"
        "**Table 1.** Not a section heading.\n"
        "**Column A**   **Column B**\n"
    )

    sections = reconcile_peer_edit_docx.build_section_map(text)

    assert "Abstract" in sections
    assert "Methods" in sections
    assert "Methods > Study Design and Setting" in sections
    assert "Methods > Table 1. Not a section heading." not in sections
    assert "Methods > Column A   Column B" not in sections


def test_summarize_section_changes_detects_changed_and_missing_sections() -> None:
    canonical_sections = {
        "Introduction": "Alpha\n",
        "Methods": "Stable\n",
        "Results": "Keep\n",
    }
    exported_sections = {
        "Introduction": "Alpha revised\n",
        "Methods": "Stable\n",
        "Discussion": "New\n",
    }

    summary = reconcile_peer_edit_docx.summarize_section_changes(
        canonical_sections,
        exported_sections,
    )

    assert summary.changed == ["Introduction"]
    assert summary.unchanged == ["Methods"]
    assert summary.export_only == ["Discussion"]
    assert summary.canonical_only == ["Results"]
