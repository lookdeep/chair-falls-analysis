# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import build_dual_track_paper


def test_parse_strobe_flow_summary_extracts_expected_counts(tmp_path: Path) -> None:
    strobe_path = tmp_path / "strobe_flow_diagram.md"
    strobe_path.write_text(
        (
            "Hourly monitors in source window (hospital_id=5): 5,531\n"
            "  -> Passed eligibility gates (min_observed_hours>=4; coverage>=0.95): 3,980\n"
            "     -> Intervention eligible: 42\n"
            "     -> Control eligible: 3,938\n\n"
            "Study-window adjudicated events (August 2024-December 2025): 43\n"
            "  -> Linked to eligible analysis base: 40\n"
            "     -> Chair hard-label events: 5\n"
            "     -> Bed hard-label events: 23\n"
            "     -> Room / no_patient labels: 12\n\n"
            "Broader monitoring feed (2022-2026, descriptive reference): 91 deduplicated events\n\n"
            "Broader observation cohort (descriptive mechanism coding only):\n"
            "  37 source annotations -> 32 included fall rows -> 32 deduplicated events\n\n"
            "Departure-aware benchmark subset (descriptive diagnostics only):\n"
            "  30 truth rows -> 31 scored sequences\n"
        ),
        encoding="utf-8",
    )

    summary = build_dual_track_paper.parse_strobe_flow_summary(strobe_path)

    assert summary.hourly_monitors == 5531
    assert summary.passed_eligibility == 3980
    assert summary.intervention_eligible == 42
    assert summary.control_eligible == 3938
    assert summary.study_window_events == 43
    assert summary.linked_events == 40
    assert summary.room_or_no_patient_hard_labels == 12
    assert summary.broader_monitoring_feed == 91
    assert summary.observation_annotations == 37
    assert summary.observation_included_rows == 32
    assert summary.observation_events == 32
    assert summary.benchmark_truth_rows == 30
    assert summary.benchmark_scored_sequences == 31


def test_validate_core_manuscript_numbers_requires_rr_marker() -> None:
    good_text = (
        "43 adjudicated fall events matched the monitoring pipeline; "
        "40 linked to eligible analysis-base hours. "
        "Broader monitoring feed (2022-2026, n=91 deduplicated events). "
        "A broader observation cohort retained 32 deduplicated fall events. "
        "The study reported 17.8 falls per 1,000 chair exposure-hours and "
        "4.3 falls per 1,000 bed exposure-hours. "
        "The adjusted RR was 2.35 (95% confidence interval 0.87 to 6.33)."
    )
    build_dual_track_paper.validate_core_manuscript_numbers(good_text)

    with pytest.raises(ValueError):
        build_dual_track_paper.validate_core_manuscript_numbers(
            good_text.replace("2.35", "2.10", 1)
        )


def test_validate_doc_patterns_requires_current_packet_markers(tmp_path: Path) -> None:
    checklist_path = tmp_path / "strobe_checklist.md"
    checklist_path.write_text(
        (
            "Participants: outcome-defined intervention/control membership, "
            "the 43-event study-window adjudicated cohort, the 40-event inferential "
            "base, and the distinct 91/32/30-31 descriptive cohorts are documented.\n"
            "Participant flow: documented in docs/strobe_flow_diagram.md, "
            "including the manuscript-facing 43 -> 40 adjudication/linkage funnel.\n"
        ),
        encoding="utf-8",
    )

    build_dual_track_paper.validate_doc_patterns(
        path=checklist_path,
        required_patterns={
            "cohort_architecture": (
                r"43-event study-window adjudicated cohort, the 40-event inferential "
                r"base, and the distinct 91/32/30-31 descriptive cohorts"
            ),
            "participant_flow": r"43 -> 40",
        },
    )

    with pytest.raises(ValueError):
        build_dual_track_paper.validate_doc_patterns(
            path=checklist_path,
            required_patterns={"stale_architecture": r"43 / 40 / 91 / 50"},
        )


def test_stage_figure_assets_copies_all_seven_manuscript_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets_root = tmp_path / "paper_assets"
    arxiv_assets_root = tmp_path / "arxiv_assets"
    mapped_sources = [
        ("figure2_fall_rates.png", "fig1_fall_rates.png", b"fig1"),
        ("figure3_sensitivity_forest.png", "fig2_sensitivity_forest.png", b"fig2"),
        ("figure4_mechanism_taxonomy.png", "fig4_mechanism_by_posture.png", b"fig4"),
        ("figure5_context_schematic.png", "fig6_context_schematic.png", b"fig6"),
        ("figure6_chair_occupancy_daypart.png", "fig5_chair_occupancy_daypart.png", b"fig5"),
        ("figureA1_operational_signals.png", "fig7_operational_signals.png", b"fig7"),
    ]
    default_map: list[tuple[str, Path]] = []
    for staged_name, source_name, marker in mapped_sources:
        source_path = tmp_path / source_name
        source_path.write_bytes(marker)
        default_map.append((staged_name, source_path))

    monkeypatch.setattr(build_dual_track_paper, "ASSETS_ROOT", assets_root)
    monkeypatch.setattr(build_dual_track_paper, "ARXIV_ASSETS_ROOT", arxiv_assets_root)
    monkeypatch.setattr(build_dual_track_paper, "DEFAULT_FIGURE_MAP", default_map)

    def fake_write_strobe(_summary: build_dual_track_paper.StrobeSummary, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"strobe")

    monkeypatch.setattr(build_dual_track_paper, "write_strobe_flow_figure", fake_write_strobe)

    summary = build_dual_track_paper.StrobeSummary(
        hourly_monitors=5531,
        passed_eligibility=3980,
        intervention_eligible=42,
        control_eligible=3938,
        study_window_events=43,
        linked_events=40,
        chair_hard_labels=5,
        bed_hard_labels=23,
        room_or_no_patient_hard_labels=12,
        broader_monitoring_feed=91,
        observation_annotations=37,
        observation_included_rows=32,
        observation_events=32,
        benchmark_truth_rows=30,
        benchmark_scored_sequences=31,
    )

    build_dual_track_paper.stage_figure_assets(summary)

    assert (assets_root / "figure1_strobe_flow.png").read_bytes() == b"strobe"
    assert (arxiv_assets_root / "figure1_strobe_flow.png").read_bytes() == b"strobe"
    for staged_name, _source_name, marker in mapped_sources:
        assert (assets_root / staged_name).read_bytes() == marker
        assert (arxiv_assets_root / staged_name).read_bytes() == marker


def test_maybe_refresh_figures_requests_restored_figure_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_map = [
        ("figure2_fall_rates.png", tmp_path / "fig1_fall_rates.png"),
        ("figure3_sensitivity_forest.png", tmp_path / "fig2_sensitivity_forest.png"),
        ("figure4_mechanism_taxonomy.png", tmp_path / "fig4_mechanism_by_posture.png"),
        ("figure5_context_schematic.png", tmp_path / "fig6_context_schematic.png"),
        ("figure6_chair_occupancy_daypart.png", tmp_path / "fig5_chair_occupancy_daypart.png"),
        ("figureA1_operational_signals.png", tmp_path / "fig7_operational_signals.png"),
    ]
    calls: list[list[str]] = []

    monkeypatch.setattr(build_dual_track_paper, "DEFAULT_FIGURE_MAP", missing_map)

    def fake_run(cmd: list[str | Path], *, cwd: Path) -> None:
        calls.append([str(part) for part in cmd])
        for _, source_path in missing_map:
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(b"generated")

    monkeypatch.setattr(build_dual_track_paper, "run", fake_run)

    build_dual_track_paper.maybe_refresh_figures("my_run_20260324T000000Z", regenerate_figures=False)

    assert len(calls) == 1
    figs_index = calls[0].index("--figs")
    assert calls[0][figs_index + 1 :] == list(build_dual_track_paper.DEFAULT_GENERATED_FIGURE_IDS)


def test_render_arxiv_tex_uses_citeproc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper_root = tmp_path / "paper"
    arxiv_root = paper_root / "arxiv"
    manuscript_path = paper_root / "manuscript.md"
    metadata_path = paper_root / "metadata.yaml"
    manuscript_path.parent.mkdir(parents=True, exist_ok=True)
    manuscript_path.write_text("Sample", encoding="utf-8")
    metadata_path.write_text("title: Sample", encoding="utf-8")

    monkeypatch.setattr(build_dual_track_paper, "PAPER_ROOT", paper_root)
    monkeypatch.setattr(build_dual_track_paper, "ARXIV_ROOT", arxiv_root)
    monkeypatch.setattr(build_dual_track_paper, "ARXIV_TEX_PATH", arxiv_root / "manuscript.tex")
    monkeypatch.setattr(build_dual_track_paper, "MANUSCRIPT_PATH", manuscript_path)
    monkeypatch.setattr(build_dual_track_paper, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(build_dual_track_paper, "require_tool", lambda _tool: None)

    calls: list[list[str]] = []

    def fake_run(cmd: list[str | Path], *, cwd: Path) -> None:
        calls.append([str(part) for part in cmd])
        assert cwd == paper_root

    monkeypatch.setattr(build_dual_track_paper, "run", fake_run)

    build_dual_track_paper.render_arxiv_tex()

    assert len(calls) == 1
    assert "--citeproc" in calls[0]


def test_render_peer_docx_uses_citeproc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper_root = tmp_path / "paper"
    peer_root = paper_root / "peer"
    manuscript_path = paper_root / "manuscript.md"
    metadata_path = paper_root / "metadata.yaml"
    reference_doc_path = paper_root / "reference.docx"
    manuscript_path.parent.mkdir(parents=True, exist_ok=True)
    manuscript_path.write_text("Sample", encoding="utf-8")
    metadata_path.write_text("title: Sample", encoding="utf-8")
    reference_doc_path.write_bytes(b"docx")

    monkeypatch.setattr(build_dual_track_paper, "PAPER_ROOT", paper_root)
    monkeypatch.setattr(build_dual_track_paper, "PEER_ROOT", peer_root)
    monkeypatch.setattr(
        build_dual_track_paper,
        "PEER_DOCX_PATH",
        peer_root / "chair_fall_paper_peer_edit.docx",
    )
    monkeypatch.setattr(build_dual_track_paper, "MANUSCRIPT_PATH", manuscript_path)
    monkeypatch.setattr(build_dual_track_paper, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(build_dual_track_paper, "REFERENCE_DOC_PATH", reference_doc_path)
    monkeypatch.setattr(build_dual_track_paper, "require_tool", lambda _tool: None)

    calls: list[list[str]] = []

    def fake_run(cmd: list[str | Path], *, cwd: Path) -> None:
        calls.append([str(part) for part in cmd])
        assert cwd == paper_root

    monkeypatch.setattr(build_dual_track_paper, "run", fake_run)

    build_dual_track_paper.render_peer_docx()

    assert len(calls) == 1
    assert "--citeproc" in calls[0]
    assert "--reference-doc" in calls[0]
