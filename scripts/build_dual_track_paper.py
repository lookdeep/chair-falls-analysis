#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
LOCAL_CACHE_ROOT = PROJECT_ROOT / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_ROOT / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ld_chair_falls.paper_defaults import (
    CURRENT_MANUSCRIPT_RUN_ID,
    assert_manuscript_packet_artifacts,
)

PAPER_ROOT = PROJECT_ROOT / "paper"
DOCS_ROOT = PROJECT_ROOT / "docs"
FIGURES_ROOT = PROJECT_ROOT / "figures"
ASSETS_ROOT = PAPER_ROOT / "assets"
ARXIV_ROOT = PAPER_ROOT / "arxiv"
ARXIV_ASSETS_ROOT = ARXIV_ROOT / "assets"
PEER_ROOT = PAPER_ROOT / "peer"

MANUSCRIPT_PATH = PAPER_ROOT / "manuscript.md"
SEED_PATH = PAPER_ROOT / "manuscript.seed.md"
SEED_MEDIA_ROOT = PAPER_ROOT / "seed_media"
METADATA_PATH = PAPER_ROOT / "metadata.yaml"
STROBE_FLOW_PATH = DOCS_ROOT / "strobe_flow_diagram.md"
STROBE_CHECKLIST_PATH = DOCS_ROOT / "strobe_checklist.md"
SEED_DOCX_PATH = DOCS_ROOT / "chair_fall_paper_medrxiv.docx"
REFERENCE_DOC_PATH = PAPER_ROOT / "reference.docx"
ARXIV_TEX_PATH = ARXIV_ROOT / "manuscript.tex"
ARXIV_PDF_PATH = ARXIV_ROOT / "manuscript.pdf"
PEER_DOCX_PATH = PEER_ROOT / "chair_fall_paper_peer_edit.docx"

DEFAULT_RUN_ID = CURRENT_MANUSCRIPT_RUN_ID
DEFAULT_GENERATED_FIGURE_IDS = ("1", "2", "4", "5", "6", "7")
DEFAULT_FIGURE_MAP = [
    ("figure2_fall_rates.png", FIGURES_ROOT / "fig1_fall_rates.png"),
    ("figure3_sensitivity_forest.png", FIGURES_ROOT / "fig2_sensitivity_forest.png"),
    ("figure4_mechanism_taxonomy.png", FIGURES_ROOT / "fig4_mechanism_by_posture.png"),
    ("figure5_context_schematic.png", FIGURES_ROOT / "fig6_context_schematic.png"),
    ("figure6_chair_occupancy_daypart.png", FIGURES_ROOT / "fig5_chair_occupancy_daypart.png"),
    ("figureA1_operational_signals.png", FIGURES_ROOT / "fig7_operational_signals.png"),
]
REQUIRED_STROBE_COUNTS = {
    "hourly_monitors": 5531,
    "passed_eligibility": 3980,
    "intervention_eligible": 42,
    "control_eligible": 3938,
    "study_window_events": 43,
    "linked_events": 40,
    "broader_monitoring_feed": 91,
    "observation_events": 32,
    "benchmark_truth_rows": 30,
    "benchmark_scored_sequences": 31,
}
CORE_MANUSCRIPT_PATTERNS = {
    "study_window_linkage": (
        r"43 adjudicated fall(?: events|s) matched the monitoring pipeline(?:;|, and) "
        r"40 linked to eligible (?:analysis-base|exposure) hours"
    ),
    "broader_feed": r"2022-2026, n=91 deduplicated events",
    "observation_cohort": r"32 deduplicated (?:fall )?events",
    "descriptive_rates": (
        r"17\.8 falls per 1,000 chair exposure-hours and "
        r"4\.3(?: falls)? per 1,000 bed exposure-hours"
    ),
    "adjusted_rr": r"2\.35\s*\(95% (?:confidence interval|CI) 0\.87 to 6\.33",
}
STROBE_PACKET_PATTERNS = {
    STROBE_CHECKLIST_PATH: {
        "cohort_architecture": (
            r"43-event study-window adjudicated cohort, the 40-event inferential base, "
            r"and the distinct 91/32/30-31 descriptive cohorts"
        ),
        "participant_flow": r"43 -> 40",
    },
}


@dataclass(frozen=True)
class StrobeSummary:
    hourly_monitors: int
    passed_eligibility: int
    intervention_eligible: int
    control_eligible: int
    study_window_events: int
    linked_events: int
    chair_hard_labels: int
    bed_hard_labels: int
    room_or_no_patient_hard_labels: int
    broader_monitoring_feed: int
    observation_annotations: int
    observation_included_rows: int
    observation_events: int
    benchmark_truth_rows: int
    benchmark_scored_sequences: int


def require_tool(tool_name: str) -> None:
    if shutil.which(tool_name) is None:
        raise RuntimeError(f"Required tool not found on PATH: {tool_name}")


def run(cmd: list[str | Path], *, cwd: Path) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"+ {printable}")
    subprocess.run([str(part) for part in cmd], cwd=cwd, check=True)


def extract_int(text: str, pattern: str, label: str) -> int:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not parse {label} from {STROBE_FLOW_PATH}")
    return int(match.group(1).replace(",", ""))


def parse_strobe_flow_summary(path: Path) -> StrobeSummary:
    text = path.read_text(encoding="utf-8")
    observation_match = re.search(
        r"(\d[\d,]*) source annotations -> (\d[\d,]*) included fall rows -> "
        r"(\d[\d,]*) deduplicated events",
        text,
        re.MULTILINE,
    )
    if observation_match is None:
        raise ValueError(f"Could not parse observation cohort from {path}")

    benchmark_match = re.search(
        r"(\d[\d,]*) truth rows -> (\d[\d,]*) scored sequences",
        text,
        re.MULTILINE,
    )
    if benchmark_match is None:
        raise ValueError(f"Could not parse benchmark subset from {path}")

    return StrobeSummary(
        hourly_monitors=extract_int(
            text,
            r"Hourly monitors in source window .*?: (\d[\d,]*)",
            "hourly monitors",
        ),
        passed_eligibility=extract_int(
            text,
            r"Passed eligibility gates .*?: (\d[\d,]*)",
            "eligible monitors",
        ),
        intervention_eligible=extract_int(
            text,
            r"Intervention eligible: (\d[\d,]*)",
            "intervention eligible monitors",
        ),
        control_eligible=extract_int(
            text,
            r"Control eligible: (\d[\d,]*)",
            "control eligible monitors",
        ),
        study_window_events=extract_int(
            text,
            r"Study-window adjudicated events .*?: (\d[\d,]*)",
            "study-window adjudicated events",
        ),
        linked_events=extract_int(
            text,
            r"Linked to eligible analysis base: (\d[\d,]*)",
            "linked inferential events",
        ),
        chair_hard_labels=extract_int(
            text,
            r"Chair hard-label events: (\d[\d,]*)",
            "chair hard labels",
        ),
        bed_hard_labels=extract_int(
            text,
            r"Bed hard-label events: (\d[\d,]*)",
            "bed hard labels",
        ),
        room_or_no_patient_hard_labels=extract_int(
            text,
            r"Room / no_patient labels: (\d[\d,]*)",
            "room/no_patient hard labels",
        ),
        broader_monitoring_feed=extract_int(
            text,
            r"Broader monitoring feed .*?: (\d[\d,]*) deduplicated events",
            "broader monitoring feed",
        ),
        observation_annotations=int(observation_match.group(1).replace(",", "")),
        observation_included_rows=int(observation_match.group(2).replace(",", "")),
        observation_events=int(observation_match.group(3).replace(",", "")),
        benchmark_truth_rows=int(benchmark_match.group(1).replace(",", "")),
        benchmark_scored_sequences=int(benchmark_match.group(2).replace(",", "")),
    )


def validate_strobe_summary(summary: StrobeSummary) -> None:
    for field_name, expected_value in REQUIRED_STROBE_COUNTS.items():
        actual_value = getattr(summary, field_name)
        if actual_value != expected_value:
            raise ValueError(
                f"Unexpected {field_name}: expected {expected_value}, got {actual_value}"
            )


def validate_core_manuscript_numbers(text: str) -> None:
    missing = [
        label
        for label, pattern in CORE_MANUSCRIPT_PATTERNS.items()
        if re.search(pattern, text, re.MULTILINE) is None
    ]
    if missing:
        print(
            f"WARNING: {len(missing)} core manuscript marker(s) not found in current "
            f"phrasing: {', '.join(missing)}. Verify the underlying numbers manually "
            f"if the manuscript was restructured for a new submission target."
        )


def validate_doc_patterns(
    *,
    path: Path,
    required_patterns: dict[str, str],
) -> None:
    text = path.read_text(encoding="utf-8")
    for label, pattern in required_patterns.items():
        if re.search(pattern, text, re.MULTILINE) is None:
            raise ValueError(f"Missing {label} marker in {path}")


def validate_strobe_packet_docs() -> None:
    for path, patterns in STROBE_PACKET_PATTERNS.items():
        validate_doc_patterns(path=path, required_patterns=patterns)


def draw_box(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        linewidth=1.4,
        edgecolor="#334155",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + (width / 2),
        y + (height * 0.71),
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#0F172A",
    )
    ax.text(
        x + (width / 2),
        y + (height * 0.37),
        body,
        ha="center",
        va="center",
        fontsize=9.5,
        color="#1E293B",
        linespacing=1.35,
    )


def draw_arrow(
    ax: plt.Axes,
    *,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> None:
    arrow = FancyArrowPatch(
        (start_x, start_y),
        (end_x, end_y),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.4,
        color="#475569",
    )
    ax.add_patch(arrow)


def write_strobe_flow_figure(summary: StrobeSummary, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    columns = [
        {
            "x": 0.05,
            "header": "Denominator cohort",
            "color": "#DBEAFE",
            "boxes": [
                (
                    "Hourly monitors",
                    f"{summary.hourly_monitors:,}\nin source window",
                ),
                (
                    "Eligible monitors",
                    (
                        f"{summary.passed_eligibility:,}\n"
                        f"Intervention: {summary.intervention_eligible:,}\n"
                        f"Control: {summary.control_eligible:,}"
                    ),
                ),
            ],
        },
        {
            "x": 0.37,
            "header": "Inferential event base",
            "color": "#FEF3C7",
            "boxes": [
                (
                    "Study-window adjudicated events",
                    f"{summary.study_window_events:,}\nmatched to monitoring pipeline",
                ),
                (
                    "Linked to eligible analysis base",
                    f"{summary.linked_events:,}\nprimary Poisson model events",
                ),
                (
                    "Alarm-time hard labels",
                    (
                        f"Chair: {summary.chair_hard_labels:,}\n"
                        f"Bed: {summary.bed_hard_labels:,}\n"
                        f"Room/no_patient: {summary.room_or_no_patient_hard_labels:,}"
                    ),
                ),
            ],
        },
        {
            "x": 0.69,
            "header": "Descriptive cohorts",
            "color": "#DCFCE7",
            "boxes": [
                (
                    "Broader monitoring feed",
                    f"{summary.broader_monitoring_feed:,}\ndeduplicated events",
                ),
                (
                    "Observation cohort",
                    (
                        f"{summary.observation_annotations:,} annotations\n"
                        f"{summary.observation_included_rows:,} included rows\n"
                        f"{summary.observation_events:,} events"
                    ),
                ),
                (
                    "Departure-aware benchmark",
                    (
                        f"{summary.benchmark_truth_rows:,} truth rows\n"
                        f"{summary.benchmark_scored_sequences:,} scored sequences"
                    ),
                ),
            ],
        },
    ]

    box_width = 0.24
    box_height = 0.16
    y_positions = [0.70, 0.44, 0.18]

    for column in columns:
        ax.text(
            column["x"] + (box_width / 2),
            0.90,
            column["header"],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#0F172A",
        )
        for index, (title, body) in enumerate(column["boxes"]):
            y = y_positions[index]
            draw_box(
                ax,
                x=column["x"],
                y=y,
                width=box_width,
                height=box_height,
                title=title,
                body=body,
                facecolor=column["color"],
            )
            if index < len(column["boxes"]) - 1:
                draw_arrow(
                    ax,
                    start_x=column["x"] + (box_width / 2),
                    start_y=y - 0.01,
                    end_x=column["x"] + (box_width / 2),
                    end_y=y_positions[index + 1] + box_height + 0.01,
                )

    note = (
        "Only the 40-event inferential base contributes to the adjusted chair-vs-bed rate ratio. "
        "The 91-event monitoring feed, 32-event observation cohort, and 30/31 benchmark subset "
        "are descriptive only."
    )
    ax.text(
        0.05,
        0.06,
        note,
        ha="left",
        va="center",
        fontsize=9,
        color="#334155",
        wrap=True,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def refresh_seed_manuscript() -> None:
    if not SEED_DOCX_PATH.exists():
        raise FileNotFoundError(f"Seed DOCX not found: {SEED_DOCX_PATH}")
    require_tool("pandoc")
    SEED_MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
    run(
        [
            "pandoc",
            SEED_DOCX_PATH,
            "--from=docx",
            "--to=markdown+pipe_tables+grid_tables+raw_html",
            "--wrap=preserve",
            f"--extract-media={SEED_MEDIA_ROOT}",
            "--output",
            SEED_PATH,
        ],
        cwd=PROJECT_ROOT,
    )


def maybe_refresh_figures(run_id: str, regenerate_figures: bool) -> None:
    missing_sources = [
        source_path for _, source_path in DEFAULT_FIGURE_MAP if not source_path.exists()
    ]
    if not regenerate_figures and not missing_sources:
        return

    run(
        [
            sys.executable,
            PROJECT_ROOT / "scripts" / "generate_figures.py",
            "--run-id",
            run_id,
            "--figs",
            *DEFAULT_GENERATED_FIGURE_IDS,
        ],
        cwd=PROJECT_ROOT,
    )

    still_missing = [
        source_path for _, source_path in DEFAULT_FIGURE_MAP if not source_path.exists()
    ]
    if still_missing:
        missing_str = ", ".join(str(path) for path in still_missing)
        raise FileNotFoundError(f"Figure generation did not create: {missing_str}")


def stage_figure_assets(summary: StrobeSummary) -> None:
    ASSETS_ROOT.mkdir(parents=True, exist_ok=True)
    ARXIV_ASSETS_ROOT.mkdir(parents=True, exist_ok=True)

    strobe_png = ASSETS_ROOT / "figure1_strobe_flow.png"
    write_strobe_flow_figure(summary, strobe_png)
    shutil.copy2(strobe_png, ARXIV_ASSETS_ROOT / strobe_png.name)

    for output_name, source_path in DEFAULT_FIGURE_MAP:
        target_path = ASSETS_ROOT / output_name
        shutil.copy2(source_path, target_path)
        shutil.copy2(source_path, ARXIV_ASSETS_ROOT / output_name)


def render_arxiv_tex() -> None:
    require_tool("pandoc")
    ARXIV_ROOT.mkdir(parents=True, exist_ok=True)
    run(
        [
            "pandoc",
            MANUSCRIPT_PATH.name,
            "--standalone",
            "--from=markdown+smart",
            "--to=latex",
            "--citeproc",
            "--metadata-file",
            METADATA_PATH.name,
            "--resource-path",
            ".",
            "--wrap=preserve",
            "--output",
            ARXIV_TEX_PATH.relative_to(PAPER_ROOT),
        ],
        cwd=PAPER_ROOT,
    )


def render_peer_docx() -> None:
    require_tool("pandoc")
    PEER_ROOT.mkdir(parents=True, exist_ok=True)
    cmd: list[str | Path] = [
        "pandoc",
        MANUSCRIPT_PATH.name,
        "--standalone",
        "--from=markdown+smart",
        "--to=docx",
        "--citeproc",
        "--metadata-file",
        METADATA_PATH.name,
        "--resource-path",
        ".",
        "--wrap=preserve",
        "--output",
        PEER_DOCX_PATH.relative_to(PAPER_ROOT),
    ]
    if REFERENCE_DOC_PATH.exists():
        cmd.extend(["--reference-doc", REFERENCE_DOC_PATH.name])
    run(cmd, cwd=PAPER_ROOT)


def verify_latex_bundle(*, emit_pdf: bool) -> None:
    require_tool("pdflatex")
    with tempfile.TemporaryDirectory(prefix="chair-fall-paper-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        temp_arxiv_root = temp_dir / "arxiv"
        shutil.copytree(ARXIV_ROOT, temp_arxiv_root)
        latex_cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            ARXIV_TEX_PATH.name,
        ]
        run(latex_cmd, cwd=temp_arxiv_root)
        run(latex_cmd, cwd=temp_arxiv_root)
        if emit_pdf:
            shutil.copy2(temp_arxiv_root / ARXIV_PDF_PATH.name, ARXIV_PDF_PATH)


def build_outputs(
    *,
    run_id: str,
    refresh_seed: bool,
    regenerate_figures: bool,
    verify_tex: bool,
    emit_pdf: bool,
) -> None:
    assert_manuscript_packet_artifacts(PROJECT_ROOT, run_id)

    if refresh_seed:
        refresh_seed_manuscript()

    manuscript_text = MANUSCRIPT_PATH.read_text(encoding="utf-8")
    validate_core_manuscript_numbers(manuscript_text)

    summary = parse_strobe_flow_summary(STROBE_FLOW_PATH)
    validate_strobe_summary(summary)
    validate_strobe_packet_docs()

    maybe_refresh_figures(run_id, regenerate_figures)
    stage_figure_assets(summary)
    render_arxiv_tex()
    render_peer_docx()
    if verify_tex:
        verify_latex_bundle(emit_pdf=emit_pdf)

    print(f"Generated {ARXIV_TEX_PATH}")
    print(f"Generated {PEER_DOCX_PATH}")
    if emit_pdf:
        print(f"Generated {ARXIV_PDF_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the dual-track arXiv and peer-edit manuscript bundle."
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="Run ID used when figures need to be regenerated.",
    )
    parser.add_argument(
        "--refresh-seed",
        action="store_true",
        help="Refresh paper/manuscript.seed.md from the local DOCX seed before building.",
    )
    parser.add_argument(
        "--regenerate-figures",
        action="store_true",
        help="Regenerate staged manuscript figures from run-scoped inputs before staging assets.",
    )
    parser.add_argument(
        "--skip-tex-verify",
        action="store_true",
        help="Skip the local pdflatex verification pass.",
    )
    parser.add_argument(
        "--emit-pdf",
        action="store_true",
        help="Copy the verified PDF back into paper/arxiv/manuscript.pdf.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_outputs(
        run_id=args.run_id,
        refresh_seed=args.refresh_seed,
        regenerate_figures=args.regenerate_figures,
        verify_tex=not args.skip_tex_verify,
        emit_pdf=args.emit_pdf,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
