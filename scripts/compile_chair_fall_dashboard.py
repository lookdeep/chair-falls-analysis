from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import pandas as pd

from ld_chair_falls.paper_defaults import (
    assert_manuscript_packet_artifacts,
    resolve_manuscript_artifact,
)
from ld_chair_falls.report_descriptive import resolve_chair_bed_inference_run_id

REFERENCE_MANUSCRIPT_TITLE = "Exposure-Normalized Bed and Chair Fall Rates via Continuous AI Monitoring"
REFERENCE_MANUSCRIPT_SOURCE = "paper/manuscript.md"
REFERENCE_PUBLIC_DATA_DIR = "data/public/"
REFERENCE_AUTHORS = (
    "Paolo Gabriel, Peter Rehani, Zack Drumm, Tyler Troy, Tiffany Wyatt, Narinder Singh"
)

MANUSCRIPT_SUMMARY_HTML = (
    "This dashboard mirrors the canonical manuscript source and the curated public aggregate-data "
    "bundle rather than the fuller internal QA packet. It keeps only the cohort counts, rate "
    "estimates, sensitivity rows, mechanism summaries, and governance statements reported in the "
    "March 2026 public release."
)

KEY_RESULTS = [
    ("Chair rate", "17.8", "falls per 1,000 chair exposure-hours"),
    ("Bed rate", "4.3", "falls per 1,000 bed exposure-hours"),
    ("Adjusted RR", "2.35", "95% CI 0.87 to 6.33; p=0.0907"),
    ("Study-window events", "43 / 40", "43 matched to pipeline; 40 linked to eligible hours"),
    ("Mechanism cohort", "32", "deduplicated broader observation events"),
    ("Direct chair signal", "6 of 7", "direct chair falls tagged footrest/positioning"),
]

COHORT_SUMMARY_ROWS = [
    ("Health system", "[Regional Health System] (analyzed at division level)"),
    ("Study period", "August 2024 - December 2025"),
    ("Total monitors (hourly data)", "5,531"),
    ("Total monitors (cohort map)", "5,570"),
    ("Eligible monitors (both gates)", "3,980 / 5,531 (72.0%)"),
    ("Intervention-eligible", "42"),
    ("Control-eligible", "3,938"),
    ("Ineligible monitors", "1,551"),
    ("Intervention-ineligible", "5"),
    ("Control-ineligible", "1,546"),
    ("Total hourly exposure rows (valid)", "356,391"),
    ("Analysis base rows (after eligibility)", "292,914"),
    ("Study-window adjudicated events matched to pipeline", "43"),
    ("Inferential fall events linked to analysis base", "40"),
    ("Broader monitoring feed (2022-2026, descriptive only)", "91"),
    ("Broader observation cohort (descriptive mechanism coding)", "32 deduplicated events"),
    ("Departure-aware benchmark subset", "30 truth rows / 31 scored sequences"),
    ("Eligibility gates applied", "min_observed_hours=4; min_coverage_ratio=0.95"),
    ("Primary model", "Poisson GLM; offset=log(exposure_hours)"),
]

UNADJUSTED_RATE_ROWS = [
    ("Chair", "5", "320.51", "15.6", "17.8"),
    ("Bed", "23", "5,121.42", "4.5", "4.3"),
]

TABLE3_SPECS = [
    {
        "label": "Primary (all hours)",
        "source": "adjusted",
        "key": "primary_adjusted",
        "fallback": {
            "rr": "2.35",
            "ci": "0.87 to 6.33",
            "p": "0.0907",
            "events": "40",
            "notes": "estimable",
        },
    },
    {
        "label": "Primary (HC3)",
        "source": "adjusted",
        "key": "primary_adjusted_hc3",
        "fallback": {
            "rr": "2.35",
            "ci": "0.93 to 5.94",
            "p": "0.0709",
            "events": "40",
            "notes": "HC3 SE",
        },
    },
    {
        "label": "Exploratory clustered SE (division)",
        "source": "adjusted",
        "key": "primary_adjusted_clustered",
        "fallback": {
            "rr": "2.35",
            "ci": "1.89 to 2.92",
            "p": "<0.0001",
            "events": "40",
            "notes": "9 intervention divisions; not a primary estimate",
        },
    },
    {
        "label": "Position-certainty hours only",
        "source": "adjusted",
        "key": "position_certain_only",
        "fallback": {
            "rr": "NE",
            "ci": "NE",
            "p": "—",
            "events": "38",
            "notes": "insufficient_data",
        },
    },
    {
        "label": "Chair-origin reclassified",
        "source": "adjusted",
        "key": "furniture_origin_reclassified",
        "fallback": {
            "rr": "2.35",
            "ci": "0.87 to 6.33",
            "p": "0.0907",
            "events": "40",
            "notes": "furniture-origin sensitivity",
        },
    },
    {
        "label": "Eligibility threshold >= 12 h",
        "source": "threshold",
        "key": 12,
        "fallback": {
            "rr": "2.39",
            "ci": "0.88 to 6.47",
            "p": "0.0859",
            "events": "38",
            "notes": "estimable",
        },
    },
    {
        "label": "Eligibility threshold >= 24 h",
        "source": "threshold",
        "key": 24,
        "fallback": {
            "rr": "2.43",
            "ci": "0.89 to 6.63",
            "p": "0.0831",
            "events": "36",
            "notes": "estimable",
        },
    },
    {
        "label": "Eligibility threshold >= 48 h",
        "source": "threshold",
        "key": 48,
        "fallback": {
            "rr": "2.42",
            "ci": "0.84 to 6.98",
            "p": "0.1008",
            "events": "30",
            "notes": "estimable",
        },
    },
]

MECHANISM_ROWS = [
    ("Other / unclassified", "18"),
    ("Transfer failure", "8"),
    ("Footrest / positioning", "6"),
]

FURNITURE_ORIGIN_ROWS = [
    ("Direct chair falls", "6"),
    ("Chair-origin room falls", "3"),
    ("Direct bed falls", "11"),
    ("Bed-origin room falls", "10"),
]

POST_DEPARTURE_ROWS = [
    ("Bed-origin events", "18 s median; IQR 7.0-56.5; max 136; n=10"),
    ("Chair-origin events", "50 s median; max 150; n=3"),
]

HARD_LABEL_ROWS = [
    ("Bed", "48", "52.7%"),
    ("No patient", "23", "25.3%"),
    ("Room", "12", "13.2%"),
    ("Chair", "8", "8.8%"),
]

LABEL_METRIC_ROWS = [
    ("Macro F1", "0.528"),
    ("ECE (10-bin)", "0.450"),
    ("Detection F1", "0.846"),
    ("Detection precision", "1.000"),
    ("Detection recall", "0.733"),
    ("Latency MAE", "37.9 s"),
    ("Latency p50", "22 s"),
    ("Latency p90", "83 s"),
]

ANCILLARY_BENCHMARK_HTML = (
    "<p>The ancillary multimodal LLM benchmark remained descriptive only. Gemini 2.5 Flash "
    "showed the highest fall sensitivity (0.842) with lower specificity (0.750), while Gemini "
    "3.1 Pro Preview showed perfect specificity (1.000) but lower fall sensitivity (0.447). The "
    "paper treats this as supporting evidence that automated fall-position classification remains a "
    "material limitation rather than as a head-to-head replacement study.</p>"
)

DISCUSSION_HTML = (
    "<p>The manuscript frames the main contribution as denominator refinement rather than a "
    "definitive effect estimate. Chair time showed a higher descriptive rate than bed time (17.8 "
    "vs 4.3 per 1,000 exposure-hours), and the adjusted chair-vs-bed RR stayed above 1.0 at "
    "2.35, but the primary confidence interval crossed 1.0.</p>"
    "<p>Mechanism findings remain hypothesis-generating. In the broader observation cohort, 6 of 7 "
    "direct chair falls carried a footrest/positioning tag, which the paper interprets as a signal "
    "for safer chair setup testing rather than as evidence to reduce chair use.</p>"
)

ETHICS_HTML = (
    "<p>The data used in this retrospective study were collected from patients admitted to one of "
    "eleven hospital partners across three different states in the USA. The study and handling of "
    "data followed CHAI standards. Access was granted through a Business Associate Agreement "
    "(BAA) for monitoring patients at high risk of falls, and patients provided written informed "
    "consent for monitoring as part of standard inpatient care.</p>"
    "<p>To protect privacy, all video data were blurred prior to storage and no identifiable "
    "information is included in the manuscript. Face-blurred frames were used only for training "
    "purposes, and the outcomes of this analysis did not influence patient care or clinical "
    "outcomes.</p>"
)


def _read_csv_if_exists(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return value
    if isinstance(value, int):
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _format_number(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{decimals}f}"


def _format_p_value(value: object) -> str:
    number = _to_float(value)
    if number is not None:
        return f"{number:.4f}"
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return "N/A"
    return text


def _lookup_adjusted_row(adjusted_rr: pd.DataFrame | None, key: str) -> dict[str, str] | None:
    if adjusted_rr is None or adjusted_rr.empty or "sensitivity_label" not in adjusted_rr.columns:
        return None
    subset = adjusted_rr.loc[adjusted_rr["sensitivity_label"] == key]
    if subset.empty:
        return None
    row = subset.iloc[0]
    rr = _to_float(row.get("rr"))
    lo = _to_float(row.get("ci_lower"))
    hi = _to_float(row.get("ci_upper"))
    events = _to_float(row.get("n_events"))
    notes_raw = row.get("model_notes", "")
    notes = "" if pd.isna(notes_raw) else str(notes_raw).strip()
    if rr is None or lo is None or hi is None:
        return {
            "rr": "NE",
            "ci": "NE",
            "p": "—",
            "events": _format_number(events, 0),
            "notes": notes or "insufficient_data",
        }
    return {
        "rr": _format_number(rr, 2),
        "ci": f"{_format_number(lo, 2)} to {_format_number(hi, 2)}",
        "p": _format_p_value(row.get("p_value")),
        "events": _format_number(events, 0),
        "notes": notes or "estimable",
    }


def _lookup_threshold_row(threshold_df: pd.DataFrame | None, hours: int) -> dict[str, str] | None:
    if threshold_df is None or threshold_df.empty:
        return None
    threshold_col = None
    for candidate in ("threshold_hours", "duration_threshold_hours"):
        if candidate in threshold_df.columns:
            threshold_col = candidate
            break
    if threshold_col is None:
        return None
    numeric_hours = pd.to_numeric(threshold_df[threshold_col], errors="coerce")
    subset = threshold_df.loc[numeric_hours == hours]
    if subset.empty:
        return None
    row = subset.iloc[0]
    rr = _to_float(row.get("rr"))
    lo = _to_float(row.get("ci_lower"))
    hi = _to_float(row.get("ci_upper"))
    events = _to_float(row.get("n_events"))
    notes_raw = row.get("model_notes", "")
    notes = "" if pd.isna(notes_raw) else str(notes_raw).strip()
    if rr is None or lo is None or hi is None:
        return {
            "rr": "NE",
            "ci": "NE",
            "p": "—",
            "events": _format_number(events, 0),
            "notes": notes or "insufficient_data",
        }
    return {
        "rr": _format_number(rr, 2),
        "ci": f"{_format_number(lo, 2)} to {_format_number(hi, 2)}",
        "p": _format_p_value(row.get("p_value")),
        "events": _format_number(events, 0),
        "notes": notes or "estimable",
    }


def _resolve_table3_rows(project_root: Path, run_id: str) -> list[dict[str, str]]:
    adjusted_rr = _read_csv_if_exists(
        resolve_manuscript_artifact(project_root, "adjusted_rr", run_id, must_exist=False)
    )
    threshold_df = _read_csv_if_exists(
        resolve_manuscript_artifact(project_root, "threshold", run_id, must_exist=False)
    )
    rows: list[dict[str, str]] = []
    for spec in TABLE3_SPECS:
        row = None
        if spec["source"] == "adjusted":
            row = _lookup_adjusted_row(adjusted_rr, str(spec["key"]))
        if spec["source"] == "threshold":
            row = _lookup_threshold_row(threshold_df, int(spec["key"]))
        row = row or dict(spec["fallback"])
        rows.append(
            {
                "label": str(spec["label"]),
                "rr": str(row["rr"]),
                "ci": str(row["ci"]),
                "p": str(row["p"]),
                "events": str(row["events"]),
                "notes": str(row["notes"]),
            }
        )
    return rows


def _render_kpi_cards(cards: list[tuple[str, str, str]]) -> str:
    rendered = []
    for label, value, detail in cards:
        rendered.extend(
            [
                '      <div class="stat-card">',
                f'        <div class="stat-label">{escape(label)}</div>',
                f'        <div class="stat-value">{escape(value)}</div>',
                f'        <div class="stat-detail">{escape(detail)}</div>',
                "      </div>",
            ]
        )
    return "\n".join(rendered)


def _render_two_column_table(
    heading: str,
    caption: str,
    rows: list[tuple[str, ...]],
) -> str:
    rendered_rows = []
    for row in rows:
        label = row[0]
        value = " · ".join(row[1:])
        rendered_rows.append(
            f"          <tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>"
        )
    body = "\n".join(rendered_rows)
    return "\n".join(
        [
            '    <div class="card">',
            f"      <h3>{escape(heading)}</h3>",
            f'      <p class="section-note">{escape(caption)}</p>',
            '      <table class="tbl">',
            "        <tbody>",
            body,
            "        </tbody>",
            "      </table>",
            "    </div>",
        ]
    )


def _render_unadjusted_table() -> str:
    row_blocks = []
    for position, hard_events, exposure, hard_rate, expected_rate in UNADJUSTED_RATE_ROWS:
        row_blocks.append(
            "\n".join(
                [
                    "          <tr>",
                    f"            <td>{escape(position)}</td>",
                    f"            <td>{escape(hard_events)}</td>",
                    f"            <td>{escape(exposure)}</td>",
                    f"            <td>{escape(hard_rate)}</td>",
                    f"            <td>{escape(expected_rate)}</td>",
                    "          </tr>",
                ]
            )
        )
    body = "\n".join(row_blocks)
    return "\n".join(
        [
            "<section>",
            "  <h2>Unadjusted Rates</h2>",
            '  <div class="card">',
            "    <h3>Table 2. Unadjusted fall rates by patient position</h3>",
            '    <p class="section-note">Intervention-eligible units only; expected rates match the '
            "probability-weighted event allocation used in adjusted modeling.</p>",
            '    <table class="tbl">',
            "      <thead>",
            "        <tr>",
            "          <th>Position</th>",
            "          <th>Hard-label fall events</th>",
            "          <th>Exposure (hours)</th>",
            "          <th>Rate per 1,000 h</th>",
            "          <th>Expected rate</th>",
            "        </tr>",
            "      </thead>",
            "      <tbody>",
            body,
            "      </tbody>",
            "    </table>",
            '    <p class="table-note">Note: Expected rates are probability-weighted descriptive rates '
            "derived from pre-fall location probabilities and align with the adjusted-model event "
            "allocation. Hard-label counts reflect AI-assigned position at event time.</p>",
            "  </div>",
            "</section>",
        ]
    )


def _render_table3(project_root: Path, run_id: str) -> str:
    rows = _resolve_table3_rows(project_root, run_id)
    row_blocks = []
    for row in rows:
        row_blocks.append(
            "\n".join(
                [
                    "          <tr>",
                    f"            <td>{escape(row['label'])}</td>",
                    f"            <td>{escape(row['rr'])}</td>",
                    f"            <td>{escape(row['ci'])}</td>",
                    f"            <td>{escape(row['p'])}</td>",
                    f"            <td>{escape(row['events'])}</td>",
                    f"            <td>{escape(row['notes'])}</td>",
                    "          </tr>",
                ]
            )
        )
    body = "\n".join(row_blocks)
    return "\n".join(
        [
            "<section>",
            "  <h2>Adjusted Modeling Summary</h2>",
            '  <div class="card">',
            "    <h3>Table 3. Adjusted rate ratios and sensitivity analyses</h3>",
            '    <table class="tbl">',
            "      <thead>",
            "        <tr>",
            "          <th>Analysis</th>",
            "          <th>RR</th>",
            "          <th>95% CI</th>",
            "          <th>p-value</th>",
            "          <th>Events (n)</th>",
            "          <th>Notes</th>",
            "        </tr>",
            "      </thead>",
            "      <tbody>",
            body,
            "      </tbody>",
            "    </table>",
            '    <p class="table-note">Note: The clustered row is exploratory only because it uses nine '
            "intervention divisions within one health system and sits within an underdispersed Poisson "
            "fit. All seven pre-specified time-window restriction analyses remained non-estimable in "
            "this run and were omitted from the manuscript table for readability.</p>",
            "  </div>",
            "</section>",
        ]
    )


def build_manuscript(project_root: Path, run_id: str, output_path: Path) -> Path:
    css = "\n".join(
        [
            "body { font-family: Georgia, serif; margin: 1.75rem auto; max-width: 1120px; "
            "line-height: 1.55; color: #1f2933; background: #fcfcfa; }",
            "h1, h2, h3 { margin-bottom: 0.35rem; }",
            "p { margin-top: 0.25rem; }",
            "section { margin-top: 1.35rem; }",
            "code { background: #f1f5f9; padding: 0.1rem 0.3rem; border-radius: 0.2rem; }",
            ".title-block { margin-bottom: 1.2rem; }",
            ".authors { font-size: 1.02rem; margin-top: 0.2rem; }",
            ".pub-date { color: #475569; font-size: 0.95rem; }",
            ".callout { background: #f5f8fb; border-left: 4px solid #315b7c; padding: 0.85rem 1rem; "
            "border-radius: 0.25rem; }",
            ".stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); "
            "gap: 0.75rem; margin-top: 0.9rem; }",
            ".two-col { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); "
            "gap: 0.9rem; }",
            ".card { border: 1px solid #d9e2ec; border-radius: 0.45rem; background: #ffffff; "
            "padding: 0.85rem 1rem; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); }",
            ".stat-card { border: 1px solid #dbe5f0; border-radius: 0.45rem; background: #ffffff; "
            "padding: 0.75rem 0.85rem; }",
            ".stat-label { color: #52606d; font-size: 0.82rem; text-transform: uppercase; "
            "letter-spacing: 0.04em; }",
            ".stat-value { font-size: 1.85rem; font-weight: 700; color: #17324d; margin-top: 0.15rem; }",
            ".stat-detail { color: #52606d; font-size: 0.9rem; margin-top: 0.2rem; }",
            ".section-note { color: #52606d; font-size: 0.94rem; }",
            ".table-note { color: #52606d; font-size: 0.9rem; margin-top: 0.7rem; }",
            ".tbl { border-collapse: collapse; width: 100%; margin-top: 0.55rem; }",
            ".tbl th, .tbl td { border: 1px solid #d9e2ec; padding: 0.45rem 0.55rem; text-align: left; "
            "vertical-align: top; }",
            ".tbl th { background: #eff4f8; }",
            ".manuscript-footer { margin-top: 1.4rem; color: #475569; font-size: 0.9rem; }",
        ]
    )

    html = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8" />',
            f"  <title>Chair Falls Risk Dashboard - {escape(run_id)}</title>",
            "  <style>",
            css,
            "  </style>",
            "</head>",
            "<body>",
            "<!--",
            f"Sync source: {REFERENCE_MANUSCRIPT_SOURCE} and {REFERENCE_PUBLIC_DATA_DIR}.",
            "Keep this dashboard aligned to the public manuscript and omit internal-only diagnostics",
            "that were not reported in the release bundle.",
            "-->",
            '  <div class="title-block">',
            "    <h1>Chair Falls Risk Dashboard</h1>",
            f"    <p class=\"authors\"><strong>Reference manuscript:</strong> "
            f"{escape(REFERENCE_MANUSCRIPT_TITLE)}</p>",
            f"    <p class=\"authors\">{escape(REFERENCE_AUTHORS)}</p>",
            "    <p class=\"pub-date\">March 2026 &middot; Public release bundle "
            f"(<code>{escape(REFERENCE_PUBLIC_DATA_DIR)}</code>) &middot; run_id: "
            f"{escape(run_id)}</p>",
            "  </div>",
            "<section>",
            "  <h2>Reference Context</h2>",
            '  <div class="callout">',
            f"    <p>{MANUSCRIPT_SUMMARY_HTML}</p>",
            "    <p><strong>Reference manuscript:</strong> "
            f"<code>{escape(REFERENCE_MANUSCRIPT_SOURCE)}</code></p>",
            "  </div>",
            '  <div class="stats-grid">',
            _render_kpi_cards(KEY_RESULTS),
            "  </div>",
            "</section>",
            "<section>",
            "  <h2>Cohort Summary</h2>",
            '  <div class="card">',
            "    <h3>Table 1. Cohort summary statistics</h3>",
            '    <table class="tbl">',
            "      <tbody>",
            *[
                f"        <tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>"
                for label, value in COHORT_SUMMARY_ROWS
            ],
            "      </tbody>",
            "    </table>",
            "  </div>",
            "</section>",
            _render_unadjusted_table(),
            _render_table3(project_root, run_id),
            "<section>",
            "  <h2>Observation Cohort</h2>",
            '  <div class="two-col">',
            _render_two_column_table(
                "Mechanism taxonomy",
                "Broader observation cohort; descriptive only, not denominator-linked.",
                MECHANISM_ROWS,
            ),
            _render_two_column_table(
                "Furniture-origin chain",
                "Including chair-origin room falls expands chair-associated events from 6 to 9.",
                FURNITURE_ORIGIN_ROWS,
            ),
            "  </div>",
            '  <div class="two-col" style="margin-top: 0.9rem;">',
            _render_two_column_table(
                "Post-departure latency",
                "Time from furniture departure to fall for events with evaluable timestamps.",
                POST_DEPARTURE_ROWS,
            ),
            '    <div class="card">',
            "      <h3>Exit concordance</h3>",
            '      <p class="section-note">Among consensus events with valid furniture departure '
            "timestamps, no events met evaluability criteria for either AI departure signal after "
            "linkage and filtering. Bias and MAE therefore remained not estimable in the submitted "
            "manuscript.</p>",
            "      <p><strong>Direct chair mechanism signal:</strong> 6 of 7 direct chair falls "
            "carried a footrest/positioning tag.</p>",
            "    </div>",
            "  </div>",
            "</section>",
            "<section>",
            "  <h2>Additional Descriptive Results</h2>",
            '  <div class="two-col">',
            _render_two_column_table(
                "Hard-label position distribution at alarm",
                "Broader monitoring feed (2022-2026; n=91 deduplicated events).",
                HARD_LABEL_ROWS,
            ),
            _render_two_column_table(
                "Label evaluation metrics",
                "v3 adjudicated benchmark subset (30 truth rows / 31 scored sequences).",
                LABEL_METRIC_ROWS,
            ),
            "  </div>",
            '  <div class="card" style="margin-top: 0.9rem;">',
            "    <h3>Ancillary multimodal LLM benchmark</h3>",
            f"    {ANCILLARY_BENCHMARK_HTML}",
            "  </div>",
            "</section>",
            "<section>",
            "  <h2>Interpretation</h2>",
            '  <div class="card">',
            f"    {DISCUSSION_HTML}",
            "  </div>",
            "</section>",
            "<section>",
            "  <h2>Ethics and Data Governance</h2>",
            '  <div class="card">',
            f"    {ETHICS_HTML}",
            "  </div>",
            "</section>",
            '  <div class="manuscript-footer">',
            "    Public-release summary dashboard &mdash; "
            f"run_id: {escape(run_id)} &middot; Reference authors: P. Gabriel, P. Rehani, "
            "Z. Drumm, T. Troy, T. Wyatt, N. Singh",
            "  </div>",
            "</body>",
            "</html>",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile docs/chair_fall_dashboard.html as a manuscript-aligned dashboard for the "
            "public release bundle."
        )
    )
    parser.add_argument("--run-id", default="", help="Run ID to compile. Defaults to latest compatible run.")
    parser.add_argument(
        "--output",
        default="docs/chair_fall_dashboard.html",
        help="Output dashboard path (default: docs/chair_fall_dashboard.html).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_id = args.run_id.strip() or resolve_chair_bed_inference_run_id(project_root)
    assert_manuscript_packet_artifacts(project_root, run_id)
    output = Path(args.output)
    if not output.is_absolute():
        output = project_root / output

    result = build_manuscript(project_root, run_id, output)
    print(result)


if __name__ == "__main__":
    main()
