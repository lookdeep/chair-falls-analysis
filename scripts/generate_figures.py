#!/usr/bin/env python3
"""
Chair Falls Risk - Manuscript Figures
Generates manuscript-facing figures as SVG and PNG from run-specific CSV inputs.

Default generation excludes the retired chair-only donut so the script matches the
current manuscript figure set. The legacy donut remains available as opt-in
figure ``3`` for ad hoc exploratory use.

Usage:
    python scripts/generate_figures.py --run-id <RUN_ID> [--figs 1 2 4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.paper_defaults import resolve_manuscript_artifact  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Shared style
BLUE = "#2563EB"
SLATE = "#64748B"
AMBER = "#F59E0B"
GREEN = "#22C55E"
GRAY = "#94A3B8"
BG = "#FFFFFF"
TEXT = "#1E293B"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "text.color": TEXT,
    }
)


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def save_figure(fig: plt.Figure, filename_stem: str) -> None:
    for suffix, fmt in (("svg", "svg"), ("png", "png")):
        out_path = OUT / f"{filename_stem}.{suffix}"
        fig.savefig(out_path, format=fmt, bbox_inches="tight", dpi=300 if fmt == "png" else None)
        print(f"Saved {out_path}")


def discover_csvs(run_id: str, project_root: Path) -> dict[str, Path]:
    csv_map = {
        "risk_rates": resolve_manuscript_artifact(project_root, "risk_rates", run_id),
        "adjusted_rr": resolve_manuscript_artifact(project_root, "adjusted_rr", run_id),
        "misclassification": resolve_manuscript_artifact(project_root, "misclassification", run_id),
        "threshold": resolve_manuscript_artifact(project_root, "threshold", run_id),
        "daypart_breakdown": resolve_manuscript_artifact(project_root, "daypart_breakdown", run_id),
        "mechanism_taxonomy": resolve_manuscript_artifact(project_root, "mechanism_taxonomy", run_id),
        "operational_event_rates": resolve_manuscript_artifact(project_root, "operational_event_rates", run_id),
    }

    return csv_map


def load_all(csv_map: dict[str, Path]) -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    for key, path in csv_map.items():
        loaded[key] = pd.read_csv(path)
    return loaded


def humanize_location(location: str) -> str:
    mapping = {
        "chair": "Chair",
        "bed": "Bed",
        "room": "Room",
        "no_patient": "No patient",
    }
    return mapping.get(location, location.replace("_", " ").title())


def humanize_mechanism(mechanism: str) -> str:
    mapping = {
        "footrest_positioning": "Footrest / positioning",
        "transfer_failure": "Transfer failure",
        "other": "Other",
    }
    return mapping.get(mechanism, mechanism.replace("_", " ").title())


def humanize_metric(metric: str) -> str:
    mapping = {
        "alarms": "Alarm Hits",
        "nudges": "Nudges",
        "announcements": "Announcements",
        "talk_event": "Talk Clicks",
        "alarm_trigger": "Alarm Triggers",
        "nudge_active_seconds": "Active Nudge Seconds",
        "safety_zone_onset": "Safety-Zone Onsets",
    }
    return mapping.get(metric, metric.replace("_", " ").title())


def require_single_row(df: pd.DataFrame, filt: pd.Series, label: str) -> pd.Series:
    subset = df.loc[filt]
    if subset.empty:
        fail(f"Expected row not found for {label}")
    return subset.iloc[0]


# Figure 1 - Unadjusted Fall Rate: Chair vs Bed
def make_fig1(risk_rates_df: pd.DataFrame) -> None:
    scoped = risk_rates_df[risk_rates_df["scope"] == "intervention_eligible"].copy()
    if scoped.empty:
        fail("Figure 1 source has no rows where scope == intervention_eligible")

    chair = require_single_row(scoped, scoped["position"] == "chair", "Figure 1 chair row")
    bed = require_single_row(scoped, scoped["position"] == "bed", "Figure 1 bed row")

    categories = ["Chair", "Bed"]
    rates = [
        float(chair["rate_per_1000_exposure_hours_expected"]),
        float(bed["rate_per_1000_exposure_hours_expected"]),
    ]
    expected_falls = [float(chair["expected_falls"]), float(bed["expected_falls"])]
    exposure_hours = [float(chair["exposure_hours"]), float(bed["exposure_hours"])]
    rr = rates[0] / rates[1]

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    bars = ax.bar(
        categories,
        rates,
        color=[BLUE, SLATE],
        width=0.5,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )

    for bar, rate, falls, exposure in zip(
        bars, rates, expected_falls, exposure_hours, strict=True
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{rate:.1f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 0.5,
            f"expected={falls:.1f}\n({exposure:.2f} h)",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )

    y_top = max(rates) * 1.3
    ax.set_ylim(0, y_top)
    ax.set_ylabel("Falls per 1,000 exposure-hours")
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_title("Unadjusted Fall Rate by Posture", fontsize=13, fontweight="bold", loc="left")
    ax.text(
        0.5,
        y_top * 0.95,
        f"Unadjusted RR (Chair/Bed) = {rr:.2f}x",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=AMBER,
    )

    fig.tight_layout()
    save_figure(fig, "fig1_fall_rates")
    plt.close(fig)
    print("Fig 1 done")


# Figure 2 - Comprehensive Sensitivity Forest Plot
def make_fig2(
    adjusted_rr_df: pd.DataFrame,
    misclassification_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
) -> None:
    model_specs = [
        ("primary_adjusted", "Primary adjusted (default SE)", True),
        ("primary_adjusted_hc3", "Primary adjusted (HC3)", False),
        ("primary_adjusted_clustered", "Primary adjusted (clustered)", False),
        ("position_certain_only", "Position-certain only", False),
        ("furniture_origin_reclassified", "Furniture-origin reclassified", False),
    ]
    misclassification_specs = [
        ("symmetric_swap_10pct", "10% symmetric swap"),
        ("symmetric_swap_20pct", "20% symmetric swap"),
        ("symmetric_swap_30pct", "30% symmetric swap"),
        ("chair_to_bed_20pct", "20% chair->bed only"),
        ("bed_to_chair_20pct", "20% bed->chair only"),
    ]
    threshold_specs = [
        (4, ">=4h (primary)"),
        (12, ">=12h"),
        (24, ">=24h"),
        (48, ">=48h"),
    ]

    rows: list[dict[str, object]] = [{"type": "header", "label": "Model Specification"}]

    for label, display, is_primary in model_specs:
        row = require_single_row(
            adjusted_rr_df,
            adjusted_rr_df["sensitivity_label"] == label,
            f"Figure 2 model spec {label}",
        )
        rr = row["rr"]
        lo = row["ci_lower"]
        hi = row["ci_upper"]
        if pd.isna(rr) or pd.isna(lo) or pd.isna(hi):
            rows.append(
                {
                    "type": "data",
                    "label": display,
                    "rr": None,
                    "lo": None,
                    "hi": None,
                    "is_primary": bool(is_primary),
                    "note": str(row.get("model_notes", "") or "insufficient_data"),
                }
            )
            continue
        rows.append(
            {
                "type": "data",
                "label": display,
                "rr": float(rr),
                "lo": float(lo),
                "hi": float(hi),
                "is_primary": bool(is_primary),
                "note": "",
            }
        )

    rows.append({"type": "header", "label": "Misclassification Sensitivity"})
    for scenario, display in misclassification_specs:
        row = require_single_row(
            misclassification_df,
            misclassification_df["scenario_label"] == scenario,
            f"Figure 2 misclassification {scenario}",
        )
        rows.append(
            {
                "type": "data",
                "label": display,
                "rr": float(row["rr"]),
                "lo": float(row["ci_lower"]),
                "hi": float(row["ci_upper"]),
                "is_primary": False,
                "note": "",
            }
        )

    rows.append({"type": "header", "label": "Eligibility Threshold"})
    for hours, display in threshold_specs:
        row = require_single_row(
            threshold_df,
            threshold_df["threshold_hours"] == hours,
            f"Figure 2 threshold {hours}",
        )
        rows.append(
            {
                "type": "data",
                "label": display,
                "rr": float(row["rr"]),
                "lo": float(row["ci_lower"]),
                "hi": float(row["ci_upper"]),
                "is_primary": False,
                "note": "",
            }
        )

    fig, ax = plt.subplots(figsize=(10, 9))
    y_positions = list(reversed(range(len(rows))))
    label_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    for idx, row in enumerate(rows):
        y = y_positions[idx]
        if row["type"] == "header":
            ax.text(
                -0.02,
                y,
                str(row["label"]),
                transform=label_transform,
                ha="right",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=TEXT,
            )
            ax.hlines(y - 0.5, xmin=0.5, xmax=30, color="#E2E8F0", linewidth=1.0, zorder=1)
            continue

        rr = row["rr"]
        lo = row["lo"]
        hi = row["hi"]
        is_primary = bool(row["is_primary"])
        color = BLUE if is_primary else SLATE

        ax.text(
            -0.02,
            y,
            str(row["label"]),
            transform=label_transform,
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold" if is_primary else "normal",
        )

        if rr is not None and lo is not None and hi is not None:
            marker = "D" if is_primary else "o"
            marker_size = 9 if is_primary else 7
            rr_value = float(rr)
            lo_value = float(lo)
            hi_value = float(hi)
            ax.plot(
                [lo_value, hi_value], [y, y], color=color, lw=2.0, solid_capstyle="round", zorder=3
            )
            ax.plot([lo_value, lo_value], [y - 0.15, y + 0.15], color=color, lw=2.0, zorder=3)
            ax.plot([hi_value, hi_value], [y - 0.15, y + 0.15], color=color, lw=2.0, zorder=3)
            ax.plot(
                rr_value,
                y,
                marker=marker,
                ms=marker_size,
                color=color,
                markeredgecolor="white",
                markeredgewidth=1.3,
                zorder=4,
            )
            summary = f"{rr_value:.2f} ({lo_value:.2f}, {hi_value:.2f})"
        else:
            summary = f"N/A ({row['note']})"

        ax.text(
            1.01,
            y,
            summary,
            transform=label_transform,
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold" if is_primary else "normal",
            color=color if rr is not None else GRAY,
        )
        ax.hlines(y, xmin=0.5, xmax=30, color="#F1F5F9", linewidth=0.8, zorder=1)

    ax.axvline(1.0, color=GRAY, linestyle="--", lw=1.2, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 30)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 30])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Rate Ratio (Chair / Bed)")
    ax.set_title(
        "Comprehensive Sensitivity Analysis", fontsize=13, fontweight="bold", loc="left"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    plt.subplots_adjust(left=0.42, right=0.77, top=0.93, bottom=0.08)

    save_figure(fig, "fig2_sensitivity_forest")
    plt.close(fig)
    print("Fig 2 done")


# Legacy Figure 3 - Chair Fall Mechanisms (retired from manuscript)
def make_fig3(mechanism_df: pd.DataFrame) -> None:
    chair_only = mechanism_df[mechanism_df["prefall_location"] == "chair"].copy()
    if chair_only.empty:
        fail("Figure 3 source has no rows where prefall_location == chair")

    counts = chair_only["charter_mechanism"].value_counts()
    total = int(counts.sum())
    labels = [humanize_mechanism(name) for name in counts.index]
    sizes = counts.to_numpy(dtype=float)

    color_map = {
        "footrest_positioning": BLUE,
        "transfer_failure": GREEN,
        "other": GRAY,
    }
    colors = [color_map.get(name, GRAY) for name in counts.index]

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.pie(
        sizes,
        labels=None,
        colors=colors,
        startangle=90,
        wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 2.0},
        explode=[0.03] * len(sizes),
    )

    ax.text(0, 0.08, f"{total}", ha="center", va="center", fontsize=22, fontweight="bold")
    ax.text(0, -0.20, "chair falls", ha="center", va="center", fontsize=9, color=GRAY)

    legend_labels = [
        f"{label} ({count / total * 100:.0f}%)"
        for label, count in zip(labels, counts.to_numpy(), strict=True)
    ]
    patches = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(colors, legend_labels, strict=True)
    ]
    ax.legend(
        handles=patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        fontsize=9,
        frameon=False,
        ncol=1,
    )
    ax.set_title(
        "Chair Fall Mechanisms (Observation Cohort)", fontsize=12, fontweight="bold", loc="center"
    )

    fig.tight_layout()
    save_figure(fig, "fig3_chair_mechanisms")
    plt.close(fig)
    print("Fig 3 done")


# Figure 4 - Mechanism Taxonomy by Pre-fall Posture
def make_fig4(mechanism_df: pd.DataFrame) -> None:
    location_order = ["chair", "bed", "room", "no_patient"]
    mechanism_order = ["footrest_positioning", "transfer_failure", "other"]
    mechanism_colors = {
        "footrest_positioning": BLUE,
        "transfer_failure": GREEN,
        "other": GRAY,
    }

    pivot = (
        mechanism_df.pivot_table(
            index="prefall_location",
            columns="charter_mechanism",
            values="event_key",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(index=location_order, fill_value=0)
        .reindex(columns=mechanism_order, fill_value=0)
    )

    totals = pivot.sum(axis=1).astype(int)
    x = np.arange(len(location_order))
    bottom = np.zeros(len(location_order), dtype=float)

    fig, ax = plt.subplots(figsize=(7.8, 4.9))

    for mechanism in mechanism_order:
        values = pivot[mechanism].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.56,
            color=mechanism_colors[mechanism],
            edgecolor="white",
            linewidth=1.0,
            label=humanize_mechanism(mechanism),
        )
        bottom += values

    labels = [f"{humanize_location(loc)}\n(n={totals[loc]})" for loc in location_order]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Number of events")
    ax.set_title(
        "Fall Mechanism Taxonomy by Pre-fall Posture", fontsize=13, fontweight="bold", loc="left"
    )
    y_max = max(1.0, float(bottom.max()) * 1.25)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for idx, total in enumerate(bottom):
        ax.text(idx, total + y_max * 0.02, f"n={int(total)}", ha="center", va="bottom", fontsize=9)

    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    save_figure(fig, "fig4_mechanism_by_posture")
    plt.close(fig)
    print("Fig 4 done")


# Generator Figure 5 (staged as manuscript Figure 6) - Daypart context figure
def make_fig5(daypart_df: pd.DataFrame) -> None:
    daypart_only = daypart_df[daypart_df["grouping"] == "daypart"].copy()
    if daypart_only.empty:
        fail("Figure 5 source has no rows where grouping == daypart")

    block_order = [
        "window_00_05",
        "window_06_08",
        "window_09_11",
        "window_12_14",
        "window_15_17",
        "window_18_20",
        "window_21_23",
    ]
    block_labels = {
        "window_00_05": "00:00-05:59",
        "window_06_08": "06:00-08:59",
        "window_09_11": "09:00-11:59",
        "window_12_14": "12:00-14:59",
        "window_15_17": "15:00-17:59",
        "window_18_20": "18:00-20:59",
        "window_21_23": "21:00-23:59",
    }
    location_order = ["chair", "bed", "room", "no_patient"]
    location_colors = {
        "chair": BLUE,
        "bed": SLATE,
        "room": GREEN,
        "no_patient": GRAY,
    }

    x = np.arange(len(block_order))
    bottom = np.zeros(len(block_order), dtype=float)

    expected_map: dict[tuple[str, str], float] = {}
    chair_probs: list[float] = []

    for block in block_order:
        block_rows = daypart_only[daypart_only["group_value"] == block]
        if block_rows.empty:
            fail(f"Figure 5 source has no rows for daypart block {block}")

        chair_row = require_single_row(
            block_rows,
            block_rows["prefall_location_label"] == "chair",
            f"Figure 5 chair row {block}",
        )
        chair_probs.append(float(chair_row["mean_probability"]))

        for location in location_order:
            row = require_single_row(
                block_rows,
                block_rows["prefall_location_label"] == location,
                f"Figure 5 {block} {location}",
            )
            expected_map[(block, location)] = float(row["expected_falls"])

    prob_min, prob_max = min(chair_probs), max(chair_probs)

    def prob_to_color(prob: float) -> tuple[float, float, float, float]:
        if prob_max == prob_min:
            return plt.cm.Blues(0.75)
        scale = (prob - prob_min) / (prob_max - prob_min)
        return plt.cm.Blues(0.45 + 0.45 * scale)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    for location in location_order:
        values = [expected_map[(block, location)] for block in block_order]
        if location == "chair":
            colors = [prob_to_color(prob) for prob in chair_probs]
        else:
            colors = [location_colors[location]] * len(values)

        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.58,
            color=colors,
            edgecolor="white",
            linewidth=1.0,
            label=humanize_location(location),
        )
        bottom += np.array(values, dtype=float)

    for idx, (_block, prob, total) in enumerate(zip(block_order, chair_probs, bottom, strict=True)):
        ax.text(
            idx,
            total + max(bottom) * 0.025,
            f"P(chair)={prob:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=prob_to_color(prob),
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([block_labels[block] for block in block_order], fontsize=10)
    ax.set_ylabel("Expected falls")
    ax.set_title(
        "Daypart Distribution of Expected Falls and Mean Chair Probability",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )
    ax.set_ylim(0, max(bottom) * 1.22)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    legend_handles = [
        mpatches.Patch(color=location_colors[loc], label=humanize_location(loc)) for loc in location_order
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout()
    save_figure(fig, "fig5_chair_occupancy_daypart")
    plt.close(fig)
    print("Fig 5 done")


# Generator Figure 6 (staged as manuscript Figure 5) - Public context schematic
def make_fig6() -> None:
    contexts = [
        {
            "label": "Bed-origin review context",
            "accent": SLATE,
            "fill": "#EFF6FF",
            "steps": (
                ("Observed in bed", "Patient supported\nin bed"),
                ("Exit / transfer", "Bed-exit or\ntransfer attempt"),
                ("Fall outcome", "Floor-level outcome\nand response"),
            ),
        },
        {
            "label": "Chair-origin review context",
            "accent": BLUE,
            "fill": "#DBEAFE",
            "steps": (
                ("Observed in chair", "Patient seated with\nchair setup in view"),
                ("Reposition / stand", "Chair movement,\nfootrest, or stand attempt"),
                ("Fall outcome", "Fall and response\nsequence"),
            ),
        },
        {
            "label": "Room fall after departure",
            "accent": GREEN,
            "fill": "#DCFCE7",
            "steps": (
                ("Last furniture", "Recent bed or chair\ndeparture"),
                ("Ambulation / transition", "Unsupported movement\nin room"),
                ("Fall outcome", "Room-level fall and\nstaff response"),
            ),
        },
    ]

    fig, ax = plt.subplots(figsize=(11.4, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_positions = [0.14, 0.42, 0.70]
    y_positions = [0.73, 0.45, 0.17]
    box_width = 0.20
    box_height = 0.14

    for y, context in zip(y_positions, contexts, strict=True):
        ax.text(
            0.02,
            y + (box_height / 2),
            context["label"],
            ha="left",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=TEXT,
        )
        for idx, ((title, body), x) in enumerate(zip(context["steps"], x_positions, strict=True)):
            patch = FancyBboxPatch(
                (x, y),
                box_width,
                box_height,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                linewidth=1.4,
                edgecolor=context["accent"],
                facecolor=context["fill"],
            )
            ax.add_patch(patch)
            ax.text(
                x + (box_width / 2),
                y + (box_height * 0.68),
                title,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )
            ax.text(
                x + (box_width / 2),
                y + (box_height * 0.31),
                body,
                ha="center",
                va="center",
                fontsize=9,
                linespacing=1.25,
            )
            if idx < len(x_positions) - 1:
                arrow = FancyArrowPatch(
                    (x + box_width + 0.015, y + (box_height / 2)),
                    (x_positions[idx + 1] - 0.015, y + (box_height / 2)),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=1.4,
                    color=context["accent"],
                )
                ax.add_patch(arrow)

    ax.text(
        0.02,
        0.96,
        "Public-release schematic for qualitative mechanism-review contexts",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.08,
        "This schematic replaces patient stills in the public repository. It preserves the review "
        "categories used during qualitative coding without publishing per-event imagery.",
        ha="left",
        va="center",
        fontsize=9.5,
        color=SLATE,
        wrap=True,
    )

    fig.tight_layout()
    save_figure(fig, "fig6_context_schematic")
    plt.close(fig)
    print("Fig 6 done")


def _make_fig7_legacy(operational_event_rates_df: pd.DataFrame) -> None:
    scoped = operational_event_rates_df[
        operational_event_rates_df["scope"] == "intervention_eligible"
    ].copy()
    if scoped.empty:
        fail("Figure 7 source has no rows where scope == intervention_eligible")

    metric_order = ["alarms", "nudges", "announcements"]
    positions = ["chair", "bed"]
    bar_colors = {"chair": BLUE, "bed": SLATE}

    fig, axes = plt.subplots(nrows=3, figsize=(8.0, 8.8))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, metric_name in zip(axes, metric_order, strict=True):
        metric_rows = scoped[scoped["metric"] == metric_name].copy()
        if metric_rows.empty:
            fail(f"Figure 7 source has no rows for metric {metric_name}")

        ordered_rows = [
            require_single_row(
                metric_rows,
                metric_rows["position"] == position,
                f"Figure 7 {metric_name} {position}",
            )
            for position in positions
        ]
        rates = [float(row["rate_per_100_exposure_hours"]) for row in ordered_rows]
        weighted_counts = [float(row["weighted_event_count"]) for row in ordered_rows]
        exposures = [float(row["exposure_hours"]) for row in ordered_rows]
        labels = [humanize_location(position) for position in positions]
        y = np.arange(len(labels))

        bars = ax.barh(
            y,
            rates,
            color=[bar_colors[position] for position in positions],
            edgecolor="white",
            linewidth=1.2,
            height=0.55,
            zorder=3,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
        ax.set_title(humanize_metric(metric_name), fontsize=12, fontweight="bold", loc="left")
        x_limit = max(rates) * 1.28 if max(rates) > 0 else 1.0
        ax.set_xlim(0, x_limit)

        for bar, rate, weighted_count, exposure in zip(
            bars,
            rates,
            weighted_counts,
            exposures,
            strict=True,
        ):
            ax.text(
                bar.get_width() + (x_limit * 0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.2f}",
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold",
            )
            ax.text(
                max(bar.get_width() * 0.5, x_limit * 0.08),
                bar.get_y() + bar.get_height() / 2,
                f"weighted={weighted_count:.2f}\n({exposure:.2f} h)",
                va="center",
                ha="center",
                fontsize=8.5,
                color="white",
                fontweight="bold",
            )

    axes[-1].set_xlabel("Events per 100 exposure-hours")
    fig.suptitle(
        "Operational Monitoring Signals by Patient Position",
        fontsize=14,
        fontweight="bold",
        x=0.11,
        ha="left",
    )
    fig.text(
        0.11,
        0.015,
        _fig7_footer_text(talk_confirmed=False),
        ha="left",
        va="bottom",
        fontsize=9,
        color=SLATE,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save_figure(fig, "fig7_operational_signals")
    plt.close(fig)
    print("Fig 7 done")


def _format_fig7_annotation(row: pd.Series, metric_name: str, value: float) -> str:
    numerator = float(row["positioned_numerator_value"])
    if metric_name == "nudge_active_seconds":
        return f"{value:.1f} sec/h\n({numerator:,.0f} active s)"
    noun = "clicks" if metric_name == "talk_event" else "triggers"
    return f"{value:.1f} /100h\n({numerator:,.0f} {noun})"


def _fig7_footer_text(
    *,
    talk_confirmed: bool,
    chair_exposure: float | None = None,
    bed_exposure: float | None = None,
) -> str:
    if talk_confirmed:
        if chair_exposure is None or bed_exposure is None:
            raise ValueError("Talk-confirmed Figure 7 footer requires chair and bed exposure totals.")
        return (
            "Main-text secondary-analysis figure. Talk clicks and manual alarm triggers are "
            "labeled with a +/-3 s modal state window around each event; the +/-5 s sensitivity "
            f"was similar. Exposure totals: chair {chair_exposure:.1f} h, bed {bed_exposure:,.1f} h. "
            "Nudge panel shows active nudge-state seconds per exposure-hour."
        )
    return (
        "Main-text secondary-analysis figure. Hourly counts are apportioned across chair and bed "
        "using the same fractional exposure weights as the primary denominator analysis."
    )


def _make_fig7_talk_confirmed(operational_event_rates_df: pd.DataFrame) -> None:
    scoped = operational_event_rates_df.copy()
    if "scope" in scoped.columns and (
        scoped["scope"] == "talk_confirmed_intervention_oneoff"
    ).any():
        scoped = scoped[scoped["scope"] == "talk_confirmed_intervention_oneoff"].copy()
    if scoped.empty:
        fail("Figure 7 source has no rows for the talk-confirmed operational study")

    metric_specs = [
        {
            "metric": "talk_event",
            "window_half_width_seconds": 3.0,
            "value_column": "rate_per_100_exposure_hours",
            "axis_label": "Talk clicks per 100 exposure-hours",
        },
        {
            "metric": "alarm_trigger",
            "window_half_width_seconds": 3.0,
            "value_column": "rate_per_100_exposure_hours",
            "axis_label": "Alarm triggers per 100 exposure-hours",
        },
        {
            "metric": "nudge_active_seconds",
            "window_half_width_seconds": None,
            "value_column": "active_seconds_per_exposure_hour",
            "axis_label": "Active nudge seconds per exposure-hour",
        },
    ]
    positions = ["chair", "bed"]
    bar_colors = {"chair": BLUE, "bed": SLATE}

    fig, axes = plt.subplots(nrows=3, figsize=(8.2, 9.4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    chair_exposure = None
    bed_exposure = None

    for ax, spec in zip(axes, metric_specs, strict=True):
        metric_rows = scoped[
            (scoped["metric"] == spec["metric"]) & (scoped["position"].isin(positions))
        ].copy()
        if spec["window_half_width_seconds"] is None:
            metric_rows = metric_rows[metric_rows["window_half_width_seconds"].isna()].copy()
        else:
            metric_rows = metric_rows[
                metric_rows["window_half_width_seconds"].astype(float)
                == spec["window_half_width_seconds"]
            ].copy()
        if metric_rows.empty:
            fail(f"Figure 7 source has no rows for metric {spec['metric']}")

        ordered_rows = [
            require_single_row(
                metric_rows,
                metric_rows["position"] == position,
                f"Figure 7 {spec['metric']} {position}",
            )
            for position in positions
        ]
        if chair_exposure is None:
            chair_exposure = float(ordered_rows[0]["exposure_hours"])
            bed_exposure = float(ordered_rows[1]["exposure_hours"])

        values = [float(row[spec["value_column"]]) for row in ordered_rows]
        labels = [humanize_location(position) for position in positions]
        y = np.arange(len(labels))

        bars = ax.barh(
            y,
            values,
            color=[bar_colors[position] for position in positions],
            edgecolor="white",
            linewidth=1.2,
            height=0.55,
            zorder=3,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
        ax.set_title(humanize_metric(spec["metric"]), fontsize=12, fontweight="bold", loc="left")
        ax.set_xlabel(spec["axis_label"], fontsize=9)
        x_limit = max(values) * 1.42 if max(values) > 0 else 1.0
        ax.set_xlim(0, x_limit)

        for bar, row, value in zip(bars, ordered_rows, values, strict=True):
            ax.text(
                bar.get_width() + (x_limit * 0.02),
                bar.get_y() + bar.get_height() / 2,
                _format_fig7_annotation(row, spec["metric"], value),
                va="center",
                ha="left",
                fontsize=9.5,
                linespacing=1.1,
            )

    fig.suptitle(
        "Operational Monitoring Signals in Talk-Confirmed Sitter Sessions",
        fontsize=14,
        fontweight="bold",
        x=0.11,
        ha="left",
    )
    fig.text(
        0.11,
        0.015,
        _fig7_footer_text(
            talk_confirmed=True,
            chair_exposure=chair_exposure,
            bed_exposure=bed_exposure,
        ),
        ha="left",
        va="bottom",
        fontsize=9,
        color=SLATE,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_figure(fig, "fig7_operational_signals")
    plt.close(fig)
    print("Fig 7 done")


def make_fig7(operational_event_rates_df: pd.DataFrame) -> None:
    talk_confirmed_columns = {
        "metric_type",
        "metric_unit",
        "window_half_width_seconds",
        "positioned_numerator_value",
        "active_seconds_per_exposure_hour",
    }
    if talk_confirmed_columns.issubset(operational_event_rates_df.columns) and {
        "talk_event",
        "alarm_trigger",
        "nudge_active_seconds",
    }.issubset(set(operational_event_rates_df["metric"].astype(str))):
        _make_fig7_talk_confirmed(operational_event_rates_df)
        return
    _make_fig7_legacy(operational_event_rates_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate chair-falls manuscript figures as SVG and PNG."
    )
    parser.add_argument("--run-id", default=None, help="Run ID suffix for QA CSV inputs (required for figs 1-5, 7)")
    parser.add_argument(
        "--figs",
        nargs="*",
        default=None,
        help=(
            "Which figures to generate (e.g. --figs 1 2 4). "
            "Default: manuscript/dashboard set (1, 2, 4, 5, 6, 7). "
            "Legacy donut figure 3 is opt-in only."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    valid_figs = {"1", "2", "3", "4", "5", "6", "7"}
    default_figs = {"1", "2", "4", "5", "6", "7"}
    figs = set(args.figs) if args.figs else default_figs.copy()

    unknown = figs - valid_figs
    if unknown:
        fail(f"Unknown figure id(s): {', '.join(sorted(unknown))}. Valid: 1 2 3 4 5 6 7")

    csv_figs = figs & {"1", "2", "3", "4", "5", "7"}
    data: dict[str, pd.DataFrame] = {}
    if csv_figs:
        if not args.run_id:
            fail("--run-id is required for figures 1-5 and 7")
        csv_map = discover_csvs(run_id=args.run_id, project_root=PROJECT_ROOT)
        data = load_all(csv_map)
        print("Using CSV sources:")
        for name, path in csv_map.items():
            print(f"  {name}: {path}")

    if "1" in figs:
        make_fig1(data["risk_rates"])
    if "2" in figs:
        make_fig2(data["adjusted_rr"], data["misclassification"], data["threshold"])
    if "3" in figs:
        make_fig3(data["mechanism_taxonomy"])
    if "4" in figs:
        make_fig4(data["mechanism_taxonomy"])
    if "5" in figs:
        make_fig5(data["daypart_breakdown"])
    if "6" in figs:
        make_fig6()
    if "7" in figs:
        make_fig7(data["operational_event_rates"])

    print("\nAll requested figures saved to", OUT)


if __name__ == "__main__":
    main()
