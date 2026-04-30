#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Compile Charter mechanism taxonomy from adjudicated consensus fall data.

Maps each annotated fall event to one of three Charter taxonomy categories:
  - footrest_positioning : footrest tag AND prefall_location == chair
  - transfer_failure     : support_bed tag (any location)
  - other                : everything else

Priority (highest first): footrest_positioning > transfer_failure > other

For events with multiple annotations (same event_key), deduplicate to one row per
event_key using the annotation that produces the most-specific mechanism per priority.

Outputs:
  outputs/mechanism_taxonomy_{run_id}.csv

The output filename is labeled with ``run_id`` for packet grouping. The input data
remain the adjudicated consensus annotations selected by ``--consensus-path``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import csv
from collections import defaultdict

from ld_chair_falls.consensus import (
    CONSENSUS_STATUS_EXCLUDED_NO_SIGNAL,
    CONSENSUS_STATUS_INCLUDED_FALL,
    load_consensus_annotations,
)


# ---------------------------------------------------------------------------
# Taxonomy logic
# ---------------------------------------------------------------------------

_MECHANISM_PRIORITY = {
    "footrest_positioning": 0,
    "transfer_failure": 1,
    "other": 2,
}


def classify_annotation(prefall_location: str, tags: set[str]) -> str:
    """Return Charter mechanism for a single annotation row."""
    is_chair = prefall_location == "chair"

    if is_chair and "footrest" in tags:
        return "footrest_positioning"
    if "support_bed" in tags:
        return "transfer_failure"
    return "other"


def parse_tags(raw: str) -> set[str]:
    """Parse a comma-separated tags string into a set of stripped tag names."""
    if not raw or not raw.strip():
        return set()
    return {t.strip() for t in raw.split(",") if t.strip()}


def _clean_value(raw: Any) -> str:
    token = "" if raw is None else str(raw).strip()
    if token in {"", "-", "nan", "None"}:
        return ""
    return token


def _parse_clock_to_seconds(raw: Any) -> int | None:
    token = _clean_value(raw)
    if not token:
        return None
    parts = token.split(":")
    if len(parts) != 3:
        return None
    try:
        hh, mm, ss = [int(float(part)) for part in parts]
    except ValueError:
        return None
    if not (0 <= hh < 24 and 0 <= mm < 60 and 0 <= ss < 60):
        return None
    return (hh * 3600) + (mm * 60) + ss


def _time_since_departure_seconds(row: dict[str, Any]) -> int | None:
    departure_seconds = _parse_clock_to_seconds(row.get("furniture_departure_time"))
    fall_seconds = _parse_clock_to_seconds(row.get("fall_time_consensus"))
    if departure_seconds is None or fall_seconds is None:
        return None
    delta_seconds = fall_seconds - departure_seconds
    if delta_seconds < 0:
        delta_seconds += 24 * 3600
    return delta_seconds


def _first_non_empty(rows: list[dict[str, Any]], field: str) -> str:
    for row in rows:
        token = _clean_value(row.get(field))
        if token:
            return token
    return ""


def _first_valid_time_since_departure(rows: list[dict[str, Any]]) -> int | str:
    for row in rows:
        delta_seconds = _time_since_departure_seconds(row)
        if delta_seconds is not None:
            return delta_seconds
    return ""


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_consensus(path: Path) -> list[dict]:
    consensus, summary = load_consensus_annotations(path)
    if summary.get("status") != "ok":
        raise ValueError(f"Unable to load consensus annotations from {path}: {summary}")

    # Include all confirmed falls, even those lacking pipeline location signal.
    # The mechanism taxonomy is a descriptive export from adjudicated consensus
    # annotations and does not depend on livestream-derived signals.
    fall_statuses = {CONSENSUS_STATUS_INCLUDED_FALL, CONSENSUS_STATUS_EXCLUDED_NO_SIGNAL}
    filtered = consensus.loc[consensus["consensus_status"].isin(fall_statuses)]
    fields = [
        "event_key",
        "last_furniture",
        "furniture_departure_time",
        "prefall_location",
        "fall_time_consensus",
        "response_time_consensus",
        "fall_tags",
    ]
    return filtered[fields].to_dict(orient="records")


def deduplicate_events(rows: list[dict]) -> list[dict]:
    """One row per event_key: keep the annotation with the highest-priority mechanism.

    Ties broken by original row order (first occurrence wins within same priority).
    """
    # Group annotations by event_key
    by_key: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_key[row["event_key"]].append(row)

    result = []
    for event_key, event_annotations in by_key.items():
        best = None
        best_priority = 999
        for ann in event_annotations:
            tags = parse_tags(ann.get("fall_tags", ""))
            loc = _clean_value(ann.get("prefall_location")).lower()
            mech = classify_annotation(loc, tags)
            prio = _MECHANISM_PRIORITY[mech]
            if best is None or prio < best_priority:
                best = ann
                best_priority = prio
                best_mech = mech

        best_loc = _clean_value(best.get("prefall_location")).lower()
        provenance_rows = [best] + [ann for ann in event_annotations if ann is not best]
        last_furniture = ""
        time_since_departure = ""
        if best_loc in {"room", "no_patient"}:
            last_furniture = _first_non_empty(provenance_rows, "last_furniture")
            time_since_departure = _first_valid_time_since_departure(provenance_rows)

        result.append({
            "event_key": event_key,
            "prefall_location": best_loc,
            "charter_mechanism": best_mech,
            "fall_tags": best.get("fall_tags", "").strip(),
            "last_furniture": last_furniture,
            "time_since_departure": time_since_departure,
            "annotation_count": len(event_annotations),
        })

    # Sort by event_key for deterministic output
    result.sort(key=lambda r: r["event_key"])
    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict]) -> None:
    mechanisms = ["footrest_positioning", "transfer_failure", "other"]
    locations = sorted({r["prefall_location"] for r in rows})

    # Build count matrix
    counts: dict[str, dict[str, int]] = {m: defaultdict(int) for m in mechanisms}
    for r in rows:
        counts[r["charter_mechanism"]][r["prefall_location"]] += 1

    # Header
    col_w = 22
    loc_w = 14
    header = f"{'mechanism':<{col_w}}" + "".join(f"{loc:>{loc_w}}" for loc in locations) + f"{'TOTAL':>{loc_w}}"
    print("\n" + "=" * len(header))
    print("Mechanism taxonomy summary (events, deduplicated)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    mech_totals: dict[str, int] = {}
    for mech in mechanisms:
        row_counts = [counts[mech][loc] for loc in locations]
        total = sum(row_counts)
        mech_totals[mech] = total
        row_str = f"{mech:<{col_w}}" + "".join(f"{c:>{loc_w}}" for c in row_counts) + f"{total:>{loc_w}}"
        print(row_str)

    print("-" * len(header))
    col_totals = [sum(counts[m][loc] for m in mechanisms) for loc in locations]
    grand_total = sum(col_totals)
    total_str = f"{'TOTAL':<{col_w}}" + "".join(f"{c:>{loc_w}}" for c in col_totals) + f"{grand_total:>{loc_w}}"
    print(total_str)
    print("=" * len(header))

    # Percentage within chair
    chair_rows = [r for r in rows if r["prefall_location"] == "chair"]
    n_chair = len(chair_rows)
    if n_chair > 0:
        print(f"\nChair-location breakdown (n={n_chair} events):")
        for mech in mechanisms:
            n = sum(1 for r in chair_rows if r["charter_mechanism"] == mech)
            pct = 100.0 * n / n_chair
            print(f"  {mech:<28}: {n:>2}  ({pct:.0f}%)")

    provenance_rows = [r for r in rows if r["prefall_location"] in {"room", "no_patient"}]
    if provenance_rows:
        provenance_n = len(provenance_rows)
        with_furniture = sum(1 for r in provenance_rows if str(r.get("last_furniture", "")).strip())
        with_delta = sum(1 for r in provenance_rows if str(r.get("time_since_departure", "")).strip())
        print(f"\nRoom/no-patient provenance coverage (n={provenance_n} events):")
        print(f"  last_furniture populated: {with_furniture}/{provenance_n}")
        print(f"  time_since_departure populated: {with_delta}/{provenance_n}")

        furniture_counts: dict[str, int] = defaultdict(int)
        for row in provenance_rows:
            label = str(row.get("last_furniture", "")).strip() or "missing"
            furniture_counts[label] += 1
        print("  last_furniture breakdown:")
        for label, count in sorted(furniture_counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {label:<12} {count:>2}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile Charter mechanism taxonomy from adjudicated consensus fall data."
    )
    parser.add_argument(
        "--run-id",
        default="20260305T043741Z",
        help=(
            "Run identifier used for packet grouping in the output filename; does not "
            "select the consensus input (default: 20260305T043741Z)"
        ),
    )
    parser.add_argument(
        "--consensus-path",
        default="data/public/falls-observations-v3-consensus.csv",
        help="Consensus CSV path relative to project root (default: data/public/falls-observations-v3-consensus.csv).",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Path to project root (default: parent of scripts/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root) if args.project_root else PROJECT_ROOT

    consensus_path = Path(args.consensus_path)
    if not consensus_path.is_absolute():
        consensus_path = project_root / consensus_path
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = outputs_dir / f"mechanism_taxonomy_{args.run_id}.csv"

    if not consensus_path.exists():
        print(f"ERROR: consensus CSV not found at {consensus_path}", file=sys.stderr)
        sys.exit(1)

    rows = read_consensus(consensus_path)
    print(f"Loaded {len(rows)} annotation rows from {consensus_path.name}")

    deduped = deduplicate_events(rows)
    print(f"Deduplicated to {len(deduped)} unique events")

    # Write CSV
    fieldnames = [
        "event_key",
        "prefall_location",
        "charter_mechanism",
        "fall_tags",
        "last_furniture",
        "time_since_departure",
        "annotation_count",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped)

    print(f"Wrote taxonomy CSV to: {out_path}")

    # Print summary
    print_summary(deduped)

    print(f"run_id={args.run_id}")
    print(f"consensus_path={consensus_path}")
    print(f"taxonomy_path={out_path}")


if __name__ == "__main__":
    main()
