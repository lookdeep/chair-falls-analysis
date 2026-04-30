#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.report_descriptive import (  # noqa: E402
    compare_descriptive_runs,
    render_descriptive_comparison_markdown,
    resolve_full_cohort_run_id,
    resolve_run_id,
    write_descriptive_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare descriptive outputs across two run IDs and flag major disagreement."
    )
    parser.add_argument(
        "--scope",
        choices=["falls", "full_cohort"],
        default="full_cohort",
        help="Run-id resolver scope when run IDs are omitted (default: full_cohort).",
    )
    parser.add_argument("--baseline-run-id", default=None, help="Baseline run id.")
    parser.add_argument("--candidate-run-id", default=None, help="Candidate run id.")
    parser.add_argument(
        "--output-json",
        default="outputs/qa/descriptive_output_comparison.json",
        help="Output JSON path (default: outputs/qa/descriptive_output_comparison.json).",
    )
    parser.add_argument(
        "--output-markdown",
        default="outputs/qa/descriptive_output_comparison.md",
        help="Output markdown path (default: outputs/qa/descriptive_output_comparison.md).",
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Project root directory. Defaults to repository project root.",
    )
    return parser.parse_args()


def _run_manifest_candidates(project_root: Path) -> list[str]:
    manifests = sorted(
        (project_root / "outputs" / "manifests").glob("run_manifest_*.yaml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return [manifest.stem.replace("run_manifest_", "", 1) for manifest in manifests]


def _has_scope_minimum_artifacts(project_root: Path, run_id: str, scope: str) -> bool:
    qa = project_root / "outputs" / "qa"
    required = {
        qa / f"source_profile_{run_id}.json",
        qa / f"qa_summary_{run_id}.json",
    }
    if scope == "falls":
        required.update(
            {
                qa / f"falls_by_site_{run_id}.csv",
                qa / f"falls_by_daypart_{run_id}.csv",
                qa / f"falls_prefall_location_{run_id}.csv",
            }
        )
    else:
        required.update(
            {
                qa / f"cohort_duration_{run_id}.csv",
                qa / f"cohort_eligibility_rates_{run_id}.csv",
                qa / f"control_denominator_coverage_{run_id}.csv",
            }
        )
    return all(path.exists() for path in required)


def _resolve_previous_run_id(project_root: Path, scope: str, current_run_id: str) -> str:
    for run_id in _run_manifest_candidates(project_root):
        if run_id == current_run_id:
            continue
        if _has_scope_minimum_artifacts(project_root, run_id, scope):
            return run_id
    raise FileNotFoundError(f"No previous comparable run found for scope={scope} before {current_run_id}.")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    if args.scope == "falls":
        candidate_run_id = args.candidate_run_id or resolve_run_id(project_root, None)
    else:
        candidate_run_id = args.candidate_run_id or resolve_full_cohort_run_id(project_root, None)

    baseline_run_id = args.baseline_run_id or _resolve_previous_run_id(
        project_root, args.scope, candidate_run_id
    )

    output_json_path, output_markdown_path = write_descriptive_comparison(
        project_root=project_root,
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    payload = compare_descriptive_runs(project_root, baseline_run_id, candidate_run_id)
    print(f"baseline_run_id={baseline_run_id}")
    print(f"candidate_run_id={candidate_run_id}")
    print(f"major_disagreement_count={payload.get('major_disagreement_count', 0)}")
    print(f"output_json={output_json_path}")
    print(f"output_markdown={output_markdown_path}")
    print(render_descriptive_comparison_markdown(payload))


if __name__ == "__main__":
    main()
