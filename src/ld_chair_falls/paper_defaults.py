from __future__ import annotations

from pathlib import Path

CURRENT_MANUSCRIPT_RUN_ID = "consensus_v3_refresh_20260320T215557Z"

PUBLIC_DATA_DIR = Path("data/public")
PUBLIC_DERIVED_DIR = PUBLIC_DATA_DIR / "derived"
PUBLIC_GEMINI_DIR = PUBLIC_DATA_DIR / "gemini"

_MANUSCRIPT_PACKET_ARTIFACT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "risk_rates": (
        "data/public/derived/chair_bed_risk_rates_{run_id}.csv",
        "outputs/qa/chair_bed_risk_rates_{run_id}.csv",
    ),
    "risk_rates_confidence_sensitivity": (
        "data/public/derived/chair_bed_risk_rates_confidence_sensitivity_{run_id}.csv",
        "outputs/qa/chair_bed_risk_rates_confidence_sensitivity_{run_id}.csv",
    ),
    "operational_event_rates": (
        "data/public/derived/chair_bed_operational_event_rates_{run_id}.csv",
        "outputs/qa/chair_bed_operational_event_rates_{run_id}.csv",
    ),
    "adjusted_rr": (
        "data/public/derived/chair_bed_adjusted_rr_{run_id}.csv",
        "outputs/qa/chair_bed_adjusted_rr_{run_id}.csv",
    ),
    "adjusted_rr_covariates": (
        "data/public/derived/chair_bed_adjusted_rr_covariates_{run_id}.csv",
        "outputs/qa/chair_bed_adjusted_rr_covariates_{run_id}.csv",
    ),
    "adjusted_rr_robustness": (
        "data/public/derived/chair_bed_adjusted_rr_robustness_{run_id}.csv",
        "outputs/qa/chair_bed_adjusted_rr_robustness_{run_id}.csv",
    ),
    "misclassification": (
        "data/public/derived/chair_bed_misclassification_sensitivity_{run_id}.csv",
        "outputs/qa/chair_bed_misclassification_sensitivity_{run_id}.csv",
    ),
    "threshold": (
        "data/public/derived/chair_bed_threshold_sensitivity_{run_id}.csv",
        "outputs/qa/chair_bed_threshold_sensitivity_{run_id}.csv",
    ),
    "analysis_reconciliation": (
        "data/public/derived/chair_bed_analysis_reconciliation_{run_id}.csv",
        "outputs/qa/chair_bed_analysis_reconciliation_{run_id}.csv",
    ),
    "consensus_adjudication": (
        "data/public/derived/consensus_adjudication_summary_{run_id}.csv",
        "outputs/qa/consensus_adjudication_summary_{run_id}.csv",
    ),
    "daypart_breakdown": (
        "data/public/derived/falls_prefall_location_probability_breakdown_{run_id}.csv",
        "outputs/qa/falls_prefall_location_probability_breakdown_{run_id}.csv",
    ),
    "furniture_origin_chain": (
        "data/public/derived/label_eval_furniture_origin_chain_{run_id}.csv",
        "outputs/qa/label_eval_furniture_origin_chain_{run_id}.csv",
    ),
    "post_departure_latency": (
        "data/public/derived/label_eval_post_departure_latency_{run_id}.csv",
        "outputs/qa/label_eval_post_departure_latency_{run_id}.csv",
    ),
    "benchmark_label_metrics": (
        "data/public/derived/label_eval_benchmark_label_metrics_{run_id}.csv",
        "outputs/qa/label_eval_benchmark_label_metrics_{run_id}.csv",
    ),
    "benchmark_sequence_metrics": (
        "data/public/derived/label_eval_benchmark_sequence_metrics_{run_id}.csv",
        "outputs/qa/label_eval_benchmark_sequence_metrics_{run_id}.csv",
    ),
    "benchmark_probability_quality": (
        "data/public/derived/label_eval_benchmark_probability_quality_{run_id}.csv",
        "outputs/qa/label_eval_benchmark_probability_quality_{run_id}.csv",
    ),
    "response_timing": (
        "data/public/derived/label_eval_response_timing_{run_id}.csv",
        "outputs/qa/label_eval_response_timing_{run_id}.csv",
    ),
    "mechanism_taxonomy": (
        "data/public/derived/mechanism_taxonomy_{run_id}.csv",
        "outputs/mechanism_taxonomy_{run_id}.csv",
    ),
    "gemini_metrics": (
        "data/public/gemini/gemini_model_characterization_metrics_20260313T180809Z.csv",
        "outputs/gemini_fall_analysis/model_characterization_20260313T180809Z/gemini_model_characterization_metrics_20260313T180809Z.csv",
    ),
    "gemini_detection": (
        "data/public/gemini/gemini_model_characterization_detection_20260313T180809Z.csv",
        "outputs/gemini_fall_analysis/model_characterization_20260313T180809Z/gemini_model_characterization_detection_20260313T180809Z.csv",
    ),
    "gemini_summary": (
        "data/public/gemini/gemini_model_characterization_summary_20260313T180809Z.md",
        "outputs/gemini_fall_analysis/model_characterization_20260313T180809Z/gemini_model_characterization_summary_20260313T180809Z.md",
    ),
}

_REQUIRED_MANUSCRIPT_PACKET_KEYS = tuple(_MANUSCRIPT_PACKET_ARTIFACT_CANDIDATES)


def manuscript_artifact_candidate_paths(
    project_root: Path,
    artifact_key: str,
    run_id: str,
) -> tuple[Path, ...]:
    templates = _MANUSCRIPT_PACKET_ARTIFACT_CANDIDATES.get(artifact_key)
    if templates is None:
        raise KeyError(f"Unknown manuscript artifact key: {artifact_key}")
    return tuple(project_root / template.format(run_id=run_id) for template in templates)


def resolve_manuscript_artifact(
    project_root: Path,
    artifact_key: str,
    run_id: str = CURRENT_MANUSCRIPT_RUN_ID,
    *,
    must_exist: bool = True,
) -> Path | None:
    candidates = manuscript_artifact_candidate_paths(project_root, artifact_key, run_id)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not must_exist:
        return None
    listing = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        f"Missing manuscript artifact '{artifact_key}' for run_id={run_id}. Checked:\n{listing}"
    )

def manuscript_packet_required_paths(project_root: Path, run_id: str) -> tuple[Path, ...]:
    return tuple(
        resolve_manuscript_artifact(project_root, artifact_key, run_id)
        for artifact_key in _REQUIRED_MANUSCRIPT_PACKET_KEYS
    )


def assert_manuscript_packet_artifacts(
    project_root: Path,
    run_id: str = CURRENT_MANUSCRIPT_RUN_ID,
) -> tuple[Path, ...]:
    resolved_paths: list[Path] = []
    missing: list[str] = []

    for artifact_key in _REQUIRED_MANUSCRIPT_PACKET_KEYS:
        candidates = manuscript_artifact_candidate_paths(project_root, artifact_key, run_id)
        match = next((path for path in candidates if path.exists()), None)
        if match is None:
            missing.append(f"- {artifact_key}: " + " or ".join(str(path) for path in candidates))
            continue
        resolved_paths.append(match)

    if missing:
        raise FileNotFoundError(
            f"Manuscript-facing packet run_id={run_id} is incomplete:\n"
            + "\n".join(missing)
            + "\nPopulate the curated `data/public/` bundle or provide the legacy `outputs/` fallback files."
        )
    return tuple(resolved_paths)
