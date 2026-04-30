from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from .dayparts import DAYPART_DISPLAY, DAYPART_ORDER, classify_hour
from .paper_defaults import CURRENT_MANUSCRIPT_RUN_ID
from .utils import ensure_dir, read_json, read_yaml, utc_now_iso, write_json

FALLS_REQUIRED_ARTIFACT_KEYS = {
    "run_manifest",
    "source_profile",
    "qa_summary",
    "falls_by_site",
    "falls_by_weekday",
    "falls_by_daypart",
    "falls_month_of_year",
    "falls_prefall_location",
    "falls_response_latency",
}

FULL_COHORT_REQUIRED_ARTIFACT_KEYS = {
    "run_manifest",
    "source_profile",
    "qa_summary",
    "cohort_duration",
    "cohort_eligibility_rates",
    "control_denominator_coverage",
}

CHAIR_BED_INFERENCE_REQUIRED_ARTIFACT_KEYS = {
    "run_manifest",
    "source_profile",
    "qa_summary",
    "chair_bed_risk_rates",
    "chair_bed_confidence_sensitivity",
}

LABEL_EVAL_SHAREHOLDER_REQUIRED_ARTIFACT_KEYS = {
    "run_manifest",
    "source_profile",
    "qa_summary",
    "label_eval_sequence_metrics",
    "label_eval_confusion_matrix",
    "label_eval_probability_quality",
    "label_eval_response_timing",
    "label_eval_population_stats",
    "label_eval_threshold_checks",
}

DEFAULT_MANUSCRIPT_RUN_ID = CURRENT_MANUSCRIPT_RUN_ID

_LOCATION_COLORS: dict[str, str] = {
    "chair": "#0B3C5D",
    "bed": "#328CC1",
    "room": "#D9B310",
    "no_patient": "#B33F62",
    "unknown": "#7C7C7C",
}
_WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "unknown"]
_DAYPART_ORDER = DAYPART_ORDER
_POSTER_TICK_FONT = 16
_POSTER_VALUE_FONT = 16
_POSTER_LEGEND_FONT = 15
_DAYPART_DISPLAY = DAYPART_DISPLAY


def resolve_run_id(project_root: Path, explicit_run_id: str | None = None) -> str:
    if explicit_run_id:
        return explicit_run_id.strip()

    return _resolve_preferred_run_id_for_required_artifacts(
        project_root,
        FALLS_REQUIRED_ARTIFACT_KEYS,
        "No compatible manifests found for falls-only descriptive artifacts.",
    )


def resolve_full_cohort_run_id(project_root: Path, explicit_run_id: str | None = None) -> str:
    if explicit_run_id:
        return explicit_run_id.strip()

    return _resolve_preferred_run_id_for_required_artifacts(
        project_root,
        FULL_COHORT_REQUIRED_ARTIFACT_KEYS,
        "No compatible manifests found for full-cohort hourly descriptive artifacts.",
    )


def resolve_chair_bed_inference_run_id(project_root: Path, explicit_run_id: str | None = None) -> str:
    if explicit_run_id:
        return explicit_run_id.strip()

    return _resolve_preferred_run_id_for_required_artifacts(
        project_root,
        CHAIR_BED_INFERENCE_REQUIRED_ARTIFACT_KEYS,
        "No compatible manifests found for chair-bed inference descriptive artifacts.",
    )


def resolve_label_eval_shareholder_run_id(project_root: Path, explicit_run_id: str | None = None) -> str:
    if explicit_run_id:
        return explicit_run_id.strip()

    return _resolve_preferred_run_id_for_required_artifacts(
        project_root,
        LABEL_EVAL_SHAREHOLDER_REQUIRED_ARTIFACT_KEYS,
        "No compatible manifests found for label-eval shareholder report artifacts.",
    )


def _resolve_preferred_run_id_for_required_artifacts(
    project_root: Path,
    required_keys: set[str],
    missing_message: str,
) -> str:
    default_artifacts = _artifact_paths(project_root, DEFAULT_MANUSCRIPT_RUN_ID)
    missing_default = _missing_required_artifacts(default_artifacts, required_keys)
    if not missing_default:
        return DEFAULT_MANUSCRIPT_RUN_ID
    if any(path.exists() for path in default_artifacts.values()):
        listing = "\n".join(f"- {path}" for path in missing_default)
        raise FileNotFoundError(
            f"Default manuscript run_id={DEFAULT_MANUSCRIPT_RUN_ID} is incomplete for this surface:\n"
            f"{listing}\n"
            "Refresh the canonical bundle or pass --run-id explicitly."
        )

    return _resolve_latest_run_id_for_required_artifacts(project_root, required_keys, missing_message)


def _resolve_latest_run_id_for_required_artifacts(
    project_root: Path,
    required_keys: set[str],
    missing_message: str,
) -> str:
    manifests_dir = project_root / "outputs" / "manifests"
    candidates = list(manifests_dir.glob("run_manifest_*.yaml"))
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    for manifest_path in candidates:
        run_id = manifest_path.stem.replace("run_manifest_", "", 1)
        artifacts = _artifact_paths(project_root, run_id)
        if not _missing_required_artifacts(artifacts, required_keys):
            return run_id

    raise FileNotFoundError(
        f"{missing_message}\nChecked manifests in {manifests_dir}. "
        "Run the descriptive pipeline first or pass --run-id explicitly."
    )


def _artifact_paths(project_root: Path, run_id: str) -> dict[str, Path]:
    qa_dir = project_root / "outputs" / "qa"
    manifests_dir = project_root / "outputs" / "manifests"
    return {
        "run_manifest": manifests_dir / f"run_manifest_{run_id}.yaml",
        "source_profile": qa_dir / f"source_profile_{run_id}.json",
        "qa_summary": qa_dir / f"qa_summary_{run_id}.json",
        "falls_by_site": qa_dir / f"falls_by_site_{run_id}.csv",
        "falls_by_weekday": qa_dir / f"falls_by_weekday_{run_id}.csv",
        "falls_by_daypart": qa_dir / f"falls_by_daypart_{run_id}.csv",
        "falls_site_name_aliases": qa_dir / f"falls_site_name_aliases_{run_id}.csv",
        "falls_month_of_year": qa_dir / f"falls_month_of_year_{run_id}.csv",
        "falls_prefall_location": qa_dir / f"falls_prefall_location_{run_id}.csv",
        "falls_prefall_location_probabilities": qa_dir / f"falls_prefall_location_probabilities_{run_id}.csv",
        "falls_prefall_location_probability_breakdown": qa_dir
        / f"falls_prefall_location_probability_breakdown_{run_id}.csv",
        "falls_response_latency": qa_dir / f"falls_response_latency_{run_id}.csv",
        "falls_patient_day_hour_location": qa_dir / f"falls_patient_day_hour_location_{run_id}.csv",
        "fall_second_level_panel": qa_dir / f"fall_second_level_panel_{run_id}.csv",
        "fall_event_localized_metrics": qa_dir / f"fall_event_localized_metrics_{run_id}.csv",
        "fall_response_curve": qa_dir / f"fall_response_curve_{run_id}.csv",
        "fall_case_crossover_sets": qa_dir / f"fall_case_crossover_sets_{run_id}.csv",
        "fall_case_crossover_effects": qa_dir / f"fall_case_crossover_effects_{run_id}.csv",
        "fall_case_crossover_diagnostics": qa_dir / f"fall_case_crossover_diagnostics_{run_id}.json",
        "fall_negative_control_sets": qa_dir / f"fall_negative_control_sets_{run_id}.csv",
        "fall_negative_control_effects": qa_dir / f"fall_negative_control_effects_{run_id}.csv",
        "fall_negative_control_diagnostics": qa_dir / f"fall_negative_control_diagnostics_{run_id}.json",
        "fall_negative_control_coverage": qa_dir / f"fall_negative_control_coverage_{run_id}.json",
        "fall_negative_control_match_quality": qa_dir / f"fall_negative_control_match_quality_{run_id}.csv",
        "cohort_analysis_markdown": qa_dir / f"cohort_analysis_{run_id}.md",
        "cohort_composition": qa_dir / f"cohort_composition_{run_id}.csv",
        "cohort_duration": qa_dir / f"cohort_duration_{run_id}.csv",
        "cohort_eligibility_rates": qa_dir / f"cohort_eligibility_rates_{run_id}.csv",
        "cohort_exclusions": qa_dir / f"cohort_exclusions_{run_id}.csv",
        "fall_density": qa_dir / f"fall_density_{run_id}.csv",
        "control_denominator_coverage": qa_dir / f"control_denominator_coverage_{run_id}.csv",
        "chair_bed_risk_rates": qa_dir / f"chair_bed_risk_rates_{run_id}.csv",
        "chair_bed_confidence_sensitivity": qa_dir
        / f"chair_bed_risk_rates_confidence_sensitivity_{run_id}.csv",
        "chair_bed_operational_event_rates": qa_dir / f"chair_bed_operational_event_rates_{run_id}.csv",
        "chair_bed_missingness_stress": qa_dir / f"chair_bed_missingness_stress_{run_id}.csv",
        "label_eval_sequence_metrics": qa_dir / f"label_eval_sequence_metrics_{run_id}.csv",
        "label_eval_confusion_matrix": qa_dir / f"label_eval_confusion_matrix_{run_id}.csv",
        "label_eval_probability_quality": qa_dir / f"label_eval_probability_quality_{run_id}.csv",
        "label_eval_response_timing": qa_dir / f"label_eval_response_timing_{run_id}.csv",
        "label_eval_population_stats": qa_dir / f"label_eval_population_stats_{run_id}.csv",
        "label_eval_truth_prefall_location": qa_dir / f"label_eval_truth_prefall_location_{run_id}.csv",
        "label_eval_tag_location_association": qa_dir / f"label_eval_tag_location_association_{run_id}.csv",
        "label_eval_signal_location_profile": qa_dir / f"label_eval_signal_location_profile_{run_id}.csv",
        "label_eval_association_summary": qa_dir / f"label_eval_association_summary_{run_id}.csv",
        "label_eval_shadow_model_metrics": qa_dir / f"label_eval_shadow_model_metrics_{run_id}.csv",
        "label_eval_shadow_model_predictions": qa_dir / f"label_eval_shadow_model_predictions_{run_id}.csv",
        "label_eval_shadow_model_comparison": qa_dir / f"label_eval_shadow_model_comparison_{run_id}.csv",
        "onset_events": qa_dir / f"onset_events_{run_id}.csv",
        "onset_case_crossover": qa_dir / f"onset_case_crossover_{run_id}.csv",
        "shadow_feature_importance": qa_dir / f"shadow_feature_importance_{run_id}.csv",
        "shadow_ablation": qa_dir / f"shadow_ablation_{run_id}.csv",
        "shadow_window_sensitivity": qa_dir / f"shadow_window_sensitivity_{run_id}.csv",
        "shadow_cv_fold_metrics": qa_dir / f"shadow_cv_fold_metrics_{run_id}.csv",
        "label_eval_threshold_checks": qa_dir / f"label_eval_threshold_checks_{run_id}.json",
        "label_eval_furniture_origin_chain": qa_dir / f"label_eval_furniture_origin_chain_{run_id}.csv",
        "label_eval_post_departure_latency": qa_dir / f"label_eval_post_departure_latency_{run_id}.csv",
        "label_eval_tag_origin_chain_crosstab": qa_dir / f"label_eval_tag_origin_chain_crosstab_{run_id}.csv",
        "furniture_origin_exit_concordance": qa_dir / f"furniture_origin_exit_concordance_{run_id}.csv",
        "furniture_origin_exit_concordance_summary": qa_dir / f"furniture_origin_exit_concordance_summary_{run_id}.csv",
        "transform_metrics": qa_dir / f"transform_metrics_{run_id}.json",
    }


def _missing_required_artifacts(artifacts: dict[str, Path], required: set[str]) -> list[Path]:
    return [path for key, path in artifacts.items() if key in required and not path.exists()]


def load_descriptive_inputs(project_root: Path, run_id: str) -> dict[str, Any]:
    artifacts = _artifact_paths(project_root, run_id)
    missing = _missing_required_artifacts(artifacts, FALLS_REQUIRED_ARTIFACT_KEYS)
    if missing:
        listing = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing required artifacts for run_id={run_id}:\n{listing}\n"
            "Re-run descriptive extraction/summary generation for this run."
        )

    prefall_location_probabilities = artifacts["falls_prefall_location_probabilities"]
    prefall_location_probability_breakdown = artifacts["falls_prefall_location_probability_breakdown"]
    patient_day_hour_location = artifacts["falls_patient_day_hour_location"]

    tables: dict[str, pd.DataFrame] = {
        "Falls by Site": pd.read_csv(artifacts["falls_by_site"]),
        "Falls by Weekday": pd.read_csv(artifacts["falls_by_weekday"]),
        "Falls by Daypart": pd.read_csv(artifacts["falls_by_daypart"]),
        "Falls by Month of Year": pd.read_csv(artifacts["falls_month_of_year"]),
        "Pre-Fall Location (Livestream)": pd.read_csv(artifacts["falls_prefall_location"]),
        "Response Latency (Livestream)": pd.read_csv(artifacts["falls_response_latency"]),
    }
    if prefall_location_probabilities.exists():
        tables["Pre-Fall Location Probabilities (Livestream)"] = pd.read_csv(prefall_location_probabilities)
    if prefall_location_probability_breakdown.exists():
        tables["Pre-Fall Location Probability Breakdown (Livestream)"] = pd.read_csv(
            prefall_location_probability_breakdown
        )
    chart_frames: dict[str, pd.DataFrame] = {}
    if patient_day_hour_location.exists():
        chart_frames["Patient-Day-Hour Location (Livestream)"] = pd.read_csv(patient_day_hour_location)
    if artifacts["fall_event_localized_metrics"].exists():
        tables["Per-Fall Localized Metrics (Livestream)"] = pd.read_csv(artifacts["fall_event_localized_metrics"])
    if artifacts["fall_response_curve"].exists():
        tables["Per-Fall Response Curve (Livestream)"] = pd.read_csv(artifacts["fall_response_curve"])
    if artifacts["fall_case_crossover_effects"].exists():
        tables["Case-Crossover Effects (Exploratory)"] = pd.read_csv(artifacts["fall_case_crossover_effects"])
    if artifacts["fall_negative_control_effects"].exists():
        tables["Negative-Control Effects (Exploratory)"] = pd.read_csv(artifacts["fall_negative_control_effects"])
    if artifacts["fall_negative_control_match_quality"].exists():
        tables["Negative-Control Match Quality"] = pd.read_csv(artifacts["fall_negative_control_match_quality"])

    return {
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "artifacts": artifacts,
        "run_manifest": read_yaml(artifacts["run_manifest"]),
        "source_profile": read_json(artifacts["source_profile"]),
        "qa_summary": read_json(artifacts["qa_summary"]),
        "negative_control_coverage": (
            read_json(artifacts["fall_negative_control_coverage"])
            if artifacts["fall_negative_control_coverage"].exists()
            else {}
        ),
        "tables": tables,
        "chart_frames": chart_frames,
    }


def load_full_cohort_hourly_inputs(project_root: Path, run_id: str) -> dict[str, Any]:
    artifacts = _artifact_paths(project_root, run_id)
    missing = _missing_required_artifacts(artifacts, FULL_COHORT_REQUIRED_ARTIFACT_KEYS)
    if missing:
        listing = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing required full-cohort artifacts for run_id={run_id}:\n{listing}\n"
            "Re-run transform/QA generation for this run."
        )

    composition = (
        pd.read_csv(artifacts["cohort_composition"])
        if artifacts["cohort_composition"].exists()
        else pd.DataFrame(columns=["cohort_type", "monitor_id", "hour_rows"])
    )
    composition_summary = pd.DataFrame()
    if not composition.empty and {"cohort_type", "monitor_id", "hour_rows"}.issubset(composition.columns):
        composition_summary = (
            composition.groupby("cohort_type", dropna=False)
            .agg(
                distinct_monitors=("monitor_id", "nunique"),
                total_hour_rows=("hour_rows", "sum"),
            )
            .reset_index()
            .sort_values(["total_hour_rows", "cohort_type"], ascending=[False, True])
        )

    tables: dict[str, pd.DataFrame] = {
        "Cohort Composition Summary": composition_summary,
        "Cohort Duration Distribution": pd.read_csv(artifacts["cohort_duration"]),
        "Cohort Eligibility Rates": pd.read_csv(artifacts["cohort_eligibility_rates"]),
        "Control Denominator Coverage": pd.read_csv(artifacts["control_denominator_coverage"]),
    }
    if artifacts["fall_density"].exists():
        tables["Fall Density by Cohort and Daypart"] = pd.read_csv(artifacts["fall_density"])
    if artifacts["chair_bed_risk_rates"].exists():
        tables["Chair-vs-Bed Risk Rates"] = pd.read_csv(artifacts["chair_bed_risk_rates"])
    if artifacts["chair_bed_confidence_sensitivity"].exists():
        tables["Chair-vs-Bed Confidence Sensitivity"] = pd.read_csv(artifacts["chair_bed_confidence_sensitivity"])

    return {
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "artifacts": artifacts,
        "run_manifest": read_yaml(artifacts["run_manifest"]),
        "source_profile": read_json(artifacts["source_profile"]),
        "qa_summary": read_json(artifacts["qa_summary"]),
        "tables": tables,
    }


def load_chair_bed_inference_inputs(project_root: Path, run_id: str) -> dict[str, Any]:
    artifacts = _artifact_paths(project_root, run_id)
    missing = _missing_required_artifacts(artifacts, CHAIR_BED_INFERENCE_REQUIRED_ARTIFACT_KEYS)
    if missing:
        listing = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing required chair-bed inference artifacts for run_id={run_id}:\n{listing}\n"
            "Re-run QA generation for this run."
        )

    tables: dict[str, pd.DataFrame] = {
        "Chair-vs-Bed Risk Rates": _read_csv_allow_empty(artifacts["chair_bed_risk_rates"]),
        "Chair-vs-Bed Confidence Sensitivity": _read_csv_allow_empty(artifacts["chair_bed_confidence_sensitivity"]),
    }
    if artifacts["cohort_eligibility_rates"].exists():
        tables["Cohort Eligibility Rates"] = _read_csv_allow_empty(artifacts["cohort_eligibility_rates"])
    if artifacts["control_denominator_coverage"].exists():
        tables["Control Denominator Coverage"] = _read_csv_allow_empty(artifacts["control_denominator_coverage"])
    if artifacts["chair_bed_missingness_stress"].exists():
        tables["Chair-vs-Bed Missingness Stress"] = _read_csv_allow_empty(artifacts["chair_bed_missingness_stress"])
    if artifacts["falls_prefall_location_probabilities"].exists():
        tables["Pre-Fall Location Probabilities (Livestream)"] = _read_csv_allow_empty(
            artifacts["falls_prefall_location_probabilities"]
        )
    if artifacts["fall_negative_control_effects"].exists():
        tables["Negative-Control Effects (Exploratory)"] = _read_csv_allow_empty(
            artifacts["fall_negative_control_effects"]
        )
    if artifacts["fall_negative_control_match_quality"].exists():
        tables["Negative-Control Match Quality"] = _read_csv_allow_empty(
            artifacts["fall_negative_control_match_quality"]
        )

    return {
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "artifacts": artifacts,
        "run_manifest": read_yaml(artifacts["run_manifest"]),
        "source_profile": read_json(artifacts["source_profile"]),
        "qa_summary": read_json(artifacts["qa_summary"]),
        "negative_control_coverage": (
            read_json(artifacts["fall_negative_control_coverage"])
            if artifacts["fall_negative_control_coverage"].exists()
            else {}
        ),
        "tables": tables,
    }


def load_label_eval_shareholder_inputs(project_root: Path, run_id: str) -> dict[str, Any]:
    artifacts = _artifact_paths(project_root, run_id)
    missing = _missing_required_artifacts(artifacts, LABEL_EVAL_SHAREHOLDER_REQUIRED_ARTIFACT_KEYS)
    if missing:
        listing = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing required label-eval shareholder artifacts for run_id={run_id}:\n{listing}\n"
            "Re-run QA generation for this run."
        )

    tables: dict[str, pd.DataFrame] = {
        "Sequence Metrics": _read_csv_allow_empty(artifacts["label_eval_sequence_metrics"]),
        "Confusion Matrix": _read_csv_allow_empty(artifacts["label_eval_confusion_matrix"]),
        "Probability Quality": _read_csv_allow_empty(artifacts["label_eval_probability_quality"]),
        "Response Timing": _read_csv_allow_empty(artifacts["label_eval_response_timing"]),
        "Population Stats": _read_csv_allow_empty(artifacts["label_eval_population_stats"]),
    }
    if artifacts["label_eval_truth_prefall_location"].exists():
        tables["Truth Pre-Fall Location"] = _read_csv_allow_empty(artifacts["label_eval_truth_prefall_location"])
    if artifacts["label_eval_tag_location_association"].exists():
        tables["Tag-Location Association"] = _read_csv_allow_empty(artifacts["label_eval_tag_location_association"])
    if artifacts["label_eval_signal_location_profile"].exists():
        tables["Signal-Location Profile"] = _read_csv_allow_empty(artifacts["label_eval_signal_location_profile"])
    if artifacts["label_eval_association_summary"].exists():
        tables["Association Summary"] = _read_csv_allow_empty(artifacts["label_eval_association_summary"])
    if artifacts["label_eval_shadow_model_metrics"].exists():
        tables["Shadow Model Metrics"] = _read_csv_allow_empty(artifacts["label_eval_shadow_model_metrics"])
    if artifacts["label_eval_shadow_model_comparison"].exists():
        tables["Shadow Model Comparison"] = _read_csv_allow_empty(artifacts["label_eval_shadow_model_comparison"])
    if artifacts["shadow_ablation"].exists():
        tables["Shadow Ablation"] = _read_csv_allow_empty(artifacts["shadow_ablation"])
    if artifacts["shadow_feature_importance"].exists():
        tables["Shadow Feature Importance"] = _read_csv_allow_empty(artifacts["shadow_feature_importance"])
    if artifacts["shadow_window_sensitivity"].exists():
        tables["Shadow Window Sensitivity"] = _read_csv_allow_empty(artifacts["shadow_window_sensitivity"])
    if artifacts["shadow_cv_fold_metrics"].exists():
        tables["Shadow CV Fold Metrics"] = _read_csv_allow_empty(artifacts["shadow_cv_fold_metrics"])
    if artifacts["onset_events"].exists():
        tables["Onset Events (Exploratory)"] = _read_csv_allow_empty(artifacts["onset_events"])
    if artifacts["onset_case_crossover"].exists():
        tables["Onset Case-Crossover (Exploratory)"] = _read_csv_allow_empty(artifacts["onset_case_crossover"])
    threshold_checks = read_json(artifacts["label_eval_threshold_checks"])

    return {
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "artifacts": artifacts,
        "run_manifest": read_yaml(artifacts["run_manifest"]),
        "source_profile": read_json(artifacts["source_profile"]),
        "qa_summary": read_json(artifacts["qa_summary"]),
        "tables": tables,
        "threshold_checks": threshold_checks,
    }


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def _read_csv_allow_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p><em>No rows available.</em></p>"

    def _fmt_float(value: float) -> str:
        return f"{value:.4f}".rstrip("0").rstrip(".")

    return df.to_html(index=False, classes="tbl", border=0, na_rep="", float_format=_fmt_float)


def _metric_value(df: pd.DataFrame, metric: str) -> float | None:
    if df.empty or "metric" not in df.columns or "value" not in df.columns:
        return None
    selected = df.loc[df["metric"] == metric, "value"]
    if selected.empty:
        return None
    try:
        return float(selected.iloc[0])
    except (TypeError, ValueError):
        return None


def _fmt_value(value: float | int | None, *, decimals: int = 1, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    return f"{value:,.{decimals}f}{suffix}"


def _truncate_label(value: str, limit: int = 22) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}..."


def _display_daypart(value: Any) -> str:
    label = str(value).strip()
    return _DAYPART_DISPLAY.get(label, label)


def _to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _gate_label(value: Any) -> str:
    if isinstance(value, bool):
        return "pass" if value else "fail"
    return "unknown"


def _report_run_context(run_manifest: dict[str, Any]) -> dict[str, Any]:
    gate = run_manifest.get("gate_preflight") if isinstance(run_manifest.get("gate_preflight"), dict) else {}
    requested_mode = str(run_manifest.get("requested_run_mode", "unknown"))
    manifest_mode = str(run_manifest.get("run_mode", "unknown"))
    gate_mode = str(gate.get("effective_run_mode", "unknown"))
    valid_modes = {"inferential_ready", "descriptive_only"}
    effective_mode = gate_mode if gate_mode in valid_modes else manifest_mode
    mode_mismatch = gate_mode in valid_modes and manifest_mode in valid_modes and gate_mode != manifest_mode
    gate_1_pass = gate.get("gate_1_pass")
    gate_2_pass = gate.get("gate_2_pass")
    gate_3_pass = gate.get("gate_3_pass")
    return {
        "requested_mode": requested_mode,
        "manifest_mode": manifest_mode,
        "gate_mode": gate_mode,
        "effective_mode": effective_mode,
        "mode_mismatch": mode_mismatch,
        "gate_1_pass": gate_1_pass,
        "gate_2_pass": gate_2_pass,
        "gate_3_pass": gate_3_pass,
        "gate_1_label": _gate_label(gate_1_pass),
        "gate_2_label": _gate_label(gate_2_pass),
        "gate_3_label": _gate_label(gate_3_pass),
        "gate_1_evidence": gate.get("gate_1_evidence"),
        "gate_2_evidence": gate.get("gate_2_evidence"),
        "gate_3_evidence": gate.get("gate_3_evidence"),
    }


def _render_run_context_html(run_manifest: dict[str, Any]) -> str:
    context = _report_run_context(run_manifest)
    cohort_definition = (
        run_manifest.get("cohort_definition")
        if isinstance(run_manifest.get("cohort_definition"), dict)
        else {}
    )
    lines = [
        "  <section>",
        "    <h2>Run Context</h2>",
        f"    <p><strong>Mode:</strong> {escape(context['effective_mode'])}</p>",
        f"    <p><strong>Requested mode:</strong> {escape(context['requested_mode'])}</p>",
        f"    <p><strong>Gate preflight mode:</strong> {escape(context['gate_mode'])}</p>",
        f"    <p><strong>Manifest mode:</strong> {escape(context['manifest_mode'])}</p>",
        f"    <p><strong>Gate status:</strong> Gate 1={escape(context['gate_1_label'])}, "
        f"Gate 2={escape(context['gate_2_label'])}, Gate 3={escape(context['gate_3_label'])}</p>",
        f"    <p><strong>Target artifact:</strong> <code>{escape(str(run_manifest.get('final_publication_artifact_target', '')))}</code></p>",
    ]
    basis = cohort_definition.get("basis")
    if basis:
        lines.append(f"    <p><strong>Cohort basis:</strong> {escape(str(basis))}</p>")
    intervention_definition = cohort_definition.get("intervention_definition")
    control_definition = cohort_definition.get("control_definition")
    if intervention_definition and control_definition:
        lines.append(
            "    <p><strong>Intervention / control definition:</strong> "
            f"{escape(str(intervention_definition))}; {escape(str(control_definition))}.</p>"
        )
    if context["mode_mismatch"]:
        lines.append(
            "    <p><strong>Consistency warning:</strong> Gate preflight mode and manifest mode disagree; "
            "report defaults to gate preflight mode.</p>"
        )
    lines.extend(["  </section>"])
    return "\n".join(lines)


def _run_mode_limitations_text(context: dict[str, Any], report_kind: str) -> str:
    effective_mode = context.get("effective_mode")
    gate_2_pass = context.get("gate_2_pass")
    if effective_mode != "inferential_ready":
        if gate_2_pass is False:
            return (
                "Gate 2 is not passing, so inferential chair-vs-bed claims remain out of scope; "
                "this report supports descriptive and QA interpretation."
            )
        return (
            "Inferential readiness is not fully confirmed; treat this report as descriptive/QA support only."
        )
    if report_kind == "falls_only":
        return (
            "Run is inference-ready, but this report remains descriptive by design and does not make "
            "population-level effect claims."
        )
    return (
        "Gate 1 and Gate 2 are marked pass for this run. Inferential analyses may proceed in dedicated "
        "model outputs."
    )


def _stakeholder_readout_html(
    run_manifest: dict[str, Any],
    source_profile: dict[str, Any],
    qa_summary: dict[str, Any],
) -> str:
    context = _report_run_context(run_manifest)
    scope_hospital = run_manifest.get("input_tables", {}).get("study_hospital_id", "unknown")
    denominator = source_profile.get("control_denominator_completeness", {})
    valid_rows = int(denominator.get("rows_with_valid_pct_sum", 0) or 0)
    total_rows = int(denominator.get("total_rows", 0) or 0)
    availability_items = [
        f"Scoped footprint is hospital_id={scope_hospital}.",
        f"Control denominator quality: {valid_rows:,}/{total_rows:,} rows with valid pct sum.",
        f"Eligibility passed: {int(qa_summary.get('eligibility_units_passed', 0) or 0):,}/"
        f"{int(qa_summary.get('eligibility_units', 0) or 0):,} units.",
    ]
    pending_items = [
        "Primary adjusted chair-vs-bed estimate (RR/CI) remains outside these descriptive outputs.",
        (
            "Sensitivity analyses (00:00-05:59, 06:00-08:59, 09:00-11:59, 12:00-14:59, "
            "15:00-17:59, 18:00-20:59, 21:00-23:59, weekday/weekend) remain pending "
            "Gate 2-driven inferential execution."
        ),
        "Clinical mechanism taxonomy and one-page nursing protocol remain separate deliverables.",
    ]
    if context.get("effective_mode") == "inferential_ready":
        pending_items[0] = (
            "Primary adjusted chair-vs-bed estimate (RR/CI) should be reported in modeling artifacts/manuscript, "
            "not these descriptive HTML summaries."
        )
    available_lines = "\n".join(f"      <li>{escape(item)}</li>" for item in availability_items)
    pending_lines = "\n".join(f"      <li>{escape(item)}</li>" for item in pending_items)
    return "\n".join(
        [
            "  <section class=\"callout\">",
            "    <h2>Stakeholder Readout (Charter Alignment)</h2>",
            "    <p><strong>Decision snapshot:</strong> "
            f"effective_mode={escape(context['effective_mode'])}, "
            f"gate_1={escape(context['gate_1_label'])}, gate_2={escape(context['gate_2_label'])}, "
            f"gate_3={escape(context['gate_3_label'])}.</p>",
            "    <p><strong>Available now</strong></p>",
            "    <ul>",
            available_lines,
            "    </ul>",
            "    <p><strong>Still pending for full charter completion</strong></p>",
            "    <ul>",
            pending_lines,
            "    </ul>",
            "  </section>",
        ]
    )


def _derive_daypart(hour: int | float | None) -> str:
    if pd.isna(hour):
        return "unknown"
    return classify_hour(hour)


def _site_pareto_chart_html(site_df: pd.DataFrame, total_falls: int) -> str:
    if site_df.empty or not {"hospital_name", "falls"}.issubset(site_df.columns):
        return ""

    full_df = site_df.copy()
    full_df["hospital_name"] = full_df["hospital_name"].astype(str).fillna("unknown")
    full_df["falls"] = _to_numeric_series(full_df["falls"])
    full_df = full_df.loc[full_df["falls"] > 0].sort_values(
        ["falls", "hospital_name"], ascending=[False, True], kind="mergesort"
    )
    chart_df = full_df.head(8).copy()
    if chart_df.empty:
        return ""

    counted = float(chart_df["falls"].sum())
    cumulative = (chart_df["falls"].cumsum() / counted).tolist() if counted else [0.0] * len(chart_df.index)
    max_falls = float(chart_df["falls"].max())

    width = 920
    height = 320
    left = 72
    right = 40
    top = 24
    bottom = 104
    plot_w = width - left - right
    plot_h = height - top - bottom
    n_sites = len(chart_df.index)
    slot = plot_w / max(n_sites, 1)
    bar_w = slot * 0.7

    bars: list[str] = []
    labels: list[str] = []
    curve_points: list[str] = []
    for idx, (_, row) in enumerate(chart_df.iterrows()):
        falls = float(row["falls"])
        site = str(row["hospital_name"])
        x = left + (idx * slot) + ((slot - bar_w) / 2)
        bar_h = (falls / max_falls) * plot_h if max_falls > 0 else 0.0
        y = top + plot_h - bar_h
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" rx="2" '
            f'fill="#1F4E79" opacity="0.92"></rect>'
        )
        labels.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{top + plot_h + 24:.2f}" text-anchor="middle" '
            f'font-size="{_POSTER_TICK_FONT}" fill="#222">{escape(_truncate_label(site, 16))}</text>'
        )
        labels.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{y - 10:.2f}" text-anchor="middle" '
            f'font-size="{_POSTER_VALUE_FONT}" fill="#0F2740">'
            f"{int(round(falls))}</text>"
        )
        cum_y = top + plot_h - (cumulative[idx] * plot_h)
        curve_points.append(f"{x + bar_w / 2:.2f},{cum_y:.2f}")

    y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_guides = []
    for tick in y_ticks:
        y = top + plot_h - (tick * plot_h)
        y_guides.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#E0E5EA"/>')
        y_guides.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="{_POSTER_TICK_FONT}" fill="#555">{tick:.0%}</text>'
        )

    shown_total = int(round(counted))
    total_label = f"{shown_total}/{total_falls}" if total_falls > 0 else str(shown_total)
    other_sites = max(len(full_df.index) - len(chart_df.index), 0)
    other_falls = max(int(round(float(full_df["falls"].sum()))) - shown_total, 0)
    other_share = (other_falls / total_falls) if total_falls else 0.0
    if other_sites > 0 and total_falls > 0:
        detail_caption = (
            f"Top sites shown ({total_label} falls in chart scope); other {other_sites} site(s) contribute "
            f"{other_falls}/{total_falls} ({other_share:.1%})."
        )
    else:
        detail_caption = f"Top sites shown ({total_label} falls in chart scope)."
    return "\n".join(
        [
            '<figure class="viz-card">',
            "  <h3>Site Pareto</h3>",
            f"  <p class=\"viz-caption\">{escape(detail_caption)}</p>",
            f'  <svg viewBox="0 0 {width} {height}" role="img" aria-label="Site pareto chart">',
            *y_guides,
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#6D7785"/>',
            *bars,
            f'<polyline points="{" ".join(curve_points)}" fill="none" stroke="#B33F62" stroke-width="2.5"/>',
            *labels,
            f'<text x="{left + plot_w - 2:.2f}" y="{top + 16:.2f}" text-anchor="end" '
            f'font-size="{_POSTER_LEGEND_FONT}" fill="#B33F62">'
            "Cumulative share</text>",
            "  </svg>",
            "</figure>",
        ]
    )


def _location_compare_chart_html(location_df: pd.DataFrame, location_prob_df: pd.DataFrame) -> str:
    if location_df.empty or location_prob_df.empty:
        return ""
    if not {"prefall_location_label", "falls"}.issubset(location_df.columns):
        return ""
    if not {"prefall_location_label", "expected_falls"}.issubset(location_prob_df.columns):
        return ""

    observed = (
        location_df[["prefall_location_label", "falls"]]
        .copy()
        .assign(falls=lambda df: _to_numeric_series(df["falls"]))
    )
    expected = (
        location_prob_df[["prefall_location_label", "expected_falls"]]
        .copy()
        .assign(expected_falls=lambda df: _to_numeric_series(df["expected_falls"]))
    )
    merged = observed.merge(expected, on="prefall_location_label", how="outer").fillna(0.0)
    merged["prefall_location_label"] = merged["prefall_location_label"].astype(str).str.strip().str.lower()
    merged = merged.loc[merged["prefall_location_label"] != ""]
    merged = (
        merged.groupby("prefall_location_label", dropna=False)[["falls", "expected_falls"]]
        .sum()
        .reset_index()
    )
    if merged.empty:
        return ""

    order = ["chair", "bed", "room", "no_patient"]
    rank_lookup = {label: idx for idx, label in enumerate(order)}
    merged["rank"] = merged["prefall_location_label"].map(rank_lookup).fillna(len(order))
    merged = merged.sort_values(
        ["rank", "prefall_location_label", "falls"], ascending=[True, True, False], kind="mergesort"
    ).reset_index(drop=True)
    max_y = float(max(merged["falls"].max(), merged["expected_falls"].max()))
    if max_y <= 0:
        return ""

    width = 920
    height = 320
    left = 68
    right = 32
    top = 22
    bottom = 104
    plot_w = width - left - right
    plot_h = height - top - bottom
    n_groups = len(merged.index)
    slot = plot_w / max(n_groups, 1)
    bar_w = slot * 0.29
    gap = slot * 0.08

    parts: list[str] = [
        '<figure class="viz-card">',
        "  <h3>Observed vs Probability-Weighted Location Burden</h3>",
        "  <p class=\"viz-caption\">Dark bar = observed hard-label falls, light bar = expected falls.</p>",
        f'  <svg viewBox="0 0 {width} {height}" role="img" aria-label="Observed vs expected location chart">',
    ]
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + plot_h - (tick * plot_h)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#E0E5EA"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#6D7785"/>')

    for idx, (_, row) in enumerate(merged.iterrows()):
        label = str(row["prefall_location_label"])
        observed_val = float(row["falls"])
        expected_val = float(row["expected_falls"])
        x0 = left + idx * slot + (slot - (2 * bar_w + gap)) / 2
        color = _LOCATION_COLORS.get(label, _LOCATION_COLORS["unknown"])
        observed_h = (observed_val / max_y) * plot_h
        expected_h = (expected_val / max_y) * plot_h
        parts.append(
            f'<rect x="{x0:.2f}" y="{top + plot_h - observed_h:.2f}" width="{bar_w:.2f}" height="{observed_h:.2f}" '
            f'fill="{color}" opacity="0.95"></rect>'
        )
        parts.append(
            f'<rect x="{x0 + bar_w + gap:.2f}" y="{top + plot_h - expected_h:.2f}" width="{bar_w:.2f}" '
            f'height="{expected_h:.2f}" fill="{color}" opacity="0.45"></rect>'
        )
        parts.append(
            f'<text x="{x0 + bar_w / 2:.2f}" y="{top + plot_h - observed_h - 6:.2f}" text-anchor="middle" '
            f'font-size="{_POSTER_VALUE_FONT}" fill="#333">{observed_val:.0f}</text>'
        )
        parts.append(
            f'<text x="{x0 + bar_w + gap + bar_w / 2:.2f}" y="{top + plot_h - expected_h - 6:.2f}" '
            f'text-anchor="middle" font-size="{_POSTER_VALUE_FONT}" fill="#555">{expected_val:.1f}</text>'
        )
        parts.append(
            f'<text x="{x0 + bar_w + gap / 2:.2f}" y="{top + plot_h + 24:.2f}" text-anchor="middle" '
            f'font-size="{_POSTER_TICK_FONT}" fill="#222">{escape(label)}</text>'
        )

    parts.extend(
        [
            f'<rect x="{left}" y="{height - 30}" width="12" height="12" fill="#334F6E"></rect>',
            f'<text x="{left + 18}" y="{height - 18}" font-size="{_POSTER_LEGEND_FONT}" fill="#334F6E">'
            "Observed falls</text>",
            f'<rect x="{left + 132}" y="{height - 30}" width="12" height="12" fill="#8EA5BD"></rect>',
            f'<text x="{left + 150}" y="{height - 18}" font-size="{_POSTER_LEGEND_FONT}" fill="#5B6978">'
            "Expected falls</text>",
            "  </svg>",
            "</figure>",
        ]
    )
    return "\n".join(parts)


def _hourly_kde_series(hours: list[float], weights: list[float], bandwidth: float = 1.35) -> list[float]:
    if not hours or not weights:
        return [0.0] * 24
    total_weight = sum(weights)
    if total_weight <= 0:
        return [0.0] * 24

    expanded_hours: list[float] = []
    expanded_weights: list[float] = []
    for hour, weight in zip(hours, weights, strict=False):
        if weight <= 0:
            continue
        for shift in (-24.0, 0.0, 24.0):
            expanded_hours.append(hour + shift)
            expanded_weights.append(weight)
    if not expanded_hours:
        return [0.0] * 24

    norm = bandwidth * math.sqrt(2.0 * math.pi)
    densities: list[float] = []
    for center in range(24):
        acc = 0.0
        for hour, weight in zip(expanded_hours, expanded_weights, strict=False):
            z = (center - hour) / bandwidth
            acc += weight * math.exp(-0.5 * z * z)
        densities.append(acc / (total_weight * norm))
    return densities


def _hourly_kde_chart_html(patient_day_hour_df: pd.DataFrame) -> str:
    if patient_day_hour_df.empty:
        return ""
    required = {"fall_hour_local", "prefall_location_label", "falls"}
    if not required.issubset(patient_day_hour_df.columns):
        return ""

    frame = patient_day_hour_df.copy()
    frame["fall_hour_local"] = pd.to_numeric(frame["fall_hour_local"], errors="coerce")
    frame["falls"] = _to_numeric_series(frame["falls"])
    frame["prefall_location_label"] = frame["prefall_location_label"].astype(str).str.strip().str.lower()
    frame = frame.dropna(subset=["fall_hour_local"])
    frame = frame.loc[(frame["fall_hour_local"] >= 0) & (frame["fall_hour_local"] <= 23) & (frame["falls"] > 0)]
    if frame.empty:
        return ""

    grouped = (
        frame.groupby(["prefall_location_label", "fall_hour_local"], dropna=False)["falls"]
        .sum()
        .reset_index()
        .sort_values(["prefall_location_label", "fall_hour_local"])
    )
    present_labels = sorted(grouped["prefall_location_label"].unique().tolist())
    preferred_order = ["chair", "bed", "room", "no_patient"]
    locations = [label for label in preferred_order if label in present_labels]
    overflow = [label for label in present_labels if label not in preferred_order]
    if len(locations) < 4:
        locations.extend(overflow[: 4 - len(locations)])
    if not locations:
        return ""
    curves: dict[str, list[float]] = {}
    for label in locations:
        subset = grouped.loc[grouped["prefall_location_label"] == label]
        curves[label] = _hourly_kde_series(
            subset["fall_hour_local"].astype(float).tolist(),
            subset["falls"].astype(float).tolist(),
        )
    y_max = max((max(series) for series in curves.values()), default=0.0)
    if y_max <= 0:
        return ""

    width = 920
    height = 340
    left = 68
    right = 36
    top = 24
    bottom = 96
    plot_w = width - left - right
    plot_h = height - top - bottom

    parts: list[str] = [
        '<figure class="viz-card">',
        "  <h3>Smoothed Hourly Density by Pre-Fall Location (KDE)</h3>",
        "  <p class=\"viz-caption\">Kernel-smoothed temporal density (circular 24-hour kernel, weighted by falls).</p>",
        f'  <svg viewBox="0 0 {width} {height}" role="img" aria-label="Hourly KDE by pre-fall location">',
    ]
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + plot_h - (tick * plot_h)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#E0E5EA"/>')
    for hour in [0, 4, 8, 12, 16, 20, 23]:
        x = left + (hour / 23.0) * plot_w if hour > 0 else left
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#F0F3F6"/>')
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 24:.2f}" text-anchor="middle" '
            f'font-size="{_POSTER_TICK_FONT}" fill="#555">{hour:02d}</text>'
        )
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#6D7785"/>')

    legend_y = top + 20
    for idx, label in enumerate(locations):
        color = _LOCATION_COLORS.get(label, _LOCATION_COLORS["unknown"])
        series = curves[label]
        points = []
        for hour, density in enumerate(series):
            x = left + (hour / 23.0) * plot_w if hour > 0 else left
            y = top + plot_h - ((density / y_max) * plot_h)
            points.append(f"{x:.2f},{y:.2f}")
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"></polyline>')
        lx = left + 10 + (idx * 170)
        parts.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 18}" y2="{legend_y}" stroke="{color}" stroke-width="2.5"></line>')
        parts.append(
            f'<text x="{lx + 24}" y="{legend_y + 5}" font-size="{_POSTER_LEGEND_FONT}" fill="#2A2A2A">'
            f"{escape(label)}</text>"
        )
    parts.extend(["  </svg>", "</figure>"])
    return "\n".join(parts)


def _weekday_daypart_heatmap_html(patient_day_hour_df: pd.DataFrame) -> str:
    required = {"fall_date_local", "fall_hour_local", "falls"}
    if patient_day_hour_df.empty or not required.issubset(patient_day_hour_df.columns):
        return ""

    frame = patient_day_hour_df.copy()
    frame["falls"] = _to_numeric_series(frame["falls"])
    frame["fall_hour_local"] = pd.to_numeric(frame["fall_hour_local"], errors="coerce")
    frame["fall_date_local"] = pd.to_datetime(frame["fall_date_local"], errors="coerce")
    frame["weekday"] = frame["fall_date_local"].dt.day_name().fillna("unknown")
    frame["daypart"] = frame["fall_hour_local"].apply(_derive_daypart)
    frame = frame.loc[frame["falls"] > 0]
    if frame.empty:
        return ""

    matrix = (
        frame.groupby(["weekday", "daypart"], dropna=False)["falls"]
        .sum()
        .reset_index()
        .pivot(index="weekday", columns="daypart", values="falls")
        .reindex(index=_WEEKDAY_ORDER, columns=_DAYPART_ORDER, fill_value=0.0)
        .fillna(0.0)
    )
    max_value = float(matrix.to_numpy().max())
    if max_value <= 0:
        return ""

    cell_w = 140
    cell_h = 40
    left = 170
    top = 48
    width = left + (cell_w * len(_DAYPART_ORDER)) + 20
    height = top + (cell_h * len(_WEEKDAY_ORDER)) + 54
    parts = [
        '<figure class="viz-card">',
        "  <h3>Weekday x Daypart Heatmap</h3>",
        "  <p class=\"viz-caption\">Falls weighted by patient-day-hour buckets.</p>",
        f'  <svg viewBox="0 0 {width} {height}" role="img" aria-label="Weekday by daypart heatmap">',
    ]
    for col_idx, daypart in enumerate(_DAYPART_ORDER):
        x = left + (col_idx * cell_w)
        label = _DAYPART_DISPLAY.get(daypart, daypart)
        parts.append(
            f'<text x="{x + (cell_w / 2):.2f}" y="{top - 14:.2f}" text-anchor="middle" '
            f'font-size="{_POSTER_LEGEND_FONT}" fill="#333">{escape(label)}</text>'
        )
    for row_idx, weekday in enumerate(_WEEKDAY_ORDER):
        y = top + (row_idx * cell_h)
        parts.append(
            f'<text x="{left - 10:.2f}" y="{y + (cell_h / 2) + 5:.2f}" text-anchor="end" '
            f'font-size="{_POSTER_LEGEND_FONT}" fill="#333">{escape(weekday)}</text>'
        )
        for col_idx, daypart in enumerate(_DAYPART_ORDER):
            x = left + (col_idx * cell_w)
            value = float(matrix.loc[weekday, daypart])
            intensity = value / max_value if max_value else 0.0
            alpha = 0.15 + (0.85 * intensity)
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w - 2:.2f}" height="{cell_h - 2:.2f}" '
                f'fill="#1F4E79" opacity="{alpha:.3f}"></rect>'
            )
            parts.append(
                f'<text x="{x + (cell_w / 2):.2f}" y="{y + (cell_h / 2) + 6:.2f}" text-anchor="middle" '
                f'font-size="{_POSTER_VALUE_FONT}" fill="#111">{int(round(value))}</text>'
            )
    parts.extend(["  </svg>", "</figure>"])
    return "\n".join(parts)


def _negative_control_effects_card_html(effects: pd.DataFrame) -> str:
    required = {"metric", "paired_mean_delta_hazard_minus_control"}
    if effects.empty or not required.issubset(effects.columns):
        return ""
    frame = effects.copy()
    frame["metric"] = frame["metric"].astype(str)
    frame["paired_mean_delta_hazard_minus_control"] = _to_numeric_series(
        frame["paired_mean_delta_hazard_minus_control"]
    )
    frame["abs_delta"] = frame["paired_mean_delta_hazard_minus_control"].abs()
    frame = frame.sort_values(["abs_delta", "metric"], ascending=[False, True], kind="mergesort").head(6)
    if frame.empty:
        return ""
    return _bar_chart_card_html(
        "Negative-Control Anchor Check",
        "Absolute paired deltas (hazard minus non-fall control); smaller is better.",
        frame["metric"].tolist(),
        frame["abs_delta"].astype(float).tolist(),
        color="#6b7280",
        label_limit=26,
    )


def _pending_control_source_card_html(status: str | None) -> str:
    if str(status or "").strip().lower() != "pending_external":
        return ""
    return "\n".join(
        [
            '<figure class="viz-card">',
            "  <h3>Non-Fall Controls Pending</h3>",
            "  <p class=\"viz-caption\">Cross-monitor negative controls are withheld until the non-fall derivatives cohort is available.</p>",
            "  <div class=\"callout\">Same-monitor case-crossover outputs can still appear independently when their source remains enabled.</div>",
            "</figure>",
        ]
    )


def _negative_control_coverage_card_html(coverage: dict[str, Any]) -> str:
    if not coverage:
        return ""
    chunk_count = _fmt_int(coverage.get("source_chunk_count", 0))
    matched_falls = _fmt_int(coverage.get("fall_events_with_selected_controls", 0))
    eligible_falls = _fmt_int(coverage.get("fall_events_eligible_for_matching", 0))
    history_status = str(coverage.get("history_status") or "unknown").replace("_", " ")
    effective_start = coverage.get("effective_source_start_ts_utc") or "n/a"
    return "\n".join(
        [
            '<figure class="viz-card">',
            "  <h3>Negative-Control Coverage</h3>",
            "  <p class=\"viz-caption\">Availability of the non-fall cohort after source-history and division matching filters.</p>",
            "  <div class=\"callout\">"
            f"Source chunks: {chunk_count}<br/>"
            f"Eligible falls: {eligible_falls}<br/>"
            f"Falls with matched controls: {matched_falls}<br/>"
            f"History status: {escape(history_status)}<br/>"
            f"Effective source start: <code>{escape(str(effective_start))}</code>"
            "</div>",
            "</figure>",
        ]
    )


def _missingness_stress_card_html(stress: pd.DataFrame) -> str:
    required = {
        "excluded_low_confidence_pct",
        "position",
        "chair_to_bed_rate_ratio_expected",
        "chair_to_bed_rate_ratio_confidence_weighted",
    }
    if stress.empty or not required.issubset(stress.columns):
        return ""
    frame = stress.copy()
    frame["position"] = frame["position"].astype(str).str.lower()
    frame["excluded_low_confidence_pct"] = pd.to_numeric(frame["excluded_low_confidence_pct"], errors="coerce")
    frame["chair_to_bed_rate_ratio_expected"] = pd.to_numeric(
        frame["chair_to_bed_rate_ratio_expected"], errors="coerce"
    )
    frame["chair_to_bed_rate_ratio_confidence_weighted"] = pd.to_numeric(
        frame["chair_to_bed_rate_ratio_confidence_weighted"], errors="coerce"
    )
    frame = frame.loc[frame["position"] == "chair"].copy()
    frame = frame.dropna(
        subset=[
            "excluded_low_confidence_pct",
            "chair_to_bed_rate_ratio_expected",
            "chair_to_bed_rate_ratio_confidence_weighted",
        ]
    )
    if frame.empty:
        return ""
    frame = frame.sort_values("excluded_low_confidence_pct", kind="mergesort")
    labels = [f"{int(value)}%" for value in frame["excluded_low_confidence_pct"].tolist()]
    expected = frame["chair_to_bed_rate_ratio_expected"].astype(float).tolist()
    weighted = frame["chair_to_bed_rate_ratio_confidence_weighted"].astype(float).tolist()
    return _dual_bar_chart_card_html(
        "Missingness Stress (Chair/Bed RR)",
        "Chair-to-bed rate ratio after excluding low-confidence events by percentile.",
        labels,
        expected,
        weighted,
        "Expected RR",
        "Confidence-weighted RR",
        left_color="#0B3C5D",
        right_color="#B45309",
        label_limit=12,
    )


def _visualizations_html(total_falls: int, tables: dict[str, pd.DataFrame], chart_frames: dict[str, pd.DataFrame]) -> str:
    latency_df = tables.get("Response Latency (Livestream)", pd.DataFrame())
    site_df = tables.get("Falls by Site", pd.DataFrame())
    location_df = tables.get("Pre-Fall Location (Livestream)", pd.DataFrame())
    location_prob_df = tables.get("Pre-Fall Location Probabilities (Livestream)", pd.DataFrame())
    negative_control_effects = tables.get("Negative-Control Effects (Exploratory)", pd.DataFrame())
    patient_day_hour_df = chart_frames.get("Patient-Day-Hour Location (Livestream)", pd.DataFrame())
    response_rate = _metric_value(latency_df, "response_rate")
    p50_latency = _metric_value(latency_df, "latency_p50_seconds")
    p90_latency = _metric_value(latency_df, "latency_p90_seconds")
    top_location = "n/a"
    if not location_df.empty and {"prefall_location_label", "falls"}.issubset(location_df.columns):
        ranked = (
            location_df.assign(falls=_to_numeric_series(location_df["falls"]))
            .sort_values(["falls", "prefall_location_label"], ascending=[False, True], kind="mergesort")
            .reset_index(drop=True)
        )
        if not ranked.empty:
            top_location = str(ranked.iloc[0]["prefall_location_label"])

    cards = [
        "\n".join(
            [
                '<figure class="viz-card viz-card-wide">',
                "  <h3>KPI Snapshot</h3>",
                '  <div class="kpi-row">',
                '    <div class="kpi-tile"><span class="kpi-label">Total Falls</span><strong class="kpi-value">'
                f"{_fmt_int(total_falls)}</strong></div>",
                '    <div class="kpi-tile"><span class="kpi-label">Response Rate</span><strong class="kpi-value">'
                f"{_fmt_value(response_rate * 100.0, decimals=1, suffix='%') if response_rate is not None else 'n/a'}</strong></div>",
                '    <div class="kpi-tile"><span class="kpi-label">Latency P50</span><strong class="kpi-value">'
                f"{_fmt_value(p50_latency, decimals=1, suffix='s')}</strong></div>",
                '    <div class="kpi-tile"><span class="kpi-label">Latency P90</span><strong class="kpi-value">'
                f"{_fmt_value(p90_latency, decimals=1, suffix='s')}</strong></div>",
                '    <div class="kpi-tile"><span class="kpi-label">Top Pre-Fall Location</span><strong class="kpi-value">'
                f"{escape(top_location)}</strong></div>",
                "  </div>",
                "</figure>",
            ]
        ),
        _site_pareto_chart_html(site_df, total_falls),
        _location_compare_chart_html(location_df, location_prob_df),
        _falls_interactive_location_focus_html(location_df, location_prob_df),
        _hourly_kde_chart_html(patient_day_hour_df),
        _weekday_daypart_heatmap_html(patient_day_hour_df),
        _negative_control_effects_card_html(negative_control_effects),
    ]
    cards = [card for card in cards if card]
    if not cards:
        return ""

    return "\n".join(
        [
            "  <section>",
            "    <h2>Visual Summary</h2>",
            "    <div class=\"viz-grid\">",
            "\n".join(cards),
            "    </div>",
            "  </section>",
        ]
    )


def _table_appendix_html(
    tables: dict[str, pd.DataFrame],
    keep_titles: set[str] | None = None,
    *,
    heading: str = "Table Appendix",
) -> str:
    if not tables:
        return ""
    selected = (
        {title: df for title, df in tables.items() if title in keep_titles}
        if keep_titles is not None
        else dict(tables)
    )
    if not selected:
        return ""
    sections = "\n".join(
        (
            f"      <section class=\"appendix-section\"><h3>{escape(title)}</h3>"
            f"{_table_html(df)}</section>"
        )
        for title, df in selected.items()
    )
    return "\n".join(
        [
            "  <section>",
            f"    <h2>{escape(heading)}</h2>",
            "    <details>",
            f"      <summary>Show detailed tables ({len(selected)})</summary>",
            sections,
            "    </details>",
            "  </section>",
        ]
    )


def _bar_chart_card_html(
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    color: str = "#1F4E79",
    label_limit: int = 20,
) -> str:
    if not labels or not values or len(labels) != len(values):
        return ""
    max_value = max(values)
    if max_value <= 0:
        return ""
    width = 920
    height = 320
    left = 68
    right = 32
    top = 24
    bottom = 96
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(values)
    slot = plot_w / max(n, 1)
    bar_w = slot * 0.62
    bars: list[str] = []
    for idx, value in enumerate(values):
        x = left + (idx * slot) + ((slot - bar_w) / 2)
        bar_h = (value / max_value) * plot_h if max_value else 0.0
        y = top + plot_h - bar_h
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{color}" opacity="0.9"></rect>'
        )
        bars.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-size="14" fill="#1f2a35">{value:,.1f}</text>'
        )
        bars.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{top + plot_h + 22:.2f}" text-anchor="middle" font-size="14" fill="#333">{escape(_truncate_label(labels[idx], label_limit))}</text>'
        )
    return "\n".join(
        [
            '<figure class="viz-card">',
            f"  <h3>{escape(title)}</h3>",
            f"  <p class=\"viz-caption\">{escape(subtitle)}</p>",
            f'  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#6D7785"></line>',
            "\n".join(bars),
            "  </svg>",
            "</figure>",
        ]
    )


def _dual_bar_chart_card_html(
    title: str,
    subtitle: str,
    labels: list[str],
    left_values: list[float],
    right_values: list[float],
    left_label: str,
    right_label: str,
    left_color: str = "#1F4E79",
    right_color: str = "#B33F62",
    label_limit: int = 20,
) -> str:
    if (
        not labels
        or not left_values
        or not right_values
        or len(labels) != len(left_values)
        or len(labels) != len(right_values)
    ):
        return ""
    max_value = max(max(left_values), max(right_values))
    if max_value <= 0:
        return ""
    width = 920
    height = 330
    left = 68
    right = 32
    top = 24
    bottom = 104
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(labels)
    slot = plot_w / max(n, 1)
    bar_w = slot * 0.28
    gap = slot * 0.08
    parts: list[str] = []
    for idx, label in enumerate(labels):
        x = left + (idx * slot) + ((slot - (2 * bar_w + gap)) / 2)
        lval = left_values[idx]
        rval = right_values[idx]
        lh = (lval / max_value) * plot_h
        rh = (rval / max_value) * plot_h
        parts.append(
            f'<rect x="{x:.2f}" y="{top + plot_h - lh:.2f}" width="{bar_w:.2f}" height="{lh:.2f}" fill="{left_color}" opacity="0.9"></rect>'
        )
        parts.append(
            f'<rect x="{x + bar_w + gap:.2f}" y="{top + plot_h - rh:.2f}" width="{bar_w:.2f}" height="{rh:.2f}" fill="{right_color}" opacity="0.8"></rect>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{top + plot_h - lh - 6:.2f}" text-anchor="middle" font-size="13" fill="#1f2a35">{lval:,.1f}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w + gap + bar_w / 2:.2f}" y="{top + plot_h - rh - 6:.2f}" text-anchor="middle" font-size="13" fill="#1f2a35">{rval:,.1f}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w + gap / 2:.2f}" y="{top + plot_h + 22:.2f}" text-anchor="middle" font-size="14" fill="#333">{escape(_truncate_label(label, label_limit))}</text>'
        )
    return "\n".join(
        [
            '<figure class="viz-card">',
            f"  <h3>{escape(title)}</h3>",
            f"  <p class=\"viz-caption\">{escape(subtitle)}</p>",
            f'  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#6D7785"></line>',
            "\n".join(parts),
            f'<rect x="{left}" y="{height - 30}" width="12" height="12" fill="{left_color}"></rect>',
            f'<text x="{left + 18}" y="{height - 18}" font-size="14" fill="#2a2a2a">{escape(left_label)}</text>',
            f'<rect x="{left + 180}" y="{height - 30}" width="12" height="12" fill="{right_color}"></rect>',
            f'<text x="{left + 198}" y="{height - 18}" font-size="14" fill="#2a2a2a">{escape(right_label)}</text>',
            "  </svg>",
            "</figure>",
        ]
    )


def _interp_hex(start_hex: str, end_hex: str, fraction: float) -> str:
    frac = max(0.0, min(1.0, fraction))
    start = start_hex.lstrip("#")
    end = end_hex.lstrip("#")
    if len(start) != 6 or len(end) != 6:
        return "#D9E6F2"
    sr, sg, sb = int(start[0:2], 16), int(start[2:4], 16), int(start[4:6], 16)
    er, eg, eb = int(end[0:2], 16), int(end[2:4], 16), int(end[4:6], 16)
    r = int(sr + (er - sr) * frac)
    g = int(sg + (eg - sg) * frac)
    b = int(sb + (eb - sb) * frac)
    return f"#{r:02X}{g:02X}{b:02X}"


def _matrix_heatmap_card_html(
    title: str,
    subtitle: str,
    x_labels: list[str],
    y_labels: list[str],
    values: dict[tuple[str, str], float],
    *,
    value_decimals: int = 2,
    x_label_limit: int = 14,
    y_label_limit: int = 16,
) -> str:
    if not x_labels or not y_labels:
        return ""
    width = max(920, 160 + len(x_labels) * 72)
    height = 280 + len(y_labels) * 36
    left = 130
    right = 40
    top = 30
    bottom = 100
    plot_w = width - left - right
    plot_h = height - top - bottom
    cell_w = plot_w / max(len(x_labels), 1)
    cell_h = plot_h / max(len(y_labels), 1)
    max_value = max([float(v) for v in values.values()] or [0.0])
    labels_x = []
    labels_y = []
    cells = []
    for col_idx, x_label in enumerate(x_labels):
        x = left + col_idx * cell_w
        labels_x.append(
            f'<text x="{x + cell_w / 2:.2f}" y="{top + plot_h + 28:.2f}" text-anchor="middle" '
            f'font-size="12" fill="#2a2a2a">{escape(_truncate_label(str(x_label), x_label_limit))}</text>'
        )
        for row_idx, y_label in enumerate(y_labels):
            y = top + row_idx * cell_h
            value = float(values.get((str(y_label), str(x_label)), 0.0))
            frac = (value / max_value) if max_value else 0.0
            fill = _interp_hex("#F2F5FA", "#1F4E79", frac)
            cells.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w - 1:.2f}" height="{cell_h - 1:.2f}" '
                f'fill="{fill}" stroke="#FFFFFF" stroke-width="0.6"></rect>'
            )
            cells.append(
                f'<text x="{x + cell_w / 2:.2f}" y="{y + cell_h / 2 + 4:.2f}" text-anchor="middle" '
                f'font-size="11" fill="#122230">{value:.{value_decimals}f}</text>'
            )
    for row_idx, y_label in enumerate(y_labels):
        y = top + row_idx * cell_h
        labels_y.append(
            f'<text x="{left - 8:.2f}" y="{y + cell_h / 2 + 4:.2f}" text-anchor="end" '
            f'font-size="12" fill="#2a2a2a">{escape(_truncate_label(str(y_label), y_label_limit))}</text>'
        )
    return "\n".join(
        [
            '<figure class="viz-card">',
            f"  <h3>{escape(title)}</h3>",
            f"  <p class=\"viz-caption\">{escape(subtitle)}</p>",
            f'  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
            *cells,
            *labels_x,
            *labels_y,
            "  </svg>",
            "</figure>",
        ]
    )


def _prefall_probability_breakdown_small_multiples_html(breakdown_df: pd.DataFrame) -> str:
    required = {"grouping", "group_value", "prefall_location_label", "pct_expected_in_group"}
    if breakdown_df.empty or not required.issubset(breakdown_df.columns):
        return ""
    frame = breakdown_df.copy()
    frame["grouping"] = frame["grouping"].astype(str)
    frame["group_value"] = frame["group_value"].astype(str)
    frame["prefall_location_label"] = frame["prefall_location_label"].astype(str)
    frame["pct_expected_in_group"] = _to_numeric_series(frame["pct_expected_in_group"])
    cards: list[str] = []
    grouping_order = ["daypart", "weekday", "hospital_id", "division_id"]
    all_groupings = list(dict.fromkeys(grouping_order + sorted(frame["grouping"].unique().tolist())))
    for grouping in all_groupings:
        sub = frame.loc[frame["grouping"] == grouping].copy()
        if sub.empty:
            continue
        if grouping == "daypart":
            x_labels = [gp for gp in _DAYPART_ORDER if gp in set(sub["group_value"])]
            y_labels = ["chair", "bed", "room", "no_patient"]
        elif grouping == "weekday":
            x_labels = [gp for gp in _WEEKDAY_ORDER if gp in set(sub["group_value"])]
            y_labels = ["chair", "bed", "room", "no_patient"]
        else:
            x_labels = sorted(sub["group_value"].unique().tolist())
            y_labels = ["chair", "bed", "room", "no_patient"]
        value_map: dict[tuple[str, str], float] = {}
        grouped = (
            sub.groupby(["prefall_location_label", "group_value"], dropna=False)["pct_expected_in_group"]
            .sum()
            .reset_index()
        )
        for _, row in grouped.iterrows():
            value_map[(str(row["prefall_location_label"]), str(row["group_value"]))] = float(row["pct_expected_in_group"])
        cards.append(
            _matrix_heatmap_card_html(
                f"Pre-Fall Probability Breakdown: {grouping}",
                "Small multiples of percent expected falls in group.",
                x_labels,
                y_labels,
                value_map,
                value_decimals=2,
            )
        )
    if not cards:
        return ""
    return "\n".join(
        [
            '<div class="viz-card viz-card-wide">',
            "  <h3>Pre-Fall Probability Breakdown Small Multiples</h3>",
            "  <p class=\"viz-caption\">Each panel shows percent expected burden by group and location label.</p>",
            '  <div class="appendix-mini-grid">',
            "\n".join(cards),
            "  </div>",
            "</div>",
        ]
    )


def _cohort_duration_interval_chart_html(duration_df: pd.DataFrame) -> str:
    required = {"cohort_type", "min", "25%", "50%", "75%", "max", "mean"}
    if duration_df.empty or not required.issubset(duration_df.columns):
        return ""
    frame = duration_df.copy()
    for col in ["min", "25%", "50%", "75%", "max", "mean"]:
        frame[col] = _to_numeric_series(frame[col])
    frame["cohort_type"] = frame["cohort_type"].astype(str)
    max_val = float(frame["max"].max())
    if max_val <= 0:
        return ""
    width = 920
    height = 320
    left = 90
    right = 30
    top = 24
    bottom = 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    slot = plot_w / max(len(frame), 1)
    rows: list[str] = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        x_mid = left + (idx * slot) + (slot / 2)
        q_min = float(row["min"])
        q25 = float(row["25%"])
        q50 = float(row["50%"])
        q75 = float(row["75%"])
        q_max = float(row["max"])
        q_mean = float(row["mean"])
        def y(value: float) -> float:
            return top + plot_h - (value / max_val) * plot_h

        rows.append(f'<line x1="{x_mid:.2f}" y1="{y(q_min):.2f}" x2="{x_mid:.2f}" y2="{y(q_max):.2f}" stroke="#1F4E79" stroke-width="2"></line>')
        rows.append(
            f'<rect x="{x_mid - 20:.2f}" y="{y(q75):.2f}" width="40" height="{max(2.0, y(q25) - y(q75)):.2f}" '
            'fill="#9FC5E8" opacity="0.85"></rect>'
        )
        rows.append(f'<line x1="{x_mid - 22:.2f}" y1="{y(q50):.2f}" x2="{x_mid + 22:.2f}" y2="{y(q50):.2f}" stroke="#0B3C5D" stroke-width="2.2"></line>')
        rows.append(f'<circle cx="{x_mid:.2f}" cy="{y(q_mean):.2f}" r="4.5" fill="#B33F62"></circle>')
        rows.append(
            f'<text x="{x_mid:.2f}" y="{top + plot_h + 24:.2f}" text-anchor="middle" font-size="14" fill="#333">{escape(_truncate_label(str(row["cohort_type"]), 16))}</text>'
        )
    return "\n".join(
        [
            '<figure class="viz-card">',
            "  <h3>Cohort Duration Distribution</h3>",
            "  <p class=\"viz-caption\">Min-max whisker, IQR box, median line, mean dot.</p>",
            f'  <svg viewBox="0 0 {width} {height}" role="img" aria-label="Cohort duration interval chart">',
            *rows,
            "  </svg>",
            "</figure>",
        ]
    )


def _confidence_sensitivity_small_multiples_html(sensitivity: pd.DataFrame) -> str:
    required = {"confidence_segment", "position", "expected_falls", "confidence_weighted_expected_falls", "hard_label_falls"}
    if sensitivity.empty or not required.issubset(sensitivity.columns):
        return ""
    sf = sensitivity.copy()
    sf["confidence_segment"] = sf["confidence_segment"].astype(str)
    sf["position"] = sf["position"].astype(str)
    sf["expected_falls"] = _to_numeric_series(sf["expected_falls"])
    sf["confidence_weighted_expected_falls"] = _to_numeric_series(sf["confidence_weighted_expected_falls"])
    sf["hard_label_falls"] = _to_numeric_series(sf["hard_label_falls"])
    cards: list[str] = []
    segment_order = ["all_events", "high_confidence", "medium_confidence", "low_confidence"]
    for segment in segment_order:
        seg_df = sf.loc[sf["confidence_segment"] == segment]
        if seg_df.empty:
            continue
        for position in ["chair", "bed"]:
            row = seg_df.loc[seg_df["position"] == position]
            if row.empty:
                continue
            cards.append(
                _bar_chart_card_html(
                    f"{segment}: {position}",
                    "Expected vs confidence-weighted vs hard-label falls.",
                    ["Expected", "Weighted", "Hard label"],
                    [
                        float(row.iloc[0]["expected_falls"]),
                        float(row.iloc[0]["confidence_weighted_expected_falls"]),
                        float(row.iloc[0]["hard_label_falls"]),
                    ],
                    color="#1F4E79",
                )
            )
    if not cards:
        return ""
    return "\n".join(
        [
            '<div class="viz-card viz-card-wide">',
            "  <h3>Confidence Sensitivity Small Multiples</h3>",
            "  <p class=\"viz-caption\">Per-segment and per-position expected, weighted, and hard-label burden.</p>",
            '  <div class="appendix-mini-grid">',
            "\n".join(cards),
            "  </div>",
            "</div>",
        ]
    )


def _falls_interactive_location_focus_html(location_df: pd.DataFrame, location_prob_df: pd.DataFrame) -> str:
    if location_df.empty or location_prob_df.empty:
        return ""
    if not {"prefall_location_label", "falls"}.issubset(location_df.columns):
        return ""
    if not {"prefall_location_label", "expected_falls"}.issubset(location_prob_df.columns):
        return ""
    observed = (
        location_df[["prefall_location_label", "falls"]]
        .assign(falls=lambda df: _to_numeric_series(df["falls"]))
        .groupby("prefall_location_label", dropna=False)["falls"]
        .sum()
        .reset_index()
    )
    expected = (
        location_prob_df[["prefall_location_label", "expected_falls"]]
        .assign(expected_falls=lambda df: _to_numeric_series(df["expected_falls"]))
        .groupby("prefall_location_label", dropna=False)["expected_falls"]
        .sum()
        .reset_index()
    )
    merged = observed.merge(expected, on="prefall_location_label", how="outer").fillna(0.0)
    merged["prefall_location_label"] = merged["prefall_location_label"].astype(str).str.strip().str.lower()
    merged = merged.loc[merged["prefall_location_label"] != ""].reset_index(drop=True)
    if merged.empty:
        return ""
    records = []
    for _, row in merged.iterrows():
        records.append(
            {
                "label": str(row["prefall_location_label"]),
                "observed": round(float(row["falls"]), 4),
                "expected": round(float(row["expected_falls"]), 4),
            }
        )
    data = json.dumps(records)
    buttons = "\n".join(
        f'      <button type="button" class="viz-btn" data-idx="{idx}">{escape(rec["label"])}</button>'
        for idx, rec in enumerate(records)
    )
    return "\n".join(
        [
            '<figure class="viz-card">',
            "  <h3>Interactive Location Drilldown</h3>",
            "  <p class=\"viz-caption\">Click a location to compare observed and probability-weighted burden.</p>",
            '  <div class="viz-btn-row">',
            buttons,
            "  </div>",
            "  <div class=\"interactive-card\">",
            "    <div><strong id=\"falls-loc-label\">-</strong></div>",
            "    <div>Observed falls: <span id=\"falls-loc-observed\">0.0</span></div>",
            "    <div>Expected falls: <span id=\"falls-loc-expected\">0.0</span></div>",
            "  </div>",
            "  <script>",
            f"    (function() {{ const data = {data};",
            "      const label = document.getElementById('falls-loc-label');",
            "      const observed = document.getElementById('falls-loc-observed');",
            "      const expected = document.getElementById('falls-loc-expected');",
            "      const btns = document.querySelectorAll('.viz-btn[data-idx]');",
            "      const paint = (idx) => { const row = data[idx] || data[0]; if (!row) return;",
            "        label.textContent = row.label; observed.textContent = row.observed.toFixed(2); expected.textContent = row.expected.toFixed(2);",
            "        btns.forEach((b) => b.classList.toggle('is-active', Number(b.dataset.idx) === idx)); };",
            "      btns.forEach((b) => b.addEventListener('click', () => paint(Number(b.dataset.idx))));",
            "      paint(0);",
            "    })();",
            "  </script>",
            "</figure>",
        ]
    )


def _full_cohort_visualizations_html(
    source_profile: dict[str, Any],
    qa_summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> str:
    hourly = source_profile.get("hourly_location_aggregation", {})
    coverage = tables.get("Control Denominator Coverage", pd.DataFrame())
    pass_fail = tables.get("Cohort Eligibility Rates", pd.DataFrame())
    composition = tables.get("Cohort Composition Summary", pd.DataFrame())
    fall_density = tables.get("Fall Density by Cohort and Daypart", pd.DataFrame())
    risk = tables.get("Chair-vs-Bed Risk Rates", pd.DataFrame())

    cards: list[str] = []
    cards.append(
        "\n".join(
            [
                '<figure class="viz-card viz-card-wide">',
                "  <h3>KPI Snapshot</h3>",
                '  <div class="kpi-row">',
                '    <div class="kpi-tile"><span class="kpi-label">Hourly Rows</span><strong class="kpi-value">'
                f"{_fmt_int(hourly.get('rows', 0))}</strong></div>",
                '    <div class="kpi-tile"><span class="kpi-label">Distinct Monitors</span><strong class="kpi-value">'
                f"{_fmt_int(hourly.get('distinct_monitor_id', 0))}</strong></div>",
                '    <div class="kpi-tile"><span class="kpi-label">Distinct Patients</span><strong class="kpi-value">'
                f"{_fmt_int(hourly.get('distinct_patient_id', 0))}</strong></div>",
                '    <div class="kpi-tile"><span class="kpi-label">Analysis Base Rows</span><strong class="kpi-value">'
                f"{_fmt_int(qa_summary.get('analysis_base_rows', 0))}</strong></div>",
                "  </div>",
                "</figure>",
            ]
        )
    )

    if not composition.empty and {"cohort_type", "total_hour_rows"}.issubset(composition.columns):
        comp = composition.copy()
        comp["total_hour_rows"] = _to_numeric_series(comp["total_hour_rows"])
        cards.append(
            _bar_chart_card_html(
                "Cohort Hour Volume",
                "Total patient-hour rows by cohort.",
                comp["cohort_type"].astype(str).tolist(),
                comp["total_hour_rows"].astype(float).tolist(),
            )
        )

    if not pass_fail.empty and {"cohort_type", "eligible", "units"}.issubset(pass_fail.columns):
        pf = pass_fail.copy()
        pf["eligible"] = pf["eligible"].astype(str).str.lower()
        pf["units"] = _to_numeric_series(pf["units"])
        grouped = (
            pf.groupby(["cohort_type", "eligible"], dropna=False)["units"].sum().unstack(fill_value=0.0).reset_index()
        )
        labels = grouped["cohort_type"].astype(str).tolist()
        eligible = grouped.get("true", pd.Series([0.0] * len(labels))).astype(float).tolist()
        ineligible = grouped.get("false", pd.Series([0.0] * len(labels))).astype(float).tolist()
        cards.append(
            _dual_bar_chart_card_html(
                "Eligibility Mix by Cohort",
                "Eligible vs ineligible units.",
                labels,
                eligible,
                ineligible,
                "Eligible",
                "Ineligible",
                left_color="#1F4E79",
                right_color="#D9B310",
            )
        )
        cards.append(_full_cohort_interactive_eligibility_html(grouped))

    if not coverage.empty and {"cohort_type", "rows", "valid_rows"}.issubset(coverage.columns):
        cov = coverage.copy()
        cov["rows"] = _to_numeric_series(cov["rows"])
        cov["valid_rows"] = _to_numeric_series(cov["valid_rows"])
        grouped_cov = cov.groupby("cohort_type", dropna=False)[["rows", "valid_rows"]].sum().reset_index()
        labels = grouped_cov["cohort_type"].astype(str).tolist()
        valid = grouped_cov["valid_rows"].astype(float).tolist()
        invalid = (grouped_cov["rows"] - grouped_cov["valid_rows"]).clip(lower=0).astype(float).tolist()
        cards.append(
            _dual_bar_chart_card_html(
                "Control Denominator Coverage",
                "Valid vs invalid rows by cohort.",
                labels,
                valid,
                invalid,
                "Valid rows",
                "Invalid rows",
                left_color="#0B3C5D",
                right_color="#B33F62",
            )
        )

    if not fall_density.empty and {"daypart", "falls"}.issubset(fall_density.columns):
        fd = fall_density.copy()
        fd["falls"] = _to_numeric_series(fd["falls"])
        grouped_fd = fd.groupby("daypart", dropna=False)["falls"].sum().reset_index()
        cards.append(
            _bar_chart_card_html(
                "Fall Density by Daypart",
                "Falls aggregated across cohorts.",
                grouped_fd["daypart"].apply(_display_daypart).astype(str).tolist(),
                grouped_fd["falls"].astype(float).tolist(),
                color="#328CC1",
                label_limit=24,
            )
        )

    if not risk.empty and {"position", "rate_per_1000_exposure_hours_expected", "rate_per_1000_exposure_hours_hard_label"}.issubset(
        risk.columns
    ):
        rb = risk.copy()
        rb["position"] = rb["position"].astype(str)
        rb["rate_per_1000_exposure_hours_expected"] = _to_numeric_series(rb["rate_per_1000_exposure_hours_expected"])
        rb["rate_per_1000_exposure_hours_hard_label"] = _to_numeric_series(rb["rate_per_1000_exposure_hours_hard_label"])
        cards.append(
            _dual_bar_chart_card_html(
                "Chair vs Bed Rates",
                "Expected vs hard-label rate per 1,000 exposure-hours.",
                rb["position"].tolist(),
                rb["rate_per_1000_exposure_hours_expected"].tolist(),
                rb["rate_per_1000_exposure_hours_hard_label"].tolist(),
                "Expected",
                "Hard label",
                left_color="#1F4E79",
                right_color="#B33F62",
            )
        )

    cards = [card for card in cards if card]
    if not cards:
        return ""
    return "\n".join(
        [
            "  <section>",
            "    <h2>Visual Summary</h2>",
            "    <div class=\"viz-grid\">",
            "\n".join(cards),
            "    </div>",
            "  </section>",
        ]
    )


def _full_cohort_interactive_eligibility_html(grouped: pd.DataFrame) -> str:
    if grouped.empty or "cohort_type" not in grouped.columns:
        return ""
    records = []
    for _, row in grouped.iterrows():
        eligible = float(row.get("true", 0.0) or 0.0)
        ineligible = float(row.get("false", 0.0) or 0.0)
        total = eligible + ineligible
        pass_rate = (eligible / total) if total else 0.0
        records.append(
            {
                "cohort": str(row["cohort_type"]),
                "eligible_units": round(eligible, 4),
                "ineligible_units": round(ineligible, 4),
                "pass_rate": round(pass_rate, 4),
            }
        )
    data = json.dumps(records)
    return "\n".join(
        [
            '<figure class="viz-card">',
            "  <h3>Interactive Eligibility Lens</h3>",
            "  <p class=\"viz-caption\">Toggle between unit counts and pass rate.</p>",
            '  <div class="viz-btn-row">',
            '    <button type="button" class="viz-btn is-active" id="elig-mode-units">Units</button>',
            '    <button type="button" class="viz-btn" id="elig-mode-rate">Pass rate</button>',
            "  </div>",
            "  <div id=\"elig-view\"></div>",
            "  <script>",
            f"    (function() {{ const data = {data};",
            "      const view = document.getElementById('elig-view');",
            "      const unitsBtn = document.getElementById('elig-mode-units');",
            "      const rateBtn = document.getElementById('elig-mode-rate');",
            "      const render = (mode) => {",
            "        const lines = data.map((row) => mode === 'units' ? `${row.cohort}: eligible ${row.eligible_units.toFixed(1)}, ineligible ${row.ineligible_units.toFixed(1)}` : `${row.cohort}: ${(row.pass_rate * 100).toFixed(1)}% pass`).join('<br/>');",
            "        view.innerHTML = `<p>${lines || 'n/a'}</p>`;",
            "        unitsBtn.classList.toggle('is-active', mode === 'units');",
            "        rateBtn.classList.toggle('is-active', mode === 'rate');",
            "      };",
            "      unitsBtn.addEventListener('click', () => render('units'));",
            "      rateBtn.addEventListener('click', () => render('rate'));",
            "      render('units');",
            "    })();",
            "  </script>",
            "</figure>",
        ]
    )


def _inference_visualizations_html(
    source_profile: dict[str, Any],
    qa_summary: dict[str, Any],
    negative_control_coverage: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> str:
    hourly = source_profile.get("hourly_location_aggregation", {})
    risk = tables.get("Chair-vs-Bed Risk Rates", pd.DataFrame())
    if not risk.empty and "daypart" in risk.columns:
        overall_risk = risk.loc[risk["daypart"].astype(str) == "all_dayparts"].copy()
        if not overall_risk.empty:
            risk = overall_risk
    sensitivity = tables.get("Chair-vs-Bed Confidence Sensitivity", pd.DataFrame())
    missingness_stress = tables.get("Chair-vs-Bed Missingness Stress", pd.DataFrame())
    negative_control_effects = tables.get("Negative-Control Effects (Exploratory)", pd.DataFrame())
    cards: list[str] = []
    cards.append(
        "\n".join(
            [
                '<figure class="viz-card viz-card-wide">',
                "  <h3>KPI Snapshot</h3>",
                '  <div class="kpi-row">',
                '    <div class="kpi-tile"><span class="kpi-label">Hourly Rows</span><strong class="kpi-value">'
                f"{_fmt_int(hourly.get('rows', 0))}</strong></div>",
                '    <div class="kpi-tile"><span class="kpi-label">Distinct Monitors</span><strong class="kpi-value">'
                f"{_fmt_int(hourly.get('distinct_monitor_id', 0))}</strong></div>",
                '    <div class="kpi-tile"><span class="kpi-label">Analysis Base Rows</span><strong class="kpi-value">'
                f"{_fmt_int(qa_summary.get('analysis_base_rows', 0))}</strong></div>",
                "  </div>",
                "</figure>",
            ]
        )
    )
    if not risk.empty and {"position", "expected_falls", "hard_label_falls"}.issubset(risk.columns):
        rf = risk.copy()
        rf["expected_falls"] = _to_numeric_series(rf["expected_falls"])
        rf["hard_label_falls"] = _to_numeric_series(rf["hard_label_falls"])
        cards.append(
            _dual_bar_chart_card_html(
                "Expected vs Hard-Label Falls",
                "Intervention-eligible scope.",
                rf["position"].astype(str).tolist(),
                rf["expected_falls"].astype(float).tolist(),
                rf["hard_label_falls"].astype(float).tolist(),
                "Expected",
                "Hard label",
            )
        )
    if not risk.empty and {"position", "rate_per_1000_exposure_hours_expected", "rate_per_1000_exposure_hours_hard_label"}.issubset(
        risk.columns
    ):
        rr = risk.copy()
        rr["rate_per_1000_exposure_hours_expected"] = _to_numeric_series(rr["rate_per_1000_exposure_hours_expected"])
        rr["rate_per_1000_exposure_hours_hard_label"] = _to_numeric_series(rr["rate_per_1000_exposure_hours_hard_label"])
        cards.append(
            _dual_bar_chart_card_html(
                "Rates per 1,000 Exposure-Hours",
                "Expected vs hard-label rates by position.",
                rr["position"].astype(str).tolist(),
                rr["rate_per_1000_exposure_hours_expected"].astype(float).tolist(),
                rr["rate_per_1000_exposure_hours_hard_label"].astype(float).tolist(),
                "Expected rate",
                "Hard-label rate",
                left_color="#0B3C5D",
                right_color="#D9B310",
            )
        )
    cards.append(_inference_interactive_confidence_html(sensitivity))
    cards.append(_missingness_stress_card_html(missingness_stress))
    cards.append(_negative_control_effects_card_html(negative_control_effects))
    cards.append(_negative_control_coverage_card_html(negative_control_coverage))
    cards.append(
        _pending_control_source_card_html(
            qa_summary.get("negative_control_source_status")
            or source_profile.get("negative_control_source_status")
            or qa_summary.get("nonfall_control_source_status")
            or source_profile.get("nonfall_control_source_status")
        )
    )
    cards = [card for card in cards if card]
    if not cards:
        return ""
    return "\n".join(
        [
            "  <section>",
            "    <h2>Visual Summary</h2>",
            "    <div class=\"viz-grid\">",
            "\n".join(cards),
            "    </div>",
            "  </section>",
        ]
    )


def _inference_interactive_confidence_html(sensitivity: pd.DataFrame) -> str:
    required = {"confidence_segment", "position", "expected_falls", "confidence_weighted_expected_falls"}
    if sensitivity.empty or not required.issubset(sensitivity.columns):
        return ""
    sf = sensitivity.copy()
    sf["confidence_segment"] = sf["confidence_segment"].astype(str).str.lower()
    sf["position"] = sf["position"].astype(str).str.lower()
    sf["expected_falls"] = _to_numeric_series(sf["expected_falls"])
    sf["confidence_weighted_expected_falls"] = _to_numeric_series(sf["confidence_weighted_expected_falls"])
    segments = []
    for segment, seg_df in sf.groupby("confidence_segment", dropna=False):
        rows = {}
        for _, row in seg_df.iterrows():
            rows[str(row["position"])] = {
                "expected": float(row["expected_falls"]),
                "weighted": float(row["confidence_weighted_expected_falls"]),
            }
        segments.append({"segment": segment, "rows": rows})
    if not segments:
        return ""
    data = json.dumps(segments)
    return "\n".join(
        [
            '<figure class="viz-card">',
            "  <h3>Interactive Confidence Segment Lens</h3>",
            "  <p class=\"viz-caption\">Switch confidence segments to compare expected vs confidence-weighted burden.</p>",
            "  <label for=\"inf-segment\"><strong>Segment:</strong></label>",
            "  <select id=\"inf-segment\"></select>",
            "  <div id=\"inf-view\"></div>",
            "  <script>",
            f"    (function() {{ const data = {data};",
            "      const select = document.getElementById('inf-segment'); const view = document.getElementById('inf-view');",
            "      data.forEach((row, idx) => { const opt = document.createElement('option'); opt.value = String(idx); opt.textContent = row.segment; select.appendChild(opt); });",
            "      const render = (idx) => { const row = data[idx] || data[0]; if (!row) { view.innerHTML = '<p>n/a</p>'; return; }",
            "        const chair = row.rows.chair || { expected: 0, weighted: 0 };",
            "        const bed = row.rows.bed || { expected: 0, weighted: 0 };",
            "        view.innerHTML = `<p>Chair: expected ${chair.expected.toFixed(2)}, weighted ${chair.weighted.toFixed(2)}<br/>Bed: expected ${bed.expected.toFixed(2)}, weighted ${bed.weighted.toFixed(2)}</p>`; };",
            "      select.addEventListener('change', () => render(Number(select.value)));",
            "      render(0);",
            "    })();",
            "  </script>",
            "</figure>",
        ]
    )


_MINIMAL_TABLES_BY_REPORT: dict[str, set[str]] = {
    "falls_only": {"Response Latency (Livestream)"},
    "full_cohort": {"Cohort Duration Distribution"},
    "inference": set(),
}


def _falls_appendix_visuals_html(tables: dict[str, pd.DataFrame]) -> str:
    cards: list[str] = []
    site_df = tables.get("Falls by Site", pd.DataFrame())
    if not site_df.empty and {"hospital_name", "falls"}.issubset(site_df.columns):
        chart_df = site_df.copy()
        chart_df["falls"] = _to_numeric_series(chart_df["falls"])
        chart_df = chart_df.sort_values(["falls", "hospital_name"], ascending=[False, True], kind="mergesort")
        cards.append(
            _bar_chart_card_html(
                "Falls by Site (All Sites)",
                "Full site distribution shown in appendix.",
                chart_df["hospital_name"].astype(str).tolist(),
                chart_df["falls"].astype(float).tolist(),
                color="#1F4E79",
            )
        )
    weekday_df = tables.get("Falls by Weekday", pd.DataFrame())
    if not weekday_df.empty and {"weekday", "falls"}.issubset(weekday_df.columns):
        chart_df = weekday_df.copy()
        chart_df["falls"] = _to_numeric_series(chart_df["falls"])
        chart_df["weekday"] = chart_df["weekday"].astype(str)
        order = {day: idx for idx, day in enumerate(_WEEKDAY_ORDER)}
        chart_df["_order"] = chart_df["weekday"].map(order).fillna(99)
        chart_df = chart_df.sort_values(["_order", "weekday"], kind="mergesort")
        cards.append(
            _bar_chart_card_html(
                "Falls by Weekday",
                "Observed falls by day of week.",
                chart_df["weekday"].tolist(),
                chart_df["falls"].astype(float).tolist(),
                color="#328CC1",
            )
        )
    daypart_df = tables.get("Falls by Daypart", pd.DataFrame())
    if not daypart_df.empty and {"daypart", "falls"}.issubset(daypart_df.columns):
        chart_df = daypart_df.copy()
        chart_df["falls"] = _to_numeric_series(chart_df["falls"])
        chart_df["daypart"] = chart_df["daypart"].astype(str)
        order = {day: idx for idx, day in enumerate(_DAYPART_ORDER)}
        chart_df["_order"] = chart_df["daypart"].map(order).fillna(99)
        chart_df = chart_df.sort_values(["_order", "daypart"], kind="mergesort")
        cards.append(
            _bar_chart_card_html(
                "Falls by Daypart",
                "Observed falls by daypart.",
                chart_df["daypart"].apply(_display_daypart).tolist(),
                chart_df["falls"].astype(float).tolist(),
                color="#0B3C5D",
                label_limit=24,
            )
        )
    month_df = tables.get("Falls by Month of Year", pd.DataFrame())
    if not month_df.empty and {"month_num", "month_of_year", "falls"}.issubset(month_df.columns):
        chart_df = month_df.copy()
        chart_df["month_num"] = _to_numeric_series(chart_df["month_num"])
        chart_df["falls"] = _to_numeric_series(chart_df["falls"])
        chart_df = chart_df.sort_values(["month_num", "month_of_year"], kind="mergesort")
        cards.append(
            _bar_chart_card_html(
                "Falls by Month of Year",
                "Seasonality view of observed falls.",
                chart_df["month_of_year"].astype(str).tolist(),
                chart_df["falls"].astype(float).tolist(),
                color="#D9B310",
            )
        )
    observed_df = tables.get("Pre-Fall Location (Livestream)", pd.DataFrame())
    if not observed_df.empty and {"prefall_location_label", "falls"}.issubset(observed_df.columns):
        chart_df = observed_df.copy()
        chart_df["falls"] = _to_numeric_series(chart_df["falls"])
        chart_df = chart_df.sort_values(["falls", "prefall_location_label"], ascending=[False, True], kind="mergesort")
        cards.append(
            _bar_chart_card_html(
                "Observed Pre-Fall Location Burden",
                "Hard-label falls by pre-fall location.",
                chart_df["prefall_location_label"].astype(str).tolist(),
                chart_df["falls"].astype(float).tolist(),
                color="#1F4E79",
            )
        )
    expected_df = tables.get("Pre-Fall Location Probabilities (Livestream)", pd.DataFrame())
    if not expected_df.empty and {"prefall_location_label", "expected_falls"}.issubset(expected_df.columns):
        chart_df = expected_df.copy()
        chart_df["expected_falls"] = _to_numeric_series(chart_df["expected_falls"])
        chart_df = chart_df.sort_values(
            ["expected_falls", "prefall_location_label"], ascending=[False, True], kind="mergesort"
        )
        cards.append(
            _bar_chart_card_html(
                "Expected Pre-Fall Location Burden",
                "Probability-weighted expected falls by location.",
                chart_df["prefall_location_label"].astype(str).tolist(),
                chart_df["expected_falls"].astype(float).tolist(),
                color="#B33F62",
            )
        )
    breakdown_df = tables.get("Pre-Fall Location Probability Breakdown (Livestream)", pd.DataFrame())
    cards.append(_prefall_probability_breakdown_small_multiples_html(breakdown_df))
    cards = [card for card in cards if card]
    if not cards:
        return ""
    return "\n".join(
        [
            "  <section>",
            "    <h2>Appendix Visuals</h2>",
            "    <div class=\"viz-grid\">",
            "\n".join(cards),
            "    </div>",
            "  </section>",
        ]
    )


def _full_cohort_appendix_visuals_html(tables: dict[str, pd.DataFrame]) -> str:
    cards: list[str] = []
    duration_df = tables.get("Cohort Duration Distribution", pd.DataFrame())
    cards.append(_cohort_duration_interval_chart_html(duration_df))
    density_df = tables.get("Fall Density by Cohort and Daypart", pd.DataFrame())
    if not density_df.empty and {"cohort_type", "daypart", "falls"}.issubset(density_df.columns):
        frame = density_df.copy()
        frame["cohort_type"] = frame["cohort_type"].astype(str)
        frame["daypart"] = frame["daypart"].astype(str)
        frame["falls"] = _to_numeric_series(frame["falls"])
        daypart_to_display = {dp: _display_daypart(dp) for dp in _DAYPART_ORDER}
        x_labels = [daypart_to_display[dp] for dp in _DAYPART_ORDER if dp in set(frame["daypart"])]
        y_labels = sorted(frame["cohort_type"].unique().tolist())
        value_map: dict[tuple[str, str], float] = {}
        for _, row in frame.iterrows():
            display_daypart = daypart_to_display.get(str(row["daypart"]), str(row["daypart"]))
            value_map[(str(row["cohort_type"]), display_daypart)] = float(row["falls"])
        cards.append(
            _matrix_heatmap_card_html(
                "Fall Density by Cohort and Daypart",
                "Heatmap with absolute falls by cohort/daypart.",
                x_labels,
                y_labels,
                value_map,
                value_decimals=1,
                x_label_limit=24,
            )
        )
    risk_df = tables.get("Chair-vs-Bed Risk Rates", pd.DataFrame())
    if not risk_df.empty and {"position", "expected_falls", "hard_label_falls"}.issubset(risk_df.columns):
        frame = risk_df.copy()
        frame["expected_falls"] = _to_numeric_series(frame["expected_falls"])
        frame["hard_label_falls"] = _to_numeric_series(frame["hard_label_falls"])
        cards.append(
            _dual_bar_chart_card_html(
                "Risk Rates: Expected vs Hard-Label Counts",
                "Appendix view of burden counts by position.",
                frame["position"].astype(str).tolist(),
                frame["expected_falls"].astype(float).tolist(),
                frame["hard_label_falls"].astype(float).tolist(),
                "Expected falls",
                "Hard-label falls",
                left_color="#0B3C5D",
                right_color="#B33F62",
            )
        )
    cards.append(_confidence_sensitivity_small_multiples_html(tables.get("Chair-vs-Bed Confidence Sensitivity", pd.DataFrame())))
    cards = [card for card in cards if card]
    if not cards:
        return ""
    return "\n".join(
        [
            "  <section>",
            "    <h2>Appendix Visuals</h2>",
            "    <div class=\"viz-grid\">",
            "\n".join(cards),
            "    </div>",
            "  </section>",
        ]
    )


def _inference_appendix_visuals_html(tables: dict[str, pd.DataFrame]) -> str:
    cards: list[str] = []
    cards.append(_confidence_sensitivity_small_multiples_html(tables.get("Chair-vs-Bed Confidence Sensitivity", pd.DataFrame())))
    cards.append(_missingness_stress_card_html(tables.get("Chair-vs-Bed Missingness Stress", pd.DataFrame())))
    pass_fail = tables.get("Cohort Eligibility Rates", pd.DataFrame())
    if not pass_fail.empty and {"cohort_type", "eligible", "units"}.issubset(pass_fail.columns):
        pf = pass_fail.copy()
        pf["eligible"] = pf["eligible"].astype(str).str.lower()
        pf["units"] = _to_numeric_series(pf["units"])
        grouped = (
            pf.groupby(["cohort_type", "eligible"], dropna=False)["units"].sum().unstack(fill_value=0.0).reset_index()
        )
        labels = grouped["cohort_type"].astype(str).tolist()
        eligible = grouped.get("true", pd.Series([0.0] * len(labels))).astype(float).tolist()
        ineligible = grouped.get("false", pd.Series([0.0] * len(labels))).astype(float).tolist()
        cards.append(
            _dual_bar_chart_card_html(
                "Eligibility Rates",
                "Eligible vs ineligible units in inference scope.",
                labels,
                eligible,
                ineligible,
                "Eligible",
                "Ineligible",
                left_color="#1F4E79",
                right_color="#D9B310",
            )
        )
    coverage = tables.get("Control Denominator Coverage", pd.DataFrame())
    if not coverage.empty and {"cohort_type", "rows", "valid_rows"}.issubset(coverage.columns):
        cov = coverage.copy()
        cov["rows"] = _to_numeric_series(cov["rows"])
        cov["valid_rows"] = _to_numeric_series(cov["valid_rows"])
        grouped_cov = cov.groupby("cohort_type", dropna=False)[["rows", "valid_rows"]].sum().reset_index()
        labels = grouped_cov["cohort_type"].astype(str).tolist()
        valid = grouped_cov["valid_rows"].astype(float).tolist()
        invalid = (grouped_cov["rows"] - grouped_cov["valid_rows"]).clip(lower=0).astype(float).tolist()
        cards.append(
            _dual_bar_chart_card_html(
                "Control Denominator Coverage",
                "Valid vs invalid denominator rows.",
                labels,
                valid,
                invalid,
                "Valid rows",
                "Invalid rows",
                left_color="#0B3C5D",
                right_color="#B33F62",
            )
        )
    location_probs = tables.get("Pre-Fall Location Probabilities (Livestream)", pd.DataFrame())
    if not location_probs.empty and {"prefall_location_label", "expected_falls"}.issubset(location_probs.columns):
        lp = location_probs.copy()
        lp["expected_falls"] = _to_numeric_series(lp["expected_falls"])
        lp = lp.sort_values(["expected_falls", "prefall_location_label"], ascending=[False, True], kind="mergesort")
        cards.append(
            _bar_chart_card_html(
                "Pre-Fall Location Probabilities",
                "Expected falls by location label.",
                lp["prefall_location_label"].astype(str).tolist(),
                lp["expected_falls"].astype(float).tolist(),
                color="#328CC1",
            )
        )
    cards = [card for card in cards if card]
    if not cards:
        return ""
    return "\n".join(
        [
            "  <section>",
            "    <h2>Appendix Visuals</h2>",
            "    <div class=\"viz-grid\">",
            "\n".join(cards),
            "    </div>",
            "  </section>",
        ]
    )


def _conclusions_html(total_falls: int, tables: dict[str, pd.DataFrame]) -> str:
    items: list[str] = []
    if total_falls > 0:
        items.append(
            "This report supports operational descriptive triage (where/when/what happened), not inferential chair-vs-bed risk claims."
        )

    site_df = tables.get("Falls by Site", pd.DataFrame())
    if not site_df.empty and {"hospital_name", "falls"}.issubset(site_df.columns):
        ranked = site_df.sort_values("falls", ascending=False).head(4)
        top = ranked.iloc[0]
        try:
            top_falls = int(top["falls"])
            top_pct = (top_falls / total_falls) if total_falls else 0.0
            top4_falls = int(ranked["falls"].astype(int).sum())
            top4_pct = (top4_falls / total_falls) if total_falls else 0.0
            items.append(
                f"Falls are concentrated by site: {escape(str(top['hospital_name']))} contributes {top_falls}/{total_falls} ({top_pct:.1%}), and the top 4 sites contribute {top4_falls}/{total_falls} ({top4_pct:.1%})."
            )
        except (TypeError, ValueError):
            pass

    daypart_df = tables.get("Falls by Daypart", pd.DataFrame())
    if not daypart_df.empty and {"daypart", "falls"}.issubset(daypart_df.columns):
        top = daypart_df.sort_values("falls", ascending=False).iloc[0]
        try:
            top_falls = int(top["falls"])
            top_pct = (top_falls / total_falls) if total_falls else 0.0
            items.append(
                f"The most common daypart is {escape(_display_daypart(top['daypart']))} with {top_falls}/{total_falls} falls ({top_pct:.1%})."
            )
        except (TypeError, ValueError):
            pass

    location_df = tables.get("Pre-Fall Location (Livestream)", pd.DataFrame())
    if not location_df.empty and {"prefall_location_label", "falls"}.issubset(location_df.columns):
        ranked = location_df.sort_values("falls", ascending=False)
        top = ranked.iloc[0]
        try:
            top_falls = int(top["falls"])
            top_pct = (top_falls / total_falls) if total_falls else 0.0
            no_patient_row = location_df.loc[
                location_df["prefall_location_label"].astype(str).str.lower() == "no_patient", "falls"
            ]
            if no_patient_row.empty:
                no_patient_row = location_df.loc[
                    location_df["prefall_location_label"].astype(str).str.lower() == "unknown", "falls"
                ]
            no_patient_falls = int(no_patient_row.iloc[0]) if not no_patient_row.empty else 0
            no_patient_pct = (no_patient_falls / total_falls) if total_falls else 0.0
            items.append(
                f"Pre-fall location is most often {escape(str(top['prefall_location_label']))} ({top_falls}/{total_falls}, {top_pct:.1%}); no-patient context remains {no_patient_falls}/{total_falls} ({no_patient_pct:.1%})."
            )
        except (TypeError, ValueError):
            pass

    location_prob_df = tables.get("Pre-Fall Location Probabilities (Livestream)", pd.DataFrame())
    if not location_prob_df.empty and {"prefall_location_label", "expected_falls"}.issubset(location_prob_df.columns):
        ranked = location_prob_df.sort_values("expected_falls", ascending=False)
        top = ranked.iloc[0]
        try:
            top_expected = float(top["expected_falls"])
            top_pct = (top_expected / total_falls) if total_falls else 0.0
            items.append(
                f"Probability-weighted pre-fall location still concentrates in {escape(str(top['prefall_location_label']))} (expected {top_expected:.1f}/{total_falls}, {top_pct:.1%})."
            )
        except (TypeError, ValueError):
            pass

    location_breakdown_df = tables.get("Pre-Fall Location Probability Breakdown (Livestream)", pd.DataFrame())
    if (
        not location_breakdown_df.empty
        and {
            "grouping",
            "group_value",
            "prefall_location_label",
            "expected_falls",
            "median_probability",
        }.issubset(location_breakdown_df.columns)
    ):
        ranked = location_breakdown_df.sort_values("expected_falls", ascending=False)
        top = ranked.iloc[0]
        try:
            items.append(
                "Highest grouped expected location burden is "
                f"{escape(str(top['prefall_location_label']))} in "
                f"{escape(str(top['grouping']))}={escape(str(top['group_value']))} "
                f"(expected {float(top['expected_falls']):.1f} falls, median probability {float(top['median_probability']):.2f})."
            )
        except (TypeError, ValueError):
            pass

    latency_df = tables.get("Response Latency (Livestream)", pd.DataFrame())
    responses = _metric_value(latency_df, "responses_detected")
    rate = _metric_value(latency_df, "response_rate")
    p50 = _metric_value(latency_df, "latency_p50_seconds")
    p90 = _metric_value(latency_df, "latency_p90_seconds")
    if rate is not None or p50 is not None or p90 is not None:
        fragments: list[str] = []
        if responses is not None and rate is not None:
            fragments.append(f"{int(responses)} detected responses ({rate:.1%})")
        elif rate is not None:
            fragments.append(f"response rate {rate:.1%}")
        if p50 is not None:
            fragments.append(f"p50 latency {p50:.1f}s")
        if p90 is not None:
            fragments.append(f"p90 latency {p90:.1f}s")
        items.append(f"Response performance: {', '.join(fragments)}.")

    if not items:
        return "<p><em>No conclusions available.</em></p>"

    return "<ul>\n" + "\n".join(f"      <li>{item}</li>" for item in items) + "\n    </ul>"


def _full_cohort_conclusions_html(
    source_profile: dict[str, Any],
    qa_summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> str:
    items: list[str] = []
    hourly = source_profile.get("hourly_location_aggregation", {})
    rows = int(hourly.get("rows", 0) or 0)
    monitors = int(hourly.get("distinct_monitor_id", 0) or 0)
    patients = int(hourly.get("distinct_patient_id", 0) or 0)
    if rows > 0:
        items.append(
            f"Hourly baseline includes {rows:,} rows across {monitors:,} monitors and {patients:,} patients."
        )

    coverage = tables.get("Control Denominator Coverage", pd.DataFrame())
    if not coverage.empty and {"rows", "valid_rows"}.issubset(coverage.columns):
        total_rows = float(pd.to_numeric(coverage["rows"], errors="coerce").fillna(0).sum())
        valid_rows = float(pd.to_numeric(coverage["valid_rows"], errors="coerce").fillna(0).sum())
        rate = (valid_rows / total_rows) if total_rows else 0.0
        items.append(
            f"Control denominator quality is {int(valid_rows):,}/{int(total_rows):,} valid rows ({rate:.1%})."
        )

    pass_fail = tables.get("Cohort Eligibility Rates", pd.DataFrame())
    if not pass_fail.empty and {"eligible", "units"}.issubset(pass_fail.columns):
        unit_total = float(pd.to_numeric(pass_fail["units"], errors="coerce").fillna(0).sum())
        eligible_units = float(
            pd.to_numeric(
                pass_fail.loc[pass_fail["eligible"].astype(str).str.lower() == "true", "units"],
                errors="coerce",
            ).fillna(0).sum()
        )
        rate = (eligible_units / unit_total) if unit_total else 0.0
        items.append(
            f"Eligibility pass rate is {int(eligible_units):,}/{int(unit_total):,} units ({rate:.1%})."
        )

    analysis_rows = int(qa_summary.get("analysis_base_rows", 0) or 0)
    if analysis_rows > 0:
        items.append(f"Analysis base contains {analysis_rows:,} rows for descriptive hourly analyses.")
    else:
        items.append("Analysis base is empty; this run supports descriptive summaries but not inferential modeling.")

    if not items:
        return "<p><em>No conclusions available.</em></p>"
    return "<ul>\n" + "\n".join(f"      <li>{item}</li>" for item in items) + "\n    </ul>"


def _chair_bed_inference_conclusions_html(
    source_profile: dict[str, Any],
    qa_summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> str:
    items: list[str] = []
    hourly = source_profile.get("hourly_location_aggregation", {})
    rows = int(hourly.get("rows", 0) or 0)
    monitors = int(hourly.get("distinct_monitor_id", 0) or 0)
    if rows > 0:
        items.append(f"Inference support scope includes {rows:,} hourly rows across {monitors:,} monitors.")

    risk = tables.get("Chair-vs-Bed Risk Rates", pd.DataFrame())
    if not risk.empty and {"scope", "position"}.issubset(risk.columns):
        scoped = risk.loc[risk["scope"].astype(str).str.lower() == "intervention_eligible"].copy()
        if scoped.empty:
            scoped = risk.copy()
        scoped["position"] = scoped["position"].astype(str).str.lower()
        chair_row = scoped.loc[scoped["position"] == "chair"]
        bed_row = scoped.loc[scoped["position"] == "bed"]
        if not chair_row.empty and not bed_row.empty:
            chair_expected_rate = _to_float(
                chair_row.iloc[0].get("rate_per_1000_exposure_hours_expected")
            )
            bed_expected_rate = _to_float(bed_row.iloc[0].get("rate_per_1000_exposure_hours_expected"))
            chair_expected_falls = _to_float(chair_row.iloc[0].get("expected_falls"))
            bed_expected_falls = _to_float(bed_row.iloc[0].get("expected_falls"))
            if chair_expected_rate is not None and bed_expected_rate is not None:
                if chair_expected_rate > bed_expected_rate:
                    leader = "chair"
                    gap = chair_expected_rate - bed_expected_rate
                else:
                    leader = "bed"
                    gap = bed_expected_rate - chair_expected_rate
                items.append(
                    f"Expected fall rate is higher for {leader} by {gap:.2f} per 1,000 exposure-hours "
                    f"(chair={chair_expected_rate:.2f}, bed={bed_expected_rate:.2f})."
                )
            if chair_expected_falls is not None and bed_expected_falls is not None:
                items.append(
                    f"Expected eligible falls: chair={chair_expected_falls:.2f}, bed={bed_expected_falls:.2f}."
                )

    sensitivity = tables.get("Chair-vs-Bed Confidence Sensitivity", pd.DataFrame())
    required_cols = {
        "confidence_segment",
        "position",
        "expected_falls",
        "confidence_weighted_expected_falls",
        "mean_confidence_weight",
    }
    if not sensitivity.empty and required_cols.issubset(sensitivity.columns):
        sensitivity = sensitivity.copy()
        sensitivity["confidence_segment"] = sensitivity["confidence_segment"].astype(str).str.lower()
        sensitivity["position"] = sensitivity["position"].astype(str).str.lower()
        all_events = sensitivity.loc[sensitivity["confidence_segment"] == "all_events"]
        if not all_events.empty:
            for position in ("chair", "bed"):
                row = all_events.loc[all_events["position"] == position]
                if row.empty:
                    continue
                expected = _to_float(row.iloc[0].get("expected_falls"))
                weighted = _to_float(row.iloc[0].get("confidence_weighted_expected_falls"))
                mean_weight = _to_float(row.iloc[0].get("mean_confidence_weight"))
                if expected is None or weighted is None:
                    continue
                ratio = (weighted / expected) if expected else 0.0
                mean_weight_text = f", mean confidence={mean_weight:.2f}" if mean_weight is not None else ""
                items.append(
                    f"{position.capitalize()} confidence-weighted expected falls are {weighted:.2f}/{expected:.2f} "
                    f"({ratio:.1%}){mean_weight_text}."
                )

    missingness_stress = tables.get("Chair-vs-Bed Missingness Stress", pd.DataFrame())
    missingness_cols = {
        "excluded_low_confidence_pct",
        "position",
        "retained_fraction",
        "chair_to_bed_rate_ratio_expected",
        "chair_to_bed_rate_ratio_confidence_weighted",
    }
    if not missingness_stress.empty and missingness_cols.issubset(missingness_stress.columns):
        ms = missingness_stress.copy()
        ms["position"] = ms["position"].astype(str).str.lower()
        ms["excluded_low_confidence_pct"] = pd.to_numeric(ms["excluded_low_confidence_pct"], errors="coerce")
        ms["retained_fraction"] = pd.to_numeric(ms["retained_fraction"], errors="coerce")
        ms["chair_to_bed_rate_ratio_expected"] = pd.to_numeric(
            ms["chair_to_bed_rate_ratio_expected"], errors="coerce"
        )
        ms["chair_to_bed_rate_ratio_confidence_weighted"] = pd.to_numeric(
            ms["chair_to_bed_rate_ratio_confidence_weighted"], errors="coerce"
        )
        chair_rows = ms.loc[ms["position"] == "chair"].dropna(
            subset=[
                "excluded_low_confidence_pct",
                "chair_to_bed_rate_ratio_expected",
                "chair_to_bed_rate_ratio_confidence_weighted",
            ]
        )
        if not chair_rows.empty:
            chair_rows = chair_rows.sort_values("excluded_low_confidence_pct", kind="mergesort")
            baseline = chair_rows.iloc[0]
            strictest = chair_rows.iloc[-1]
            base_rr = _to_float(baseline.get("chair_to_bed_rate_ratio_expected"))
            strict_rr = _to_float(strictest.get("chair_to_bed_rate_ratio_expected"))
            strict_w_rr = _to_float(strictest.get("chair_to_bed_rate_ratio_confidence_weighted"))
            retained_fraction = _to_float(strictest.get("retained_fraction"))
            if base_rr is not None and strict_rr is not None and strict_w_rr is not None:
                retained_text = (
                    f"; retained fraction={retained_fraction:.1%}"
                    if retained_fraction is not None
                    else ""
                )
                items.append(
                    f"Missingness stress at {int(strictest['excluded_low_confidence_pct'])}% exclusion: "
                    f"chair/bed RR expected={strict_rr:.3f} (baseline={base_rr:.3f}), "
                    f"confidence-weighted={strict_w_rr:.3f}{retained_text}."
                )

    negative_effects = tables.get("Negative-Control Effects (Exploratory)", pd.DataFrame())
    if not negative_effects.empty and {
        "metric",
        "paired_mean_delta_hazard_minus_control",
    }.issubset(negative_effects.columns):
        ne = negative_effects.copy()
        ne["metric"] = ne["metric"].astype(str)
        ne["paired_mean_delta_hazard_minus_control"] = pd.to_numeric(
            ne["paired_mean_delta_hazard_minus_control"], errors="coerce"
        )
        ne = ne.dropna(subset=["paired_mean_delta_hazard_minus_control"])
        if not ne.empty:
            ne["abs_delta"] = ne["paired_mean_delta_hazard_minus_control"].abs()
            top = ne.sort_values(["abs_delta", "metric"], ascending=[False, True], kind="mergesort").iloc[0]
            top_metric = str(top["metric"])
            top_delta = float(top["paired_mean_delta_hazard_minus_control"])
            items.append(
                f"Negative-control anchor check top absolute delta: {top_metric}={top_delta:.4f} "
                "(hazard minus non-fall control)."
            )

    analysis_rows = int(qa_summary.get("analysis_base_rows", 0) or 0)
    if analysis_rows > 0:
        items.append(f"Analysis base contributes {analysis_rows:,} rows to inference support tables.")
    else:
        items.append("Analysis base is empty; inference support tables may be incomplete.")

    if not items:
        return "<p><em>No conclusions available.</em></p>"
    return "<ul>\n" + "\n".join(f"      <li>{item}</li>" for item in items) + "\n    </ul>"


def render_falls_only_descriptive_html(payload: dict[str, Any]) -> str:
    run_id = payload["run_id"]
    run_manifest = payload["run_manifest"]
    source_profile = payload["source_profile"]
    qa_summary = payload["qa_summary"]
    tables: dict[str, pd.DataFrame] = payload["tables"]
    chart_frames: dict[str, pd.DataFrame] = payload.get("chart_frames", {})
    artifacts: dict[str, Path] = payload["artifacts"]
    project_root = payload.get("project_root")

    falls = source_profile.get("fall_events_source", {})
    denominator = source_profile.get("control_denominator_completeness", {})
    total_falls = int(falls.get("rows", 0) or 0)
    context = _report_run_context(run_manifest)

    summary_items = [
        ("Fall events", _fmt_int(falls.get("rows", 0))),
        ("Distinct patients", _fmt_int(falls.get("distinct_patient_id", 0))),
        ("Distinct monitors", _fmt_int(falls.get("distinct_monitor_id", 0))),
        ("Distinct hospitals", _fmt_int(falls.get("distinct_hospital_id", 0))),
        ("Distinct divisions", _fmt_int(falls.get("distinct_division_id", 0))),
        ("Analysis base rows", _fmt_int(qa_summary.get("analysis_base_rows", 0))),
    ]
    summary_list = "\n".join(
        f"      <li><strong>{escape(label)}:</strong> {escape(value)}</li>"
        for label, value in summary_items
    )

    visualizations = _visualizations_html(total_falls, tables, chart_frames)
    appendix_visuals = _falls_appendix_visuals_html(tables)
    table_appendix = _table_appendix_html(
        tables,
        _MINIMAL_TABLES_BY_REPORT["falls_only"],
        heading="Minimal Table Appendix",
    )

    artifact_list = "\n".join(
        f"      <li><code>{escape(str(path.relative_to(project_root)) if isinstance(project_root, Path) and path.is_relative_to(project_root) else str(path))}</code></li>"
        for path in artifacts.values()
    )

    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "  <meta charset=\"utf-8\" />",
            f"  <title>Falls-Only Descriptive Report - {escape(run_id)}</title>",
            "  <style>",
            "    body { font-family: Georgia, serif; margin: 2rem auto; max-width: 1080px; line-height: 1.45; }",
            "    h1, h2 { margin-bottom: 0.35rem; }",
            "    p { margin-top: 0.25rem; }",
            "    .meta { color: #444; margin-bottom: 1rem; }",
            "    .callout { background: #f7f7f7; border-left: 4px solid #555; padding: 0.75rem 1rem; }",
            "    .tbl { border-collapse: collapse; width: 100%; margin-bottom: 1rem; }",
            "    .tbl th, .tbl td { border: 1px solid #ddd; padding: 0.4rem 0.5rem; text-align: left; }",
            "    .tbl th { background: #f2f2f2; }",
            "    .viz-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 0.9rem; }",
            "    .viz-card { border: 1px solid #e4e7eb; border-radius: 0.45rem; background: #fff; padding: 0.55rem 0.65rem; margin: 0; }",
            "    .viz-card-wide { grid-column: 1 / -1; }",
            "    .viz-card h3 { margin: 0 0 0.24rem 0; font-size: 1.25rem; }",
            "    .viz-caption { margin: 0 0 0.5rem 0; color: #555; font-size: 1.0rem; }",
            "    .viz-card svg { width: 100%; height: auto; display: block; }",
            "    .kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.45rem; }",
            "    .kpi-tile { border: 1px solid #e4e7eb; border-radius: 0.35rem; padding: 0.45rem 0.55rem; background: #fafbfd; }",
            "    .kpi-label { display: block; font-size: 0.76rem; color: #5a6571; margin-bottom: 0.22rem; }",
            "    .kpi-value { font-size: 1.08rem; color: #1d2a38; }",
            "    .viz-btn-row { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.35rem 0 0.65rem 0; }",
            "    .viz-btn { border: 1px solid #9aa8b6; background: #fff; color: #1d2a38; border-radius: 0.25rem; padding: 0.22rem 0.55rem; cursor: pointer; }",
            "    .viz-btn.is-active { background: #1F4E79; color: #fff; border-color: #1F4E79; }",
            "    .interactive-card { border: 1px solid #e5e9ee; border-radius: 0.35rem; padding: 0.45rem 0.55rem; background: #fafbfd; }",
            "    .appendix-mini-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 0.6rem; }",
            "    details { border: 1px solid #e4e7eb; border-radius: 0.35rem; background: #fff; padding: 0.35rem 0.5rem; }",
            "    details summary { cursor: pointer; font-weight: 600; }",
            "    .appendix-section { margin-top: 0.9rem; }",
            "    code { background: #f3f3f3; padding: 0.1rem 0.3rem; border-radius: 0.2rem; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Falls-Only Descriptive Report</h1>",
            f"  <p class=\"meta\"><strong>Run ID:</strong> {escape(run_id)} | "
            f"<strong>Generated:</strong> {escape(payload['generated_at_utc'])}</p>",
            _render_run_context_html(run_manifest),
            "  <section>",
            "    <h2>Core Summary</h2>",
            "    <ul>",
            summary_list,
            "    </ul>",
            "  </section>",
            "  <section class=\"callout\">",
            "    <h2>Data Limitations</h2>",
            f"    <p><strong>Control denominator rows with valid pct sum:</strong> "
            f"{_fmt_int(denominator.get('rows_with_valid_pct_sum', 0))} / "
            f"{_fmt_int(denominator.get('total_rows', 0))}</p>",
            f"    <p><strong>Eligibility units passed:</strong> "
            f"{_fmt_int(qa_summary.get('eligibility_units_passed', 0))} / "
            f"{_fmt_int(qa_summary.get('eligibility_units', 0))}</p>",
            f"    <p>{escape(_run_mode_limitations_text(context, 'falls_only'))}</p>",
            "  </section>",
            _stakeholder_readout_html(run_manifest, source_profile, qa_summary),
            "  <section>",
            "    <h2>Conclusions</h2>",
            _conclusions_html(total_falls, tables),
            "  </section>",
            visualizations,
            appendix_visuals,
            table_appendix,
            "  <section>",
            "    <h2>Source Artifacts Used</h2>",
            "    <ul>",
            artifact_list,
            "    </ul>",
            "  </section>",
            "</body>",
            "</html>",
        ]
    )


def render_full_cohort_hourly_descriptive_html(payload: dict[str, Any]) -> str:
    run_id = payload["run_id"]
    run_manifest = payload["run_manifest"]
    source_profile = payload["source_profile"]
    qa_summary = payload["qa_summary"]
    tables: dict[str, pd.DataFrame] = payload["tables"]
    artifacts: dict[str, Path] = payload["artifacts"]
    project_root = payload.get("project_root")

    hourly = source_profile.get("hourly_location_aggregation", {})
    denominator = source_profile.get("control_denominator_completeness", {})
    context = _report_run_context(run_manifest)

    summary_items = [
        ("Hourly rows", _fmt_int(hourly.get("rows", 0))),
        ("Distinct monitors", _fmt_int(hourly.get("distinct_monitor_id", 0))),
        ("Distinct patients", _fmt_int(hourly.get("distinct_patient_id", 0))),
        ("Eligibility units", _fmt_int(qa_summary.get("eligibility_units", 0))),
        ("Eligibility units passed", _fmt_int(qa_summary.get("eligibility_units_passed", 0))),
        ("Analysis base rows", _fmt_int(qa_summary.get("analysis_base_rows", 0))),
    ]
    summary_list = "\n".join(
        f"      <li><strong>{escape(label)}:</strong> {escape(value)}</li>"
        for label, value in summary_items
    )

    visualizations = _full_cohort_visualizations_html(source_profile, qa_summary, tables)
    appendix_visuals = _full_cohort_appendix_visuals_html(tables)
    table_appendix = _table_appendix_html(
        tables,
        _MINIMAL_TABLES_BY_REPORT["full_cohort"],
        heading="Minimal Table Appendix",
    )

    artifact_keys = [
        "run_manifest",
        "source_profile",
        "qa_summary",
        "falls_site_name_aliases",
        "cohort_analysis_markdown",
        "cohort_composition",
        "cohort_duration",
        "cohort_eligibility_rates",
        "cohort_exclusions",
        "fall_density",
        "control_denominator_coverage",
        "chair_bed_risk_rates",
        "chair_bed_confidence_sensitivity",
        "transform_metrics",
    ]
    artifact_list = "\n".join(
        f"      <li><code>{escape(str(path.relative_to(project_root)) if isinstance(project_root, Path) and path.is_relative_to(project_root) else str(path))}</code></li>"
        for key, path in artifacts.items()
        if key in artifact_keys and path.exists()
    )

    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "  <meta charset=\"utf-8\" />",
            f"  <title>Full-Cohort Hourly Descriptive Report - {escape(run_id)}</title>",
            "  <style>",
            "    body { font-family: Georgia, serif; margin: 2rem auto; max-width: 1080px; line-height: 1.45; }",
            "    h1, h2 { margin-bottom: 0.35rem; }",
            "    p { margin-top: 0.25rem; }",
            "    .meta { color: #444; margin-bottom: 1rem; }",
            "    .callout { background: #f7f7f7; border-left: 4px solid #555; padding: 0.75rem 1rem; }",
            "    .tbl { border-collapse: collapse; width: 100%; margin-bottom: 1rem; }",
            "    .tbl th, .tbl td { border: 1px solid #ddd; padding: 0.4rem 0.5rem; text-align: left; }",
            "    .tbl th { background: #f2f2f2; }",
            "    .viz-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 0.9rem; }",
            "    .viz-card { border: 1px solid #e4e7eb; border-radius: 0.45rem; background: #fff; padding: 0.55rem 0.65rem; margin: 0; }",
            "    .viz-card-wide { grid-column: 1 / -1; }",
            "    .viz-card h3 { margin: 0 0 0.24rem 0; font-size: 1.25rem; }",
            "    .viz-caption { margin: 0 0 0.5rem 0; color: #555; font-size: 1.0rem; }",
            "    .viz-card svg { width: 100%; height: auto; display: block; }",
            "    .kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.45rem; }",
            "    .kpi-tile { border: 1px solid #e4e7eb; border-radius: 0.35rem; padding: 0.45rem 0.55rem; background: #fafbfd; }",
            "    .kpi-label { display: block; font-size: 0.76rem; color: #5a6571; margin-bottom: 0.22rem; }",
            "    .kpi-value { font-size: 1.08rem; color: #1d2a38; }",
            "    .viz-btn-row { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.35rem 0 0.65rem 0; }",
            "    .viz-btn { border: 1px solid #9aa8b6; background: #fff; color: #1d2a38; border-radius: 0.25rem; padding: 0.22rem 0.55rem; cursor: pointer; }",
            "    .viz-btn.is-active { background: #1F4E79; color: #fff; border-color: #1F4E79; }",
            "    .appendix-mini-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 0.6rem; }",
            "    details { border: 1px solid #e4e7eb; border-radius: 0.35rem; background: #fff; padding: 0.35rem 0.5rem; }",
            "    details summary { cursor: pointer; font-weight: 600; }",
            "    .appendix-section { margin-top: 0.9rem; }",
            "    code { background: #f3f3f3; padding: 0.1rem 0.3rem; border-radius: 0.2rem; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Full-Cohort Hourly Descriptive Report</h1>",
            f"  <p class=\"meta\"><strong>Run ID:</strong> {escape(run_id)} | "
            f"<strong>Generated:</strong> {escape(payload['generated_at_utc'])}</p>",
            _render_run_context_html(run_manifest),
            "  <section>",
            "    <h2>Core Summary</h2>",
            "    <ul>",
            summary_list,
            "    </ul>",
            "  </section>",
            "  <section>",
            "    <h2>Cohort Definitions</h2>",
            "    <p><strong>Intervention cohort:</strong> patients who experienced at least one fall during their stay.</p>",
            "    <p><strong>Control cohort:</strong> patients with no documented fall during their stay.</p>",
            "  </section>",
            "  <section class=\"callout\">",
            "    <h2>Data Limitations</h2>",
            f"    <p><strong>Control denominator rows with valid pct sum:</strong> "
            f"{_fmt_int(denominator.get('rows_with_valid_pct_sum', 0))} / "
            f"{_fmt_int(denominator.get('total_rows', 0))}</p>",
            f"    <p>{escape(_run_mode_limitations_text(context, 'full_cohort'))}</p>",
            "  </section>",
            _stakeholder_readout_html(run_manifest, source_profile, qa_summary),
            "  <section>",
            "    <h2>Conclusions</h2>",
            _full_cohort_conclusions_html(source_profile, qa_summary, tables),
            "  </section>",
            visualizations,
            appendix_visuals,
            table_appendix,
            "  <section>",
            "    <h2>Source Artifacts Used</h2>",
            "    <ul>",
            artifact_list,
            "    </ul>",
            "  </section>",
            "</body>",
            "</html>",
        ]
    )


def render_chair_bed_inference_html(payload: dict[str, Any]) -> str:
    run_id = payload["run_id"]
    run_manifest = payload["run_manifest"]
    source_profile = payload["source_profile"]
    qa_summary = payload["qa_summary"]
    tables: dict[str, pd.DataFrame] = payload["tables"]
    artifacts: dict[str, Path] = payload["artifacts"]
    project_root = payload.get("project_root")

    hourly = source_profile.get("hourly_location_aggregation", {})
    denominator = source_profile.get("control_denominator_completeness", {})
    context = _report_run_context(run_manifest)

    summary_items = [
        ("Hourly rows", _fmt_int(hourly.get("rows", 0))),
        ("Distinct monitors", _fmt_int(hourly.get("distinct_monitor_id", 0))),
        ("Distinct patients", _fmt_int(hourly.get("distinct_patient_id", 0))),
        ("Eligibility units", _fmt_int(qa_summary.get("eligibility_units", 0))),
        ("Eligibility units passed", _fmt_int(qa_summary.get("eligibility_units_passed", 0))),
        ("Analysis base rows", _fmt_int(qa_summary.get("analysis_base_rows", 0))),
    ]
    summary_list = "\n".join(
        f"      <li><strong>{escape(label)}:</strong> {escape(value)}</li>"
        for label, value in summary_items
    )

    visualizations = _inference_visualizations_html(
        source_profile,
        qa_summary,
        payload.get("negative_control_coverage", {}),
        tables,
    )
    appendix_visuals = _inference_appendix_visuals_html(tables)
    table_appendix = _table_appendix_html(
        tables,
        _MINIMAL_TABLES_BY_REPORT["inference"],
        heading="Minimal Table Appendix",
    )

    artifact_keys = [
        "run_manifest",
        "source_profile",
        "qa_summary",
        "falls_site_name_aliases",
        "cohort_eligibility_rates",
        "control_denominator_coverage",
        "chair_bed_risk_rates",
        "chair_bed_confidence_sensitivity",
        "chair_bed_missingness_stress",
        "fall_negative_control_effects",
        "fall_negative_control_diagnostics",
        "falls_prefall_location_probabilities",
        "transform_metrics",
    ]
    artifact_list = "\n".join(
        f"      <li><code>{escape(str(path.relative_to(project_root)) if isinstance(project_root, Path) and path.is_relative_to(project_root) else str(path))}</code></li>"
        for key, path in artifacts.items()
        if key in artifact_keys and path.exists()
    )

    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "  <meta charset=\"utf-8\" />",
            f"  <title>Chair-Bed Inference Support Report - {escape(run_id)}</title>",
            "  <style>",
            "    body { font-family: Georgia, serif; margin: 2rem auto; max-width: 1080px; line-height: 1.45; }",
            "    h1, h2 { margin-bottom: 0.35rem; }",
            "    p { margin-top: 0.25rem; }",
            "    .meta { color: #444; margin-bottom: 1rem; }",
            "    .callout { background: #f7f7f7; border-left: 4px solid #555; padding: 0.75rem 1rem; }",
            "    .tbl { border-collapse: collapse; width: 100%; margin-bottom: 1rem; }",
            "    .tbl th, .tbl td { border: 1px solid #ddd; padding: 0.4rem 0.5rem; text-align: left; }",
            "    .tbl th { background: #f2f2f2; }",
            "    .viz-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 0.9rem; }",
            "    .viz-card { border: 1px solid #e4e7eb; border-radius: 0.45rem; background: #fff; padding: 0.55rem 0.65rem; margin: 0; }",
            "    .viz-card-wide { grid-column: 1 / -1; }",
            "    .viz-card h3 { margin: 0 0 0.24rem 0; font-size: 1.25rem; }",
            "    .viz-caption { margin: 0 0 0.5rem 0; color: #555; font-size: 1.0rem; }",
            "    .viz-card svg { width: 100%; height: auto; display: block; }",
            "    .kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.45rem; }",
            "    .kpi-tile { border: 1px solid #e4e7eb; border-radius: 0.35rem; padding: 0.45rem 0.55rem; background: #fafbfd; }",
            "    .kpi-label { display: block; font-size: 0.76rem; color: #5a6571; margin-bottom: 0.22rem; }",
            "    .kpi-value { font-size: 1.08rem; color: #1d2a38; }",
            "    .viz-btn-row { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.35rem 0 0.65rem 0; }",
            "    .viz-btn { border: 1px solid #9aa8b6; background: #fff; color: #1d2a38; border-radius: 0.25rem; padding: 0.22rem 0.55rem; cursor: pointer; }",
            "    .viz-btn.is-active { background: #1F4E79; color: #fff; border-color: #1F4E79; }",
            "    .appendix-mini-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 0.6rem; }",
            "    details { border: 1px solid #e4e7eb; border-radius: 0.35rem; background: #fff; padding: 0.35rem 0.5rem; }",
            "    details summary { cursor: pointer; font-weight: 600; }",
            "    .appendix-section { margin-top: 0.9rem; }",
            "    code { background: #f3f3f3; padding: 0.1rem 0.3rem; border-radius: 0.2rem; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Chair-Bed Inference Support Report</h1>",
            f"  <p class=\"meta\"><strong>Run ID:</strong> {escape(run_id)} | "
            f"<strong>Generated:</strong> {escape(payload['generated_at_utc'])}</p>",
            _render_run_context_html(run_manifest),
            "  <section>",
            "    <h2>Core Summary</h2>",
            "    <ul>",
            summary_list,
            "    </ul>",
            "  </section>",
            "  <section class=\"callout\">",
            "    <h2>Interpretation Guardrails</h2>",
            f"    <p><strong>Control denominator rows with valid pct sum:</strong> "
            f"{_fmt_int(denominator.get('rows_with_valid_pct_sum', 0))} / "
            f"{_fmt_int(denominator.get('total_rows', 0))}</p>",
            "    <p>This report summarizes inference-supporting QA outputs and confidence sensitivity. "
            "It does not, by itself, establish causal chair-vs-bed effects.</p>",
            f"    <p>{escape(_run_mode_limitations_text(context, 'inference'))}</p>",
            "  </section>",
            _stakeholder_readout_html(run_manifest, source_profile, qa_summary),
            "  <section>",
            "    <h2>Conclusions</h2>",
            _chair_bed_inference_conclusions_html(source_profile, qa_summary, tables),
            "  </section>",
            visualizations,
            appendix_visuals,
            table_appendix,
            "  <section>",
            "    <h2>Source Artifacts Used</h2>",
            "    <ul>",
            artifact_list,
            "    </ul>",
            "  </section>",
            "</body>",
            "</html>",
        ]
    )


def write_falls_only_descriptive(
    project_root: Path,
    run_id: str,
    output_filename: str = "outputs/falls_only_descriptive.html",
) -> Path:
    payload = load_descriptive_inputs(project_root, run_id)
    payload["project_root"] = project_root
    html = render_falls_only_descriptive_html(payload)
    output_path = Path(output_filename)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    ensure_dir(output_path.parent)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def write_full_cohort_hourly_descriptive(
    project_root: Path,
    run_id: str,
    output_filename: str = "outputs/full_cohort_hourly_descriptive.html",
) -> Path:
    payload = load_full_cohort_hourly_inputs(project_root, run_id)
    payload["project_root"] = project_root
    html = render_full_cohort_hourly_descriptive_html(payload)
    output_path = Path(output_filename)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    ensure_dir(output_path.parent)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def write_chair_bed_inference_report(
    project_root: Path,
    run_id: str,
    output_filename: str = "outputs/chair_bed_inference_descriptive.html",
) -> Path:
    payload = load_chair_bed_inference_inputs(project_root, run_id)
    payload["project_root"] = project_root
    html = render_chair_bed_inference_html(payload)
    output_path = Path(output_filename)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    ensure_dir(output_path.parent)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _label_eval_metric_map(df: pd.DataFrame) -> dict[str, float]:
    if df.empty or not {"metric", "value"}.issubset(df.columns):
        return {}
    result: dict[str, float] = {}
    for _, row in df.iterrows():
        key = str(row["metric"])
        value = _to_float(row["value"])
        if value is not None:
            result[key] = value
    return result


def _kpi_card(label: str, value: str, subtitle: str, tone: str = "neutral") -> str:
    tone_colors = {
        "good": ("#e9f8ef", "#1f7a3d"),
        "warn": ("#fff7e8", "#a16207"),
        "bad": ("#fdecec", "#b42318"),
        "neutral": ("#eef2f6", "#334155"),
    }
    bg, fg = tone_colors.get(tone, tone_colors["neutral"])
    return "\n".join(
        [
            f'      <div class="kpi-tile" style="background:{bg};border-color:{fg}33">',
            f'        <span class="kpi-label" style="color:{fg}">{escape(label)}</span>',
            f'        <span class="kpi-value" style="color:{fg}">{escape(value)}</span>',
            f'        <div class="kpi-sub">{escape(subtitle)}</div>',
            "      </div>",
        ]
    )


def _score_tone(actual: float | None, threshold: float | None, operator: str) -> str:
    if actual is None or threshold is None:
        return "neutral"
    passed = (actual >= threshold) if operator == ">=" else (actual <= threshold)
    return "good" if passed else "bad"


def _bars_html(title: str, subtitle: str, values: list[tuple[str, float]], *, color: str = "#1f4e79") -> str:
    if not values:
        return ""
    max_value = max(value for _, value in values)
    max_value = max(max_value, 1e-9)
    rows = []
    for label, value in values:
        width = max(2.0, (value / max_value) * 100.0)
        rows.append(
            "\n".join(
                [
                    '      <div class="bar-row">',
                    f'        <span class="bar-label">{escape(label)}</span>',
                    "        <div class=\"bar-track\">",
                    f'          <div class="bar-fill" style="width:{width:.1f}%;background:{color}"></div>',
                    "        </div>",
                    f'        <span class="bar-value">{value:.3f}</span>',
                    "      </div>",
                ]
            )
        )
    return "\n".join(
        [
            '    <figure class="viz-card">',
            f"      <h3>{escape(title)}</h3>",
            f'      <p class="viz-caption">{escape(subtitle)}</p>',
            '      <div class="bar-grid">',
            *rows,
            "      </div>",
            "    </figure>",
        ]
    )


def _confusion_heatmap_html(confusion_df: pd.DataFrame) -> str:
    if confusion_df.empty or not {"truth_label", "predicted_label", "count"}.issubset(confusion_df.columns):
        return ""
    labels = ["chair", "bed", "room", "no_patient"]
    matrix = confusion_df.groupby(["truth_label", "predicted_label"], dropna=False)["count"].sum().reset_index()
    value_map = {
        (str(row["truth_label"]), str(row["predicted_label"])): float(_to_float(row["count"]) or 0.0)
        for _, row in matrix.iterrows()
    }
    max_count = max(value_map.values()) if value_map else 1.0
    max_count = max(max_count, 1.0)
    cells = []
    for t_label in labels:
        row_cells = [f'<th class="heat-head">{escape(t_label)}</th>']
        for p_label in labels:
            value = value_map.get((t_label, p_label), 0.0)
            alpha = 0.08 + (0.72 * (value / max_count))
            row_cells.append(
                f'<td class="heat-cell" style="background: rgba(31,78,121,{alpha:.3f})">{int(value)}</td>'
            )
        cells.append("<tr>" + "".join(row_cells) + "</tr>")
    return "\n".join(
        [
            '    <figure class="viz-card">',
            "      <h3>Confusion Matrix (Sequence-Level)</h3>",
            '      <p class="viz-caption">Truth labels by predicted labels. Darker cells indicate larger counts.</p>',
            '      <table class="heat-table">',
            "        <thead><tr><th>Truth \\ Pred</th><th>chair</th><th>bed</th><th>room</th><th>no_patient</th></tr></thead>",
            "        <tbody>",
            *cells,
            "        </tbody>",
            "      </table>",
            "    </figure>",
        ]
    )


def _shadow_model_comparison_card_html(comparison_df: pd.DataFrame) -> str:
    required = {"metric", "baseline_value", "shadow_value", "delta"}
    if comparison_df.empty or not required.issubset(comparison_df.columns):
        return ""
    frame = comparison_df.copy()
    frame["metric"] = frame["metric"].astype(str)
    for column in ("baseline_value", "shadow_value", "delta"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["baseline_value", "shadow_value", "delta"])
    if frame.empty:
        return ""

    preferred_order = [
        "accuracy",
        "macro_f1",
        "macro_roc_auc",
        "macro_pr_auc",
        "log_loss",
        "ece_10_bin",
    ]
    rank = {metric: idx for idx, metric in enumerate(preferred_order)}
    frame["metric_rank"] = frame["metric"].map(rank).fillna(len(rank)).astype(int)
    frame = frame.sort_values(["metric_rank", "metric"], kind="mergesort").head(8)

    rows = "\n".join(
        (
            "        <tr>"
            f"<td>{escape(str(row['metric']))}</td>"
            f"<td>{float(row['baseline_value']):.4f}</td>"
            f"<td>{float(row['shadow_value']):.4f}</td>"
            f"<td>{float(row['delta']):+.4f}</td>"
            "</tr>"
        )
        for _, row in frame.iterrows()
    )
    return "\n".join(
        [
            '<figure class="viz-card">',
            "  <h3>Shadow Model Delta (Report-Only)</h3>",
            "  <p class=\"viz-caption\">Cross-validated shadow model vs baseline heuristic.</p>",
            '  <table class="tbl">',
            "    <thead><tr><th>Metric</th><th>Baseline</th><th>Shadow</th><th>Delta</th></tr></thead>",
            "    <tbody>",
            rows,
            "    </tbody>",
            "  </table>",
            "</figure>",
        ]
    )


def _label_eval_visualizations_html(tables: dict[str, pd.DataFrame]) -> str:
    seq = _label_eval_metric_map(tables.get("Sequence Metrics", pd.DataFrame()))
    prob = _label_eval_metric_map(tables.get("Probability Quality", pd.DataFrame()))
    resp = _label_eval_metric_map(tables.get("Response Timing", pd.DataFrame()))
    pop = _label_eval_metric_map(tables.get("Population Stats", pd.DataFrame()))
    confusion = tables.get("Confusion Matrix", pd.DataFrame())
    shadow_comparison = tables.get("Shadow Model Comparison", pd.DataFrame())

    quality_bars = _bars_html(
        "Core Quality Metrics",
        "Higher is better for accuracy/F1; lower is better for ECE.",
        [
            ("accuracy", float(seq.get("accuracy", 0.0))),
            ("macro_f1", float(seq.get("macro_f1", 0.0))),
            ("detection_f1", float(resp.get("detection_f1", 0.0))),
            ("1 - ece", max(0.0, 1.0 - float(prob.get("ece_10_bin", 0.0)))),
        ],
        color="#0b3c5d",
    )
    latency_bars = _bars_html(
        "Response Timing Error (Seconds)",
        "Lower is better; compares derived timing vs consensus timing.",
        [
            ("MAE", float(resp.get("latency_mae_seconds", 0.0))),
            ("P50 abs error", float(resp.get("latency_abs_error_p50_seconds", 0.0))),
            ("P90 abs error", float(resp.get("latency_abs_error_p90_seconds", 0.0))),
        ],
        color="#b45309",
    )
    population_bars = _bars_html(
        "Population Context",
        "Coverage and composition for current run.",
        [
            ("truth rows", float(pop.get("truth_rows", 0.0))),
            ("sequence count", float(pop.get("sequence_count", 0.0))),
            ("offscreen sequences", float(pop.get("sequence_offscreen_count", 0.0))),
            ("unique monitors", float(pop.get("truth_unique_monitors", 0.0))),
        ],
        color="#334155",
    )
    parts = [
        "  <section>",
        "    <h2>Visual Summary</h2>",
        '    <div class="viz-grid">',
        _confusion_heatmap_html(confusion),
        quality_bars,
        latency_bars,
        population_bars,
        _shadow_model_comparison_card_html(shadow_comparison),
        "    </div>",
        "  </section>",
    ]
    return "\n".join([part for part in parts if part and part.strip()])


def _label_eval_labeled_cohort_html(tables: dict[str, pd.DataFrame]) -> str:
    truth = tables.get("Truth Pre-Fall Location", pd.DataFrame())
    tag_assoc = tables.get("Tag-Location Association", pd.DataFrame())
    signal_profile = tables.get("Signal-Location Profile", pd.DataFrame())
    assoc_summary = _label_eval_metric_map(tables.get("Association Summary", pd.DataFrame()))

    if truth.empty and tag_assoc.empty and signal_profile.empty:
        return ""

    truth_values: list[tuple[str, float]] = []
    if not truth.empty and {"prefall_location", "truth_rows"}.issubset(truth.columns):
        sorted_truth = truth.sort_values(["truth_rows", "prefall_location"], ascending=[False, True], kind="mergesort")
        truth_values = [
            (str(row["prefall_location"]), float(_to_float(row["truth_rows"]) or 0.0))
            for _, row in sorted_truth.iterrows()
        ]

    truth_bars = _bars_html(
        "Labeled Cohort Pre-Fall Location",
        "Consensus labels in the adjudicated cohort.",
        truth_values,
        color="#1f4e79",
    )
    cramers_v = _fmt_value(assoc_summary.get("tag_location_cramers_v"), decimals=3)
    top_share = _fmt_value(assoc_summary.get("top_truth_location_share"), decimals=3)

    return "\n".join(
        [
            "  <section>",
            "    <h2>Labeled Cohort Snapshot</h2>",
            "    <p>Reasonable descriptive readout for stakeholder review: counts/proportions and effect sizes only.</p>",
            "    <div class=\"kpi-row\">",
            _kpi_card(
                "Tag-Location Cramer V",
                cramers_v,
                "association strength from consensus fall tags",
                "neutral",
            ),
            _kpi_card(
                "Top Location Share",
                top_share,
                "share of truth labels in top pre-fall location",
                "neutral",
            ),
            "    </div>",
            '    <div class="viz-grid">',
            truth_bars,
            "    </div>",
            "    <details>",
            "      <summary>Tag-to-Location Association Table</summary>",
            _table_html(tag_assoc),
            "    </details>",
            "    <details>",
            "      <summary>Signal Profile by Truth Location</summary>",
            _table_html(signal_profile),
            "    </details>",
            "  </section>",
        ]
    )


def render_label_eval_shareholder_html(payload: dict[str, Any]) -> str:
    run_id = payload["run_id"]
    run_manifest = payload["run_manifest"]
    source_profile = payload["source_profile"]
    qa_summary = payload["qa_summary"]
    tables: dict[str, pd.DataFrame] = payload["tables"]
    artifacts: dict[str, Path] = payload["artifacts"]
    threshold_checks: dict[str, Any] = payload.get("threshold_checks", {})
    project_root = payload.get("project_root")
    context = _report_run_context(run_manifest)

    seq = _label_eval_metric_map(tables.get("Sequence Metrics", pd.DataFrame()))
    prob = _label_eval_metric_map(tables.get("Probability Quality", pd.DataFrame()))
    resp = _label_eval_metric_map(tables.get("Response Timing", pd.DataFrame()))
    pop = _label_eval_metric_map(tables.get("Population Stats", pd.DataFrame()))

    checks = threshold_checks.get("checks", []) if isinstance(threshold_checks, dict) else []
    check_map = {str(item.get("name")): item for item in checks}
    c_macro = check_map.get("macro_f1", {})
    c_ece = check_map.get("ece_10_bin", {})
    c_det = check_map.get("response_detection_f1", {})
    c_lat = check_map.get("response_latency_mae_seconds", {})

    kpis = "\n".join(
        [
            _kpi_card(
                "Macro F1",
                _fmt_value(seq.get("macro_f1"), decimals=3),
                f"threshold: >= {_fmt_value(_to_float(c_macro.get('threshold')), decimals=2)}",
                _score_tone(_to_float(c_macro.get("actual")), _to_float(c_macro.get("threshold")), ">="),
            ),
            _kpi_card(
                "ECE",
                _fmt_value(prob.get("ece_10_bin"), decimals=3),
                f"threshold: <= {_fmt_value(_to_float(c_ece.get('threshold')), decimals=2)}",
                _score_tone(_to_float(c_ece.get("actual")), _to_float(c_ece.get("threshold")), "<="),
            ),
            _kpi_card(
                "Response F1",
                _fmt_value(resp.get("detection_f1"), decimals=3),
                f"threshold: >= {_fmt_value(_to_float(c_det.get('threshold')), decimals=2)}",
                _score_tone(_to_float(c_det.get("actual")), _to_float(c_det.get("threshold")), ">="),
            ),
            _kpi_card(
                "Latency MAE (s)",
                _fmt_value(resp.get("latency_mae_seconds"), decimals=1),
                f"threshold: <= {_fmt_value(_to_float(c_lat.get('threshold')), decimals=1)}",
                _score_tone(_to_float(c_lat.get("actual")), _to_float(c_lat.get("threshold")), "<="),
            ),
            _kpi_card(
                "Scored Sequences",
                _fmt_value(seq.get("scored_sequences"), decimals=0),
                f"offscreen sequences: {_fmt_value(pop.get('sequence_offscreen_count'), decimals=0)}",
                "neutral",
            ),
        ]
    )

    eval_state = "Not evaluated"
    eval_tone = "warn"
    if bool(threshold_checks.get("evaluated")):
        overall = bool(threshold_checks.get("overall_pass"))
        eval_state = "Pass" if overall else "Fail"
        eval_tone = "good" if overall else "bad"

    gate_card = _kpi_card(
        "Threshold Gate",
        eval_state,
        f"mode: {threshold_checks.get('mode', 'dual')}",
        eval_tone,
    )

    visualizations = _label_eval_visualizations_html(tables)
    labeled_cohort = _label_eval_labeled_cohort_html(tables)
    table_appendix = _table_appendix_html(
        tables,
        [
            "Sequence Metrics",
            "Probability Quality",
            "Response Timing",
            "Population Stats",
            "Truth Pre-Fall Location",
            "Association Summary",
            "Shadow Model Metrics",
            "Shadow Model Comparison",
            "Shadow Ablation",
            "Shadow Feature Importance",
            "Shadow Window Sensitivity",
        ],
        heading="Minimal Table Appendix",
    )
    artifact_keys = [
        "run_manifest",
        "source_profile",
        "qa_summary",
        "label_eval_sequence_metrics",
        "label_eval_confusion_matrix",
        "label_eval_probability_quality",
        "label_eval_response_timing",
        "label_eval_population_stats",
        "label_eval_truth_prefall_location",
        "label_eval_tag_location_association",
        "label_eval_signal_location_profile",
        "label_eval_association_summary",
        "label_eval_shadow_model_metrics",
        "label_eval_shadow_model_comparison",
        "onset_events",
        "onset_case_crossover",
        "shadow_feature_importance",
        "shadow_ablation",
        "shadow_window_sensitivity",
        "shadow_cv_fold_metrics",
        "label_eval_threshold_checks",
    ]
    artifact_list = "\n".join(
        f"      <li><code>{escape(str(path.relative_to(project_root)) if isinstance(project_root, Path) and path.is_relative_to(project_root) else str(path))}</code></li>"
        for key, path in artifacts.items()
        if key in artifact_keys and path.exists()
    )

    summary_items = [
        ("Run mode", context.get("effective_mode", "unknown")),
        ("Truth rows", _fmt_int(pop.get("truth_rows", 0))),
        ("Unique monitors", _fmt_int(pop.get("truth_unique_monitors", 0))),
        ("Sequence count", _fmt_int(pop.get("sequence_count", 0))),
        ("Scored sequences", _fmt_int(seq.get("scored_sequences", 0))),
        ("Analysis base rows", _fmt_int(qa_summary.get("analysis_base_rows", 0))),
    ]
    summary_list = "\n".join(
        f"      <li><strong>{escape(label)}:</strong> {escape(value)}</li>"
        for label, value in summary_items
    )

    denominator = source_profile.get("control_denominator_completeness", {})
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "  <meta charset=\"utf-8\" />",
            f"  <title>Label Evaluation Shareholder Report - {escape(run_id)}</title>",
            "  <style>",
            "    body { font-family: Georgia, serif; margin: 2rem auto; max-width: 1120px; line-height: 1.45; }",
            "    h1, h2 { margin-bottom: 0.35rem; }",
            "    p { margin-top: 0.25rem; }",
            "    .meta { color: #444; margin-bottom: 1rem; }",
            "    .callout { background: #f7f7f7; border-left: 4px solid #555; padding: 0.75rem 1rem; }",
            "    .tbl { border-collapse: collapse; width: 100%; margin-bottom: 1rem; }",
            "    .tbl th, .tbl td { border: 1px solid #ddd; padding: 0.4rem 0.5rem; text-align: left; }",
            "    .tbl th { background: #f2f2f2; }",
            "    .viz-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 0.9rem; }",
            "    .viz-card { border: 1px solid #e4e7eb; border-radius: 0.45rem; background: #fff; padding: 0.55rem 0.65rem; margin: 0; }",
            "    .viz-card h3 { margin: 0 0 0.24rem 0; font-size: 1.25rem; }",
            "    .viz-caption { margin: 0 0 0.5rem 0; color: #555; font-size: 1.0rem; }",
            "    .kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.55rem; }",
            "    .kpi-tile { border: 1px solid #e4e7eb; border-radius: 0.35rem; padding: 0.45rem 0.55rem; background: #fafbfd; }",
            "    .kpi-label { display: block; font-size: 0.76rem; color: #5a6571; margin-bottom: 0.22rem; }",
            "    .kpi-value { font-size: 1.12rem; color: #1d2a38; font-weight: 700; }",
            "    .kpi-sub { margin-top: 0.2rem; color: #4b5563; font-size: 0.74rem; }",
            "    .bar-grid { display: grid; gap: 0.36rem; }",
            "    .bar-row { display: grid; grid-template-columns: 150px 1fr 82px; align-items: center; gap: 0.5rem; }",
            "    .bar-label { font-size: 0.88rem; color: #1f2937; }",
            "    .bar-track { width: 100%; height: 12px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }",
            "    .bar-fill { height: 100%; border-radius: 999px; }",
            "    .bar-value { text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.84rem; color: #334155; }",
            "    .heat-table { border-collapse: collapse; width: 100%; margin-top: 0.25rem; }",
            "    .heat-table th, .heat-table td { border: 1px solid #d1d5db; padding: 0.45rem 0.5rem; text-align: center; }",
            "    .heat-head { background: #f8fafc; text-transform: lowercase; }",
            "    .heat-cell { color: #fff; font-weight: 700; }",
            "    details { border: 1px solid #e4e7eb; border-radius: 0.35rem; background: #fff; padding: 0.35rem 0.5rem; }",
            "    details summary { cursor: pointer; font-weight: 600; }",
            "    code { background: #f3f3f3; padding: 0.1rem 0.3rem; border-radius: 0.2rem; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Shareholder Report: Livestream Label Evaluation</h1>",
            f"  <p class=\"meta\"><strong>Run ID:</strong> {escape(run_id)} | "
            f"<strong>Generated:</strong> {escape(payload['generated_at_utc'])}</p>",
            _render_run_context_html(run_manifest),
            "  <section>",
            "    <h2>Executive Summary</h2>",
            "    <ul>",
            summary_list,
            "    </ul>",
            "  </section>",
            "  <section class=\"callout\">",
            "    <h2>Gate Status</h2>",
            f"    <p><strong>Threshold checks enabled:</strong> {escape(str(bool(threshold_checks.get('enabled'))))}</p>",
            f"    <p><strong>Thresholds evaluated:</strong> {escape(str(bool(threshold_checks.get('evaluated'))))}</p>",
            f"    <p><strong>Control denominator rows with valid pct sum:</strong> {_fmt_int(denominator.get('rows_with_valid_pct_sum', 0))} / {_fmt_int(denominator.get('total_rows', 0))}</p>",
            "  </section>",
            "  <section>",
            "    <h2>KPI Scorecards</h2>",
            '    <div class="kpi-row">',
            gate_card,
            kpis,
            "    </div>",
            "  </section>",
            visualizations,
            labeled_cohort,
            table_appendix,
            "  <section>",
            "    <h2>Source Artifacts Used</h2>",
            "    <ul>",
            artifact_list,
            "    </ul>",
            "  </section>",
            "</body>",
            "</html>",
        ]
    )


def write_label_eval_shareholder_report(
    project_root: Path,
    run_id: str,
    output_filename: str = "outputs/label_eval_shareholder_report.html",
) -> Path:
    payload = load_label_eval_shareholder_inputs(project_root, run_id)
    payload["project_root"] = project_root
    html = render_label_eval_shareholder_html(payload)
    output_path = Path(output_filename)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    ensure_dir(output_path.parent)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def _sum_numeric(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())


def _collect_comparison_metrics(project_root: Path, run_id: str) -> dict[str, float]:
    artifacts = _artifact_paths(project_root, run_id)
    source_profile = read_json(artifacts["source_profile"])
    qa_summary = read_json(artifacts["qa_summary"])
    transform_metrics = _read_optional_json(artifacts["transform_metrics"])

    fall_events = source_profile.get("fall_events_source", {})
    hourly = source_profile.get("hourly_location_aggregation", {})
    denominator = source_profile.get("control_denominator_completeness", {})

    metrics = {
        "fall_events_rows": float(fall_events.get("rows", 0) or 0),
        "hourly_rows": float(hourly.get("rows", 0) or 0),
        "control_valid_rows": float(denominator.get("rows_with_valid_pct_sum", 0) or 0),
        "control_total_rows": float(denominator.get("total_rows", 0) or 0),
        "eligibility_units": float(qa_summary.get("eligibility_units", 0) or 0),
        "eligibility_units_passed": float(qa_summary.get("eligibility_units_passed", 0) or 0),
        "analysis_base_rows": float(qa_summary.get("analysis_base_rows", 0) or 0),
    }

    mapping = transform_metrics.get("mapping_report", {})
    for key in ("hourly_mapped_rate", "events_mapped_rate"):
        value = _to_float(mapping.get(key))
        if value is not None:
            metrics[key] = value

    if artifacts["control_denominator_coverage"].exists():
        coverage = pd.read_csv(artifacts["control_denominator_coverage"])
        metrics["coverage_rows_sum"] = _sum_numeric(coverage, "rows")
        metrics["coverage_valid_rows_sum"] = _sum_numeric(coverage, "valid_rows")

    if artifacts["cohort_eligibility_rates"].exists():
        pass_fail = pd.read_csv(artifacts["cohort_eligibility_rates"])
        metrics["cohort_units_sum"] = _sum_numeric(pass_fail, "units")
        if "eligible" in pass_fail.columns:
            passed = pass_fail.loc[pass_fail["eligible"].astype(str).str.lower() == "true"]
            metrics["cohort_units_passed"] = _sum_numeric(passed, "units")

    if artifacts["falls_by_site"].exists():
        falls_site = pd.read_csv(artifacts["falls_by_site"])
        metrics["falls_by_site_sum"] = _sum_numeric(falls_site, "falls")

    if artifacts["falls_prefall_location_probabilities"].exists():
        pre_prob = pd.read_csv(artifacts["falls_prefall_location_probabilities"])
        metrics["expected_falls_sum"] = _sum_numeric(pre_prob, "expected_falls")

    return metrics


def _is_major_disagreement(metric_key: str, baseline: float, candidate: float) -> bool:
    delta = abs(candidate - baseline)
    relative = (delta / abs(baseline)) if baseline else None
    thresholds = {
        "fall_events_rows": {"abs": 5.0, "rel": 0.05},
        "hourly_rows": {"abs": 5000.0, "rel": 0.05},
        "control_valid_rows": {"abs": 20000.0, "rel": 0.1},
        "control_total_rows": {"abs": 20000.0, "rel": 0.1},
        "eligibility_units": {"abs": 250.0, "rel": 0.1},
        "eligibility_units_passed": {"abs": 100.0, "rel": 0.2},
        "analysis_base_rows": {"abs": 10000.0, "rel": 0.25},
        "hourly_mapped_rate": {"abs": 0.05},
        "events_mapped_rate": {"abs": 0.05},
        "coverage_rows_sum": {"abs": 20000.0, "rel": 0.1},
        "coverage_valid_rows_sum": {"abs": 20000.0, "rel": 0.1},
        "cohort_units_sum": {"abs": 250.0, "rel": 0.1},
        "cohort_units_passed": {"abs": 100.0, "rel": 0.2},
        "falls_by_site_sum": {"abs": 5.0, "rel": 0.05},
        "expected_falls_sum": {"abs": 5.0, "rel": 0.1},
    }
    rule = thresholds.get(metric_key)
    if not rule:
        return False
    if baseline == 0:
        return delta >= rule["abs"]
    rel_threshold = rule.get("rel")
    if rel_threshold is None:
        return delta >= rule["abs"]
    return delta >= rule["abs"] and relative is not None and relative >= rel_threshold


def compare_descriptive_runs(project_root: Path, baseline_run_id: str, candidate_run_id: str) -> dict[str, Any]:
    baseline_metrics = _collect_comparison_metrics(project_root, baseline_run_id)
    candidate_metrics = _collect_comparison_metrics(project_root, candidate_run_id)
    keys = sorted(set(baseline_metrics) | set(candidate_metrics))

    comparisons: list[dict[str, Any]] = []
    major = []
    for key in keys:
        base_value = float(baseline_metrics.get(key, 0.0))
        cand_value = float(candidate_metrics.get(key, 0.0))
        delta = cand_value - base_value
        pct_delta = None if base_value == 0 else (delta / base_value)
        is_major = _is_major_disagreement(key, base_value, cand_value)
        row = {
            "metric": key,
            "baseline": base_value,
            "candidate": cand_value,
            "delta": delta,
            "pct_delta": pct_delta,
            "major_disagreement": is_major,
        }
        comparisons.append(row)
        if is_major:
            major.append(row)

    return {
        "generated_at_utc": utc_now_iso(),
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "metrics": comparisons,
        "major_disagreements": major,
        "major_disagreement_count": len(major),
    }


def render_descriptive_comparison_markdown(payload: dict[str, Any]) -> str:
    baseline_run_id = payload["baseline_run_id"]
    candidate_run_id = payload["candidate_run_id"]
    major = payload.get("major_disagreements", [])
    lines = [
        f"# Descriptive Output Comparison: {baseline_run_id} vs {candidate_run_id}",
        "",
        f"Generated: {payload.get('generated_at_utc', utc_now_iso())}",
        "",
        f"Major disagreements: {len(major)}",
        "",
        "## Major Disagreements",
    ]
    if major:
        for row in major:
            pct_text = "n/a" if row["pct_delta"] is None else f"{row['pct_delta']:.1%}"
            lines.append(
                f"- {row['metric']}: baseline={row['baseline']:.4f}, candidate={row['candidate']:.4f}, "
                f"delta={row['delta']:.4f}, pct_delta={pct_text}"
            )
    else:
        lines.append("- None.")

    lines.extend(["", "## All Compared Metrics"])
    for row in payload.get("metrics", []):
        pct_text = "n/a" if row["pct_delta"] is None else f"{row['pct_delta']:.1%}"
        lines.append(
            f"- {row['metric']}: baseline={row['baseline']:.4f}, candidate={row['candidate']:.4f}, "
            f"delta={row['delta']:.4f}, pct_delta={pct_text}, major={row['major_disagreement']}"
        )

    return "\n".join(lines)


def write_descriptive_comparison(
    project_root: Path,
    baseline_run_id: str,
    candidate_run_id: str,
    output_json: str = "outputs/qa/descriptive_output_comparison.json",
    output_markdown: str = "outputs/qa/descriptive_output_comparison.md",
) -> tuple[Path, Path]:
    payload = compare_descriptive_runs(project_root, baseline_run_id, candidate_run_id)

    output_json_path = Path(output_json)
    if not output_json_path.is_absolute():
        output_json_path = project_root / output_json_path
    ensure_dir(output_json_path.parent)
    write_json(output_json_path, payload)

    markdown = render_descriptive_comparison_markdown(payload)
    output_markdown_path = Path(output_markdown)
    if not output_markdown_path.is_absolute():
        output_markdown_path = project_root / output_markdown_path
    ensure_dir(output_markdown_path.parent)
    output_markdown_path.write_text(markdown, encoding="utf-8")

    return output_json_path, output_markdown_path
