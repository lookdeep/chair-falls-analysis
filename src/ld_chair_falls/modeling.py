from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .config import Settings
from .consensus import included_fall_annotations, load_consensus_annotations, parse_event_key
from .dayparts import DAYPART_SENSITIVITY_LABELS, DAYPART_SPECS

FORMULA = (
    "fall_count ~ C(position, Treatment(reference='bed'))"
    " + C(daypart) + C(day_of_week) + C(calendar_quarter) + C(division_id)"
)
COEF_NAME = "C(position, Treatment(reference='bed'))[T.chair]"


def _load_analysis_base(settings: Settings) -> pd.DataFrame:
    path = settings.paths.staged_run_dir / "prep.patient_hour_analysis_base_v1.parquet"
    df = pd.read_parquet(path)
    df["division_id"] = df["division_id"].astype(str)
    return df


def _build_position_cells(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["division_id", "daypart", "day_of_week", "calendar_quarter"]

    chair_rows = df.copy()
    chair_rows["position"] = "chair"
    chair_rows["exposure_hours"] = df["pct_chair"]
    # for fall rows: attribute falls proportionally by pct_chair / (pct_chair + pct_bed)
    total_pos = (df["pct_chair"] + df["pct_bed"]).clip(lower=1e-9)
    chair_rows["fall_count"] = df["falls_in_hour"] * df["pct_chair"] / total_pos

    bed_rows = df.copy()
    bed_rows["position"] = "bed"
    bed_rows["exposure_hours"] = df["pct_bed"]
    bed_rows["fall_count"] = df["falls_in_hour"] * df["pct_bed"] / total_pos

    combined = pd.concat([chair_rows, bed_rows], ignore_index=True)

    agg = (
        combined.groupby(group_cols + ["position"], as_index=False)
        .agg(exposure_hours=("exposure_hours", "sum"), fall_count=("fall_count", "sum"))
    )
    agg["exposure_hours"] = agg["exposure_hours"].clip(lower=1e-6)
    return agg


def _insufficient_data_result(cells: pd.DataFrame, label: str, *, model_family: str, se_type: str) -> dict:
    n_events = float(cells["fall_count"].sum())
    return {
        "sensitivity_label": label,
        "position": "chair",
        "rr": None,
        "ci_lower": None,
        "ci_upper": None,
        "p_value": None,
        "n_events": n_events,
        "exposure_hours": float(cells.loc[cells["position"] == "chair", "exposure_hours"].sum()),
        "model_family": model_family,
        "se_type": se_type,
        "cluster_variable": "division_id" if se_type == "cluster" else "",
        "deviance_df_ratio": None,
        "pearson_df_ratio": None,
        "model_notes": "insufficient_data",
    }


def _fit_rate_model(
    cells: pd.DataFrame,
    label: str,
    *,
    model_family: str = "poisson",
    se_type: str = "default",
) -> dict:
    n_events_chair = cells.loc[cells["position"] == "chair", "fall_count"].sum()
    n_events_bed = cells.loc[cells["position"] == "bed", "fall_count"].sum()
    n_events = n_events_chair + n_events_bed

    if n_events_chair < 5 or n_events_bed < 5:
        return _insufficient_data_result(
            cells,
            label,
            model_family=model_family,
            se_type=se_type,
        )

    try:
        offset = np.log(cells["exposure_hours"].clip(lower=1e-6))
        family = sm.families.Poisson() if model_family == "poisson" else sm.families.NegativeBinomial()
        fit_kwargs: dict[str, object] = {}
        if se_type == "hc3":
            fit_kwargs["cov_type"] = "HC3"
        elif se_type == "cluster":
            fit_kwargs["cov_type"] = "cluster"
            fit_kwargs["cov_kwds"] = {"groups": cells["division_id"].astype(str)}
        model = smf.glm(FORMULA, data=cells, family=family, offset=offset).fit(**fit_kwargs)

        coef = model.params[COEF_NAME]
        se = model.bse[COEF_NAME]
        pval = model.pvalues[COEF_NAME]
        dev_df_ratio = float(model.deviance / model.df_resid) if float(model.df_resid) > 0 else None
        pearson_df_ratio = (
            float(model.pearson_chi2 / model.df_resid) if float(model.df_resid) > 0 else None
        )

        rr = float(np.exp(coef))
        ci_lower = float(np.exp(coef - 1.96 * se))
        ci_upper = float(np.exp(coef + 1.96 * se))

        return {
            "sensitivity_label": label,
            "position": "chair",
            "rr": rr,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": float(pval),
            "n_events": float(n_events),
            "exposure_hours": float(cells.loc[cells["position"] == "chair", "exposure_hours"].sum()),
            "model_family": model_family,
            "se_type": se_type,
            "cluster_variable": "division_id" if se_type == "cluster" else "",
            "deviance_df_ratio": dev_df_ratio,
            "pearson_df_ratio": pearson_df_ratio,
            "model_notes": "",
        }
    except Exception as exc:
        return {
            "sensitivity_label": label,
            "position": "chair",
            "rr": None,
            "ci_lower": None,
            "ci_upper": None,
            "p_value": None,
            "n_events": float(n_events),
            "exposure_hours": float(cells.loc[cells["position"] == "chair", "exposure_hours"].sum()),
            "model_family": model_family,
            "se_type": se_type,
            "cluster_variable": "division_id" if se_type == "cluster" else "",
            "deviance_df_ratio": None,
            "pearson_df_ratio": None,
            "model_notes": f"model_error: {exc}",
        }


def _primary_covariate_table(cells: pd.DataFrame) -> pd.DataFrame:
    offset = np.log(cells["exposure_hours"].clip(lower=1e-6))
    model = smf.glm(FORMULA, data=cells, family=sm.families.Poisson(), offset=offset).fit()

    rows = []
    for name in model.params.index:
        coef = model.params[name]
        se = model.bse[name]
        pval = model.pvalues[name]
        rows.append({
            "covariate": name,
            "coef": float(coef),
            "se": float(se),
            "rr": float(np.exp(coef)),
            "ci_lower": float(np.exp(coef - 1.96 * se)),
            "ci_upper": float(np.exp(coef + 1.96 * se)),
            "p_value": float(pval),
        })
    return pd.DataFrame(rows)


def _ambulatory_rate(df: pd.DataFrame) -> dict:
    total_pos = (df["pct_chair"] + df["pct_bed"] + df["pct_ambulatory"]).clip(lower=1e-9)
    amb_falls = (df["falls_in_hour"] * df["pct_ambulatory"] / total_pos).sum()
    amb_hours = df["pct_ambulatory"].sum()
    rate = (amb_falls / max(amb_hours, 1e-9)) * 1000.0
    return {
        "sensitivity_label": "exploratory_unadjusted",
        "position": "ambulatory",
        "rr": None,
        "ci_lower": None,
        "ci_upper": None,
        "p_value": None,
        "n_events": float(amb_falls),
        "exposure_hours": float(amb_hours),
        "model_family": "unadjusted",
        "se_type": "na",
        "cluster_variable": "",
        "deviance_df_ratio": None,
        "pearson_df_ratio": None,
        "model_notes": f"unadjusted_rate_per_1000h={rate:.4f}",
    }


def _slice_daypart_window(df: pd.DataFrame, daypart: str) -> pd.DataFrame:
    if "daypart" in df.columns:
        return df[df["daypart"] == daypart].copy()

    if "hour_ts" not in df.columns:
        return df.iloc[0:0].copy()

    local_hour = pd.to_datetime(df["hour_ts"], errors="coerce").dt.hour
    for key, start_hour, end_hour, _display in DAYPART_SPECS:
        if key != daypart:
            continue
        return df[(local_hour >= start_hour) & (local_hour <= end_hour)].copy()
    return df.iloc[0:0].copy()


def _apply_misclassification_swap(
    cells: pd.DataFrame,
    *,
    chair_to_bed_rate: float,
    bed_to_chair_rate: float,
) -> pd.DataFrame:
    strata_cols = ["division_id", "daypart", "day_of_week", "calendar_quarter"]
    chair = cells.loc[cells["position"] == "chair", strata_cols + ["exposure_hours", "fall_count"]].rename(
        columns={"exposure_hours": "chair_exposure", "fall_count": "chair_fall"}
    )
    bed = cells.loc[cells["position"] == "bed", strata_cols + ["exposure_hours", "fall_count"]].rename(
        columns={"exposure_hours": "bed_exposure", "fall_count": "bed_fall"}
    )
    merged = chair.merge(bed, on=strata_cols, how="outer").fillna(0.0)
    merged["chair_adj"] = (
        (1.0 - chair_to_bed_rate) * merged["chair_fall"] + bed_to_chair_rate * merged["bed_fall"]
    )
    merged["bed_adj"] = (
        (1.0 - bed_to_chair_rate) * merged["bed_fall"] + chair_to_bed_rate * merged["chair_fall"]
    )
    chair_out = merged[strata_cols].copy()
    chair_out["position"] = "chair"
    chair_out["exposure_hours"] = merged["chair_exposure"].clip(lower=1e-6)
    chair_out["fall_count"] = merged["chair_adj"].clip(lower=0.0)
    bed_out = merged[strata_cols].copy()
    bed_out["position"] = "bed"
    bed_out["exposure_hours"] = merged["bed_exposure"].clip(lower=1e-6)
    bed_out["fall_count"] = merged["bed_adj"].clip(lower=0.0)
    return pd.concat([chair_out, bed_out], ignore_index=True)


def _run_misclassification_sensitivity(cells_primary: pd.DataFrame) -> pd.DataFrame:
    scenarios = [
        ("symmetric_swap_10pct", 0.10, 0.10),
        ("symmetric_swap_20pct", 0.20, 0.20),
        ("symmetric_swap_30pct", 0.30, 0.30),
        ("chair_to_bed_20pct", 0.20, 0.00),
        ("bed_to_chair_20pct", 0.00, 0.20),
    ]
    rows = []
    for label, chair_to_bed, bed_to_chair in scenarios:
        adjusted_cells = _apply_misclassification_swap(
            cells_primary,
            chair_to_bed_rate=chair_to_bed,
            bed_to_chair_rate=bed_to_chair,
        )
        result = _fit_rate_model(
            adjusted_cells,
            label=label,
            model_family="poisson",
            se_type="default",
        )
        rows.append(
            {
                "scenario_label": label,
                "chair_to_bed_rate": chair_to_bed,
                "bed_to_chair_rate": bed_to_chair,
                "rr": result["rr"],
                "ci_lower": result["ci_lower"],
                "ci_upper": result["ci_upper"],
                "p_value": result["p_value"],
                "n_events": result["n_events"],
                "model_notes": result["model_notes"],
            }
        )
    return pd.DataFrame(rows)


def _run_threshold_sensitivity(settings: Settings, thresholds: tuple[int, ...]) -> pd.DataFrame:
    from .transform import build_analysis_base, build_eligibility

    staged_run_dir = settings.paths.staged_run_dir
    hourly_path = staged_run_dir / "prep.patient_hour_location_with_cohort_v1.parquet"
    events_path = staged_run_dir / "prep.fall_events_with_cohort_v1.parquet"
    if not hourly_path.exists():
        return pd.DataFrame(
            columns=[
                "threshold_hours",
                "eligible_units",
                "analysis_rows",
                "rr",
                "ci_lower",
                "ci_upper",
                "p_value",
                "n_events",
                "model_notes",
            ]
        )
    hourly = pd.read_parquet(hourly_path)
    events = pd.read_parquet(events_path) if events_path.exists() else pd.DataFrame()
    rows = []
    for threshold in thresholds:
        eligibility = build_eligibility(hourly, min_observed_hours=threshold)
        base = build_analysis_base(hourly, events, eligibility)
        if "division_id" in base.columns:
            base["division_id"] = base["division_id"].astype(str)
        # Restrict to intervention cohort per Charter s4 — control units provide denominator context only
        if "cohort_type" in base.columns:
            base = base[base["cohort_type"] == "intervention"].copy()
        cells = _build_position_cells(base)
        result = _fit_rate_model(cells, label=f"min_observed_hours_{threshold}")
        rows.append(
            {
                "threshold_hours": threshold,
                "eligible_units": int(eligibility["eligible"].sum()) if not eligibility.empty else 0,
                "analysis_rows": int(len(base.index)),
                "rr": result["rr"],
                "ci_lower": result["ci_lower"],
                "ci_upper": result["ci_upper"],
                "p_value": result["p_value"],
                "n_events": result["n_events"],
                "model_notes": result["model_notes"],
            }
        )
    return pd.DataFrame(rows)


def _load_consensus_furniture_origin(settings: Settings) -> pd.DataFrame:
    """Load consensus CSV and extract chair-origin room/no_patient events."""
    consensus_path = settings.project_root / settings.fall_labels_consensus_csv_path
    consensus, summary = load_consensus_annotations(consensus_path)
    if summary.get("status") != "ok":
        return pd.DataFrame(columns=["event_key", "monitor_id", "hour_ts"])
    consensus = parse_event_key(included_fall_annotations(consensus))
    if consensus.empty or "last_furniture" not in consensus.columns:
        return pd.DataFrame(columns=["event_key", "monitor_id", "hour_ts"])

    # Derive hour_ts from date + minute (floor to hour)
    hour_str = consensus["date_local"] + " " + consensus["minute_local"].str.split(":").str[0].str.zfill(2) + ":00:00"
    consensus["hour_ts"] = pd.to_datetime(hour_str, errors="coerce")

    # Filter to chair-origin room/no_patient events
    mask = consensus["prefall_location"].isin(["room", "no_patient"]) & consensus["last_furniture"].eq("chair")

    result = consensus.loc[mask, ["event_key", "monitor_id", "hour_ts", "last_furniture", "prefall_location"]].copy()
    return result.drop_duplicates(subset=["event_key"]).reset_index(drop=True)


def _reclassify_chair_origin_falls(
    df: pd.DataFrame,
    chair_origin_events: pd.DataFrame,
) -> pd.DataFrame:
    """Override fall attribution for chair-origin events to 100% chair."""
    if chair_origin_events.empty:
        return df.copy()
    result = df.copy()
    # Build set of (monitor_id, hour_ts) for chair-origin events
    co_keys = set()
    for _, row in chair_origin_events.iterrows():
        mid = row["monitor_id"]
        hts = row["hour_ts"]
        if pd.notna(mid) and pd.notna(hts):
            co_keys.add((int(mid), pd.Timestamp(hts)))

    if not co_keys or "monitor_id" not in result.columns or "hour_ts" not in result.columns:
        return result

    for idx, row in result.iterrows():
        mid = row.get("monitor_id")
        hts = row.get("hour_ts")
        if mid is not None and hts is not None and row.get("falls_in_hour", 0) > 0:
            try:
                key = (int(mid), pd.Timestamp(hts))
            except (ValueError, TypeError):
                continue
            if key in co_keys:
                result.at[idx, "pct_chair"] = 1.0
                result.at[idx, "pct_bed"] = 0.0
    return result


def _write_furniture_origin_linkage_diagnostic(
    settings: Settings,
    analysis_base: pd.DataFrame,
    chair_origin_events: pd.DataFrame,
) -> Path:
    """Write diagnostic showing which consensus events matched the analysis base."""
    qa_dir = settings.paths.qa_dir
    qa_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, event in chair_origin_events.iterrows():
        mid = event.get("monitor_id")
        hts = event.get("hour_ts")
        matched = False
        falls_in_hour = 0
        pct_chair_original = None

        if pd.notna(mid) and pd.notna(hts) and "monitor_id" in analysis_base.columns:
            mask = (
                (analysis_base["monitor_id"].astype("Int64") == int(mid))
                & (analysis_base["hour_ts"] == pd.Timestamp(hts))
            )
            matches = analysis_base.loc[mask]
            if not matches.empty:
                matched = True
                falls_in_hour = float(matches["falls_in_hour"].sum())
                pct_chair_original = float(matches["pct_chair"].iloc[0])

        rows.append({
            "event_key": event.get("event_key", ""),
            "monitor_id": mid,
            "hour_ts": hts,
            "last_furniture": event.get("last_furniture", ""),
            "prefall_location": event.get("prefall_location", ""),
            "matched_in_analysis_base": matched,
            "falls_in_matched_hour": falls_in_hour,
            "pct_chair_original": pct_chair_original,
        })

    diag_df = pd.DataFrame(rows)
    path = qa_dir / f"furniture_origin_reclassification_diagnostic_{settings.run_id}.csv"
    diag_df.to_csv(path, index=False)
    return path


def run_adjusted_rr_model(settings: Settings) -> tuple[Path, Path]:
    """Fit Poisson RR model + sensitivity analyses. Returns (adjusted_rr_path, sensitivity_path)."""
    df = _load_analysis_base(settings)
    # Restrict to intervention cohort per Charter s4 — control units provide denominator context only
    df = df[df["cohort_type"] == "intervention"].copy()
    cells_primary = _build_position_cells(df)

    results = []

    primary = _fit_rate_model(cells_primary, "primary_adjusted", model_family="poisson", se_type="default")
    results.append(primary)
    results.append(_fit_rate_model(cells_primary, "primary_adjusted_hc3", model_family="poisson", se_type="hc3"))
    results.append(
        _fit_rate_model(cells_primary, "primary_adjusted_clustered", model_family="poisson", se_type="cluster")
    )
    if primary["deviance_df_ratio"] is not None and primary["deviance_df_ratio"] > 1.5:
        results.append(
            _fit_rate_model(
                cells_primary,
                "primary_adjusted_negative_binomial",
                model_family="negative_binomial",
                se_type="default",
            )
        )

    sensitivities = [
        ("weekday_only", df[df["day_of_week"].isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])]),
        ("weekend_only", df[df["day_of_week"].isin(["Saturday", "Sunday"])]),
        ("position_certain_only", df[(df["pct_chair"] >= 0.5) | (df["pct_bed"] >= 0.5)]),
    ]
    sensitivities = [
        *[(label, _slice_daypart_window(df, daypart)) for daypart, label in DAYPART_SENSITIVITY_LABELS.items()],
        *sensitivities,
    ]

    for label, subset in sensitivities:
        cells = _build_position_cells(subset.copy())
        results.append(_fit_rate_model(cells, label, model_family="poisson", se_type="default"))

    results.append(_ambulatory_rate(df))

    # Furniture-origin reclassification sensitivity
    chair_origin_events = _load_consensus_furniture_origin(settings)
    if not chair_origin_events.empty:
        _write_furniture_origin_linkage_diagnostic(settings, df, chair_origin_events)
        df_reclassified = _reclassify_chair_origin_falls(df, chair_origin_events)
        cells_reclassified = _build_position_cells(df_reclassified)
        results.append(
            _fit_rate_model(
                cells_reclassified,
                "furniture_origin_reclassified",
                model_family="poisson",
                se_type="default",
            )
        )

    rr_df = pd.DataFrame(results, columns=[
        "sensitivity_label", "position", "rr", "ci_lower", "ci_upper",
        "p_value", "n_events", "exposure_hours", "model_family",
        "se_type", "cluster_variable", "deviance_df_ratio",
        "pearson_df_ratio", "model_notes",
    ])

    run_id = settings.run_id
    qa_dir = settings.paths.qa_dir
    qa_dir.mkdir(parents=True, exist_ok=True)

    adjusted_rr_path = qa_dir / f"chair_bed_adjusted_rr_{run_id}.csv"
    rr_df.to_csv(adjusted_rr_path, index=False)

    cov_df = pd.DataFrame(
        columns=["covariate", "coef", "se", "rr", "ci_lower", "ci_upper", "p_value"]
    )
    primary_n_chair = cells_primary.loc[cells_primary["position"] == "chair", "fall_count"].sum()
    primary_n_bed = cells_primary.loc[cells_primary["position"] == "bed", "fall_count"].sum()
    if primary_n_chair >= 5 and primary_n_bed >= 5:
        try:
            cov_df = _primary_covariate_table(cells_primary)
        except Exception:
            pass

    sensitivity_path = qa_dir / f"chair_bed_adjusted_rr_covariates_{run_id}.csv"
    cov_df.to_csv(sensitivity_path, index=False)

    robustness_path = qa_dir / f"chair_bed_adjusted_rr_robustness_{run_id}.csv"
    rr_df.loc[
        rr_df["sensitivity_label"].isin(
            ["primary_adjusted_hc3", "primary_adjusted_clustered", "primary_adjusted_negative_binomial"]
        )
    ].to_csv(robustness_path, index=False)

    misclassification_path = qa_dir / f"chair_bed_misclassification_sensitivity_{run_id}.csv"
    _run_misclassification_sensitivity(cells_primary).to_csv(misclassification_path, index=False)

    threshold_path = qa_dir / f"chair_bed_threshold_sensitivity_{run_id}.csv"
    _run_threshold_sensitivity(settings, thresholds=(4, 12, 24, 48)).to_csv(threshold_path, index=False)

    return adjusted_rr_path, sensitivity_path
