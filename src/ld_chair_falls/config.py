from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .utils import default_run_id

RunMode = Literal["inferential_ready", "descriptive_only"]
DEFAULT_FALL_LABELS_CONSENSUS_CSV_PATH = Path("data/public/falls-observations-v3-consensus.csv")


@dataclass(frozen=True)
class RunPaths:
    project_root: Path
    run_id: str

    @property
    def raw_run_dir(self) -> Path:
        return self.project_root / "data" / "raw" / self.run_id

    @property
    def staged_run_dir(self) -> Path:
        return self.project_root / "data" / "staged" / self.run_id

    @property
    def manifests_dir(self) -> Path:
        return self.project_root / "outputs" / "manifests"

    @property
    def qa_dir(self) -> Path:
        return self.project_root / "outputs" / "qa"

    @property
    def audit_dir(self) -> Path:
        return self.project_root / "outputs" / "audit"


class Settings(BaseModel):
    project_root: Path
    run_id: str = Field(default_factory=default_run_id)

    study_hospital_id: int = 5
    google_cloud_project: str = "ld-restricted"
    bq_dataset: str = "chair_falls_analysis"
    fall_events_table: str = "bq_falls_directory"
    live_stream_derivatives_table: str = "live_stream_falls_derivatives_cache"
    negative_control_derivatives_table: str = "live_stream_non_falls_derivatives_cache"
    fall_labels_consensus_csv_path: Path = DEFAULT_FALL_LABELS_CONSENSUS_CSV_PATH
    fall_labels_raw_logs_csv_path: Path = Path("docs/falls-observations-v1 - raw_logs.csv")
    fall_labels_rubric_csv_path: Path = Path("docs/falls-observations-v1 - rubric.csv")

    hourly_location_sql_path: Path = Path("sql/02_extract/hourly_location_agg.sql")
    key_dimensions_sql_path: Path = Path("sql/02_extract/key_dimensions.sql")
    fall_events_sql_path: Path = Path("sql/02_extract/fall_events.sql")
    fall_livestream_event_windows_sql_path: Path = Path("sql/02_extract/fall_livestream_event_windows.sql")
    fall_livestream_second_level_sql_path: Path = Path("sql/02_extract/fall_livestream_second_level.sql")
    negative_control_source_inventory_sql_path: Path = Path("sql/02_extract/negative_control_source_inventory.sql")
    fall_case_crossover_windows_sql_path: Path = Path("sql/02_extract/fall_case_crossover_windows.sql")
    fall_case_crossover_second_level_sql_path: Path = Path(
        "sql/02_extract/fall_case_crossover_second_level.sql"
    )
    fall_negative_control_windows_sql_path: Path = Path("sql/02_extract/fall_negative_control_windows.sql")
    fall_negative_control_second_level_sql_path: Path = Path(
        "sql/02_extract/fall_negative_control_second_level.sql"
    )

    cohort_map_path: Path | None = None
    allowed_cohorts: tuple[str, ...] = ("control", "intervention", "observational")
    manual_intervention_monitor_ids: tuple[int, ...] = (2834,)
    min_observed_hours: int = 4
    hospital_timezone: str = "America/Chicago"
    study_start_date: date = date(2024, 8, 1)
    study_end_date: date = date(2025, 12, 31)

    requested_run_mode: RunMode = "descriptive_only"
    gate_1_pass: bool = False
    gate_2_pass: bool = False
    gate_1_evidence: str | None = None
    gate_2_evidence: str | None = None

    dry_run: bool = True
    label_eval_mode: Literal["report_only", "threshold_only", "dual"] = "dual"
    label_eval_thresholds_enabled: bool = False
    label_eval_expected_monitor_count: int = 48
    label_eval_threshold_macro_f1: float = 0.55
    label_eval_threshold_ece: float = 0.15
    label_eval_threshold_response_detection_f1: float = 0.70
    label_eval_threshold_response_latency_mae_seconds: float = 45.0
    label_eval_sweep_points: int = 101
    label_eval_operating_objective: Literal["balanced_macro_f1"] = "balanced_macro_f1"
    label_eval_calibration_bins: int = 10
    label_eval_v2_recency_half_life_seconds: int = 45
    label_eval_v2_recent_visible_gap_seconds: int = 60
    label_eval_v2_recent_visible_min_strength: float = 0.55
    label_eval_v2_no_patient_strong_threshold: float = 0.75
    label_eval_v2_room_floor_min: float = 0.10
    label_eval_v2_room_floor_max: float = 0.25
    prefall_panel_anchor_weight: float = 0.45
    prefall_panel_room_bonus: float = 0.28
    prefall_panel_overlap_no_patient_cap: float = 0.20
    prefall_panel_safety_zone_threshold: float = 0.50
    prefall_panel_v3_visible_no_patient_cap: float = 0.35
    prefall_panel_v3_transient_no_patient_cap: float = 0.50
    prefall_panel_v3_present_sparse_no_patient_cap: float = 0.45
    prefall_panel_v3_out_of_room_no_patient_floor: float = 0.55
    prefall_panel_v3_presence_min_support: float = 0.35
    prefall_panel_v3_departure_min_confidence: float = 0.35
    prefall_panel_v3_bed_in_bed_min_score: float = 0.55
    prefall_panel_v3_bed_in_bed_invisible_min_score: float = 0.95
    prefall_panel_v3_motion_presence_scale: float = 0.15
    prefall_panel_v3_room_object_mean_min: float = 0.50
    prefall_panel_v3_sparse_corroboration_min: float = 0.60
    prefall_panel_v3_posture_min_frames: int = 6
    prefall_panel_v3_posture_max_switch_count: int = 2
    prefall_panel_v3_posture_bed_min_lying_prob: float = 0.60
    prefall_panel_v3_posture_max_chair_bed_gap: float = 0.20
    prefall_panel_v3_velocity_tiebreak_margin: float = 0.10
    prefall_panel_v3_velocity_min_approach: float = 0.05
    label_eval_v2_target_macro_f1: float = 0.55
    label_eval_v2_target_chair_f1: float = 0.55
    label_eval_v2_target_room_f1: float = 0.45
    label_eval_v2_target_bed_f1_drop: float = 0.05
    label_eval_shadow_cv_folds: int = 5
    label_eval_shadow_cv_repeats: int = 3
    label_eval_shadow_alpha: float = 0.15
    fall_window_pre_anchor_seconds: int = 300
    fall_window_focus_seconds: int = 30
    fall_window_dropout_seconds: int = 180
    fall_window_post_anchor_seconds: int = 180
    fall_case_crossover_anchor_days: tuple[int, ...] = (-1, -2)
    case_crossover_source_status: Literal["pending_external", "ready"] = "ready"
    case_crossover_source_ready: bool = True
    negative_control_source_status: Literal["pending_external", "ready"] = "ready"
    negative_control_source_ready: bool = True
    negative_control_expected_start_date: date = date(2024, 1, 1)
    negative_control_match_limit: int = 3
    label_eval_shadow_window_seconds: tuple[int, ...] = (60, 300)
    label_eval_min_sample_presence_ratio: float = 0.95
    label_eval_min_frames_per_window: int = 30
    label_eval_inter_frame_gap_seconds: int = 3
    label_eval_onset_baseline_seconds: int = 30
    label_eval_onset_distance_std_floor_meters: float = 0.3
    label_eval_onset_distance_zscore: float = 2.0
    label_eval_onset_nudge_threshold: float = 0.5
    label_eval_posture_transition_min_score: float = 0.5
    label_eval_posture_transition_min_run_length: int = 2
    label_eval_onset_min_prevalence_ratio: float = 0.10
    label_eval_shadow_feature_cap: int = 100
    confidence_missingness_cutpoints: tuple[int, ...] = (0, 10, 20, 30)

    @property
    def effective_run_mode(self) -> RunMode:
        return "inferential_ready" if self.gate_1_pass and self.gate_2_pass else "descriptive_only"

    @property
    def paths(self) -> RunPaths:
        return RunPaths(project_root=self.project_root, run_id=self.run_id)

    @property
    def nonfall_control_source_status(self) -> Literal["pending_external", "ready"]:
        return self.negative_control_source_status

    @property
    def nonfall_control_source_ready(self) -> bool:
        return self.negative_control_source_ready


_BOOL_TRUE = {"1", "true", "yes", "y", "on"}


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _BOOL_TRUE


def _read_bool_alias(name: str, alias: str | None, default: bool) -> bool:
    if os.getenv(name) is not None:
        return _read_bool(name, default)
    if alias and os.getenv(alias) is not None:
        return _read_bool(alias, default)
    return default


def _read_env_alias(name: str, alias: str | None, default: str) -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    if alias:
        alias_value = os.getenv(alias)
        if alias_value is not None:
            return alias_value
    return default


def _read_status_alias(name: str, alias: str | None, default: Literal["pending_external", "ready"]) -> Literal["pending_external", "ready"]:
    value = _read_env_alias(name, alias, default).strip().lower()
    return "ready" if value == "ready" else "pending_external"


def _read_date(name: str, default: date) -> date:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return default


def _read_allowed_cohorts(default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv("ALLOWED_COHORTS")
    if not value:
        return default
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    return parsed or default


def _read_manual_intervention_monitors(default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.getenv("MANUAL_INTERVENTION_MONITOR_IDS")
    if not value:
        return default
    monitors: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            monitors.append(int(token))
        except ValueError:
            continue
    return tuple(monitors) if monitors else default


def _read_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.getenv(name)
    if not value:
        return default
    parsed: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            parsed.append(int(token))
        except ValueError:
            continue
    return tuple(parsed) if parsed else default


def load_settings(
    run_id: str | None = None,
    dry_run: bool = True,
    cohort_map_path: str | None = None,
    gate_1_pass: bool | None = None,
    gate_2_pass: bool | None = None,
) -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env", override=False)

    requested_mode = os.getenv("RUN_MODE", "descriptive_only").strip()
    if requested_mode not in {"inferential_ready", "descriptive_only"}:
        requested_mode = "descriptive_only"

    resolved_cohort_map_path = cohort_map_path or os.getenv("COHORT_MAP_PATH")

    return Settings(
        project_root=project_root,
        run_id=run_id or os.getenv("RUN_ID") or default_run_id(),
        study_hospital_id=int(os.getenv("STUDY_HOSPITAL_ID", "5")),
        google_cloud_project=os.getenv("GOOGLE_CLOUD_PROJECT", "ld-restricted"),
        bq_dataset=os.getenv("BQ_DATASET", "chair_falls_analysis"),
        fall_events_table=os.getenv("FALL_EVENTS_TABLE", "bq_falls_directory"),
        live_stream_derivatives_table=os.getenv(
            "LIVE_STREAM_DERIVATIVES_TABLE", "live_stream_falls_derivatives_cache"
        ),
        negative_control_derivatives_table=_read_env_alias(
            "NEGATIVE_CONTROL_DERIVATIVES_TABLE",
            "NONFALL_CONTROL_DERIVATIVES_TABLE",
            "live_stream_non_falls_derivatives_cache",
        ),
        hourly_location_sql_path=Path(
            os.getenv("HOURLY_LOCATION_SQL_PATH", "sql/02_extract/hourly_location_agg.sql")
        ),
        key_dimensions_sql_path=Path(
            os.getenv("KEY_DIMENSIONS_SQL_PATH", "sql/02_extract/key_dimensions.sql")
        ),
        fall_events_sql_path=Path(os.getenv("FALL_EVENTS_SQL_PATH", "sql/02_extract/fall_events.sql")),
        fall_livestream_event_windows_sql_path=Path(
            os.getenv(
                "FALL_LIVESTREAM_EVENT_WINDOWS_SQL_PATH",
                "sql/02_extract/fall_livestream_event_windows.sql",
            )
        ),
        fall_livestream_second_level_sql_path=Path(
            os.getenv(
                "FALL_LIVESTREAM_SECOND_LEVEL_SQL_PATH",
                "sql/02_extract/fall_livestream_second_level.sql",
            )
        ),
        negative_control_source_inventory_sql_path=Path(
            os.getenv(
                "NEGATIVE_CONTROL_SOURCE_INVENTORY_SQL_PATH",
                "sql/02_extract/negative_control_source_inventory.sql",
            )
        ),
        fall_case_crossover_windows_sql_path=Path(
            os.getenv(
                "FALL_CASE_CROSSOVER_WINDOWS_SQL_PATH",
                "sql/02_extract/fall_case_crossover_windows.sql",
            )
        ),
        fall_case_crossover_second_level_sql_path=Path(
            os.getenv(
                "FALL_CASE_CROSSOVER_SECOND_LEVEL_SQL_PATH",
                "sql/02_extract/fall_case_crossover_second_level.sql",
            )
        ),
        fall_negative_control_windows_sql_path=Path(
            os.getenv(
                "FALL_NEGATIVE_CONTROL_WINDOWS_SQL_PATH",
                "sql/02_extract/fall_negative_control_windows.sql",
            )
        ),
        fall_negative_control_second_level_sql_path=Path(
            os.getenv(
                "FALL_NEGATIVE_CONTROL_SECOND_LEVEL_SQL_PATH",
                "sql/02_extract/fall_negative_control_second_level.sql",
            )
        ),
        fall_labels_consensus_csv_path=Path(
            os.getenv(
                "FALL_LABELS_CONSENSUS_CSV_PATH",
                str(DEFAULT_FALL_LABELS_CONSENSUS_CSV_PATH),
            )
        ),
        fall_labels_raw_logs_csv_path=Path(
            os.getenv("FALL_LABELS_RAW_LOGS_CSV_PATH", "docs/falls-observations-v1 - raw_logs.csv")
        ),
        fall_labels_rubric_csv_path=Path(
            os.getenv("FALL_LABELS_RUBRIC_CSV_PATH", "docs/falls-observations-v1 - rubric.csv")
        ),
        cohort_map_path=Path(resolved_cohort_map_path) if resolved_cohort_map_path else None,
        allowed_cohorts=_read_allowed_cohorts(("control", "intervention", "observational")),
        manual_intervention_monitor_ids=_read_manual_intervention_monitors((2834,)),
        min_observed_hours=int(os.getenv("MIN_OBSERVED_HOURS", "4")),
        hospital_timezone=os.getenv("HOSPITAL_TIMEZONE", "America/Chicago"),
        study_start_date=_read_date("STUDY_START_DATE", date(2024, 8, 1)),
        study_end_date=_read_date("STUDY_END_DATE", date(2025, 12, 31)),
        requested_run_mode=requested_mode,
        gate_1_pass=gate_1_pass if gate_1_pass is not None else _read_bool("GATE_1_PASS", False),
        gate_2_pass=gate_2_pass if gate_2_pass is not None else _read_bool("GATE_2_PASS", False),
        gate_1_evidence=os.getenv("GATE_1_EVIDENCE") or None,
        gate_2_evidence=os.getenv("GATE_2_EVIDENCE") or None,
        dry_run=dry_run,
        label_eval_mode=(
            os.getenv("LABEL_EVAL_MODE", "dual").strip().lower()
            if os.getenv("LABEL_EVAL_MODE", "dual").strip().lower() in {"report_only", "threshold_only", "dual"}
            else "dual"
        ),
        label_eval_thresholds_enabled=_read_bool("LABEL_EVAL_THRESHOLDS_ENABLED", False),
        label_eval_expected_monitor_count=int(os.getenv("LABEL_EVAL_EXPECTED_MONITOR_COUNT", "48")),
        label_eval_threshold_macro_f1=float(os.getenv("LABEL_EVAL_THRESHOLD_MACRO_F1", "0.55")),
        label_eval_threshold_ece=float(os.getenv("LABEL_EVAL_THRESHOLD_ECE", "0.15")),
        label_eval_threshold_response_detection_f1=float(
            os.getenv("LABEL_EVAL_THRESHOLD_RESPONSE_DETECTION_F1", "0.70")
        ),
        label_eval_threshold_response_latency_mae_seconds=float(
            os.getenv("LABEL_EVAL_THRESHOLD_RESPONSE_LATENCY_MAE_SECONDS", "45")
        ),
        label_eval_sweep_points=int(os.getenv("LABEL_EVAL_SWEEP_POINTS", "101")),
        label_eval_operating_objective=(
            os.getenv("LABEL_EVAL_OP_OBJECTIVE", "balanced_macro_f1").strip().lower()
            if os.getenv("LABEL_EVAL_OP_OBJECTIVE", "balanced_macro_f1").strip().lower()
            in {"balanced_macro_f1"}
            else "balanced_macro_f1"
        ),
        label_eval_calibration_bins=int(os.getenv("LABEL_EVAL_CALIBRATION_BINS", "10")),
        label_eval_v2_recency_half_life_seconds=int(
            os.getenv("LABEL_EVAL_V2_RECENCY_HALF_LIFE_SECONDS", "45")
        ),
        label_eval_v2_recent_visible_gap_seconds=int(
            os.getenv("LABEL_EVAL_V2_RECENT_VISIBLE_GAP_SECONDS", "60")
        ),
        label_eval_v2_recent_visible_min_strength=float(
            os.getenv("LABEL_EVAL_V2_RECENT_VISIBLE_MIN_STRENGTH", "0.55")
        ),
        label_eval_v2_no_patient_strong_threshold=float(
            os.getenv("LABEL_EVAL_V2_NO_PATIENT_STRONG_THRESHOLD", "0.75")
        ),
        label_eval_v2_room_floor_min=float(os.getenv("LABEL_EVAL_V2_ROOM_FLOOR_MIN", "0.10")),
        label_eval_v2_room_floor_max=float(os.getenv("LABEL_EVAL_V2_ROOM_FLOOR_MAX", "0.25")),
        prefall_panel_anchor_weight=float(os.getenv("PREFALL_PANEL_ANCHOR_WEIGHT", "0.45")),
        prefall_panel_room_bonus=float(os.getenv("PREFALL_PANEL_ROOM_BONUS", "0.28")),
        prefall_panel_overlap_no_patient_cap=float(
            os.getenv("PREFALL_PANEL_OVERLAP_NO_PATIENT_CAP", "0.20")
        ),
        prefall_panel_safety_zone_threshold=float(
            os.getenv("PREFALL_PANEL_SAFETY_ZONE_THRESHOLD", "0.50")
        ),
        prefall_panel_v3_visible_no_patient_cap=float(
            os.getenv("PREFALL_PANEL_V3_VISIBLE_NO_PATIENT_CAP", "0.35")
        ),
        prefall_panel_v3_transient_no_patient_cap=float(
            os.getenv("PREFALL_PANEL_V3_TRANSIENT_NO_PATIENT_CAP", "0.50")
        ),
        prefall_panel_v3_present_sparse_no_patient_cap=float(
            os.getenv("PREFALL_PANEL_V3_PRESENT_SPARSE_NO_PATIENT_CAP", "0.45")
        ),
        prefall_panel_v3_out_of_room_no_patient_floor=float(
            os.getenv("PREFALL_PANEL_V3_OUT_OF_ROOM_NO_PATIENT_FLOOR", "0.55")
        ),
        prefall_panel_v3_presence_min_support=float(
            os.getenv("PREFALL_PANEL_V3_PRESENCE_MIN_SUPPORT", "0.35")
        ),
        prefall_panel_v3_departure_min_confidence=float(
            os.getenv("PREFALL_PANEL_V3_DEPARTURE_MIN_CONFIDENCE", "0.35")
        ),
        prefall_panel_v3_bed_in_bed_min_score=float(
            os.getenv("PREFALL_PANEL_V3_BED_IN_BED_MIN_SCORE", "0.55")
        ),
        prefall_panel_v3_bed_in_bed_invisible_min_score=float(
            os.getenv("PREFALL_PANEL_V3_BED_IN_BED_INVISIBLE_MIN_SCORE", "0.95")
        ),
        prefall_panel_v3_motion_presence_scale=float(
            os.getenv("PREFALL_PANEL_V3_MOTION_PRESENCE_SCALE", "0.15")
        ),
        prefall_panel_v3_room_object_mean_min=float(
            os.getenv("PREFALL_PANEL_V3_ROOM_OBJECT_MEAN_MIN", "0.50")
        ),
        prefall_panel_v3_sparse_corroboration_min=float(
            os.getenv("PREFALL_PANEL_V3_SPARSE_CORROBORATION_MIN", "0.60")
        ),
        prefall_panel_v3_posture_min_frames=int(
            os.getenv("PREFALL_PANEL_V3_POSTURE_MIN_FRAMES", "6")
        ),
        prefall_panel_v3_posture_max_switch_count=int(
            os.getenv("PREFALL_PANEL_V3_POSTURE_MAX_SWITCH_COUNT", "2")
        ),
        prefall_panel_v3_posture_bed_min_lying_prob=float(
            os.getenv("PREFALL_PANEL_V3_POSTURE_BED_MIN_LYING_PROB", "0.60")
        ),
        prefall_panel_v3_posture_max_chair_bed_gap=float(
            os.getenv("PREFALL_PANEL_V3_POSTURE_MAX_CHAIR_BED_GAP", "0.20")
        ),
        prefall_panel_v3_velocity_tiebreak_margin=float(
            os.getenv("PREFALL_PANEL_V3_VELOCITY_TIEBREAK_MARGIN", "0.10")
        ),
        prefall_panel_v3_velocity_min_approach=float(
            os.getenv("PREFALL_PANEL_V3_VELOCITY_MIN_APPROACH", "0.05")
        ),
        label_eval_v2_target_macro_f1=float(os.getenv("LABEL_EVAL_V2_TARGET_MACRO_F1", "0.55")),
        label_eval_v2_target_chair_f1=float(os.getenv("LABEL_EVAL_V2_TARGET_CHAIR_F1", "0.55")),
        label_eval_v2_target_room_f1=float(os.getenv("LABEL_EVAL_V2_TARGET_ROOM_F1", "0.45")),
        label_eval_v2_target_bed_f1_drop=float(
            os.getenv("LABEL_EVAL_V2_TARGET_BED_F1_DROP", "0.05")
        ),
        label_eval_shadow_cv_folds=int(os.getenv("LABEL_EVAL_SHADOW_CV_FOLDS", "5")),
        label_eval_shadow_cv_repeats=int(os.getenv("LABEL_EVAL_SHADOW_CV_REPEATS", "3")),
        label_eval_shadow_alpha=float(os.getenv("LABEL_EVAL_SHADOW_ALPHA", "0.15")),
        fall_window_pre_anchor_seconds=int(os.getenv("FALL_WINDOW_PRE_ANCHOR_SECONDS", "300")),
        fall_window_focus_seconds=int(os.getenv("FALL_WINDOW_FOCUS_SECONDS", "30")),
        fall_window_dropout_seconds=int(os.getenv("FALL_WINDOW_DROPOUT_SECONDS", "180")),
        fall_window_post_anchor_seconds=int(os.getenv("FALL_WINDOW_POST_ANCHOR_SECONDS", "180")),
        fall_case_crossover_anchor_days=_read_int_tuple("FALL_CASE_CROSSOVER_ANCHOR_DAYS", (-1, -2)),
        case_crossover_source_status=_read_status_alias(
            "CASE_CROSSOVER_SOURCE_STATUS",
            None,
            "ready",
        ),
        case_crossover_source_ready=_read_bool_alias("CASE_CROSSOVER_SOURCE_READY", None, True),
        negative_control_source_status=_read_status_alias(
            "NEGATIVE_CONTROL_SOURCE_STATUS",
            "NONFALL_CONTROL_SOURCE_STATUS",
            "ready",
        ),
        negative_control_source_ready=_read_bool_alias(
            "NEGATIVE_CONTROL_SOURCE_READY",
            "NONFALL_CONTROL_SOURCE_READY",
            True,
        ),
        negative_control_expected_start_date=_read_date(
            "NEGATIVE_CONTROL_EXPECTED_START_DATE",
            date(2024, 1, 1),
        ),
        negative_control_match_limit=int(os.getenv("NEGATIVE_CONTROL_MATCH_LIMIT", "3")),
        label_eval_shadow_window_seconds=_read_int_tuple("LABEL_EVAL_SHADOW_WINDOW_SECONDS", (60, 300)),
        label_eval_min_sample_presence_ratio=float(
            os.getenv("LABEL_EVAL_MIN_SAMPLE_PRESENCE_RATIO", "0.95")
        ),
        label_eval_min_frames_per_window=int(os.getenv("LABEL_EVAL_MIN_FRAMES_PER_WINDOW", "30")),
        label_eval_inter_frame_gap_seconds=int(os.getenv("LABEL_EVAL_INTER_FRAME_GAP_SECONDS", "3")),
        label_eval_onset_baseline_seconds=int(os.getenv("LABEL_EVAL_ONSET_BASELINE_SECONDS", "30")),
        label_eval_onset_distance_std_floor_meters=float(
            os.getenv("LABEL_EVAL_ONSET_DISTANCE_STD_FLOOR_METERS", "0.3")
        ),
        label_eval_onset_distance_zscore=float(os.getenv("LABEL_EVAL_ONSET_DISTANCE_ZSCORE", "2.0")),
        label_eval_onset_nudge_threshold=float(os.getenv("LABEL_EVAL_ONSET_NUDGE_THRESHOLD", "0.5")),
        label_eval_posture_transition_min_score=float(
            os.getenv("LABEL_EVAL_POSTURE_TRANSITION_MIN_SCORE", "0.5")
        ),
        label_eval_posture_transition_min_run_length=int(
            os.getenv("LABEL_EVAL_POSTURE_TRANSITION_MIN_RUN_LENGTH", "2")
        ),
        label_eval_onset_min_prevalence_ratio=float(
            os.getenv("LABEL_EVAL_ONSET_MIN_PREVALENCE_RATIO", "0.10")
        ),
        label_eval_shadow_feature_cap=int(os.getenv("LABEL_EVAL_SHADOW_FEATURE_CAP", "100")),
        confidence_missingness_cutpoints=_read_int_tuple(
            "CONFIDENCE_MISSINGNESS_CUTPOINTS",
            (0, 10, 20, 30),
        ),
    )
