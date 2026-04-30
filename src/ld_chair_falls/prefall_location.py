from __future__ import annotations

from typing import Any

import pandas as pd

LOCATION_CLASSES = ("chair", "bed", "room", "no_patient")
VISIBLE_LOCATION_CLASSES = ("chair", "bed", "room")
POSTURE_CLASSES = ("sitting", "standing", "lying")

PREDICTION_REASON_DIRECT_FOCUS_ARGMAX = "direct_focus_argmax"
PREDICTION_REASON_OVERLAP_REALLOCATION = "overlap_present_reallocation"
PREDICTION_REASON_RECENT_VISIBLE_FALLBACK = "recent_visible_fallback"
PREDICTION_REASON_STRONG_NO_PATIENT = "strong_no_patient"
PREDICTION_REASON_ZERO_SIGNAL_DEFAULT = "zero_signal_default"
PREDICTION_REASON_ROOM_RECOVERY_V2 = "room_recovery_v2"
PREDICTION_REASON_SPARSE_PANEL_SQL_FALLBACK = "sparse_panel_sql_fallback"
PREDICTION_REASON_POSTURE_STATE_COLLAPSE_V3 = "posture_state_collapse_v3"
PREDICTION_REASON_SPARSE_SIGNAL_CARRY_FORWARD_V3 = "sparse_signal_carry_forward_v3"
PREDICTION_REASON_ROOM_DEPARTURE_RECOVERY_V3 = "room_departure_recovery_v3"
PREDICTION_REASON_VELOCITY_TIEBREAK_V3 = "velocity_tiebreak_v3"
PREDICTION_REASON_BED_IN_BED_RECOVERY_V3 = "bed_in_bed_recovery_v3"

ANCHOR_START_SECONDS = -180
ANCHOR_END_SECONDS = -61
TAIL_START_SECONDS = -60
TAIL_END_SECONDS = -1


def _safe_float(value: Any) -> float:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted):
        return 0.0
    return float(converted)


def _safe_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _series_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.mean())


def _series_max(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.max())


def _scaled_support(value: Any, scale: float = 1.0) -> float:
    if scale <= 0.0:
        return 0.0
    return min(1.0, max(0.0, _safe_float(value) / float(scale)))


def _normalize_probability_dict(probs: dict[str, float]) -> dict[str, float]:
    clean = {
        label: max(0.0, _safe_float(probs.get(label, 0.0)))
        for label in LOCATION_CLASSES
    }
    total = float(sum(clean.values()))
    if total <= 0.0:
        return {
            "chair": 0.0,
            "bed": 0.0,
            "room": 0.0,
            "no_patient": 1.0,
        }
    return {label: clean[label] / total for label in LOCATION_CLASSES}


def _normalize_visible_probs(probs: dict[str, float]) -> dict[str, float]:
    total = float(sum(max(0.0, _safe_float(probs.get(label, 0.0))) for label in VISIBLE_LOCATION_CLASSES))
    if total <= 0.0:
        return {label: 0.0 for label in VISIBLE_LOCATION_CLASSES}
    return {
        label: max(0.0, _safe_float(probs.get(label, 0.0))) / total
        for label in VISIBLE_LOCATION_CLASSES
    }


def _normalize_posture_probs(probs: dict[str, float]) -> dict[str, float]:
    total = float(sum(max(0.0, _safe_float(probs.get(label, 0.0))) for label in POSTURE_CLASSES))
    if total <= 0.0:
        return {label: 0.0 for label in POSTURE_CLASSES}
    return {
        label: max(0.0, _safe_float(probs.get(label, 0.0))) / total
        for label in POSTURE_CLASSES
    }


def _normalize_nudge_scores(series: pd.Series) -> pd.Series:
    scores = pd.to_numeric(series, errors="coerce")
    max_score = pd.to_numeric(scores.max(), errors="coerce")
    if pd.isna(max_score) or float(max_score) <= 0.0:
        return scores.fillna(0.0)
    if float(max_score) > 1.0:
        divisor = 100.0 if float(max_score) <= 100.0 else float(max_score)
        scores = scores / divisor
    return scores.fillna(0.0).clip(lower=0.0, upper=1.0)


def _normalize_posture_frame_scores(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel.copy()
    frame = panel.copy()
    for label in POSTURE_CLASSES:
        column = f"primary_patient_posture_score_{label}"
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce").clip(lower=0.0, upper=1.0)
    score_cols = [f"primary_patient_posture_score_{label}" for label in POSTURE_CLASSES]
    row_sum = frame[score_cols].sum(axis=1, min_count=1)
    has_score = row_sum.fillna(0.0) > 0.0
    for label in POSTURE_CLASSES:
        column = f"primary_patient_posture_score_{label}"
        frame.loc[has_score, column] = frame.loc[has_score, column].fillna(0.0) / row_sum.loc[has_score]
    return frame


def _window_posture_score_probs(panel: pd.DataFrame, half_life_seconds: int) -> dict[str, float]:
    if panel.empty:
        return {label: 0.0 for label in POSTURE_CLASSES}
    posture = _normalize_posture_frame_scores(panel)
    observed = posture.loc[posture["frame_has_posture_signal"]].copy()
    if observed.empty:
        return {label: 0.0 for label in POSTURE_CLASSES}

    if int(half_life_seconds) > 0:
        observed["recency_weight"] = 2.0 ** (
            pd.to_numeric(observed["second_offset"], errors="coerce").fillna(-999.0) / float(half_life_seconds)
        )
    else:
        observed["recency_weight"] = 1.0

    weighted: dict[str, float] = {}
    for label in POSTURE_CLASSES:
        column = f"primary_patient_posture_score_{label}"
        scores = pd.to_numeric(observed[column], errors="coerce").fillna(0.0)
        weighted[label] = float((scores * observed["recency_weight"]).sum())
    return _normalize_posture_probs(weighted)


def _window_posture_label_shares(panel: pd.DataFrame) -> dict[str, float]:
    if panel.empty:
        return {label: 0.0 for label in POSTURE_CLASSES}
    posture = (
        panel.loc[panel["frame_has_posture_signal"], "primary_patient_posture_label"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )
    posture = posture.loc[posture.isin(POSTURE_CLASSES)]
    if posture.empty:
        return {label: 0.0 for label in POSTURE_CLASSES}
    counts = posture.value_counts(normalize=True)
    return {label: float(counts.get(label, 0.0)) for label in POSTURE_CLASSES}


def _window_distance_probs(panel: pd.DataFrame, half_life_seconds: int) -> dict[str, float]:
    visible = panel.loc[panel["frame_has_location_signal"]].copy()
    if visible.empty:
        return {label: 0.0 for label in VISIBLE_LOCATION_CLASSES}

    if int(half_life_seconds) > 0:
        visible["recency_weight"] = 2.0 ** (
            pd.to_numeric(visible["second_offset"], errors="coerce").fillna(-999.0) / float(half_life_seconds)
        )
    else:
        visible["recency_weight"] = 1.0

    distance_totals: dict[str, float] = {}
    for label, column in [
        ("chair", "patient_chair_distance"),
        ("bed", "patient_bed_distance"),
        ("room", "patient_room_distance"),
    ]:
        distance = pd.to_numeric(visible[column], errors="coerce")
        inverse_distance = (
            visible["recency_weight"]
            / distance.where(distance > 1e-6, other=1e-6)
        ).where(distance.notna(), other=0.0)
        distance_totals[label] = float(inverse_distance.sum())

    distance_probs = _normalize_visible_probs(distance_totals)
    dominant_counts = (
        visible["dominant_location_label"]
        .astype("string")
        .fillna("no_patient")
        .str.strip()
        .str.lower()
        .value_counts(normalize=True)
    )
    dominant_probs = {
        label: float(dominant_counts.get(label, 0.0))
        for label in VISIBLE_LOCATION_CLASSES
    }
    dominant_probs = _normalize_visible_probs(dominant_probs)

    combined = {
        label: (0.75 * distance_probs[label]) + (0.25 * dominant_probs[label])
        for label in VISIBLE_LOCATION_CLASSES
    }
    return _normalize_visible_probs(combined)


def _window_label_shares(panel: pd.DataFrame) -> dict[str, float]:
    if panel.empty:
        return {label: 0.0 for label in LOCATION_CLASSES}
    dominant = (
        panel["dominant_location_label"]
        .astype("string")
        .fillna("no_patient")
        .str.strip()
        .str.lower()
    )
    counts = dominant.value_counts(normalize=True)
    return {label: float(counts.get(label, 0.0)) for label in LOCATION_CLASSES}


def _window_nearest_visible_shares(panel: pd.DataFrame) -> dict[str, float]:
    visible = panel.loc[panel["frame_has_location_signal"]].copy()
    if visible.empty:
        return {label: 0.0 for label in VISIBLE_LOCATION_CLASSES}

    distance_frame = pd.DataFrame(
        {
            "chair": pd.to_numeric(visible["patient_chair_distance"], errors="coerce"),
            "bed": pd.to_numeric(visible["patient_bed_distance"], errors="coerce"),
            "room": pd.to_numeric(visible["patient_room_distance"], errors="coerce"),
        },
        index=visible.index,
    )
    valid = distance_frame.notna().any(axis=1)
    if not valid.any():
        return {label: 0.0 for label in VISIBLE_LOCATION_CLASSES}

    nearest = distance_frame.loc[valid].idxmin(axis=1)
    counts = nearest.value_counts(normalize=True)
    return {label: float(counts.get(label, 0.0)) for label in VISIBLE_LOCATION_CLASSES}


def _window_room_bed_margin_mean(panel: pd.DataFrame) -> float | None:
    visible = panel.loc[panel["frame_has_location_signal"]].copy()
    if visible.empty:
        return None
    bed = pd.to_numeric(visible["patient_bed_distance"], errors="coerce")
    room = pd.to_numeric(visible["patient_room_distance"], errors="coerce")
    margin = (bed - room).dropna()
    if margin.empty:
        return None
    return float(margin.mean())


def _positive_margin_score(value: Any) -> float:
    margin = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(margin):
        return 0.0
    margin = max(0.0, float(margin))
    if margin <= 0.0:
        return 0.0
    return min(1.0, margin / (margin + 0.05))


def _blend_visible_prob_sources(*sources: tuple[float, dict[str, float]]) -> dict[str, float]:
    blended = {label: 0.0 for label in VISIBLE_LOCATION_CLASSES}
    total_weight = 0.0
    for weight, probs in sources:
        if weight <= 0.0:
            continue
        normalized = _normalize_visible_probs(probs)
        if sum(normalized.values()) <= 0.0:
            continue
        total_weight += float(weight)
        for label in VISIBLE_LOCATION_CLASSES:
            blended[label] += float(weight) * normalized[label]
    if total_weight <= 0.0:
        return {label: 0.0 for label in VISIBLE_LOCATION_CLASSES}
    return _normalize_visible_probs(blended)


def _dominant_visible_label(probs: dict[str, float], min_confidence: float = 0.0) -> str:
    if sum(max(0.0, _safe_float(probs.get(label, 0.0))) for label in probs) <= 0.0:
        return "none"
    label = _argmax_label(probs)
    if _safe_float(probs.get(label, 0.0)) < float(min_confidence):
        return "none"
    return label


def _dominant_posture_label_from_row(
    row: pd.Series,
    prefix: str,
    min_confidence: float = 0.45,
) -> str:
    probs = _normalize_posture_probs(
        {
            label: row.get(f"{prefix}_posture_prob_{label}", 0.0)
            for label in POSTURE_CLASSES
        }
    )
    if sum(probs.values()) <= 0.0:
        shares = _normalize_posture_probs(
            {
                label: row.get(f"{prefix}_posture_share_{label}", 0.0)
                for label in POSTURE_CLASSES
            }
        )
        probs = shares
    if sum(probs.values()) <= 0.0:
        return "none"
    label = max(probs, key=probs.get)
    return label if probs[label] >= float(min_confidence) else "none"


def _window_distance_velocity_means(panel: pd.DataFrame) -> dict[str, float]:
    visible = panel.loc[panel["frame_has_location_signal"]].copy()
    if len(visible.index) < 2:
        return {label: 0.0 for label in VISIBLE_LOCATION_CLASSES}
    visible = visible.sort_values("second_offset")
    second_delta = pd.to_numeric(visible["second_offset"], errors="coerce").diff()
    velocities: dict[str, float] = {}
    for label, column in [
        ("chair", "patient_chair_distance"),
        ("bed", "patient_bed_distance"),
        ("room", "patient_room_distance"),
    ]:
        distance = pd.to_numeric(visible[column], errors="coerce")
        velocity = (distance.diff() / second_delta).replace([float("inf"), float("-inf")], pd.NA).dropna()
        velocities[label] = float(velocity.mean()) if not velocity.empty else 0.0
    return velocities


def _argmax_label(probabilities: dict[str, float]) -> str:
    return max(probabilities, key=lambda label: (_safe_float(probabilities.get(label, 0.0)), label))


def _posture_adjusted_visible_probs(
    visible_probs: dict[str, float],
    posture_probs: dict[str, float],
) -> dict[str, float]:
    room_visible = max(0.0, _safe_float(visible_probs.get("room", 0.0)))
    chair_visible = max(0.0, _safe_float(visible_probs.get("chair", 0.0)))
    bed_visible = max(0.0, _safe_float(visible_probs.get("bed", 0.0)))
    non_room_visible = chair_visible + bed_visible
    if non_room_visible <= 0.0:
        return _normalize_visible_probs(
            {
                "chair": chair_visible,
                "bed": bed_visible,
                "room": room_visible,
            }
        )

    lying_bias = max(
        0.0,
        posture_probs["lying"] - max(posture_probs["sitting"], posture_probs["standing"]),
    )
    if lying_bias <= 0.0:
        return _normalize_visible_probs(
            {
                "chair": chair_visible,
                "bed": bed_visible,
                "room": room_visible,
            }
        )

    chair_signal = chair_visible
    bed_signal = bed_visible + (lying_bias * non_room_visible)
    chair_share = chair_signal / max(1e-6, chair_signal + bed_signal)
    bed_share = bed_signal / max(1e-6, chair_signal + bed_signal)

    # Posture is only a high-precision chair-vs-bed cue in v3. It must preserve
    # room evidence and never create room/no-patient moves by itself.
    return _normalize_visible_probs(
        {
            "chair": non_room_visible * chair_share,
            "bed": non_room_visible * bed_share,
            "room": room_visible,
        }
    )


def _event_window_visible_prior(row: pd.Series, legacy_visible: dict[str, float]) -> dict[str, float]:
    focus_visible = _normalize_visible_probs(
        {
            "chair": row.get("pre_focus_weight_chair", 0.0),
            "bed": row.get("pre_focus_weight_bed", 0.0),
            "room": row.get("pre_focus_weight_room", 0.0),
        }
    )
    conditional_visible = _normalize_visible_probs(
        {
            "chair": row.get("pre_cond_prob_chair", 0.0),
            "bed": row.get("pre_cond_prob_bed", 0.0),
            "room": row.get("pre_cond_prob_room", 0.0),
        }
    )
    return _blend_visible_prob_sources(
        (0.45, legacy_visible),
        (0.35, focus_visible),
        (0.20, conditional_visible),
    )


def _infer_signal_regime(
    settings: Any,
    row: pd.Series,
    legacy_visible: dict[str, float],
    event_window_visible: dict[str, float],
    presence_support: float,
) -> tuple[str, float, float, float, float]:
    distance_signal = min(1.0, _safe_float(row.get("pre_distance_signal_count")) / 20.0)
    visible_support = max(
        max(legacy_visible.values()),
        max(event_window_visible.values()),
        _safe_float(row.get("pre_state_visible_prob")),
        _safe_float(row.get("pre_dropout_visibility_ratio")),
        distance_signal,
    )
    transient_support = max(
        _safe_float(row.get("pre_state_not_visible_prob")),
        1.0 - _safe_float(row.get("pre_dropout_visibility_ratio")),
        _safe_float(row.get("pre_prob_not_visible")),
    )
    out_of_room_support = max(
        _safe_float(row.get("pre_state_out_of_room_prob")),
        _safe_float(row.get("pre_out_of_room_share")),
        _safe_float(row.get("pre_prob_out_of_room")),
    )
    if visible_support >= max(transient_support, out_of_room_support) and visible_support >= 0.35:
        return "visible", visible_support, transient_support, out_of_room_support, presence_support
    if out_of_room_support >= max(visible_support, transient_support) and out_of_room_support >= 0.45:
        return "out_of_room", visible_support, transient_support, out_of_room_support, presence_support
    if (
        transient_support >= max(visible_support, out_of_room_support)
        and transient_support >= 0.35
        and presence_support >= float(settings.prefall_panel_v3_presence_min_support)
    ):
        return "present_but_sparse", visible_support, transient_support, out_of_room_support, presence_support
    if transient_support >= max(visible_support, out_of_room_support) and transient_support >= 0.35:
        return "temporarily_not_visible", visible_support, transient_support, out_of_room_support, presence_support
    return "unknown", visible_support, transient_support, out_of_room_support, presence_support


def _set_no_patient_probability(
    base_probs: dict[str, float],
    target_no_patient: float,
    visible_prior: dict[str, float],
) -> dict[str, float]:
    probs = _normalize_probability_dict(base_probs)
    target = min(0.95, max(0.0, float(target_no_patient)))
    current = probs["no_patient"]
    if abs(target - current) <= 1e-6:
        return probs

    updated = probs.copy()
    if target < current:
        freed_mass = current - target
        updated["no_patient"] = target
        visible_mix = _blend_visible_prob_sources((0.60, _normalize_visible_probs(updated)), (0.40, visible_prior))
        for label in VISIBLE_LOCATION_CLASSES:
            updated[label] += freed_mass * visible_mix[label]
        return _normalize_probability_dict(updated)

    needed_mass = target - current
    updated["no_patient"] = target
    donor_total = sum(updated[label] for label in VISIBLE_LOCATION_CLASSES)
    if donor_total <= 0.0:
        return _normalize_probability_dict(updated)
    for label in VISIBLE_LOCATION_CLASSES:
        donor_share = updated[label] / donor_total
        updated[label] = max(0.0, updated[label] - (needed_mass * donor_share))
    return _normalize_probability_dict(updated)


def _shift_probability_to_label(
    base_probs: dict[str, float],
    target_label: str,
    target_probability: float,
) -> dict[str, float]:
    probs = _normalize_probability_dict(base_probs)
    target = max(0.0, min(0.95, float(target_probability)))
    if probs.get(target_label, 0.0) >= target:
        return probs

    updated = probs.copy()
    remaining = target - updated[target_label]
    for donor in sorted(
        [label for label in LOCATION_CLASSES if label != target_label],
        key=lambda label: updated[label],
        reverse=True,
    ):
        donor_cap = 0.55 if donor == "no_patient" else 0.45
        transferable = min(remaining, updated[donor] * donor_cap)
        if transferable <= 0.0:
            continue
        updated[donor] -= transferable
        updated[target_label] += transferable
        remaining -= transferable
        if remaining <= 1e-6:
            break
    return _normalize_probability_dict(updated)


def _dynamic_anchor_weight(
    settings: Any,
    row: pd.Series,
    anchor_probs: dict[str, float],
    tail_probs: dict[str, float],
    anchor_frames: int,
    tail_frames: int,
    last_gap: Any,
) -> float:
    if tail_frames <= 0:
        return 1.0
    if anchor_frames <= 0:
        return 0.0

    weight = float(settings.prefall_panel_anchor_weight)
    anchor_conf = max(anchor_probs.values()) * min(1.0, anchor_frames / 4.0)
    tail_conf = max(tail_probs.values()) * min(1.0, tail_frames / 4.0)
    if tail_conf > anchor_conf + 0.15:
        weight -= 0.15
    elif anchor_conf > tail_conf + 0.15:
        weight += 0.15

    if pd.notna(last_gap):
        gap_ratio = min(
            1.0,
            max(0.0, float(last_gap) / max(1.0, float(settings.label_eval_v2_recent_visible_gap_seconds))),
        )
        weight += 0.15 * gap_ratio

    anchor_label = _dominant_visible_label(anchor_probs, min_confidence=0.40)
    tail_label = _dominant_visible_label(tail_probs, min_confidence=0.40)
    if anchor_label != "none" and tail_label != "none" and anchor_label != tail_label:
        if tail_frames < 3:
            weight += 0.10
        else:
            weight -= 0.05
    return min(0.80, max(0.20, weight))


def _room_departure_confidence(row: pd.Series) -> float:
    confidence = 0.0
    if _safe_bool(row.get("panel_departure_detected", False)):
        confidence += 0.30
    if (
        str(row.get("panel_anchor_label", "none")).strip().lower() in {"chair", "bed"}
        and str(row.get("panel_tail_label", "none")).strip().lower() == "room"
    ):
        confidence += 0.10
    confidence += 0.25 * min(1.0, _safe_float(row.get("panel_tail_share_room")) / 0.25)
    confidence += 0.20 * min(1.0, _safe_float(row.get("panel_tail_room_nearest_share")) / 0.35)
    confidence += 0.10 * _positive_margin_score(row.get("panel_room_bed_margin_delta"))
    confidence += 0.03 * min(1.0, _safe_float(row.get("panel_tail_overlap_max")) / 0.25)
    confidence += 0.02 * min(1.0, _safe_float(row.get("panel_tail_nudge_max")) / 0.70)
    return min(1.0, max(0.0, confidence))


def _cache_presence_support(settings: Any, row: pd.Series) -> dict[str, float]:
    motion_support = _scaled_support(
        max(
            _safe_float(row.get("panel_tail_motion_bac_max")),
            _safe_float(row.get("panel_tail_motion_bac_mean")),
        ),
        float(settings.prefall_panel_v3_motion_presence_scale),
    )
    patient_candidate_support = max(
        _scaled_support(row.get("panel_anchor_patient_candidate_count_mean")),
        _scaled_support(row.get("panel_tail_patient_candidate_count_mean")),
    )
    other_candidate_support = _scaled_support(
        row.get("panel_tail_other_candidate_count_mean"),
        float(settings.prefall_panel_v3_room_object_mean_min),
    )
    bed_candidate_support = _scaled_support(row.get("panel_tail_bed_candidate_count_mean"))
    chair_candidate_support = _scaled_support(row.get("panel_tail_chair_candidate_count_mean"))
    posture_support = _scaled_support(
        row.get("panel_tail_posture_observed_frames"),
        float(settings.prefall_panel_v3_posture_min_frames),
    )
    bed_in_bed_support = max(
        _safe_float(row.get("panel_tail_bed_in_bed_score_max")),
        _safe_float(row.get("panel_tail_bed_in_bed_score_mean")),
    )
    presence_support = max(
        patient_candidate_support,
        posture_support,
        0.75 * bed_in_bed_support,
        0.60 * motion_support,
        0.50 * max(other_candidate_support, bed_candidate_support, chair_candidate_support),
    )
    room_presence_support = max(
        _safe_float(row.get("panel_tail_share_room")),
        _safe_float(row.get("panel_tail_room_nearest_share")),
        0.60 * other_candidate_support,
        0.40 * motion_support,
    )
    bed_presence_support = max(
        _safe_float(row.get("panel_tail_prob_bed")),
        _safe_float(row.get("panel_tail_bed_nearest_share")),
        0.65 * bed_in_bed_support,
        0.50 * bed_candidate_support,
    )
    chair_presence_support = max(
        _safe_float(row.get("panel_tail_prob_chair")),
        0.50 * chair_candidate_support,
        0.35 * patient_candidate_support,
    )
    return {
        "presence": min(1.0, max(0.0, presence_support)),
        "room": min(1.0, max(0.0, room_presence_support)),
        "bed": min(1.0, max(0.0, bed_presence_support)),
        "chair": min(1.0, max(0.0, chair_presence_support)),
        "bed_in_bed": min(1.0, max(0.0, bed_in_bed_support)),
        "motion": min(1.0, max(0.0, motion_support)),
        "patient_candidate": min(1.0, max(0.0, patient_candidate_support)),
        "other_candidate": min(1.0, max(0.0, other_candidate_support)),
        "bed_candidate": min(1.0, max(0.0, bed_candidate_support)),
        "chair_candidate": min(1.0, max(0.0, chair_candidate_support)),
        "posture": min(1.0, max(0.0, posture_support)),
    }


def _apply_velocity_tiebreak(
    settings: Any,
    row: pd.Series,
    base_probs: dict[str, float],
) -> tuple[dict[str, float], bool]:
    top_visible = sorted(
        ((label, _safe_float(base_probs.get(label, 0.0))) for label in VISIBLE_LOCATION_CLASSES),
        key=lambda item: item[1],
        reverse=True,
    )
    if len(top_visible) < 2 or (top_visible[0][1] - top_visible[1][1]) > float(settings.prefall_panel_v3_velocity_tiebreak_margin):
        return base_probs, False

    approach_scores = {}
    for label in VISIBLE_LOCATION_CLASSES:
        velocity = _safe_float(row.get(f"panel_tail_{label}_distance_velocity_mean"))
        approach_scores[label] = max(0.0, -velocity)
    winner, winner_score = max(approach_scores.items(), key=lambda item: item[1])
    if winner_score < float(settings.prefall_panel_v3_velocity_min_approach):
        return base_probs, False
    if winner == "room":
        room_support = max(
            _safe_float(row.get("panel_tail_share_room")),
            _safe_float(row.get("panel_tail_room_nearest_share")),
        )
        if room_support < 0.10:
            return base_probs, False
    target_probability = min(0.70, max(top_visible[0][1] + 0.03, 0.40 + winner_score))
    updated = _shift_probability_to_label(base_probs, winner, target_probability)
    changed = _dominant_visible_label(_normalize_visible_probs(updated)) != _dominant_visible_label(
        _normalize_visible_probs(base_probs)
    )
    return updated, changed


def _panel_summary(settings: Any, second_level_panel: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "fall_event_id",
        "panel_anchor_visible_frames",
        "panel_tail_visible_frames",
        "panel_last_visible_gap_seconds",
        "panel_anchor_prob_chair",
        "panel_anchor_prob_bed",
        "panel_anchor_prob_room",
        "panel_tail_prob_chair",
        "panel_tail_prob_bed",
        "panel_tail_prob_room",
        "panel_tail_share_room",
        "panel_tail_share_no_patient",
        "panel_tail_overlap_mean",
        "panel_tail_overlap_max",
        "panel_anchor_nudge_mean",
        "panel_tail_nudge_mean",
        "panel_tail_nudge_max",
        "panel_anchor_bed_in_bed_score_mean",
        "panel_tail_bed_in_bed_score_mean",
        "panel_tail_bed_in_bed_score_max",
        "panel_anchor_motion_bac_mean",
        "panel_tail_motion_bac_mean",
        "panel_tail_motion_bac_max",
        "panel_anchor_patient_candidate_count_mean",
        "panel_tail_patient_candidate_count_mean",
        "panel_tail_staff_candidate_count_mean",
        "panel_tail_other_candidate_count_mean",
        "panel_tail_bed_candidate_count_mean",
        "panel_tail_chair_candidate_count_mean",
        "panel_anchor_room_nearest_share",
        "panel_tail_room_nearest_share",
        "panel_tail_bed_nearest_share",
        "panel_anchor_room_bed_margin_mean",
        "panel_tail_room_bed_margin_mean",
        "panel_room_bed_margin_delta",
        "panel_tail_chair_distance_velocity_mean",
        "panel_tail_bed_distance_velocity_mean",
        "panel_tail_room_distance_velocity_mean",
        "panel_anchor_posture_observed_frames",
        "panel_tail_posture_observed_frames",
        "panel_posture_observed_frames",
        "panel_posture_switch_count",
        "panel_anchor_posture_prob_sitting",
        "panel_anchor_posture_prob_standing",
        "panel_anchor_posture_prob_lying",
        "panel_tail_posture_prob_sitting",
        "panel_tail_posture_prob_standing",
        "panel_tail_posture_prob_lying",
        "panel_anchor_posture_share_sitting",
        "panel_anchor_posture_share_standing",
        "panel_anchor_posture_share_lying",
        "panel_tail_posture_share_sitting",
        "panel_tail_posture_share_standing",
        "panel_tail_posture_share_lying",
        "panel_anchor_label",
        "panel_tail_label",
        "panel_departure_detected",
    ]
    if second_level_panel.empty:
        return pd.DataFrame(columns=columns)

    required = {
        "fall_event_id",
        "second_offset",
        "frame_has_location_signal",
        "patient_chair_distance",
        "patient_bed_distance",
        "patient_room_distance",
        "dominant_location_label",
    }
    if not required.issubset(second_level_panel.columns):
        return pd.DataFrame(columns=columns)

    panel = second_level_panel.copy()
    panel["fall_event_id"] = pd.to_numeric(panel["fall_event_id"], errors="coerce").astype("Int64")
    panel["second_offset"] = pd.to_numeric(panel["second_offset"], errors="coerce")
    panel["frame_has_location_signal"] = panel["frame_has_location_signal"].fillna(False).astype(bool)
    panel = panel.loc[
        panel["fall_event_id"].notna()
        & panel["second_offset"].notna()
        & (panel["second_offset"] < 0)
        & (panel["second_offset"] >= -int(settings.fall_window_dropout_seconds))
    ].copy()
    if panel.empty:
        return pd.DataFrame(columns=columns)

    if "patient_staff_iou" not in panel.columns:
        panel["patient_staff_iou"] = 0.0
    for column in ["patient_chair_distance", "patient_bed_distance", "patient_room_distance", "patient_staff_iou"]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel["dominant_location_label"] = (
        panel["dominant_location_label"].astype("string").fillna("no_patient").str.strip().str.lower()
    )
    if "nudge_score" not in panel.columns:
        panel["nudge_score"] = 0.0
    panel["nudge_score_normalized"] = _normalize_nudge_scores(panel["nudge_score"])
    for column in ["bed_in_bed_score", "motion_bac"]:
        if column not in panel.columns:
            panel[column] = pd.NA
        panel[column] = pd.to_numeric(panel[column], errors="coerce").clip(lower=0.0)
    for column in [
        "patient_candidate_count",
        "staff_candidate_count",
        "other_candidate_count",
        "bed_candidate_count",
        "chair_candidate_count",
    ]:
        if column not in panel.columns:
            panel[column] = 0.0
        panel[column] = pd.to_numeric(panel[column], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "primary_patient_posture_label" not in panel.columns:
        panel["primary_patient_posture_label"] = pd.NA
    panel["primary_patient_posture_label"] = (
        panel["primary_patient_posture_label"].astype("string").fillna("").str.strip().str.lower()
    )
    panel = _normalize_posture_frame_scores(panel)
    panel["frame_has_posture_signal"] = (
        panel["primary_patient_posture_label"].isin(POSTURE_CLASSES)
        | panel[[f"primary_patient_posture_score_{label}" for label in POSTURE_CLASSES]].notna().any(axis=1)
    )

    rows: list[dict[str, Any]] = []
    for fall_event_id, group in panel.groupby("fall_event_id", dropna=False):
        anchor = group.loc[group["second_offset"].between(ANCHOR_START_SECONDS, ANCHOR_END_SECONDS)].copy()
        tail = group.loc[group["second_offset"].between(TAIL_START_SECONDS, TAIL_END_SECONDS)].copy()
        anchor_visible = anchor.loc[anchor["frame_has_location_signal"]].copy()
        tail_visible = tail.loc[tail["frame_has_location_signal"]].copy()
        all_visible = group.loc[group["frame_has_location_signal"]].copy()

        anchor_probs = _window_distance_probs(anchor, settings.label_eval_v2_recency_half_life_seconds)
        tail_probs = _window_distance_probs(tail, settings.label_eval_v2_recency_half_life_seconds)
        tail_shares = _window_label_shares(tail)
        anchor_nearest_shares = _window_nearest_visible_shares(anchor)
        tail_nearest_shares = _window_nearest_visible_shares(tail)
        anchor_room_bed_margin_mean = _window_room_bed_margin_mean(anchor)
        tail_room_bed_margin_mean = _window_room_bed_margin_mean(tail)
        anchor_posture_probs = _window_posture_score_probs(anchor, settings.label_eval_v2_recency_half_life_seconds)
        tail_posture_probs = _window_posture_score_probs(tail, settings.label_eval_v2_recency_half_life_seconds)
        anchor_posture_shares = _window_posture_label_shares(anchor)
        tail_posture_shares = _window_posture_label_shares(tail)
        tail_distance_velocity = _window_distance_velocity_means(tail)
        room_bed_margin_delta = None
        if anchor_room_bed_margin_mean is not None and tail_room_bed_margin_mean is not None:
            room_bed_margin_delta = tail_room_bed_margin_mean - anchor_room_bed_margin_mean
        last_gap = pd.NA
        if not all_visible.empty:
            last_gap = float(abs(float(pd.to_numeric(all_visible["second_offset"], errors="coerce").max())))
        posture_observed = group.loc[group["frame_has_posture_signal"]].copy()
        posture_switch_count = 0.0
        if not posture_observed.empty:
            prev_posture = posture_observed["primary_patient_posture_label"].shift(1)
            posture_switch_count = float(
                (
                    prev_posture.notna()
                    & posture_observed["primary_patient_posture_label"].notna()
                    & (posture_observed["primary_patient_posture_label"] != prev_posture)
                ).sum()
            )

        room_direct_support = max(
            tail_probs["room"],
            tail_shares["room"],
            _safe_float(tail_visible["patient_staff_iou"].max()),
            _safe_float(tail["nudge_score_normalized"].max()),
        )
        departure_detected = bool(room_direct_support >= 0.35 and tail_shares["room"] >= 0.10)

        rows.append(
            {
                "fall_event_id": int(fall_event_id),
                "panel_anchor_visible_frames": float(len(anchor_visible.index)),
                "panel_tail_visible_frames": float(len(tail_visible.index)),
                "panel_last_visible_gap_seconds": last_gap,
                "panel_anchor_prob_chair": round(anchor_probs["chair"], 6),
                "panel_anchor_prob_bed": round(anchor_probs["bed"], 6),
                "panel_anchor_prob_room": round(anchor_probs["room"], 6),
                "panel_tail_prob_chair": round(tail_probs["chair"], 6),
                "panel_tail_prob_bed": round(tail_probs["bed"], 6),
                "panel_tail_prob_room": round(tail_probs["room"], 6),
                "panel_tail_share_room": round(float(tail_shares["room"]), 6),
                "panel_tail_share_no_patient": round(float(tail_shares["no_patient"]), 6),
                "panel_tail_overlap_mean": round(float(pd.to_numeric(tail["patient_staff_iou"], errors="coerce").mean() or 0.0), 6),
                "panel_tail_overlap_max": round(float(pd.to_numeric(tail["patient_staff_iou"], errors="coerce").max() or 0.0), 6),
                "panel_anchor_nudge_mean": round(float(pd.to_numeric(anchor["nudge_score_normalized"], errors="coerce").mean() or 0.0), 6),
                "panel_tail_nudge_mean": round(float(pd.to_numeric(tail["nudge_score_normalized"], errors="coerce").mean() or 0.0), 6),
                "panel_tail_nudge_max": round(float(pd.to_numeric(tail["nudge_score_normalized"], errors="coerce").max() or 0.0), 6),
                "panel_anchor_bed_in_bed_score_mean": round(_series_mean(anchor["bed_in_bed_score"]), 6),
                "panel_tail_bed_in_bed_score_mean": round(_series_mean(tail["bed_in_bed_score"]), 6),
                "panel_tail_bed_in_bed_score_max": round(_series_max(tail["bed_in_bed_score"]), 6),
                "panel_anchor_motion_bac_mean": round(_series_mean(anchor["motion_bac"]), 6),
                "panel_tail_motion_bac_mean": round(_series_mean(tail["motion_bac"]), 6),
                "panel_tail_motion_bac_max": round(_series_max(tail["motion_bac"]), 6),
                "panel_anchor_patient_candidate_count_mean": round(_series_mean(anchor["patient_candidate_count"]), 6),
                "panel_tail_patient_candidate_count_mean": round(_series_mean(tail["patient_candidate_count"]), 6),
                "panel_tail_staff_candidate_count_mean": round(_series_mean(tail["staff_candidate_count"]), 6),
                "panel_tail_other_candidate_count_mean": round(_series_mean(tail["other_candidate_count"]), 6),
                "panel_tail_bed_candidate_count_mean": round(_series_mean(tail["bed_candidate_count"]), 6),
                "panel_tail_chair_candidate_count_mean": round(_series_mean(tail["chair_candidate_count"]), 6),
                "panel_anchor_room_nearest_share": round(float(anchor_nearest_shares["room"]), 6),
                "panel_tail_room_nearest_share": round(float(tail_nearest_shares["room"]), 6),
                "panel_tail_bed_nearest_share": round(float(tail_nearest_shares["bed"]), 6),
                "panel_anchor_room_bed_margin_mean": round(float(anchor_room_bed_margin_mean), 6)
                if anchor_room_bed_margin_mean is not None
                else pd.NA,
                "panel_tail_room_bed_margin_mean": round(float(tail_room_bed_margin_mean), 6)
                if tail_room_bed_margin_mean is not None
                else pd.NA,
                "panel_room_bed_margin_delta": round(float(room_bed_margin_delta), 6)
                if room_bed_margin_delta is not None
                else pd.NA,
                "panel_tail_chair_distance_velocity_mean": round(float(tail_distance_velocity["chair"]), 6),
                "panel_tail_bed_distance_velocity_mean": round(float(tail_distance_velocity["bed"]), 6),
                "panel_tail_room_distance_velocity_mean": round(float(tail_distance_velocity["room"]), 6),
                "panel_anchor_posture_observed_frames": float(int(anchor["frame_has_posture_signal"].sum())),
                "panel_tail_posture_observed_frames": float(int(tail["frame_has_posture_signal"].sum())),
                "panel_posture_observed_frames": float(int(group["frame_has_posture_signal"].sum())),
                "panel_posture_switch_count": float(posture_switch_count),
                "panel_anchor_posture_prob_sitting": round(float(anchor_posture_probs["sitting"]), 6),
                "panel_anchor_posture_prob_standing": round(float(anchor_posture_probs["standing"]), 6),
                "panel_anchor_posture_prob_lying": round(float(anchor_posture_probs["lying"]), 6),
                "panel_tail_posture_prob_sitting": round(float(tail_posture_probs["sitting"]), 6),
                "panel_tail_posture_prob_standing": round(float(tail_posture_probs["standing"]), 6),
                "panel_tail_posture_prob_lying": round(float(tail_posture_probs["lying"]), 6),
                "panel_anchor_posture_share_sitting": round(float(anchor_posture_shares["sitting"]), 6),
                "panel_anchor_posture_share_standing": round(float(anchor_posture_shares["standing"]), 6),
                "panel_anchor_posture_share_lying": round(float(anchor_posture_shares["lying"]), 6),
                "panel_tail_posture_share_sitting": round(float(tail_posture_shares["sitting"]), 6),
                "panel_tail_posture_share_standing": round(float(tail_posture_shares["standing"]), 6),
                "panel_tail_posture_share_lying": round(float(tail_posture_shares["lying"]), 6),
                "panel_anchor_label": _argmax_label(anchor_probs) if sum(anchor_probs.values()) > 0 else "no_patient",
                "panel_tail_label": _argmax_label(tail_probs) if sum(tail_probs.values()) > 0 else "no_patient",
                "panel_departure_detected": departure_detected,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def build_panel_prefall_event_windows(
    settings: Any,
    event_windows: pd.DataFrame,
    second_level_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if event_windows.empty:
        return pd.DataFrame()

    enriched = event_windows.copy()
    for source, target in [
        ("pre_prob_chair", "sql_pre_prob_chair"),
        ("pre_prob_bed", "sql_pre_prob_bed"),
        ("pre_prob_room", "sql_pre_prob_room"),
        ("pre_prob_no_patient", "sql_pre_prob_no_patient"),
        ("prefall_location_label", "sql_prefall_location_label"),
        ("prefall_location_margin", "sql_prefall_location_margin"),
    ]:
        enriched[target] = enriched.get(source)

    summary = _panel_summary(settings, second_level_panel if second_level_panel is not None else pd.DataFrame())
    if not summary.empty and "fall_event_id" in enriched.columns:
        enriched["fall_event_id"] = pd.to_numeric(enriched.get("fall_event_id"), errors="coerce").astype("Int64")
        enriched = enriched.merge(summary, on="fall_event_id", how="left")
    else:
        for column in summary.columns:
            if column != "fall_event_id":
                enriched[column] = pd.NA

    rows: list[dict[str, Any]] = []
    for _, row in enriched.iterrows():
        legacy_probs = _normalize_probability_dict(
            {
                "chair": row.get("sql_pre_prob_chair", row.get("pre_prob_chair", 0.0)),
                "bed": row.get("sql_pre_prob_bed", row.get("pre_prob_bed", 0.0)),
                "room": row.get("sql_pre_prob_room", row.get("pre_prob_room", 0.0)),
                "no_patient": row.get("sql_pre_prob_no_patient", row.get("pre_prob_no_patient", 0.0)),
            }
        )
        legacy_visible = _normalize_visible_probs(legacy_probs)
        event_window_visible = _event_window_visible_prior(row, legacy_visible)
        anchor_probs = _normalize_visible_probs(
            {
                "chair": row.get("panel_anchor_prob_chair", 0.0),
                "bed": row.get("panel_anchor_prob_bed", 0.0),
                "room": row.get("panel_anchor_prob_room", 0.0),
            }
        )
        tail_probs = _normalize_visible_probs(
            {
                "chair": row.get("panel_tail_prob_chair", 0.0),
                "bed": row.get("panel_tail_prob_bed", 0.0),
                "room": row.get("panel_tail_prob_room", 0.0),
            }
        )
        anchor_frames = int(_safe_float(row.get("panel_anchor_visible_frames")))
        tail_frames = int(_safe_float(row.get("panel_tail_visible_frames")))
        last_gap = pd.to_numeric(pd.Series([row.get("panel_last_visible_gap_seconds")]), errors="coerce").iloc[0]

        if anchor_frames <= 0 and tail_frames <= 0:
            sparse_visible_confidence = max(legacy_visible.values())
            if sparse_visible_confidence >= 0.45 and legacy_probs["no_patient"] < 0.80:
                no_patient_prob = min(0.35, max(0.10, legacy_probs["no_patient"]))
                visible_mass = 1.0 - no_patient_prob
                probs = _normalize_probability_dict(
                    {
                        "chair": visible_mass * legacy_visible["chair"],
                        "bed": visible_mass * legacy_visible["bed"],
                        "room": visible_mass * legacy_visible["room"],
                        "no_patient": no_patient_prob,
                    }
                )
                reason = PREDICTION_REASON_SPARSE_PANEL_SQL_FALLBACK
            else:
                probs = {
                    "chair": 0.0,
                    "bed": 0.0,
                    "room": 0.0,
                    "no_patient": 1.0,
                }
                reason = PREDICTION_REASON_ZERO_SIGNAL_DEFAULT
            room_transfer = 0.0
            no_patient_capped = False
        else:
            anchor_weight = float(settings.prefall_panel_anchor_weight)
            panel_visible_probs = {
                label: (anchor_weight * anchor_probs[label]) + ((1.0 - anchor_weight) * tail_probs[label])
                for label in VISIBLE_LOCATION_CLASSES
            }
            if anchor_frames <= 0:
                panel_visible_probs = tail_probs.copy()
            elif tail_frames <= 0:
                panel_visible_probs = anchor_probs.copy()

            overlap_strength = min(
                1.0,
                max(
                    _safe_float(row.get("panel_tail_overlap_max")),
                    _safe_float(row.get("panel_tail_overlap_mean")) * 2.0,
                ),
            )
            safety_threshold = float(settings.prefall_panel_safety_zone_threshold)
            tail_nudge_max = _safe_float(row.get("panel_tail_nudge_max"))
            tail_nudge_mean = _safe_float(row.get("panel_tail_nudge_mean"))
            anchor_nudge_mean = _safe_float(row.get("panel_anchor_nudge_mean"))
            safety_level = 0.0
            if tail_nudge_max > safety_threshold:
                safety_level = min(1.0, (tail_nudge_max - safety_threshold) / max(1e-6, 1.0 - safety_threshold))
            safety_rise = max(0.0, tail_nudge_mean - anchor_nudge_mean)
            safety_strength = min(1.0, max(safety_level, safety_rise))
            gap_ratio = 1.0
            if pd.notna(last_gap):
                gap_ratio = min(1.0, max(0.0, float(last_gap) / float(settings.fall_window_dropout_seconds)))
            tail_no_patient_share = _safe_float(row.get("panel_tail_share_no_patient"))
            tail_room_share = _safe_float(row.get("panel_tail_share_room"))
            room_signal = max(tail_probs["room"], tail_room_share)
            room_support_gate = max(tail_room_share, overlap_strength, safety_strength)
            legacy_visible_weight = 0.35 if room_support_gate >= 0.15 else 0.15
            if (
                room_support_gate < 0.15
                and legacy_probs["room"] >= 0.70
                and anchor_probs["chair"] >= 0.60
            ):
                legacy_visible_weight = max(legacy_visible_weight, 0.30)
            visible_probs = _normalize_visible_probs(
                {
                    label: (legacy_visible_weight * legacy_visible[label])
                    + ((1.0 - legacy_visible_weight) * panel_visible_probs[label])
                    for label in VISIBLE_LOCATION_CLASSES
                }
            )
            base_room_signal = min(
                1.0,
                (0.60 * room_signal) + (0.25 * overlap_strength) + (0.15 * safety_strength),
            )
            if room_support_gate < 0.15:
                base_room_signal = min(base_room_signal, 0.34)
            room_distance_support = max(
                _safe_float(row.get("panel_tail_room_nearest_share")),
                _positive_margin_score(row.get("panel_tail_room_bed_margin_mean")),
                _positive_margin_score(row.get("panel_room_bed_margin_delta")),
            )
            if (
                legacy_probs["chair"] >= 0.50
                and tail_room_share < 0.05
                and room_distance_support < 0.25
            ):
                legacy_visible_weight = max(legacy_visible_weight, 0.55)
                visible_probs = _normalize_visible_probs(
                    {
                        label: (legacy_visible_weight * legacy_visible[label])
                        + ((1.0 - legacy_visible_weight) * panel_visible_probs[label])
                        for label in VISIBLE_LOCATION_CLASSES
                    }
                )
            tail_bed_nearest_share = _safe_float(row.get("panel_tail_bed_nearest_share"))
            bed_lock = (
                tail_probs["bed"] >= 0.60
                and tail_bed_nearest_share >= 0.60
                and room_support_gate < 0.15
                and room_distance_support < 0.55
            )
            distance_room_signal = 0.0
            if room_support_gate >= 0.15 and not bed_lock:
                distance_room_signal = min(
                    1.0,
                    (0.65 * room_distance_support)
                    + (0.20 * room_signal)
                    + (0.10 * overlap_strength)
                    + (0.05 * safety_strength),
                )
            direct_room_signal = max(base_room_signal, distance_room_signal)
            if (
                legacy_probs["chair"] >= 0.50
                and tail_room_share < 0.05
                and room_distance_support < 0.25
            ):
                direct_room_signal = min(direct_room_signal, 0.34)

            no_patient_prob = max(
                min(0.85, legacy_probs["no_patient"]),
                (0.55 * tail_no_patient_share) + (0.45 * gap_ratio),
            )
            no_patient_prob = min(0.85, max(0.05, no_patient_prob))
            recent_visible_override = False
            recent_visible_strength = max(tail_probs.values()) if tail_frames > 0 else 0.0
            if (
                tail_frames > 0
                and recent_visible_strength >= 0.65
                and (pd.isna(last_gap) or float(last_gap) <= float(settings.label_eval_v2_recent_visible_gap_seconds))
            ):
                no_patient_prob = min(no_patient_prob, 0.25)
                recent_visible_override = True
            no_patient_capped = False
            if overlap_strength >= 0.05 and max(visible_probs.values()) >= 0.45:
                capped_value = float(settings.prefall_panel_overlap_no_patient_cap)
                if no_patient_prob > capped_value:
                    no_patient_prob = capped_value
                    no_patient_capped = True

            visible_mass = max(0.15, 1.0 - no_patient_prob)
            probs = {
                "chair": visible_mass * visible_probs["chair"],
                "bed": visible_mass * visible_probs["bed"],
                "room": visible_mass * visible_probs["room"],
                "no_patient": no_patient_prob,
            }

            room_transfer = 0.0
            if direct_room_signal >= 0.35 and max(room_signal, room_distance_support) >= 0.15 and not bed_lock:
                room_transfer_scale = float(settings.prefall_panel_room_bonus) * direct_room_signal
                if (
                    str(row.get("sql_prefall_location_label", "")).strip().lower() == "room"
                    and safety_strength >= 0.50
                    and room_signal >= 0.25
                ):
                    room_transfer_scale *= 1.25
                donor = max(("chair", "bed", "no_patient"), key=lambda label: probs[label])
                donor_cap = 0.40 if donor == "no_patient" else 0.35
                room_transfer = min(
                    room_transfer_scale,
                    probs[donor] * donor_cap,
                )
                if room_transfer > 0.0:
                    probs[donor] -= room_transfer
                    probs["room"] += room_transfer

            probs = _normalize_probability_dict(probs)
            if room_transfer > 0.0:
                reason = PREDICTION_REASON_ROOM_RECOVERY_V2
            elif recent_visible_override:
                reason = PREDICTION_REASON_RECENT_VISIBLE_FALLBACK
            elif no_patient_capped:
                reason = PREDICTION_REASON_OVERLAP_REALLOCATION
            elif probs["no_patient"] >= float(settings.label_eval_v2_no_patient_strong_threshold):
                reason = PREDICTION_REASON_STRONG_NO_PATIENT
            else:
                reason = PREDICTION_REASON_DIRECT_FOCUS_ARGMAX

        v2_probs = _normalize_probability_dict(probs)
        ordered = sorted(v2_probs.items(), key=lambda item: item[1], reverse=True)
        label = ordered[0][0]
        margin = float(ordered[0][1] - ordered[1][1]) if len(ordered) > 1 else float(ordered[0][1])

        dynamic_anchor_weight = _dynamic_anchor_weight(
            settings,
            row,
            anchor_probs,
            tail_probs,
            anchor_frames,
            tail_frames,
            last_gap,
        )
        tail_weight = 1.0 - dynamic_anchor_weight
        anchor_posture_probs = _normalize_posture_probs(
            {
                "sitting": row.get("panel_anchor_posture_prob_sitting", 0.0),
                "standing": row.get("panel_anchor_posture_prob_standing", 0.0),
                "lying": row.get("panel_anchor_posture_prob_lying", 0.0),
            }
        )
        tail_posture_probs = _normalize_posture_probs(
            {
                "sitting": row.get("panel_tail_posture_prob_sitting", 0.0),
                "standing": row.get("panel_tail_posture_prob_standing", 0.0),
                "lying": row.get("panel_tail_posture_prob_lying", 0.0),
            }
        )
        anchor_posture_shares = _normalize_posture_probs(
            {
                "sitting": row.get("panel_anchor_posture_share_sitting", 0.0),
                "standing": row.get("panel_anchor_posture_share_standing", 0.0),
                "lying": row.get("panel_anchor_posture_share_lying", 0.0),
            }
        )
        tail_posture_shares = _normalize_posture_probs(
            {
                "sitting": row.get("panel_tail_posture_share_sitting", 0.0),
                "standing": row.get("panel_tail_posture_share_standing", 0.0),
                "lying": row.get("panel_tail_posture_share_lying", 0.0),
            }
        )
        posture_probs = _normalize_posture_probs(
            {
                label_name: (
                    0.75
                    * (
                        (dynamic_anchor_weight * anchor_posture_probs[label_name])
                        + (tail_weight * tail_posture_probs[label_name])
                    )
                )
                + (
                    0.25
                    * (
                        (dynamic_anchor_weight * anchor_posture_shares[label_name])
                        + (tail_weight * tail_posture_shares[label_name])
                    )
                )
                for label_name in POSTURE_CLASSES
            }
        )
        posture_observed_frames = max(
            int(_safe_float(row.get("panel_posture_observed_frames"))),
            int(_safe_float(row.get("pre_posture_observed_frame_count"))),
        )
        posture_switch_count = float(_safe_float(row.get("panel_posture_switch_count")))
        anchor_posture_label = _dominant_posture_label_from_row(row, "panel_anchor")
        tail_posture_label = _dominant_posture_label_from_row(row, "panel_tail")
        dynamic_panel_visible = _blend_visible_prob_sources(
            (dynamic_anchor_weight, anchor_probs),
            (tail_weight, tail_probs),
        )
        presence_supports = _cache_presence_support(settings, row)
        signal_regime, visible_support, transient_support, out_of_room_support, presence_support = _infer_signal_regime(
            settings,
            row,
            legacy_visible,
            event_window_visible,
            presence_supports["presence"],
        )
        room_presence_support = presence_supports["room"]
        bed_presence_support = presence_supports["bed"]
        chair_presence_support = presence_supports["chair"]
        bed_in_bed_support = presence_supports["bed_in_bed"]
        motion_support = presence_supports["motion"]
        patient_candidate_support = presence_supports["patient_candidate"]
        bed_candidate_support = presence_supports["bed_candidate"]
        chair_candidate_support = presence_supports["chair_candidate"]
        posture_support = presence_supports["posture"]
        departure_confidence = _room_departure_confidence(row)

        visible_corroboration = max(
            _scaled_support(tail_frames, 2.0),
            _scaled_support(row.get("pre_distance_signal_count"), 2.0),
        )
        tail_patient_corroboration = _scaled_support(row.get("panel_tail_patient_candidate_count_mean"))
        tail_motion_corroboration = _scaled_support(
            max(
                _safe_float(row.get("panel_tail_motion_bac_max")),
                _safe_float(row.get("panel_tail_motion_bac_mean")),
            ),
            float(settings.prefall_panel_v3_motion_presence_scale),
        )
        tail_posture_corroboration = _scaled_support(
            row.get("panel_tail_posture_observed_frames"),
            float(settings.prefall_panel_v3_posture_min_frames),
        )
        sparse_corroboration = max(
            visible_corroboration,
            tail_patient_corroboration,
            tail_posture_corroboration,
            0.75 * tail_motion_corroboration,
        )
        room_sparse_corroboration = max(
            visible_corroboration,
            tail_patient_corroboration,
            0.75 * tail_motion_corroboration,
        )
        invisible_bed_rescue_corroboration = max(
            patient_candidate_support,
            posture_support,
            0.75 * motion_support,
            0.75 * chair_candidate_support,
        )
        sparse_corroboration_min = float(settings.prefall_panel_v3_sparse_corroboration_min)
        sparse_corroborated = sparse_corroboration >= sparse_corroboration_min
        room_sparse_corroborated = room_sparse_corroboration >= sparse_corroboration_min
        has_visible_tail_support = bool(anchor_frames > 0 or tail_frames > 0)
        invisible_bed_rescue_allowed = has_visible_tail_support or (
            bed_in_bed_support >= float(settings.prefall_panel_v3_bed_in_bed_invisible_min_score)
            and invisible_bed_rescue_corroboration >= sparse_corroboration_min
        )

        v3_visible_probs = _blend_visible_prob_sources(
            (0.70, _normalize_visible_probs(v2_probs)),
            (0.20, dynamic_panel_visible),
            (0.10, event_window_visible),
        )
        visible_mass = max(0.0, 1.0 - v2_probs["no_patient"])
        v3_probs = _normalize_probability_dict(
            {
                "chair": visible_mass * v3_visible_probs["chair"],
                "bed": visible_mass * v3_visible_probs["bed"],
                "room": visible_mass * v3_visible_probs["room"],
                "no_patient": v2_probs["no_patient"],
            }
        )

        v3_reason = reason
        v3_reason_primary = reason
        v3_base_recovery_reason = ""
        v3_carry_forward_used = False
        v3_room_override_used = False
        v3_velocity_used = False
        v3_posture_applied = False
        v3_posture_mode = "none"
        v3_posture_gate_reason = "not_evaluated"
        posture_blend = 0.0

        preserve_v2_room = label == "room"
        if preserve_v2_room:
            v3_probs = v2_probs.copy()
            v3_visible_probs = _normalize_visible_probs(v3_probs)

        strong_visible_carry = visible_support >= 0.80 and max(event_window_visible.values()) >= 0.60
        if signal_regime == "visible" and sparse_corroborated:
            target_no_patient = min(
                v3_probs["no_patient"],
                float(
                    settings.prefall_panel_v3_visible_no_patient_cap
                    if strong_visible_carry
                    else settings.prefall_panel_v3_transient_no_patient_cap
                ),
            )
        elif (
            signal_regime == "temporarily_not_visible"
            and max(event_window_visible.values()) >= 0.45
            and sparse_corroborated
        ):
            target_no_patient = min(
                v3_probs["no_patient"],
                float(settings.prefall_panel_v3_transient_no_patient_cap),
            )
        elif signal_regime == "present_but_sparse" and sparse_corroborated:
            target_no_patient = min(
                v3_probs["no_patient"],
                float(settings.prefall_panel_v3_present_sparse_no_patient_cap),
            )
        elif signal_regime == "out_of_room":
            target_no_patient = max(
                v3_probs["no_patient"],
                float(settings.prefall_panel_v3_out_of_room_no_patient_floor),
            )
        else:
            target_no_patient = v3_probs["no_patient"]

        if abs(target_no_patient - v3_probs["no_patient"]) > 0.02:
            v3_probs = _set_no_patient_probability(
                v3_probs,
                target_no_patient,
                _blend_visible_prob_sources((0.60, v3_visible_probs), (0.40, event_window_visible)),
            )
            v3_visible_probs = _normalize_visible_probs(v3_probs)
            if target_no_patient < v2_probs["no_patient"] and (
                strong_visible_carry or signal_regime == "present_but_sparse"
            ):
                v3_carry_forward_used = bool(
                    signal_regime == "present_but_sparse"
                    or (strong_visible_carry and anchor_frames <= 0 and tail_frames <= 0)
                )
                v3_base_recovery_reason = PREDICTION_REASON_SPARSE_SIGNAL_CARRY_FORWARD_V3

        bed_lock_v3 = (
            (
                _safe_float(row.get("panel_tail_bed_nearest_share")) >= 0.75
                and _safe_float(row.get("panel_tail_share_room")) < 0.10
                and _safe_float(row.get("panel_tail_room_nearest_share")) < 0.15
            )
            or (
                bed_in_bed_support >= float(settings.prefall_panel_v3_bed_in_bed_min_score)
                and _safe_float(row.get("panel_tail_share_room")) < 0.20
                and _safe_float(row.get("panel_tail_room_nearest_share")) < 0.25
            )
        )
        room_nearest_share = _safe_float(row.get("panel_tail_room_nearest_share"))
        room_tail_share = _safe_float(row.get("panel_tail_share_room"))
        room_velocity_approach = _safe_float(row.get("panel_tail_room_distance_velocity_mean")) <= -float(
            settings.prefall_panel_v3_velocity_min_approach
        )
        room_override_evidence = (
            room_nearest_share >= 0.35
            or room_velocity_approach
            or (
                signal_regime == "present_but_sparse"
                and room_presence_support >= float(settings.prefall_panel_v3_presence_min_support)
                and room_sparse_corroborated
            )
            or (
                _safe_bool(row.get("panel_departure_detected", False))
                and (room_nearest_share >= 0.35 or room_tail_share >= 0.35 or room_velocity_approach)
            )
        )
        if (
            not preserve_v2_room
            and not bed_lock_v3
            and (
                (
                    departure_confidence >= float(settings.prefall_panel_v3_departure_min_confidence)
                    and room_override_evidence
                    and room_sparse_corroborated
                )
                or (
                    signal_regime == "present_but_sparse"
                    and room_presence_support >= float(settings.prefall_panel_v3_presence_min_support)
                    and room_sparse_corroborated
                )
            )
        ):
            desired_room = min(
                0.70,
                max(v3_probs["room"], 0.18 + (0.60 * max(departure_confidence, room_presence_support))),
            )
            room_shifted = _shift_probability_to_label(v3_probs, "room", desired_room)
            if room_shifted["room"] > v3_probs["room"] + 0.02:
                v3_probs = room_shifted
                v3_visible_probs = _normalize_visible_probs(v3_probs)
                v3_room_override_used = True
                v3_base_recovery_reason = PREDICTION_REASON_ROOM_DEPARTURE_RECOVERY_V3

        velocity_shifted, v3_velocity_used = _apply_velocity_tiebreak(settings, row, v3_probs)
        if v3_velocity_used:
            v3_probs = velocity_shifted
            v3_visible_probs = _normalize_visible_probs(v3_probs)
            if not v3_room_override_used and not v3_carry_forward_used:
                v3_base_recovery_reason = PREDICTION_REASON_VELOCITY_TIEBREAK_V3

        v3_bed_rescue_used = False
        if (
            not preserve_v2_room
            and not v3_room_override_used
            and bed_in_bed_support >= float(settings.prefall_panel_v3_bed_in_bed_min_score)
            and bed_presence_support >= max(room_presence_support, chair_presence_support)
            and v3_probs["room"] < 0.30
            and (
                has_visible_tail_support
                or (
                    invisible_bed_rescue_allowed
                    and bed_candidate_support >= float(settings.prefall_panel_v3_presence_min_support)
                )
            )
        ):
            desired_bed = min(0.80, max(v3_probs["bed"], 0.20 + (0.60 * bed_in_bed_support)))
            bed_shifted = _shift_probability_to_label(v3_probs, "bed", desired_bed)
            if bed_shifted["bed"] > v3_probs["bed"] + 0.02:
                v3_probs = bed_shifted
                v3_visible_probs = _normalize_visible_probs(v3_probs)
                v3_bed_rescue_used = True
                v3_base_recovery_reason = PREDICTION_REASON_BED_IN_BED_RECOVERY_V3

        lying_probability = posture_probs["lying"]
        chair_bed_gap = abs(v3_probs["chair"] - v3_probs["bed"])
        current_v3_label = _argmax_label(v3_probs)
        if current_v3_label not in {"chair", "bed"}:
            v3_posture_gate_reason = "winner_not_furniture"
        elif max(v3_probs["room"], v3_probs["no_patient"]) >= 0.35:
            v3_posture_gate_reason = "room_or_no_patient_evidence"
        elif departure_confidence >= float(settings.prefall_panel_v3_departure_min_confidence):
            v3_posture_gate_reason = "departure_evidence"
        elif posture_observed_frames < int(settings.prefall_panel_v3_posture_min_frames):
            v3_posture_gate_reason = "insufficient_posture_frames"
        elif posture_switch_count > float(settings.prefall_panel_v3_posture_max_switch_count):
            v3_posture_gate_reason = "unstable_posture"
        elif (
            anchor_posture_label != "none"
            and tail_posture_label != "none"
            and anchor_posture_label != tail_posture_label
        ):
            v3_posture_gate_reason = "anchor_tail_posture_disagree"
        elif lying_probability < float(settings.prefall_panel_v3_posture_bed_min_lying_prob):
            v3_posture_gate_reason = "lying_not_dominant"
        elif chair_bed_gap > float(settings.prefall_panel_v3_posture_max_chair_bed_gap):
            v3_posture_gate_reason = "chair_bed_gap_too_wide"
        else:
            v3_posture_gate_reason = "applied"
            v3_posture_mode = "lying_bed_bias"
            lying_dominance = max(
                0.0,
                lying_probability - max(posture_probs["sitting"], posture_probs["standing"]),
            )
            posture_support = min(
                1.0,
                posture_observed_frames / max(1.0, float(settings.prefall_panel_v3_posture_min_frames)),
            )
            posture_blend = min(1.0, max(0.0, lying_dominance / 0.25)) * posture_support
            if posture_blend > 0.0:
                posture_adjusted_visible = _posture_adjusted_visible_probs(v3_visible_probs, posture_probs)
                blended_visible = _blend_visible_prob_sources(
                    (1.0 - posture_blend, v3_visible_probs),
                    (posture_blend, posture_adjusted_visible),
                )
                visible_mass = max(0.0, 1.0 - v3_probs["no_patient"])
                posture_shifted = _normalize_probability_dict(
                    {
                        "chair": visible_mass * blended_visible["chair"],
                        "bed": visible_mass * blended_visible["bed"],
                        "room": visible_mass * blended_visible["room"],
                        "no_patient": v3_probs["no_patient"],
                    }
                )
                if posture_shifted["bed"] > v3_probs["bed"] + 0.02:
                    v3_probs = posture_shifted
                    v3_visible_probs = blended_visible
                    v3_posture_applied = True
                    v3_reason_primary = PREDICTION_REASON_POSTURE_STATE_COLLAPSE_V3

        if v3_posture_applied:
            v3_reason = PREDICTION_REASON_POSTURE_STATE_COLLAPSE_V3
        elif v3_room_override_used:
            v3_reason = PREDICTION_REASON_ROOM_DEPARTURE_RECOVERY_V3
            v3_reason_primary = v3_reason
        elif v3_bed_rescue_used:
            v3_reason = PREDICTION_REASON_BED_IN_BED_RECOVERY_V3
            v3_reason_primary = v3_reason
        elif v3_velocity_used:
            v3_reason = PREDICTION_REASON_VELOCITY_TIEBREAK_V3
            v3_reason_primary = v3_reason
        elif v3_base_recovery_reason:
            v3_reason = v3_base_recovery_reason
            v3_reason_primary = v3_reason

        v3_ordered = sorted(v3_probs.items(), key=lambda item: item[1], reverse=True)
        v3_label = v3_ordered[0][0]
        v3_margin = (
            float(v3_ordered[0][1] - v3_ordered[1][1])
            if len(v3_ordered) > 1
            else float(v3_ordered[0][1])
        )
        signal_profile = (
            f"anchor={row.get('panel_anchor_label', 'no_patient')};"
            f"tail={row.get('panel_tail_label', 'no_patient')};"
            f"gap={'' if pd.isna(last_gap) else round(float(last_gap), 1)};"
            f"anchor_weight={round(float(dynamic_anchor_weight), 3)};"
            f"room_tail={round(_safe_float(row.get('panel_tail_share_room')), 3)};"
            f"room_nearest={round(_safe_float(row.get('panel_tail_room_nearest_share')), 3)};"
            f"room_margin_delta={round(_safe_float(row.get('panel_room_bed_margin_delta')), 3)};"
            f"overlap={round(_safe_float(row.get('panel_tail_overlap_max')), 3)};"
            f"nudge={round(_safe_float(row.get('panel_tail_nudge_max')), 3)};"
            f"signal_regime={signal_regime};"
            f"visible_support={round(float(visible_support), 3)};"
            f"transient_support={round(float(transient_support), 3)};"
            f"out_of_room_support={round(float(out_of_room_support), 3)};"
            f"presence_support={round(float(presence_support), 3)};"
            f"room_presence_support={round(float(room_presence_support), 3)};"
            f"bed_in_bed={round(float(bed_in_bed_support), 3)};"
            f"departure_conf={round(float(departure_confidence), 3)};"
            f"posture={max(posture_probs, key=posture_probs.get) if posture_probs else 'none'};"
            f"posture_blend={round(float(posture_blend), 3)};"
            f"posture_gate={v3_posture_gate_reason};"
            f"posture_switch={round(float(posture_switch_count), 1)}"
        )
        rows.append(
            {
                **row.to_dict(),
                "pre_prob_v2_chair": round(v2_probs["chair"], 6),
                "pre_prob_v2_bed": round(v2_probs["bed"], 6),
                "pre_prob_v2_room": round(v2_probs["room"], 6),
                "pre_prob_v2_no_patient": round(v2_probs["no_patient"], 6),
                "prefall_location_v2_label": label,
                "prefall_location_v2_margin": round(margin, 6),
                "prefall_location_v2_reason": reason,
                "pre_prob_v3_chair": round(v3_probs["chair"], 6),
                "pre_prob_v3_bed": round(v3_probs["bed"], 6),
                "pre_prob_v3_room": round(v3_probs["room"], 6),
                "pre_prob_v3_no_patient": round(v3_probs["no_patient"], 6),
                "prefall_location_v3_label": v3_label,
                "prefall_location_v3_margin": round(v3_margin, 6),
                "prefall_location_v3_reason": v3_reason,
                "prefall_location_v3_reason_primary": v3_reason_primary,
                "prefall_location_v3_signal_regime": signal_regime,
                "prefall_location_v3_departure_confidence": round(float(departure_confidence), 6),
                "prefall_location_v3_presence_support": round(float(presence_support), 6),
                "prefall_location_v3_room_presence_support": round(float(room_presence_support), 6),
                "prefall_location_v3_bed_in_bed_support": round(float(bed_in_bed_support), 6),
                "prefall_location_v3_carry_forward_used": v3_carry_forward_used,
                "prefall_location_v3_posture_applied": v3_posture_applied,
                "prefall_location_v3_posture_mode": v3_posture_mode,
                "prefall_location_v3_posture_gate_reason": v3_posture_gate_reason,
                "prefall_location_v3_base_recovery_reason": v3_base_recovery_reason or reason,
                "prefall_location_reason": v3_reason,
                "prefall_location_signal_profile": signal_profile,
                "prefall_location_v2_anchor_label": row.get("panel_anchor_label", "no_patient"),
                "prefall_location_v2_tail_label": row.get("panel_tail_label", "no_patient"),
                "prefall_location_v2_departure_detected": _safe_bool(row.get("panel_departure_detected", False)),
                "prefall_location_v2_last_visible_gap_seconds": last_gap,
                "pre_prob_chair": round(v3_probs["chair"], 6),
                "pre_prob_bed": round(v3_probs["bed"], 6),
                "pre_prob_room": round(v3_probs["room"], 6),
                "pre_prob_no_patient": round(v3_probs["no_patient"], 6),
                "prefall_location_label": v3_label,
                "prefall_location_margin": round(v3_margin, 6),
            }
        )

    result = pd.DataFrame(rows)
    if "prefall_location_v2_departure_detected" in result.columns:
        result["prefall_location_v2_departure_detected"] = result[
            "prefall_location_v2_departure_detected"
        ].map(_safe_bool).astype(object)
    return result
