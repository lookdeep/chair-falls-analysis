"""Readable, non-production Python companion for graph-facts v3 SQL validation.

This file is a handoff reference only. It is intentionally not imported by the
chair-falls pipeline, and it is not the canonical implementation.

Canonical executable behavior lives in:
`src/ld_chair_falls/prefall_location.py::build_panel_prefall_event_windows`

The goal here is different:
- mirror the SQL handoff phases in readable Python
- accept simple dict/list inputs instead of pandas DataFrames
- make manual SQL validation and re-implementation easier

Use this file to understand or validate the logic, not to drive production.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

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

# These defaults are copied from src/ld_chair_falls/config.py so readers can
# trace current thresholds in one place while translating the logic to SQL.
DEFAULTS = {
    "label_eval_v2_recency_half_life_seconds": 45,
    "label_eval_v2_recent_visible_gap_seconds": 60,
    "label_eval_v2_no_patient_strong_threshold": 0.75,
    "prefall_panel_anchor_weight": 0.45,
    "prefall_panel_room_bonus": 0.28,
    "prefall_panel_overlap_no_patient_cap": 0.20,
    "prefall_panel_safety_zone_threshold": 0.50,
    "prefall_panel_v3_visible_no_patient_cap": 0.35,
    "prefall_panel_v3_transient_no_patient_cap": 0.50,
    "prefall_panel_v3_present_sparse_no_patient_cap": 0.45,
    "prefall_panel_v3_out_of_room_no_patient_floor": 0.55,
    "prefall_panel_v3_presence_min_support": 0.35,
    "prefall_panel_v3_departure_min_confidence": 0.35,
    "prefall_panel_v3_bed_in_bed_min_score": 0.55,
    "prefall_panel_v3_bed_in_bed_invisible_min_score": 0.95,
    "prefall_panel_v3_motion_presence_scale": 0.15,
    "prefall_panel_v3_room_object_mean_min": 0.50,
    "prefall_panel_v3_sparse_corroboration_min": 0.60,
    "prefall_panel_v3_posture_min_frames": 6,
    "prefall_panel_v3_posture_max_switch_count": 2,
    "prefall_panel_v3_posture_bed_min_lying_prob": 0.60,
    "prefall_panel_v3_posture_max_chair_bed_gap": 0.20,
    "prefall_panel_v3_velocity_tiebreak_margin": 0.10,
    "prefall_panel_v3_velocity_min_approach": 0.05,
    "fall_window_dropout_seconds": 180,
}

__all__ = ["DEFAULTS", "summarize_panel_history", "derive_prefall_location_v3"]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
        return None if result != result else result
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return None if result != result else result


def _safe_float(value: Any, default: float = 0.0) -> float:
    result = _optional_float(value)
    return default if result is None else result


def _safe_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def clamp01(value: Any) -> float:
    return min(1.0, max(0.0, _safe_float(value)))


def safe_ratio(numerator: Any, denominator: Any) -> float:
    den = _safe_float(denominator)
    if abs(den) <= 1e-12:
        return 0.0
    return _safe_float(numerator) / den


def scaled_support(value: Any, scale: float = 1.0) -> float:
    if scale <= 0.0:
        return 0.0
    return clamp01(safe_ratio(value, scale))


def positive_margin_score(margin: Any) -> float:
    value = _optional_float(margin)
    if value is None or value <= 0.0:
        return 0.0
    return clamp01(safe_ratio(value, value + 0.05))


def _normalize_probability_dict(probs: Mapping[str, Any]) -> dict[str, float]:
    clean = {label: max(0.0, _safe_float(probs.get(label, 0.0))) for label in LOCATION_CLASSES}
    total = sum(clean.values())
    if total <= 0.0:
        return {"chair": 0.0, "bed": 0.0, "room": 0.0, "no_patient": 1.0}
    return {label: clean[label] / total for label in LOCATION_CLASSES}


def _normalize_visible_probs(probs: Mapping[str, Any]) -> dict[str, float]:
    clean = {label: max(0.0, _safe_float(probs.get(label, 0.0))) for label in VISIBLE_LOCATION_CLASSES}
    total = sum(clean.values())
    if total <= 0.0:
        return {label: 0.0 for label in VISIBLE_LOCATION_CLASSES}
    return {label: clean[label] / total for label in VISIBLE_LOCATION_CLASSES}


def _normalize_posture_probs(probs: Mapping[str, Any]) -> dict[str, float]:
    clean = {label: max(0.0, _safe_float(probs.get(label, 0.0))) for label in POSTURE_CLASSES}
    total = sum(clean.values())
    if total <= 0.0:
        return {label: 0.0 for label in POSTURE_CLASSES}
    return {label: clean[label] / total for label in POSTURE_CLASSES}


def _blend_visible_prob_sources(*weighted_sources: tuple[float, Mapping[str, Any]]) -> dict[str, float]:
    blended = {label: 0.0 for label in VISIBLE_LOCATION_CLASSES}
    total_weight = 0.0
    for weight, probs in weighted_sources:
        if weight <= 0.0:
            continue
        norm = _normalize_visible_probs(probs)
        for label in VISIBLE_LOCATION_CLASSES:
            blended[label] += weight * norm[label]
        total_weight += weight
    if total_weight <= 0.0:
        return {label: 0.0 for label in VISIBLE_LOCATION_CLASSES}
    return _normalize_visible_probs(blended)


def _argmax_label(probabilities: Mapping[str, Any]) -> str:
    return max(probabilities, key=lambda label: (_safe_float(probabilities.get(label, 0.0)), label))


def _dominant_visible_label(probs: Mapping[str, Any], min_confidence: float = 0.40) -> str:
    visible = _normalize_visible_probs(probs)
    if sum(visible.values()) <= 0.0:
        return "none"
    label = max(visible, key=visible.get)
    return label if visible[label] >= min_confidence else "none"


def _dominant_posture_label(
    score_probs: Mapping[str, Any],
    share_probs: Mapping[str, Any],
    min_confidence: float = 0.45,
) -> str:
    normalized_scores = _normalize_posture_probs(score_probs)
    normalized_shares = _normalize_posture_probs(share_probs)
    chosen = normalized_scores if sum(normalized_scores.values()) > 0.0 else normalized_shares
    if sum(chosen.values()) <= 0.0:
        return "none"
    label = max(chosen, key=chosen.get)
    return label if chosen[label] >= min_confidence else "none"


def _mean(values: Iterable[float | None]) -> float:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else 0.0


def _max_value(values: Iterable[float | None]) -> float:
    clean = [value for value in values if value is not None]
    return max(clean) if clean else 0.0


def _normalize_label(value: Any, *, allow_no_patient: bool = True) -> str:
    text = str(value or "").strip().lower()
    allowed = set(LOCATION_CLASSES if allow_no_patient else VISIBLE_LOCATION_CLASSES)
    return text if text in allowed else ("no_patient" if allow_no_patient else "none")


def _normalize_posture_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in POSTURE_CLASSES else ""


def _prepare_panel_rows(second_level_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    max_nudge = 0.0
    for row in second_level_rows:
        second_offset = _optional_float(row.get("second_offset"))
        if second_offset is None:
            continue
        prepared_row = {
            "fall_event_id": row.get("fall_event_id"),
            "second_offset": second_offset,
            "frame_has_location_signal": _safe_bool(row.get("frame_has_location_signal")),
            "patient_chair_distance": _optional_float(row.get("patient_chair_distance")),
            "patient_bed_distance": _optional_float(row.get("patient_bed_distance")),
            "patient_room_distance": _optional_float(row.get("patient_room_distance")),
            "patient_staff_iou": clamp01(row.get("patient_staff_iou")),
            "dominant_location_label": _normalize_label(row.get("dominant_location_label")),
            "nudge_score_raw": _safe_float(row.get("nudge_score")),
            "bed_in_bed_score": max(0.0, _safe_float(row.get("bed_in_bed_score"))),
            "motion_bac": max(0.0, _safe_float(row.get("motion_bac"))),
            "patient_candidate_count": max(0.0, _safe_float(row.get("patient_candidate_count"))),
            "staff_candidate_count": max(0.0, _safe_float(row.get("staff_candidate_count"))),
            "other_candidate_count": max(0.0, _safe_float(row.get("other_candidate_count"))),
            "bed_candidate_count": max(0.0, _safe_float(row.get("bed_candidate_count"))),
            "chair_candidate_count": max(0.0, _safe_float(row.get("chair_candidate_count"))),
        }
        posture_label = _normalize_posture_label(row.get("primary_patient_posture_label"))
        sitting_raw = clamp01(row.get("primary_patient_posture_score_sitting"))
        standing_raw = clamp01(row.get("primary_patient_posture_score_standing"))
        lying_raw = clamp01(row.get("primary_patient_posture_score_lying"))
        posture_total = sitting_raw + standing_raw + lying_raw
        if posture_total > 0.0:
            posture_scores = {
                "sitting": sitting_raw / posture_total,
                "standing": standing_raw / posture_total,
                "lying": lying_raw / posture_total,
            }
        else:
            posture_scores = {label: 0.0 for label in POSTURE_CLASSES}
        prepared_row["primary_patient_posture_label"] = posture_label
        prepared_row["frame_has_posture_signal"] = posture_label in POSTURE_CLASSES or posture_total > 0.0
        for label in POSTURE_CLASSES:
            prepared_row[f"posture_score_{label}"] = posture_scores[label]
        prepared.append(prepared_row)
        max_nudge = max(max_nudge, prepared_row["nudge_score_raw"])

    if max_nudge <= 0.0:
        divisor = 1.0
    elif max_nudge <= 1.0:
        divisor = 1.0
    elif max_nudge <= 100.0:
        divisor = 100.0
    else:
        divisor = max_nudge

    for row in prepared:
        row["nudge_score_normalized"] = clamp01(row["nudge_score_raw"] / divisor if divisor > 0.0 else 0.0)
    return sorted(prepared, key=lambda item: item["second_offset"])


def _window_rows(rows: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    return [row for row in rows if start <= row["second_offset"] <= end]


def _window_distance_probs(rows: list[dict[str, Any]], half_life_seconds: int) -> dict[str, float]:
    visible = [row for row in rows if row["frame_has_location_signal"]]
    if not visible:
        return {label: 0.0 for label in VISIBLE_LOCATION_CLASSES}

    distance_totals = {label: 0.0 for label in VISIBLE_LOCATION_CLASSES}
    dominant_counts = {label: 0.0 for label in VISIBLE_LOCATION_CLASSES}
    for row in visible:
        recency_weight = 2.0 ** (row["second_offset"] / half_life_seconds) if half_life_seconds > 0 else 1.0
        for label, column in (
            ("chair", "patient_chair_distance"),
            ("bed", "patient_bed_distance"),
            ("room", "patient_room_distance"),
        ):
            distance = row[column]
            if distance is None:
                continue
            distance_totals[label] += recency_weight / max(distance, 1e-6)
        dominant = row["dominant_location_label"]
        if dominant in VISIBLE_LOCATION_CLASSES:
            dominant_counts[dominant] += 1.0

    distance_probs = _normalize_visible_probs(distance_totals)
    dominant_probs = _normalize_visible_probs(dominant_counts)
    combined = {
        label: (0.75 * distance_probs[label]) + (0.25 * dominant_probs[label])
        for label in VISIBLE_LOCATION_CLASSES
    }
    return _normalize_visible_probs(combined)


def _window_label_shares(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {label: 0.0 for label in LOCATION_CLASSES}
    counts = {label: 0.0 for label in LOCATION_CLASSES}
    for row in rows:
        counts[_normalize_label(row.get("dominant_location_label"))] += 1.0
    total = sum(counts.values())
    return {label: safe_ratio(counts[label], total) for label in LOCATION_CLASSES}


def _window_nearest_visible_shares(rows: list[dict[str, Any]]) -> dict[str, float]:
    visible = [row for row in rows if row["frame_has_location_signal"]]
    nearest_counts = {label: 0.0 for label in VISIBLE_LOCATION_CLASSES}
    valid_rows = 0
    for row in visible:
        distances = {
            "chair": row["patient_chair_distance"],
            "bed": row["patient_bed_distance"],
            "room": row["patient_room_distance"],
        }
        available = {label: value for label, value in distances.items() if value is not None}
        if not available:
            continue
        winner = min(available, key=available.get)
        nearest_counts[winner] += 1.0
        valid_rows += 1
    if valid_rows <= 0:
        return {label: 0.0 for label in VISIBLE_LOCATION_CLASSES}
    return {label: nearest_counts[label] / valid_rows for label in VISIBLE_LOCATION_CLASSES}


def _window_room_bed_margin_mean(rows: list[dict[str, Any]]) -> float | None:
    margins: list[float] = []
    for row in rows:
        if not row["frame_has_location_signal"]:
            continue
        bed = row["patient_bed_distance"]
        room = row["patient_room_distance"]
        if bed is None or room is None:
            continue
        margins.append(bed - room)
    return None if not margins else sum(margins) / len(margins)


def _window_distance_velocity_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    visible = [row for row in rows if row["frame_has_location_signal"]]
    if len(visible) < 2:
        return {label: 0.0 for label in VISIBLE_LOCATION_CLASSES}
    visible = sorted(visible, key=lambda item: item["second_offset"])
    velocities = {label: [] for label in VISIBLE_LOCATION_CLASSES}
    for previous, current in zip(visible, visible[1:]):
        second_delta = current["second_offset"] - previous["second_offset"]
        if abs(second_delta) <= 1e-12:
            continue
        for label, column in (
            ("chair", "patient_chair_distance"),
            ("bed", "patient_bed_distance"),
            ("room", "patient_room_distance"),
        ):
            prev_distance = previous[column]
            curr_distance = current[column]
            if prev_distance is None or curr_distance is None:
                continue
            velocities[label].append((curr_distance - prev_distance) / second_delta)
    return {label: _mean(velocities[label]) for label in VISIBLE_LOCATION_CLASSES}


def _window_posture_score_probs(rows: list[dict[str, Any]], half_life_seconds: int) -> dict[str, float]:
    observed = [row for row in rows if row["frame_has_posture_signal"]]
    if not observed:
        return {label: 0.0 for label in POSTURE_CLASSES}
    weighted = {label: 0.0 for label in POSTURE_CLASSES}
    for row in observed:
        recency_weight = 2.0 ** (row["second_offset"] / half_life_seconds) if half_life_seconds > 0 else 1.0
        for label in POSTURE_CLASSES:
            weighted[label] += recency_weight * row[f"posture_score_{label}"]
    return _normalize_posture_probs(weighted)


def _window_posture_label_shares(rows: list[dict[str, Any]]) -> dict[str, float]:
    observed = [row for row in rows if row["frame_has_posture_signal"] and row["primary_patient_posture_label"]]
    if not observed:
        return {label: 0.0 for label in POSTURE_CLASSES}
    counts = {label: 0.0 for label in POSTURE_CLASSES}
    for row in observed:
        counts[row["primary_patient_posture_label"]] += 1.0
    total = sum(counts.values())
    return {label: safe_ratio(counts[label], total) for label in POSTURE_CLASSES}


def summarize_panel_history(second_level_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one event's per-second history into anchor/tail validation features."""

    rows = _prepare_panel_rows(list(second_level_rows))
    if not rows:
        return {
            "fall_event_id": None,
            "anchor_visible_frames": 0.0,
            "tail_visible_frames": 0.0,
            "last_visible_gap_seconds": None,
            "anchor_visible_probs": {label: 0.0 for label in VISIBLE_LOCATION_CLASSES},
            "tail_visible_probs": {label: 0.0 for label in VISIBLE_LOCATION_CLASSES},
            "tail_share_room": 0.0,
            "tail_share_no_patient": 0.0,
            "tail_overlap_mean": 0.0,
            "tail_overlap_max": 0.0,
            "anchor_nudge_mean": 0.0,
            "tail_nudge_mean": 0.0,
            "tail_nudge_max": 0.0,
            "anchor_bed_in_bed_score_mean": 0.0,
            "tail_bed_in_bed_score_mean": 0.0,
            "tail_bed_in_bed_score_max": 0.0,
            "anchor_motion_bac_mean": 0.0,
            "tail_motion_bac_mean": 0.0,
            "tail_motion_bac_max": 0.0,
            "anchor_patient_candidate_count_mean": 0.0,
            "tail_patient_candidate_count_mean": 0.0,
            "tail_staff_candidate_count_mean": 0.0,
            "tail_other_candidate_count_mean": 0.0,
            "tail_bed_candidate_count_mean": 0.0,
            "tail_chair_candidate_count_mean": 0.0,
            "anchor_room_nearest_share": 0.0,
            "tail_room_nearest_share": 0.0,
            "tail_bed_nearest_share": 0.0,
            "anchor_room_bed_margin_mean": None,
            "tail_room_bed_margin_mean": None,
            "room_bed_margin_delta": None,
            "tail_chair_distance_velocity_mean": 0.0,
            "tail_bed_distance_velocity_mean": 0.0,
            "tail_room_distance_velocity_mean": 0.0,
            "anchor_posture_observed_frames": 0.0,
            "tail_posture_observed_frames": 0.0,
            "posture_observed_frames": 0.0,
            "posture_switch_count": 0.0,
            "anchor_posture_score_probs": {label: 0.0 for label in POSTURE_CLASSES},
            "tail_posture_score_probs": {label: 0.0 for label in POSTURE_CLASSES},
            "anchor_posture_shares": {label: 0.0 for label in POSTURE_CLASSES},
            "tail_posture_shares": {label: 0.0 for label in POSTURE_CLASSES},
            "anchor_label": "none",
            "tail_label": "none",
            "anchor_posture_label": "none",
            "tail_posture_label": "none",
            "departure_detected": False,
        }

    half_life = int(DEFAULTS["label_eval_v2_recency_half_life_seconds"])
    dropout_seconds = int(DEFAULTS["fall_window_dropout_seconds"])
    rows = [row for row in rows if row["second_offset"] < 0 and row["second_offset"] >= -dropout_seconds]
    anchor = _window_rows(rows, ANCHOR_START_SECONDS, ANCHOR_END_SECONDS)
    tail = _window_rows(rows, TAIL_START_SECONDS, TAIL_END_SECONDS)
    visible_all = [row for row in rows if row["frame_has_location_signal"]]
    visible_anchor = [row for row in anchor if row["frame_has_location_signal"]]
    visible_tail = [row for row in tail if row["frame_has_location_signal"]]

    anchor_visible_probs = _window_distance_probs(anchor, half_life)
    tail_visible_probs = _window_distance_probs(tail, half_life)
    tail_label_shares = _window_label_shares(tail)
    anchor_nearest = _window_nearest_visible_shares(anchor)
    tail_nearest = _window_nearest_visible_shares(tail)
    anchor_margin = _window_room_bed_margin_mean(anchor)
    tail_margin = _window_room_bed_margin_mean(tail)
    tail_velocity = _window_distance_velocity_means(tail)
    anchor_posture_score_probs = _window_posture_score_probs(anchor, half_life)
    tail_posture_score_probs = _window_posture_score_probs(tail, half_life)
    anchor_posture_shares = _window_posture_label_shares(anchor)
    tail_posture_shares = _window_posture_label_shares(tail)
    posture_observed = [row for row in rows if row["frame_has_posture_signal"]]

    posture_switch_count = 0.0
    previous_label = None
    for row in posture_observed:
        current_label = row["primary_patient_posture_label"]
        if previous_label and current_label and current_label != previous_label:
            posture_switch_count += 1.0
        if current_label:
            previous_label = current_label

    room_bed_margin_delta = None
    if anchor_margin is not None and tail_margin is not None:
        room_bed_margin_delta = tail_margin - anchor_margin

    last_visible_gap_seconds = None
    if visible_all:
        last_visible_gap_seconds = abs(max(row["second_offset"] for row in visible_all))

    room_direct_support = max(
        tail_visible_probs["room"],
        tail_label_shares["room"],
        _max_value(row["patient_staff_iou"] for row in visible_tail),
        _max_value(row["nudge_score_normalized"] for row in tail),
    )
    departure_detected = room_direct_support >= 0.35 and tail_label_shares["room"] >= 0.10

    return {
        "fall_event_id": rows[0].get("fall_event_id"),
        "anchor_visible_frames": float(len(visible_anchor)),
        "tail_visible_frames": float(len(visible_tail)),
        "last_visible_gap_seconds": last_visible_gap_seconds,
        "anchor_visible_probs": anchor_visible_probs,
        "tail_visible_probs": tail_visible_probs,
        "tail_share_room": tail_label_shares["room"],
        "tail_share_no_patient": tail_label_shares["no_patient"],
        "tail_overlap_mean": _mean(row["patient_staff_iou"] for row in tail),
        "tail_overlap_max": _max_value(row["patient_staff_iou"] for row in tail),
        "anchor_nudge_mean": _mean(row["nudge_score_normalized"] for row in anchor),
        "tail_nudge_mean": _mean(row["nudge_score_normalized"] for row in tail),
        "tail_nudge_max": _max_value(row["nudge_score_normalized"] for row in tail),
        "anchor_bed_in_bed_score_mean": _mean(row["bed_in_bed_score"] for row in anchor),
        "tail_bed_in_bed_score_mean": _mean(row["bed_in_bed_score"] for row in tail),
        "tail_bed_in_bed_score_max": _max_value(row["bed_in_bed_score"] for row in tail),
        "anchor_motion_bac_mean": _mean(row["motion_bac"] for row in anchor),
        "tail_motion_bac_mean": _mean(row["motion_bac"] for row in tail),
        "tail_motion_bac_max": _max_value(row["motion_bac"] for row in tail),
        "anchor_patient_candidate_count_mean": _mean(row["patient_candidate_count"] for row in anchor),
        "tail_patient_candidate_count_mean": _mean(row["patient_candidate_count"] for row in tail),
        "tail_staff_candidate_count_mean": _mean(row["staff_candidate_count"] for row in tail),
        "tail_other_candidate_count_mean": _mean(row["other_candidate_count"] for row in tail),
        "tail_bed_candidate_count_mean": _mean(row["bed_candidate_count"] for row in tail),
        "tail_chair_candidate_count_mean": _mean(row["chair_candidate_count"] for row in tail),
        "anchor_room_nearest_share": anchor_nearest["room"],
        "tail_room_nearest_share": tail_nearest["room"],
        "tail_bed_nearest_share": tail_nearest["bed"],
        "anchor_room_bed_margin_mean": anchor_margin,
        "tail_room_bed_margin_mean": tail_margin,
        "room_bed_margin_delta": room_bed_margin_delta,
        "tail_chair_distance_velocity_mean": tail_velocity["chair"],
        "tail_bed_distance_velocity_mean": tail_velocity["bed"],
        "tail_room_distance_velocity_mean": tail_velocity["room"],
        "anchor_posture_observed_frames": float(sum(1 for row in anchor if row["frame_has_posture_signal"])),
        "tail_posture_observed_frames": float(sum(1 for row in tail if row["frame_has_posture_signal"])),
        "posture_observed_frames": float(sum(1 for row in rows if row["frame_has_posture_signal"])),
        "posture_switch_count": posture_switch_count,
        "anchor_posture_score_probs": anchor_posture_score_probs,
        "tail_posture_score_probs": tail_posture_score_probs,
        "anchor_posture_shares": anchor_posture_shares,
        "tail_posture_shares": tail_posture_shares,
        "anchor_label": _dominant_visible_label(anchor_visible_probs),
        "tail_label": _dominant_visible_label(tail_visible_probs),
        "anchor_posture_label": _dominant_posture_label(anchor_posture_score_probs, anchor_posture_shares),
        "tail_posture_label": _dominant_posture_label(tail_posture_score_probs, tail_posture_shares),
        "departure_detected": departure_detected,
    }


def _event_window_visible_prior(event_window: Mapping[str, Any], legacy_visible: Mapping[str, Any]) -> dict[str, float]:
    focus_visible = _normalize_visible_probs(
        {
            "chair": event_window.get("pre_focus_weight_chair", 0.0),
            "bed": event_window.get("pre_focus_weight_bed", 0.0),
            "room": event_window.get("pre_focus_weight_room", 0.0),
        }
    )
    conditional_visible = _normalize_visible_probs(
        {
            "chair": event_window.get("pre_cond_prob_chair", 0.0),
            "bed": event_window.get("pre_cond_prob_bed", 0.0),
            "room": event_window.get("pre_cond_prob_room", 0.0),
        }
    )
    return _blend_visible_prob_sources(
        (0.45, legacy_visible),
        (0.35, focus_visible),
        (0.20, conditional_visible),
    )


def _posture_adjusted_visible_probs(
    visible_probs: Mapping[str, Any],
    posture_probs: Mapping[str, Any],
) -> dict[str, float]:
    room_visible = max(0.0, _safe_float(visible_probs.get("room", 0.0)))
    chair_visible = max(0.0, _safe_float(visible_probs.get("chair", 0.0)))
    bed_visible = max(0.0, _safe_float(visible_probs.get("bed", 0.0)))
    non_room_visible = chair_visible + bed_visible
    if non_room_visible <= 0.0:
        return _normalize_visible_probs(
            {"chair": chair_visible, "bed": bed_visible, "room": room_visible}
        )

    lying_bias = max(
        0.0,
        _safe_float(posture_probs.get("lying", 0.0))
        - max(_safe_float(posture_probs.get("sitting", 0.0)), _safe_float(posture_probs.get("standing", 0.0))),
    )
    if lying_bias <= 0.0:
        return _normalize_visible_probs(
            {"chair": chair_visible, "bed": bed_visible, "room": room_visible}
        )

    chair_signal = chair_visible
    bed_signal = bed_visible + (lying_bias * non_room_visible)
    chair_share = chair_signal / max(1e-6, chair_signal + bed_signal)
    bed_share = bed_signal / max(1e-6, chair_signal + bed_signal)
    return _normalize_visible_probs(
        {
            "chair": non_room_visible * chair_share,
            "bed": non_room_visible * bed_share,
            "room": room_visible,
        }
    )


def _infer_signal_regime(
    defaults: Mapping[str, Any],
    event_window: Mapping[str, Any],
    legacy_visible: Mapping[str, Any],
    event_window_visible: Mapping[str, Any],
    presence_support: float,
) -> tuple[str, float, float, float, float]:
    distance_signal = min(1.0, _safe_float(event_window.get("pre_distance_signal_count")) / 20.0)
    visible_support = max(
        max(legacy_visible.values()),
        max(event_window_visible.values()),
        _safe_float(event_window.get("pre_state_visible_prob")),
        _safe_float(event_window.get("pre_dropout_visibility_ratio")),
        distance_signal,
    )
    transient_support = max(
        _safe_float(event_window.get("pre_state_not_visible_prob")),
        1.0 - _safe_float(event_window.get("pre_dropout_visibility_ratio")),
        _safe_float(event_window.get("pre_prob_not_visible")),
    )
    out_of_room_support = max(
        _safe_float(event_window.get("pre_state_out_of_room_prob")),
        _safe_float(event_window.get("pre_out_of_room_share")),
        _safe_float(event_window.get("pre_prob_out_of_room")),
    )
    if visible_support >= max(transient_support, out_of_room_support) and visible_support >= 0.35:
        return "visible", visible_support, transient_support, out_of_room_support, presence_support
    if out_of_room_support >= max(visible_support, transient_support) and out_of_room_support >= 0.45:
        return "out_of_room", visible_support, transient_support, out_of_room_support, presence_support
    if (
        transient_support >= max(visible_support, out_of_room_support)
        and transient_support >= 0.35
        and presence_support >= _safe_float(defaults["prefall_panel_v3_presence_min_support"])
    ):
        return "present_but_sparse", visible_support, transient_support, out_of_room_support, presence_support
    if transient_support >= max(visible_support, out_of_room_support) and transient_support >= 0.35:
        return "temporarily_not_visible", visible_support, transient_support, out_of_room_support, presence_support
    return "unknown", visible_support, transient_support, out_of_room_support, presence_support


def _set_no_patient_probability(
    base_probs: Mapping[str, Any],
    target_no_patient: float,
    visible_prior: Mapping[str, Any],
) -> dict[str, float]:
    probs = _normalize_probability_dict(base_probs)
    target = min(0.95, max(0.0, target_no_patient))
    current = probs["no_patient"]
    if abs(target - current) <= 1e-6:
        return probs

    updated = probs.copy()
    if target < current:
        freed_mass = current - target
        updated["no_patient"] = target
        visible_mix = _blend_visible_prob_sources(
            (0.60, _normalize_visible_probs(updated)),
            (0.40, visible_prior),
        )
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
    base_probs: Mapping[str, Any],
    target_label: str,
    target_probability: float,
) -> dict[str, float]:
    probs = _normalize_probability_dict(base_probs)
    target = max(0.0, min(0.95, target_probability))
    if probs.get(target_label, 0.0) >= target:
        return probs

    updated = probs.copy()
    remaining = target - updated[target_label]
    donors = sorted(
        (label for label in LOCATION_CLASSES if label != target_label),
        key=lambda label: updated[label],
        reverse=True,
    )
    for donor in donors:
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
    defaults: Mapping[str, Any],
    anchor_probs: Mapping[str, Any],
    tail_probs: Mapping[str, Any],
    anchor_frames: int,
    tail_frames: int,
    last_gap: float | None,
) -> float:
    if tail_frames <= 0:
        return 1.0
    if anchor_frames <= 0:
        return 0.0

    weight = _safe_float(defaults["prefall_panel_anchor_weight"])
    anchor_conf = max(anchor_probs.values()) * min(1.0, anchor_frames / 4.0)
    tail_conf = max(tail_probs.values()) * min(1.0, tail_frames / 4.0)
    if tail_conf > anchor_conf + 0.15:
        weight -= 0.15
    elif anchor_conf > tail_conf + 0.15:
        weight += 0.15

    if last_gap is not None:
        gap_ratio = min(
            1.0,
            max(0.0, last_gap / max(1.0, _safe_float(defaults["label_eval_v2_recent_visible_gap_seconds"]))),
        )
        weight += 0.15 * gap_ratio

    anchor_label = _dominant_visible_label(anchor_probs)
    tail_label = _dominant_visible_label(tail_probs)
    if anchor_label != "none" and tail_label != "none" and anchor_label != tail_label:
        if tail_frames < 3:
            weight += 0.10
        else:
            weight -= 0.05
    return min(0.80, max(0.20, weight))


def _room_departure_confidence(summary: Mapping[str, Any]) -> float:
    confidence = 0.0
    if _safe_bool(summary.get("departure_detected")):
        confidence += 0.30
    if str(summary.get("anchor_label", "none")).strip().lower() in {"chair", "bed"} and str(
        summary.get("tail_label", "none")
    ).strip().lower() == "room":
        confidence += 0.10
    confidence += 0.25 * min(1.0, safe_ratio(summary.get("tail_share_room"), 0.25))
    confidence += 0.20 * min(1.0, safe_ratio(summary.get("tail_room_nearest_share"), 0.35))
    confidence += 0.10 * positive_margin_score(summary.get("room_bed_margin_delta"))
    confidence += 0.03 * min(1.0, safe_ratio(summary.get("tail_overlap_max"), 0.25))
    confidence += 0.02 * min(1.0, safe_ratio(summary.get("tail_nudge_max"), 0.70))
    return min(1.0, max(0.0, confidence))


def _cache_presence_support(defaults: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, float]:
    motion_support = scaled_support(
        max(_safe_float(summary.get("tail_motion_bac_max")), _safe_float(summary.get("tail_motion_bac_mean"))),
        _safe_float(defaults["prefall_panel_v3_motion_presence_scale"]),
    )
    patient_candidate_support = max(
        scaled_support(summary.get("anchor_patient_candidate_count_mean")),
        scaled_support(summary.get("tail_patient_candidate_count_mean")),
    )
    other_candidate_support = scaled_support(
        summary.get("tail_other_candidate_count_mean"),
        _safe_float(defaults["prefall_panel_v3_room_object_mean_min"]),
    )
    bed_candidate_support = scaled_support(summary.get("tail_bed_candidate_count_mean"))
    chair_candidate_support = scaled_support(summary.get("tail_chair_candidate_count_mean"))
    posture_support = scaled_support(
        summary.get("tail_posture_observed_frames"),
        _safe_float(defaults["prefall_panel_v3_posture_min_frames"]),
    )
    bed_in_bed_support = max(
        _safe_float(summary.get("tail_bed_in_bed_score_max")),
        _safe_float(summary.get("tail_bed_in_bed_score_mean")),
    )
    tail_visible = _normalize_visible_probs(summary.get("tail_visible_probs", {}))
    presence_support = max(
        patient_candidate_support,
        posture_support,
        0.75 * bed_in_bed_support,
        0.60 * motion_support,
        0.50 * max(other_candidate_support, bed_candidate_support, chair_candidate_support),
    )
    room_presence_support = max(
        _safe_float(summary.get("tail_share_room")),
        _safe_float(summary.get("tail_room_nearest_share")),
        0.60 * other_candidate_support,
        0.40 * motion_support,
    )
    bed_presence_support = max(
        tail_visible["bed"],
        _safe_float(summary.get("tail_bed_nearest_share")),
        0.65 * bed_in_bed_support,
        0.50 * bed_candidate_support,
    )
    chair_presence_support = max(
        tail_visible["chair"],
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
    defaults: Mapping[str, Any],
    summary: Mapping[str, Any],
    base_probs: Mapping[str, Any],
) -> tuple[dict[str, float], bool]:
    top_visible = sorted(
        ((label, _safe_float(base_probs.get(label, 0.0))) for label in VISIBLE_LOCATION_CLASSES),
        key=lambda item: item[1],
        reverse=True,
    )
    if len(top_visible) < 2 or (
        top_visible[0][1] - top_visible[1][1]
    ) > _safe_float(defaults["prefall_panel_v3_velocity_tiebreak_margin"]):
        return _normalize_probability_dict(base_probs), False

    approach_scores = {
        "chair": max(0.0, -_safe_float(summary.get("tail_chair_distance_velocity_mean"))),
        "bed": max(0.0, -_safe_float(summary.get("tail_bed_distance_velocity_mean"))),
        "room": max(0.0, -_safe_float(summary.get("tail_room_distance_velocity_mean"))),
    }
    winner, winner_score = max(approach_scores.items(), key=lambda item: item[1])
    if winner_score < _safe_float(defaults["prefall_panel_v3_velocity_min_approach"]):
        return _normalize_probability_dict(base_probs), False
    if winner == "room":
        room_support = max(
            _safe_float(summary.get("tail_share_room")),
            _safe_float(summary.get("tail_room_nearest_share")),
        )
        if room_support < 0.10:
            return _normalize_probability_dict(base_probs), False
    target_probability = min(0.70, max(top_visible[0][1] + 0.03, 0.40 + winner_score))
    updated = _shift_probability_to_label(base_probs, winner, target_probability)
    changed = _dominant_visible_label(_normalize_visible_probs(updated)) != _dominant_visible_label(
        _normalize_visible_probs(base_probs)
    )
    return updated, changed


def derive_prefall_location_v3(
    event_window: Mapping[str, Any],
    panel_summary: Mapping[str, Any],
    defaults: Mapping[str, Any] = DEFAULTS,
) -> dict[str, Any]:
    """Derive v2/v3 pre-fall probabilities from one event window plus one panel summary."""

    legacy_probs = _normalize_probability_dict(
        {
            "chair": event_window.get("pre_prob_chair", 0.0),
            "bed": event_window.get("pre_prob_bed", 0.0),
            "room": event_window.get("pre_prob_room", 0.0),
            "no_patient": event_window.get("pre_prob_no_patient", 0.0),
        }
    )
    legacy_visible = _normalize_visible_probs(legacy_probs)
    event_window_visible = _event_window_visible_prior(event_window, legacy_visible)
    anchor_probs = _normalize_visible_probs(panel_summary.get("anchor_visible_probs", {}))
    tail_probs = _normalize_visible_probs(panel_summary.get("tail_visible_probs", {}))
    anchor_frames = int(round(_safe_float(panel_summary.get("anchor_visible_frames"))))
    tail_frames = int(round(_safe_float(panel_summary.get("tail_visible_frames"))))
    last_gap = _optional_float(panel_summary.get("last_visible_gap_seconds"))

    # Phase 1: V2 blend and room/no-patient correction.
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
            probs = {"chair": 0.0, "bed": 0.0, "room": 0.0, "no_patient": 1.0}
            reason = PREDICTION_REASON_ZERO_SIGNAL_DEFAULT
        room_transfer = 0.0
        no_patient_capped = False
    else:
        anchor_weight = _safe_float(defaults["prefall_panel_anchor_weight"])
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
                _safe_float(panel_summary.get("tail_overlap_max")),
                _safe_float(panel_summary.get("tail_overlap_mean")) * 2.0,
            ),
        )
        safety_threshold = _safe_float(defaults["prefall_panel_safety_zone_threshold"])
        tail_nudge_max = _safe_float(panel_summary.get("tail_nudge_max"))
        tail_nudge_mean = _safe_float(panel_summary.get("tail_nudge_mean"))
        anchor_nudge_mean = _safe_float(panel_summary.get("anchor_nudge_mean"))
        safety_level = 0.0
        if tail_nudge_max > safety_threshold:
            safety_level = min(1.0, (tail_nudge_max - safety_threshold) / max(1e-6, 1.0 - safety_threshold))
        safety_rise = max(0.0, tail_nudge_mean - anchor_nudge_mean)
        safety_strength = min(1.0, max(safety_level, safety_rise))
        gap_ratio = 1.0 if last_gap is None else min(
            1.0,
            max(0.0, last_gap / _safe_float(defaults["fall_window_dropout_seconds"])),
        )
        tail_no_patient_share = _safe_float(panel_summary.get("tail_share_no_patient"))
        tail_room_share = _safe_float(panel_summary.get("tail_share_room"))
        room_signal = max(tail_probs["room"], tail_room_share)
        room_support_gate = max(tail_room_share, overlap_strength, safety_strength)
        legacy_visible_weight = 0.35 if room_support_gate >= 0.15 else 0.15
        if room_support_gate < 0.15 and legacy_probs["room"] >= 0.70 and anchor_probs["chair"] >= 0.60:
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
            _safe_float(panel_summary.get("tail_room_nearest_share")),
            positive_margin_score(panel_summary.get("tail_room_bed_margin_mean")),
            positive_margin_score(panel_summary.get("room_bed_margin_delta")),
        )
        if legacy_probs["chair"] >= 0.50 and tail_room_share < 0.05 and room_distance_support < 0.25:
            legacy_visible_weight = max(legacy_visible_weight, 0.55)
            visible_probs = _normalize_visible_probs(
                {
                    label: (legacy_visible_weight * legacy_visible[label])
                    + ((1.0 - legacy_visible_weight) * panel_visible_probs[label])
                    for label in VISIBLE_LOCATION_CLASSES
                }
            )
        tail_bed_nearest_share = _safe_float(panel_summary.get("tail_bed_nearest_share"))
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
        if legacy_probs["chair"] >= 0.50 and tail_room_share < 0.05 and room_distance_support < 0.25:
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
            and (last_gap is None or last_gap <= _safe_float(defaults["label_eval_v2_recent_visible_gap_seconds"]))
        ):
            no_patient_prob = min(no_patient_prob, 0.25)
            recent_visible_override = True
        no_patient_capped = False
        if overlap_strength >= 0.05 and max(visible_probs.values()) >= 0.45:
            capped_value = _safe_float(defaults["prefall_panel_overlap_no_patient_cap"])
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
        sql_prefall_label = str(
            event_window.get("sql_prefall_location_label", event_window.get("prefall_location_label", ""))
        ).strip().lower()
        if direct_room_signal >= 0.35 and max(room_signal, room_distance_support) >= 0.15 and not bed_lock:
            room_transfer_scale = _safe_float(defaults["prefall_panel_room_bonus"]) * direct_room_signal
            if sql_prefall_label == "room" and safety_strength >= 0.50 and room_signal >= 0.25:
                room_transfer_scale *= 1.25
            donor = max(("chair", "bed", "no_patient"), key=lambda label: probs[label])
            donor_cap = 0.40 if donor == "no_patient" else 0.35
            room_transfer = min(room_transfer_scale, probs[donor] * donor_cap)
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
        elif probs["no_patient"] >= _safe_float(defaults["label_eval_v2_no_patient_strong_threshold"]):
            reason = PREDICTION_REASON_STRONG_NO_PATIENT
        else:
            reason = PREDICTION_REASON_DIRECT_FOCUS_ARGMAX

    v2_probs = _normalize_probability_dict(probs)
    ordered = sorted(v2_probs.items(), key=lambda item: item[1], reverse=True)
    v2_label = ordered[0][0]
    v2_margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else ordered[0][1]

    # Phase 2: dynamic anchor blend and posture summary.
    dynamic_anchor_weight = _dynamic_anchor_weight(
        defaults,
        anchor_probs,
        tail_probs,
        anchor_frames,
        tail_frames,
        last_gap,
    )
    tail_weight = 1.0 - dynamic_anchor_weight
    anchor_posture_probs = _normalize_posture_probs(panel_summary.get("anchor_posture_score_probs", {}))
    tail_posture_probs = _normalize_posture_probs(panel_summary.get("tail_posture_score_probs", {}))
    anchor_posture_shares = _normalize_posture_probs(panel_summary.get("anchor_posture_shares", {}))
    tail_posture_shares = _normalize_posture_probs(panel_summary.get("tail_posture_shares", {}))
    posture_probs = _normalize_posture_probs(
        {
            label: (
                0.75
                * (
                    (dynamic_anchor_weight * anchor_posture_probs[label])
                    + (tail_weight * tail_posture_probs[label])
                )
            )
            + (
                0.25
                * (
                    (dynamic_anchor_weight * anchor_posture_shares[label])
                    + (tail_weight * tail_posture_shares[label])
                )
            )
            for label in POSTURE_CLASSES
        }
    )
    posture_observed_frames = max(
        int(round(_safe_float(panel_summary.get("posture_observed_frames")))),
        int(round(_safe_float(event_window.get("pre_posture_observed_frame_count")))),
    )
    posture_switch_count = _safe_float(panel_summary.get("posture_switch_count"))
    dynamic_panel_visible = _blend_visible_prob_sources(
        (dynamic_anchor_weight, anchor_probs),
        (tail_weight, tail_probs),
    )

    # Phase 3: cache/presence classification.
    presence_supports = _cache_presence_support(defaults, panel_summary)
    signal_regime, visible_support, transient_support, out_of_room_support, presence_support = _infer_signal_regime(
        defaults,
        event_window,
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
    departure_confidence = _room_departure_confidence(panel_summary)

    visible_corroboration = max(
        scaled_support(tail_frames, 2.0),
        scaled_support(event_window.get("pre_distance_signal_count"), 2.0),
    )
    tail_patient_corroboration = scaled_support(panel_summary.get("tail_patient_candidate_count_mean"))
    tail_motion_corroboration = scaled_support(
        max(
            _safe_float(panel_summary.get("tail_motion_bac_max")),
            _safe_float(panel_summary.get("tail_motion_bac_mean")),
        ),
        _safe_float(defaults["prefall_panel_v3_motion_presence_scale"]),
    )
    tail_posture_corroboration = scaled_support(
        panel_summary.get("tail_posture_observed_frames"),
        _safe_float(defaults["prefall_panel_v3_posture_min_frames"]),
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
    sparse_corroboration_min = _safe_float(defaults["prefall_panel_v3_sparse_corroboration_min"])
    sparse_corroborated = sparse_corroboration >= sparse_corroboration_min
    room_sparse_corroborated = room_sparse_corroboration >= sparse_corroboration_min
    has_visible_tail_support = anchor_frames > 0 or tail_frames > 0
    invisible_bed_rescue_allowed = has_visible_tail_support or (
        bed_in_bed_support >= _safe_float(defaults["prefall_panel_v3_bed_in_bed_invisible_min_score"])
        and invisible_bed_rescue_corroboration >= sparse_corroboration_min
    )

    # Phase 4: initial v3 blend and no-patient rebalance.
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

    preserve_v2_room = v2_label == "room"
    if preserve_v2_room:
        v3_probs = v2_probs.copy()
        v3_visible_probs = _normalize_visible_probs(v3_probs)

    strong_visible_carry = visible_support >= 0.80 and max(event_window_visible.values()) >= 0.60
    if signal_regime == "visible" and sparse_corroborated:
        target_no_patient = min(
            v3_probs["no_patient"],
            _safe_float(
                defaults["prefall_panel_v3_visible_no_patient_cap"]
                if strong_visible_carry
                else defaults["prefall_panel_v3_transient_no_patient_cap"]
            ),
        )
    elif signal_regime == "temporarily_not_visible" and max(event_window_visible.values()) >= 0.45 and sparse_corroborated:
        target_no_patient = min(
            v3_probs["no_patient"],
            _safe_float(defaults["prefall_panel_v3_transient_no_patient_cap"]),
        )
    elif signal_regime == "present_but_sparse" and sparse_corroborated:
        target_no_patient = min(
            v3_probs["no_patient"],
            _safe_float(defaults["prefall_panel_v3_present_sparse_no_patient_cap"]),
        )
    elif signal_regime == "out_of_room":
        target_no_patient = max(
            v3_probs["no_patient"],
            _safe_float(defaults["prefall_panel_v3_out_of_room_no_patient_floor"]),
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
        if target_no_patient < v2_probs["no_patient"] and (strong_visible_carry or signal_regime == "present_but_sparse"):
            v3_carry_forward_used = signal_regime == "present_but_sparse" or (
                strong_visible_carry and anchor_frames <= 0 and tail_frames <= 0
            )
            v3_base_recovery_reason = PREDICTION_REASON_SPARSE_SIGNAL_CARRY_FORWARD_V3

    # Phase 5: room departure recovery.
    bed_lock_v3 = (
        (
            _safe_float(panel_summary.get("tail_bed_nearest_share")) >= 0.75
            and _safe_float(panel_summary.get("tail_share_room")) < 0.10
            and _safe_float(panel_summary.get("tail_room_nearest_share")) < 0.15
        )
        or (
            bed_in_bed_support >= _safe_float(defaults["prefall_panel_v3_bed_in_bed_min_score"])
            and _safe_float(panel_summary.get("tail_share_room")) < 0.20
            and _safe_float(panel_summary.get("tail_room_nearest_share")) < 0.25
        )
    )
    room_nearest_share = _safe_float(panel_summary.get("tail_room_nearest_share"))
    room_tail_share = _safe_float(panel_summary.get("tail_share_room"))
    room_velocity_approach = _safe_float(panel_summary.get("tail_room_distance_velocity_mean")) <= -_safe_float(
        defaults["prefall_panel_v3_velocity_min_approach"]
    )
    room_override_evidence = (
        room_nearest_share >= 0.35
        or room_velocity_approach
        or (
            signal_regime == "present_but_sparse"
            and room_presence_support >= _safe_float(defaults["prefall_panel_v3_presence_min_support"])
            and room_sparse_corroborated
        )
        or (
            _safe_bool(panel_summary.get("departure_detected"))
            and (room_nearest_share >= 0.35 or room_tail_share >= 0.35 or room_velocity_approach)
        )
    )
    if (
        not preserve_v2_room
        and not bed_lock_v3
        and (
            (
                departure_confidence >= _safe_float(defaults["prefall_panel_v3_departure_min_confidence"])
                and room_override_evidence
                and room_sparse_corroborated
            )
            or (
                signal_regime == "present_but_sparse"
                and room_presence_support >= _safe_float(defaults["prefall_panel_v3_presence_min_support"])
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

    # Phase 6: bed rescue, velocity tie-break, and posture rebalance.
    velocity_shifted, v3_velocity_used = _apply_velocity_tiebreak(defaults, panel_summary, v3_probs)
    if v3_velocity_used:
        v3_probs = velocity_shifted
        v3_visible_probs = _normalize_visible_probs(v3_probs)
        if not v3_room_override_used and not v3_carry_forward_used:
            v3_base_recovery_reason = PREDICTION_REASON_VELOCITY_TIEBREAK_V3

    v3_bed_rescue_used = False
    if (
        not preserve_v2_room
        and not v3_room_override_used
        and bed_in_bed_support >= _safe_float(defaults["prefall_panel_v3_bed_in_bed_min_score"])
        and bed_presence_support >= max(room_presence_support, chair_presence_support)
        and v3_probs["room"] < 0.30
        and (
            has_visible_tail_support
            or (
                invisible_bed_rescue_allowed
                and bed_candidate_support >= _safe_float(defaults["prefall_panel_v3_presence_min_support"])
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
    elif departure_confidence >= _safe_float(defaults["prefall_panel_v3_departure_min_confidence"]):
        v3_posture_gate_reason = "departure_evidence"
    elif posture_observed_frames < int(_safe_float(defaults["prefall_panel_v3_posture_min_frames"])):
        v3_posture_gate_reason = "insufficient_posture_frames"
    elif posture_switch_count > _safe_float(defaults["prefall_panel_v3_posture_max_switch_count"]):
        v3_posture_gate_reason = "unstable_posture"
    elif (
        str(panel_summary.get("anchor_posture_label", "none")) != "none"
        and str(panel_summary.get("tail_posture_label", "none")) != "none"
        and str(panel_summary.get("anchor_posture_label", "none")) != str(panel_summary.get("tail_posture_label", "none"))
    ):
        v3_posture_gate_reason = "anchor_tail_posture_disagree"
    elif lying_probability < _safe_float(defaults["prefall_panel_v3_posture_bed_min_lying_prob"]):
        v3_posture_gate_reason = "lying_not_dominant"
    elif chair_bed_gap > _safe_float(defaults["prefall_panel_v3_posture_max_chair_bed_gap"]):
        v3_posture_gate_reason = "chair_bed_gap_too_wide"
    else:
        v3_posture_gate_reason = "applied"
        v3_posture_mode = "lying_bed_bias"
        lying_dominance = max(
            0.0,
            lying_probability - max(posture_probs["sitting"], posture_probs["standing"]),
        )
        posture_support_weight = min(
            1.0,
            posture_observed_frames / max(1.0, _safe_float(defaults["prefall_panel_v3_posture_min_frames"])),
        )
        posture_blend = min(1.0, max(0.0, lying_dominance / 0.25)) * posture_support_weight
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
    v3_margin = v3_ordered[0][1] - v3_ordered[1][1] if len(v3_ordered) > 1 else v3_ordered[0][1]
    signal_profile = (
        f"anchor={panel_summary.get('anchor_label', 'none')};"
        f"tail={panel_summary.get('tail_label', 'none')};"
        f"gap={'' if last_gap is None else round(last_gap, 1)};"
        f"anchor_weight={round(dynamic_anchor_weight, 3)};"
        f"room_tail={round(_safe_float(panel_summary.get('tail_share_room')), 3)};"
        f"room_nearest={round(_safe_float(panel_summary.get('tail_room_nearest_share')), 3)};"
        f"room_margin_delta={round(_safe_float(panel_summary.get('room_bed_margin_delta')), 3)};"
        f"overlap={round(_safe_float(panel_summary.get('tail_overlap_max')), 3)};"
        f"nudge={round(_safe_float(panel_summary.get('tail_nudge_max')), 3)};"
        f"signal_regime={signal_regime};"
        f"visible_support={round(visible_support, 3)};"
        f"transient_support={round(transient_support, 3)};"
        f"out_of_room_support={round(out_of_room_support, 3)};"
        f"presence_support={round(presence_support, 3)};"
        f"room_presence_support={round(room_presence_support, 3)};"
        f"bed_in_bed={round(bed_in_bed_support, 3)};"
        f"departure_conf={round(departure_confidence, 3)};"
        f"posture={max(posture_probs, key=posture_probs.get) if posture_probs else 'none'};"
        f"posture_blend={round(posture_blend, 3)};"
        f"posture_gate={v3_posture_gate_reason};"
        f"posture_switch={round(posture_switch_count, 1)}"
    )

    return {
        "pre_prob_v2_chair": round(v2_probs["chair"], 6),
        "pre_prob_v2_bed": round(v2_probs["bed"], 6),
        "pre_prob_v2_room": round(v2_probs["room"], 6),
        "pre_prob_v2_no_patient": round(v2_probs["no_patient"], 6),
        "prefall_location_v2_label": v2_label,
        "prefall_location_v2_margin": round(v2_margin, 6),
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
        "prefall_location_v3_departure_confidence": round(departure_confidence, 6),
        "prefall_location_v3_presence_support": round(presence_support, 6),
        "prefall_location_v3_room_presence_support": round(room_presence_support, 6),
        "prefall_location_v3_bed_in_bed_support": round(bed_in_bed_support, 6),
        "prefall_location_v3_carry_forward_used": v3_carry_forward_used,
        "prefall_location_v3_posture_applied": v3_posture_applied,
        "prefall_location_v3_posture_mode": v3_posture_mode,
        "prefall_location_v3_posture_gate_reason": v3_posture_gate_reason,
        "prefall_location_v3_base_recovery_reason": v3_base_recovery_reason or reason,
        "prefall_location_reason": v3_reason,
        "prefall_location_signal_profile": signal_profile,
        "prefall_location_v2_anchor_label": panel_summary.get("anchor_label", "none"),
        "prefall_location_v2_tail_label": panel_summary.get("tail_label", "none"),
        "prefall_location_v2_departure_detected": _safe_bool(panel_summary.get("departure_detected")),
        "prefall_location_v2_last_visible_gap_seconds": last_gap,
        "pre_prob_chair": round(v3_probs["chair"], 6),
        "pre_prob_bed": round(v3_probs["bed"], 6),
        "pre_prob_room": round(v3_probs["room"], 6),
        "pre_prob_no_patient": round(v3_probs["no_patient"], 6),
        "prefall_location_label": v3_label,
        "prefall_location_margin": round(v3_margin, 6),
    }
