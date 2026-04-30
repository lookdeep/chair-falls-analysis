#!/usr/bin/env python3
# ruff: noqa: E402
"""
Stand-alone script to run fall analysis on a folder of videos using Gemini.

This file intentionally remains the prompt-editing surface. The cached
preprocessing, upload reuse, and multi-model orchestration live under
`src/ld_chair_falls/gemini_fall_analysis.py`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.gemini_fall_analysis import (
    RetryPolicy,
    parse_args,
    parse_model_names,
    parse_rpm_limits,
    run_analysis_batch,
)


class FallEvent(BaseModel):
    fall_detected: bool = Field(
        ...,
        description=(
            "Whether a fall was observed or strongly inferred in this video clip. "
            "Set true if the fall is directly seen, or if the patient is later "
            "clearly seen on the ground in a way consistent with a fall. "
            "False if no fall evidence is present."
        ),
    )
    prefall_location: str | None = Field(
        None,
        description=(
            "Where was the patient immediately before the fall. "
            "One of: 'chair', 'bed', 'room'. "
            "Use 'chair' if seated in a chair or wheelchair, 'bed' if in or on the bed, "
            "'room' if standing, walking, or elsewhere in the room. "
            "Null if the fall is not detected, the patient is not visible, "
            "or the pre-fall context cannot be inferred."
        ),
    )
    last_furniture: str | None = Field(
        None,
        description=(
            "The last piece of furniture the patient was on or touching "
            "before the fall (for example 'chair' or 'bed'). Null if unclear, "
            "not visible, or no fall is detected."
        ),
    )
    fall_time_seconds: int | None = Field(
        None,
        description=(
            "Time in seconds from the start of the clip when the patient "
            "hits the ground or completes the fall, if that moment is visible. "
            "At 1 fps this equals the frame number (0-indexed). "
            "Null if no fall is detected or the patient is only discovered "
            "already on the ground."
        ),
    )
    staff_entry_time_seconds: int | None = Field(
        None,
        description=(
            "Time in seconds from the start of the clip when a staff member "
            "first enters the room after the fall. Null if staff was already "
            "present, no staff entered during the clip, no fall is detected, "
            "or the post-fall sequence is unclear."
        ),
    )
    fall_tags: list[str] = Field(
        default_factory=list,
        description=(
            "Tags describing the fall mechanism and context. "
            "Choose from: 'slip', 'trip', 'tumble', 'collapse', 'reach', "
            "'roll', 'back', 'offscreen', 'bathroom', 'no_fall', "
            "'bad_video', 'no_video'. "
            "Include support tags if the patient grabbed something: "
            "'support_table', 'support_door', 'support_wall', "
            "'support_person', 'support_bed', 'support_walker', "
            "'support_chair', 'support_lift', 'footrest'. "
            "Empty list if no relevant tags apply."
        ),
    )


class ChairFallsResponseSchema(BaseModel):
    fall_event: FallEvent = Field(
        ...,
        description="Structured observation of the fall event in this video clip.",
    )
    staff_response_time_seconds: int | None = Field(
        None,
        description=(
            "Derived field: seconds between fall_time and staff_entry_time. "
            "Null if either timestamp is missing."
        ),
    )
    confidence: str = Field(
        ...,
        description=(
            "Confidence in the observations given video quality "
            "(blurred, de-identified). One of: 'high', 'medium', 'low'."
        ),
    )
    report: str = Field(
        ...,
        description=(
            "Brief narrative of what was observed: patient position, "
            "fall sequence, and staff response. Note any visibility issues, "
            "and state whether the fall was directly observed or inferred from "
            "later on-ground observation."
        ),
    )


SYSTEM_INSTRUCTIONS = """\
You are an expert clinical video reviewer analysing blurred, de-identified \
hospital room footage.

Each video represents 1 hour of real-time activity encoded at 1 frame per \
second (3600 frames total). A subtitle caption on each frame shows the \
real wall-clock time — use it to orient yourself, but report all times as \
**seconds from the start of the clip** (frame 0 = second 0). There is no \
audio.

Your task is to watch the entire video and extract structured information \
about any patient fall that occurs.

**Step-by-step instructions:**

1. **Determine whether a fall occurs.** A fall is any event where the \
patient transitions from a supported position (chair, bed, standing) to \
the floor unintentionally. If the transition itself is not visible but the \
patient is later clearly seen on the ground in a way consistent with a fall, \
you may still set `fall_detected` to true.

2. **Identify the pre-fall location.** Where was the patient immediately \
before the fall? Classify as 'chair' (seated in chair or wheelchair), \
'bed' (in or on the bed), or 'room' (standing, walking, or elsewhere in \
the room). If the patient is only found on the ground and you cannot infer \
the pre-fall context, leave this null.

3. **Identify the last furniture.** What was the last piece of furniture \
the patient was on or touching before the fall (for example 'chair' or 'bed')? \
Leave this null if it cannot be inferred.

4. **Record the fall time.** Count the seconds from the start of the \
clip to the frame where the patient hits the ground, but only if that \
moment is visible. If the patient is first discovered already on the ground, \
leave `fall_time_seconds` null.

5. **Record the staff entry time.** Count the seconds from the start of \
the clip to the frame where a staff member first enters the room *after* \
the fall. If staff was already present before the fall, no staff enters \
during the clip, or you cannot tell that the entry occurred after the \
fall or on-ground state, leave null.

6. **Tag the fall mechanism.** Choose all applicable tags: slip, trip, \
tumble, collapse, reach, roll, back. Add support tags if the patient \
grabbed or braced against something during the fall. Use `no_fall` only \
when there is no visible fall and no credible evidence that the patient was \
found on the ground because of a fall.

7. **Assess confidence.** Given the video is blurred and de-identified, \
rate your confidence as 'high', 'medium', or 'low'.

8. **Write a brief report** summarising what you observed. Explicitly say \
whether the fall was directly observed or inferred because the patient was \
later seen on the ground.

If the fall itself is not visible but the patient is later clearly seen on \
the ground in a way consistent with a fall, set `fall_detected` to true, \
leave `fall_time_seconds` null unless the transition is visible, and explain \
that inference in the report.

If there is no visible fall and no credible on-ground evidence of a fall, \
set `fall_detected` to false and leave timing fields null. Still note what \
you can see (for example patient in bed or no activity).\
"""

USER_PROMPT = (
    "Analyze the provided video and respond in JSON format according to "
    "the provided schema. The video has been processed to 1 fps for analysis."
)


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in environment or .env", file=sys.stderr)
        raise SystemExit(1)

    try:
        model_names = parse_model_names(args.model, args.models)
        rpm_limits = parse_rpm_limits(args.rpm, model_names)
        retry_policy = RetryPolicy(
            max_attempts=args.retry_max,
            base_delay_seconds=args.retry_base_seconds,
        )
        run_analysis_batch(
            video_folder=args.video_folder,
            model_names=model_names,
            output_dir=args.output,
            cache_dir=args.cache_dir,
            refresh_cache=args.refresh_cache,
            fallback_sleep_seconds=args.sleep,
            rpm_limits=rpm_limits,
            retry_policy=retry_policy,
            api_key=api_key,
            system_instructions=SYSTEM_INSTRUCTIONS,
            user_prompt=USER_PROMPT,
            response_schema=ChairFallsResponseSchema,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
