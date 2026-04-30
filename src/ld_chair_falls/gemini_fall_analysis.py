from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from google.genai import Client

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gemini_fall_analysis"
DEFAULT_SINGLE_MODEL = "gemini-2.5-flash"
DEFAULT_BATCH_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview",
)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
TRANSIENT_ERROR_TOKENS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "deadline exceeded",
    "internal",
    "rate limit",
    "resource exhausted",
    "temporarily unavailable",
    "timeout",
    "unavailable",
)
CSV_FIELDNAMES = [
    "file",
    "fall_detected",
    "prefall_location",
    "last_furniture",
    "fall_time_seconds",
    "fall_time_wallclock",
    "staff_entry_time_seconds",
    "staff_entry_time_wallclock",
    "staff_response_time_seconds",
    "fall_tags",
    "confidence",
    "model",
    "batch_id",
    "prompt_hash",
    "source_video",
    "cached_video",
    "cache_key",
    "cache_hit",
    "upload_s",
    "total_s",
]


def default_cache_dir() -> Path:
    return Path.home() / "Documents" / "chair-falls-risk" / "gemini_cache"


DEFAULT_CACHE_DIR = default_cache_dir()


class FallEvent(BaseModel):
    fall_detected: bool = Field(
        ...,
        description=(
            "Whether a fall was observed in this video clip. "
            "False if the patient never falls, or the event is not visible."
        ),
    )
    prefall_location: str | None = Field(
        None,
        description=(
            "Where the patient was immediately before the fall. "
            "One of: 'chair', 'bed', 'room'. "
            "Use 'chair' if seated in a chair/wheelchair, 'bed' if in/on the bed, "
            "'room' if standing or moving in the room. "
            "Null if no fall detected or patient not visible."
        ),
    )
    last_furniture: str | None = Field(
        None,
        description=(
            "The last piece of furniture the patient was on or touching "
            "before the fall (e.g. 'chair', 'bed'). Null if unclear or no fall."
        ),
    )
    fall_time_seconds: int | None = Field(
        None,
        description=(
            "Time in seconds from the start of the clip when the patient "
            "hits the ground or completes the fall. At 1 fps this equals "
            "the frame number (0-indexed). Null if no fall detected."
        ),
    )
    staff_entry_time_seconds: int | None = Field(
        None,
        description=(
            "Time in seconds from the start of the clip when a staff member "
            "first enters the room after the fall. Null if staff was already "
            "present, no staff entered during the clip, or no fall detected."
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
            "fall sequence, and staff response. Note any visibility issues."
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
the floor unintentionally.

2. **Identify the pre-fall location.** Where was the patient immediately \
before the fall? Classify as 'chair' (seated in chair or wheelchair), \
'bed' (in or on the bed), or 'room' (standing, walking, or elsewhere in \
the room).

3. **Identify the last furniture.** What was the last piece of furniture \
the patient was on or touching before the fall (e.g. 'chair', 'bed')?

4. **Record the fall time.** Count the seconds from the start of the \
clip to the frame where the patient hits the ground.

5. **Record the staff entry time.** Count the seconds from the start of \
the clip to the frame where a staff member first enters the room *after* \
the fall. If staff was already present before the fall, or no staff \
enters during the clip, leave null.

6. **Tag the fall mechanism.** Choose all applicable tags: slip, trip, \
tumble, collapse, reach, roll, back. Add support tags if the patient \
grabbed or braced against something during the fall.

7. **Assess confidence.** Given the video is blurred and de-identified, \
rate your confidence as 'high', 'medium', or 'low'.

8. **Write a brief report** summarising what you observed.

If no fall is visible, set fall_detected to false and leave timing fields \
null. Still note what you can see (e.g. patient in bed, no activity).\
"""

USER_PROMPT = (
    "Analyze the provided video and respond in JSON format according to "
    "the provided schema. The video has been processed to 1 fps for analysis."
)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 5.0


@dataclass(frozen=True)
class CachedVideoArtifact:
    source_video: Path
    cached_video: Path
    cache_key: str
    subtitle_timestamps: list[str]
    cache_hit: bool


@dataclass
class ModelRateLimiter:
    rpm_limits: dict[str, float]
    fallback_sleep_seconds: float = 0.0
    last_request_started: dict[str, float] = field(default_factory=dict)

    def wait_for_turn(self, model_name: str) -> None:
        rpm_limit = self.rpm_limits.get(model_name)
        min_interval = 0.0
        if rpm_limit is not None:
            if rpm_limit <= 0:
                raise ValueError(f"RPM limit must be positive for model '{model_name}'")
            min_interval = 60.0 / rpm_limit
        elif self.fallback_sleep_seconds > 0:
            min_interval = self.fallback_sleep_seconds

        if min_interval <= 0:
            self.last_request_started[model_name] = time.perf_counter()
            return

        last_started = self.last_request_started.get(model_name)
        if last_started is not None:
            remaining = min_interval - (time.perf_counter() - last_started)
            if remaining > 0:
                time.sleep(remaining)
        self.last_request_started[model_name] = time.perf_counter()


class ModelOutputWriter:
    def __init__(self, model_name: str, jsonl_path: Path, summary_path: Path) -> None:
        self.model_name = model_name
        self.jsonl_path = jsonl_path
        self.summary_path = summary_path
        self.summary_rows: list[dict[str, Any]] = []

    def write_record(self, record: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if "error" in record:
            return
        fall_event = record["result"].get("fall_event", {})
        self.summary_rows.append(
            {
                "file": record["file"],
                "fall_detected": fall_event.get("fall_detected", ""),
                "prefall_location": fall_event.get("prefall_location", ""),
                "last_furniture": fall_event.get("last_furniture", ""),
                "fall_time_seconds": fall_event.get("fall_time_seconds", ""),
                "fall_time_wallclock": fall_event.get("fall_time_wallclock", ""),
                "staff_entry_time_seconds": fall_event.get("staff_entry_time_seconds", ""),
                "staff_entry_time_wallclock": fall_event.get("staff_entry_time_wallclock", ""),
                "staff_response_time_seconds": record["result"].get("staff_response_time_seconds", ""),
                "fall_tags": "|".join(fall_event.get("fall_tags", [])),
                "confidence": record["result"].get("confidence", ""),
                "model": record["model"],
                "batch_id": record.get("batch_id", ""),
                "prompt_hash": record.get("prompt_hash", ""),
                "source_video": record.get("source_video", ""),
                "cached_video": record.get("cached_video", ""),
                "cache_key": record.get("cache_key", ""),
                "cache_hit": record.get("cache_hit", ""),
                "upload_s": record["timing"]["upload_seconds"],
                "total_s": record["timing"]["total_seconds"],
            }
        )

    def finalize(self) -> None:
        with self.summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.summary_rows)


def discover_videos(folder: Path) -> list[Path]:
    videos = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos)


def _get_video_fps(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    num, den = out.stdout.strip().split("/")
    return int(num) / int(den)


def slow_down_to_1fps(src: Path, dst: Path) -> Path:
    src_fps = _get_video_fps(src)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vf",
            f"setpts=PTS*{src_fps}",
            "-r",
            "1",
            "-c:s",
            "copy",
            "-vcodec",
            "libx264",
            "-crf",
            "28",
            "-f",
            "mp4",
            str(dst),
        ],
        capture_output=True,
        check=True,
    )
    return dst


def extract_subtitle_timestamps(video_path: Path, tmp_dir: Path) -> list[str]:
    srt_path = tmp_dir / f"{video_path.stem}.srt"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-map", "0:s:0", str(srt_path)],
        capture_output=True,
    )
    if proc.returncode != 0 or not srt_path.exists():
        return []
    pattern = re.compile(r"(\d{1,2}:\d{2}:\d{2})\.\d+")
    timestamps: list[str] = []
    for line in srt_path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            timestamps.append(match.group(1))
    srt_path.unlink(missing_ok=True)
    return timestamps


def seconds_to_wallclock(
    seconds: int | None,
    subtitle_timestamps: list[str],
) -> str | None:
    if seconds is None or not subtitle_timestamps:
        return None
    idx = max(0, min(seconds, len(subtitle_timestamps) - 1))
    return subtitle_timestamps[idx]


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_response_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    fall_event = parsed.setdefault("fall_event", {})
    fall_time = _coerce_optional_int(fall_event.get("fall_time_seconds"))
    staff_entry_time = _coerce_optional_int(fall_event.get("staff_entry_time_seconds"))

    fall_event["fall_time_seconds"] = fall_time
    fall_event["staff_entry_time_seconds"] = staff_entry_time

    if fall_time is None or staff_entry_time is None or staff_entry_time < fall_time:
        parsed["staff_response_time_seconds"] = None
    else:
        parsed["staff_response_time_seconds"] = staff_entry_time - fall_time
    return parsed


def slugify_model_name(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


def _source_fingerprint(video_path: Path) -> dict[str, Any]:
    stat = video_path.stat()
    return {
        "source_path": str(video_path.resolve()),
        "source_name": video_path.name,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def build_cache_key(video_path: Path) -> str:
    payload = json.dumps(_source_fingerprint(video_path), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_prompt_hash(
    *,
    system_instructions: str = SYSTEM_INSTRUCTIONS,
    user_prompt: str = USER_PROMPT,
) -> str:
    payload = f"{system_instructions}\n---\n{user_prompt}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _cache_paths(cache_dir: Path, video_path: Path) -> tuple[str, Path, Path]:
    cache_key = build_cache_key(video_path)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", video_path.stem).strip("_") or "video"
    cached_video = cache_dir / f"{stem}.{cache_key}.1fps.mp4"
    metadata_path = cache_dir / f"{stem}.{cache_key}.json"
    return cache_key, cached_video, metadata_path


def _load_cache_metadata(metadata_path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _is_cache_valid(
    metadata: dict[str, Any] | None,
    *,
    cached_video: Path,
    video_path: Path,
) -> bool:
    if metadata is None or not cached_video.exists():
        return False
    required_keys = {"cache_key", "subtitle_timestamps", "source_path", "source_size", "source_mtime_ns"}
    if not required_keys.issubset(metadata):
        return False
    fingerprint = _source_fingerprint(video_path)
    return (
        metadata["source_path"] == fingerprint["source_path"]
        and metadata["source_size"] == fingerprint["source_size"]
        and metadata["source_mtime_ns"] == fingerprint["source_mtime_ns"]
    )


def prepare_cached_video(
    video_path: Path,
    *,
    cache_dir: Path,
    scratch_dir: Path,
    refresh_cache: bool = False,
) -> CachedVideoArtifact:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key, cached_video, metadata_path = _cache_paths(cache_dir, video_path)
    metadata = None if refresh_cache else _load_cache_metadata(metadata_path)
    if not refresh_cache and _is_cache_valid(metadata, cached_video=cached_video, video_path=video_path):
        return CachedVideoArtifact(
            source_video=video_path,
            cached_video=cached_video,
            cache_key=cache_key,
            subtitle_timestamps=list(metadata["subtitle_timestamps"]),
            cache_hit=True,
        )

    subtitle_timestamps = extract_subtitle_timestamps(video_path, scratch_dir)
    tmp_output = scratch_dir / f"{cached_video.stem}.part{cached_video.suffix}"
    tmp_output.unlink(missing_ok=True)
    slow_down_to_1fps(video_path, tmp_output)
    shutil.move(str(tmp_output), cached_video)

    payload = {
        "cache_key": cache_key,
        "cached_video": str(cached_video.resolve()),
        "subtitle_timestamps": subtitle_timestamps,
        "created_at": datetime.now(UTC).isoformat(),
        **_source_fingerprint(video_path),
    }
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return CachedVideoArtifact(
        source_video=video_path,
        cached_video=cached_video,
        cache_key=cache_key,
        subtitle_timestamps=subtitle_timestamps,
        cache_hit=False,
    )


def parse_response_payload(raw_text: str) -> dict[str, Any]:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        return json.loads(cleaned)


def connect(api_key: str) -> Client:
    from google import genai

    return genai.Client(api_key=api_key)


def _status_code_from_exception(exc: Exception) -> int | None:
    for attr in ("status_code", "code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status
    return None


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    for key in ("retry-after", "Retry-After"):
        value = headers.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _is_transient_error(exc: Exception) -> bool:
    status_code = _status_code_from_exception(exc)
    if status_code in TRANSIENT_STATUS_CODES:
        return True
    message = str(exc).lower()
    return any(token in message for token in TRANSIENT_ERROR_TOKENS)


def run_with_retries[T](
    operation: Callable[[], T],
    *,
    retry_policy: RetryPolicy,
    operation_name: str,
    rate_limiter: ModelRateLimiter | None = None,
    model_name: str | None = None,
) -> T:
    for attempt in range(1, retry_policy.max_attempts + 1):
        if rate_limiter is not None and model_name is not None:
            rate_limiter.wait_for_turn(model_name)
        try:
            return operation()
        except Exception as exc:
            if attempt >= retry_policy.max_attempts or not _is_transient_error(exc):
                raise
            retry_after = _extract_retry_after_seconds(exc)
            if retry_after is not None:
                delay = retry_after
            else:
                delay = retry_policy.base_delay_seconds * (2 ** (attempt - 1))
                delay += random.uniform(0.0, retry_policy.base_delay_seconds)
            print(
                f"retrying {operation_name} attempt {attempt + 1}/{retry_policy.max_attempts} "
                f"after {delay:.1f}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise RuntimeError(f"unreachable retry loop for {operation_name}")


def upload_and_activate(
    client: Client,
    file_path: str,
    *,
    retry_policy: RetryPolicy,
    max_wait: int = 300,
    poll_interval: int = 5,
) -> Any:
    video_file = run_with_retries(
        lambda: client.files.upload(file=file_path),
        retry_policy=retry_policy,
        operation_name="files.upload",
    )
    started = time.perf_counter()

    while getattr(getattr(video_file, "state", None), "name", None) != "ACTIVE":
        elapsed = time.perf_counter() - started
        if elapsed > max_wait:
            raise TimeoutError(f"File activation timed out after {max_wait}s")
        if getattr(getattr(video_file, "state", None), "name", None) == "FAILED":
            raise RuntimeError("File processing failed on Gemini servers")
        time.sleep(poll_interval)
        video_file = run_with_retries(
            lambda remote_name=video_file.name: client.files.get(name=remote_name),
            retry_policy=retry_policy,
            operation_name="files.get",
        )

    return video_file


def call_gemini(
    client: Client,
    video_file: Any,
    model_name: str,
    *,
    system_instructions: str = SYSTEM_INSTRUCTIONS,
    user_prompt: str = USER_PROMPT,
    response_schema: type[BaseModel] = ChairFallsResponseSchema,
) -> Any:
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=system_instructions,
        temperature=0.0,
        response_schema=response_schema,
        response_mime_type="application/json",
    )
    return client.models.generate_content(
        model=model_name,
        contents=[video_file, user_prompt],
        config=config,
    )


def cleanup_remote(client: Client, video_file: Any) -> None:
    try:
        client.files.delete(name=video_file.name)
    except Exception:
        pass


def _usage_payload(result: Any) -> dict[str, Any]:
    metadata = getattr(result, "usage_metadata", None)
    if metadata is None:
        return {}
    return {
        "prompt_token_count": getattr(metadata, "prompt_token_count", None),
        "candidates_token_count": getattr(metadata, "candidates_token_count", None),
        "total_token_count": getattr(metadata, "total_token_count", None),
    }


def _record_base(
    *,
    video_path: Path,
    model_name: str,
    cached_video: CachedVideoArtifact,
    batch_id: str,
    prompt_hash: str,
) -> dict[str, Any]:
    return {
        "file": video_path.name,
        "model": model_name,
        "batch_id": batch_id,
        "prompt_hash": prompt_hash,
        "source_video": str(video_path.resolve()),
        "cached_video": str(cached_video.cached_video.resolve()),
        "cache_key": cached_video.cache_key,
        "cache_hit": cached_video.cache_hit,
    }


def build_success_record(
    *,
    video_path: Path,
    model_name: str,
    cached_video: CachedVideoArtifact,
    batch_id: str,
    prompt_hash: str,
    upload_seconds: float,
    request_seconds: float,
    result: Any,
) -> dict[str, Any]:
    parsed = normalize_response_payload(parse_response_payload(getattr(result, "text", "")))
    fall_event = parsed.get("fall_event", {})
    if cached_video.subtitle_timestamps:
        fall_event["fall_time_wallclock"] = seconds_to_wallclock(
            fall_event.get("fall_time_seconds"),
            cached_video.subtitle_timestamps,
        )
        fall_event["staff_entry_time_wallclock"] = seconds_to_wallclock(
            fall_event.get("staff_entry_time_seconds"),
            cached_video.subtitle_timestamps,
        )
    return {
        **_record_base(
            video_path=video_path,
            model_name=model_name,
            cached_video=cached_video,
            batch_id=batch_id,
            prompt_hash=prompt_hash,
        ),
        "result": parsed,
        "usage": _usage_payload(result),
        "timing": {
            "upload_seconds": round(upload_seconds, 2),
            "request_seconds": round(request_seconds, 2),
            "total_seconds": round(upload_seconds + request_seconds, 2),
        },
    }


def build_error_record(
    *,
    video_path: Path,
    model_name: str,
    cached_video: CachedVideoArtifact | None,
    batch_id: str,
    prompt_hash: str,
    stage: str,
    error: Exception | str,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "file": video_path.name,
        "model": model_name,
        "batch_id": batch_id,
        "prompt_hash": prompt_hash,
        "error_stage": stage,
        "error": str(error),
    }
    if cached_video is not None:
        base.update(
            {
                "source_video": str(video_path.resolve()),
                "cached_video": str(cached_video.cached_video.resolve()),
                "cache_key": cached_video.cache_key,
                "cache_hit": cached_video.cache_hit,
            }
        )
    return base


def process_video_across_models(
    *,
    client: Client,
    video_path: Path,
    cached_video: CachedVideoArtifact,
    model_names: list[str],
    batch_id: str,
    prompt_hash: str,
    rate_limiter: ModelRateLimiter,
    retry_policy: RetryPolicy,
    system_instructions: str = SYSTEM_INSTRUCTIONS,
    user_prompt: str = USER_PROMPT,
    response_schema: type[BaseModel] = ChairFallsResponseSchema,
) -> list[dict[str, Any]]:
    upload_started = time.perf_counter()
    try:
        video_file = upload_and_activate(
            client,
            str(cached_video.cached_video),
            retry_policy=retry_policy,
        )
    except Exception as exc:
        return [
            build_error_record(
                video_path=video_path,
                model_name=model_name,
                cached_video=cached_video,
                batch_id=batch_id,
                prompt_hash=prompt_hash,
                stage="upload",
                error=exc,
            )
            for model_name in model_names
        ]

    upload_seconds = time.perf_counter() - upload_started
    records: list[dict[str, Any]] = []
    try:
        for model_name in model_names:
            request_started = time.perf_counter()
            try:
                result = run_with_retries(
                    lambda requested_model=model_name: call_gemini(
                        client,
                        video_file,
                        requested_model,
                        system_instructions=system_instructions,
                        user_prompt=user_prompt,
                        response_schema=response_schema,
                    ),
                    retry_policy=retry_policy,
                    operation_name=f"models.generate_content[{model_name}]",
                    rate_limiter=rate_limiter,
                    model_name=model_name,
                )
                records.append(
                    build_success_record(
                        video_path=video_path,
                        model_name=model_name,
                        cached_video=cached_video,
                        batch_id=batch_id,
                        prompt_hash=prompt_hash,
                        upload_seconds=upload_seconds,
                        request_seconds=time.perf_counter() - request_started,
                        result=result,
                    )
                )
            except Exception as exc:
                records.append(
                    build_error_record(
                        video_path=video_path,
                        model_name=model_name,
                        cached_video=cached_video,
                        batch_id=batch_id,
                        prompt_hash=prompt_hash,
                        stage="generate",
                        error=exc,
                    )
                )
    finally:
        cleanup_remote(client, video_file)
    return records


def _resolve_output_paths(
    output_dir: Path,
    *,
    batch_id: str,
    model_names: list[str],
) -> dict[str, tuple[Path, Path]]:
    multi_model = len(model_names) > 1
    paths: dict[str, tuple[Path, Path]] = {}
    for model_name in model_names:
        suffix = f"_{slugify_model_name(model_name)}" if multi_model else ""
        paths[model_name] = (
            output_dir / f"results_{batch_id}{suffix}.jsonl",
            output_dir / f"summary_{batch_id}{suffix}.csv",
        )
    return paths


def create_output_writers(
    output_dir: Path,
    *,
    batch_id: str,
    model_names: list[str],
) -> dict[str, ModelOutputWriter]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path_map = _resolve_output_paths(output_dir, batch_id=batch_id, model_names=model_names)
    return {
        model_name: ModelOutputWriter(model_name, jsonl_path, summary_path)
        for model_name, (jsonl_path, summary_path) in path_map.items()
    }


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def parse_model_names(model_arg: str, models_arg: str | None) -> list[str]:
    if models_arg:
        models = [token.strip() for token in models_arg.split(",") if token.strip()]
        if not models:
            raise ValueError("--models must include at least one model name")
        return _dedupe_preserve_order(models)
    return [model_arg.strip()]


def parse_rpm_limits(rpm_args: list[str] | None, model_names: list[str]) -> dict[str, float]:
    if not rpm_args:
        return {}
    slug_map = {slugify_model_name(name): name for name in model_names}
    exact_map = {name: name for name in model_names}
    resolved: dict[str, float] = {}
    for token in rpm_args:
        if "=" not in token:
            raise ValueError(f"RPM override must be model=value, received '{token}'")
        raw_name, raw_value = token.split("=", 1)
        candidate = raw_name.strip()
        if not candidate:
            raise ValueError(f"RPM override is missing a model name: '{token}'")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"RPM override has invalid numeric value: '{token}'") from exc
        if value <= 0:
            raise ValueError(f"RPM override must be positive: '{token}'")
        resolved_name = exact_map.get(candidate) or slug_map.get(candidate)
        if resolved_name is None:
            raise ValueError(
                f"RPM override references an unknown selected model '{candidate}'. "
                f"Expected one of {sorted(model_names)} or their slug forms."
            )
        resolved[resolved_name] = value
    return resolved


def ensure_ffmpeg_available() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be installed")


def run_analysis_batch(
    *,
    video_folder: Path,
    model_names: list[str],
    output_dir: Path,
    cache_dir: Path,
    refresh_cache: bool,
    fallback_sleep_seconds: float,
    rpm_limits: dict[str, float],
    retry_policy: RetryPolicy,
    api_key: str,
    batch_id: str | None = None,
    system_instructions: str = SYSTEM_INSTRUCTIONS,
    user_prompt: str = USER_PROMPT,
    response_schema: type[BaseModel] = ChairFallsResponseSchema,
) -> dict[str, Any]:
    video_folder = video_folder.resolve()
    output_dir = output_dir.resolve()
    cache_dir = cache_dir.resolve()

    if not video_folder.is_dir():
        raise NotADirectoryError(f"{video_folder} is not a directory")

    videos = discover_videos(video_folder)
    if not videos:
        raise FileNotFoundError(f"No video files found in {video_folder}")

    ensure_ffmpeg_available()

    batch_id = batch_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    prompt_hash = build_prompt_hash(
        system_instructions=system_instructions,
        user_prompt=user_prompt,
    )
    writers = create_output_writers(output_dir, batch_id=batch_id, model_names=model_names)
    limiter = ModelRateLimiter(rpm_limits=rpm_limits, fallback_sleep_seconds=fallback_sleep_seconds)
    client = connect(api_key)
    scratch_dir = Path(tempfile.mkdtemp(prefix="gemini_fall_analysis_"))
    manifest_path = output_dir / f"batch_manifest_{batch_id}.json"

    print(f"Found {len(videos)} videos in {video_folder}")
    print(f"Models: {', '.join(model_names)}")
    print(f"Output dir: {output_dir}")
    print(f"Cache dir:  {cache_dir}")
    print(f"Prompt hash: {prompt_hash}")
    print()

    try:
        for index, video_path in enumerate(videos, start=1):
            print(f"[{index}/{len(videos)}] {video_path.name}", flush=True)
            try:
                cached_video = prepare_cached_video(
                    video_path,
                    cache_dir=cache_dir,
                    scratch_dir=scratch_dir,
                    refresh_cache=refresh_cache,
                )
            except Exception as exc:
                print(f"  preprocess failed: {exc}", flush=True)
                for model_name in model_names:
                    writers[model_name].write_record(
                        build_error_record(
                            video_path=video_path,
                            model_name=model_name,
                            cached_video=None,
                            batch_id=batch_id,
                            prompt_hash=prompt_hash,
                            stage="preprocess",
                            error=exc,
                        )
                    )
                continue

            print(
                f"  cache={'hit' if cached_video.cache_hit else 'miss'} "
                f"cached_video={cached_video.cached_video.name}",
                flush=True,
            )
            records = process_video_across_models(
                client=client,
                video_path=video_path,
                cached_video=cached_video,
                model_names=model_names,
                batch_id=batch_id,
                prompt_hash=prompt_hash,
                rate_limiter=limiter,
                retry_policy=retry_policy,
                system_instructions=system_instructions,
                user_prompt=user_prompt,
                response_schema=response_schema,
            )
            for record in records:
                writers[record["model"]].write_record(record)

            parts: list[str] = []
            for record in records:
                if "error" in record:
                    parts.append(f"{record['model']}=FAILED[{record['error_stage']}]")
                else:
                    timing = record["timing"]
                    tokens = record["usage"].get("total_token_count", "?")
                    parts.append(
                        f"{record['model']}=ok(total={timing['total_seconds']}s,tokens={tokens})"
                    )
            print("  " + ", ".join(parts), flush=True)
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    for writer in writers.values():
        writer.finalize()

    manifest = {
        "batch_id": batch_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "video_folder": str(video_folder),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "prompt_hash": prompt_hash,
        "models": model_names,
        "rpm_limits": rpm_limits,
        "fallback_sleep_seconds": fallback_sleep_seconds,
        "retry_max_attempts": retry_policy.max_attempts,
        "retry_base_delay_seconds": retry_policy.base_delay_seconds,
        "video_count": len(videos),
        "artifacts": {
            model_name: {
                "results_jsonl": str(writer.jsonl_path),
                "summary_csv": str(writer.summary_path),
            }
            for model_name, writer in writers.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)

    print()
    for model_name in model_names:
        writer = writers[model_name]
        print(f"{model_name}:")
        print(f"  Results: {writer.jsonl_path}")
        print(f"  Summary: {writer.summary_path}")
    print(f"Manifest: {manifest_path}")

    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gemini fall analysis on a folder of videos.",
    )
    parser.add_argument(
        "video_folder",
        type=Path,
        help="Path to folder containing video files.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_SINGLE_MODEL,
        help=f"Single Gemini model name (default: {DEFAULT_SINGLE_MODEL}).",
    )
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated Gemini model list. "
            f"Recommended comparison set: {', '.join(DEFAULT_BATCH_MODELS)}."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Persistent cache directory for slowed videos (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force re-encoding even when a cached 1 fps artifact exists.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Fallback seconds to sleep between calls for models without explicit RPM overrides.",
    )
    parser.add_argument(
        "--rpm",
        action="append",
        default=None,
        help="Per-model rate limit override as model=value. May be repeated. Slug forms are accepted.",
    )
    parser.add_argument(
        "--retry-max",
        type=int,
        default=3,
        help="Maximum attempts for transient Gemini API failures (default: 3).",
    )
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=5.0,
        help="Base seconds for exponential backoff on transient Gemini API failures (default: 5).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
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
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
