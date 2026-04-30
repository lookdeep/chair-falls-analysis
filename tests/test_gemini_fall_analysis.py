# ruff: noqa: E402

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import ld_chair_falls.gemini_fall_analysis as gfa


def test_parse_args_uses_external_documents_cache_by_default() -> None:
    args = gfa.parse_args(["/tmp/videos"])

    assert args.cache_dir == Path.home() / "Documents" / "chair-falls-risk" / "gemini_cache"


def test_prepare_cached_video_reuses_existing_artifact(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "1456_2023-05-04_23-07.mp4"
    source.write_bytes(b"video")
    cache_dir = tmp_path / "cache"
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    calls = {"slow": 0, "subs": 0}

    def fake_subtitles(video_path: Path, tmp_dir: Path) -> list[str]:
        assert video_path == source
        assert tmp_dir == scratch_dir
        calls["subs"] += 1
        return ["10:00:00", "10:00:01"]

    def fake_slow(src: Path, dst: Path) -> Path:
        assert src == source
        calls["slow"] += 1
        dst.write_bytes(b"slowed")
        return dst

    monkeypatch.setattr(gfa, "extract_subtitle_timestamps", fake_subtitles)
    monkeypatch.setattr(gfa, "slow_down_to_1fps", fake_slow)

    first = gfa.prepare_cached_video(
        source,
        cache_dir=cache_dir,
        scratch_dir=scratch_dir,
    )
    second = gfa.prepare_cached_video(
        source,
        cache_dir=cache_dir,
        scratch_dir=scratch_dir,
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.cached_video == second.cached_video
    assert second.subtitle_timestamps == ["10:00:00", "10:00:01"]
    assert calls == {"slow": 1, "subs": 1}


def test_prepare_cached_video_rebuilds_when_metadata_is_corrupt(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "1456_2023-05-04_23-07.mp4"
    source.write_bytes(b"video")
    cache_dir = tmp_path / "cache"
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    slow_calls = 0

    def fake_subtitles(video_path: Path, tmp_dir: Path) -> list[str]:
        return ["10:00:00"]

    def fake_slow(src: Path, dst: Path) -> Path:
        nonlocal slow_calls
        slow_calls += 1
        dst.write_bytes(f"slowed-{slow_calls}".encode())
        return dst

    monkeypatch.setattr(gfa, "extract_subtitle_timestamps", fake_subtitles)
    monkeypatch.setattr(gfa, "slow_down_to_1fps", fake_slow)

    first = gfa.prepare_cached_video(
        source,
        cache_dir=cache_dir,
        scratch_dir=scratch_dir,
    )
    metadata_path = next(cache_dir.glob("*.json"))
    metadata_path.write_text("{bad json", encoding="utf-8")

    second = gfa.prepare_cached_video(
        source,
        cache_dir=cache_dir,
        scratch_dir=scratch_dir,
    )

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert slow_calls == 2
    assert second.cached_video.read_bytes() == b"slowed-2"


def test_prepare_cached_video_rebuilds_when_source_fingerprint_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "1456_2023-05-04_23-07.mp4"
    source.write_bytes(b"video")
    cache_dir = tmp_path / "cache"
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    slow_calls = 0

    def fake_subtitles(video_path: Path, tmp_dir: Path) -> list[str]:
        return ["10:00:00"]

    def fake_slow(src: Path, dst: Path) -> Path:
        nonlocal slow_calls
        slow_calls += 1
        dst.write_bytes(f"slowed-{slow_calls}".encode())
        return dst

    monkeypatch.setattr(gfa, "extract_subtitle_timestamps", fake_subtitles)
    monkeypatch.setattr(gfa, "slow_down_to_1fps", fake_slow)

    gfa.prepare_cached_video(
        source,
        cache_dir=cache_dir,
        scratch_dir=scratch_dir,
    )

    source.write_bytes(b"video-updated")

    refreshed = gfa.prepare_cached_video(
        source,
        cache_dir=cache_dir,
        scratch_dir=scratch_dir,
    )

    assert refreshed.cache_hit is False
    assert slow_calls == 2
    assert refreshed.cached_video.read_bytes() == b"slowed-2"


def test_prepare_cached_video_uses_mp4_temp_output(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "1456_2023-05-04_23-07.mp4"
    source.write_bytes(b"video")
    cache_dir = tmp_path / "cache"
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    seen_dst: Path | None = None

    def fake_subtitles(video_path: Path, tmp_dir: Path) -> list[str]:
        return []

    def fake_slow(src: Path, dst: Path) -> Path:
        nonlocal seen_dst
        seen_dst = dst
        dst.write_bytes(b"slowed")
        return dst

    monkeypatch.setattr(gfa, "extract_subtitle_timestamps", fake_subtitles)
    monkeypatch.setattr(gfa, "slow_down_to_1fps", fake_slow)

    gfa.prepare_cached_video(
        source,
        cache_dir=cache_dir,
        scratch_dir=scratch_dir,
    )

    assert seen_dst is not None
    assert seen_dst.suffix == ".mp4"
    assert seen_dst.name.endswith(".part.mp4")


def test_slow_down_to_1fps_explicitly_sets_mp4_muxer(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    target = tmp_path / "target.mp4"
    source.write_bytes(b"video")

    recorded_cmd: list[str] = []

    def fake_get_video_fps(path: Path) -> float:
        assert path == source
        return 6.0

    def fake_run(cmd: list[str], *, capture_output: bool, check: bool, text: bool = False):
        nonlocal recorded_cmd
        recorded_cmd = cmd
        assert capture_output is True
        assert check is True
        assert text is False
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(gfa, "_get_video_fps", fake_get_video_fps)
    monkeypatch.setattr(gfa.subprocess, "run", fake_run)

    result = gfa.slow_down_to_1fps(source, target)

    assert result == target
    assert recorded_cmd[-3:] == ["-f", "mp4", str(target)]


def test_process_video_across_models_reuses_upload_and_continues_after_model_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "1456_2023-05-04_23-07.mp4"
    source.write_bytes(b"video")
    cached = tmp_path / "1456_2023-05-04_23-07.cached.mp4"
    cached.write_bytes(b"cached")
    artifact = gfa.CachedVideoArtifact(
        source_video=source,
        cached_video=cached,
        cache_key="cache1234",
        subtitle_timestamps=["10:00:00", "10:00:01", "10:00:02"],
        cache_hit=True,
    )

    upload_calls: list[str] = []
    deleted_names: list[str] = []
    requested_models: list[str] = []

    class FakeUsage:
        prompt_token_count = 11
        candidates_token_count = 12
        total_token_count = 23

    class FakeResult:
        def __init__(self, payload: str) -> None:
            self.text = payload
            self.usage_metadata = FakeUsage()

    class FakeFiles:
        def delete(self, *, name: str) -> None:
            deleted_names.append(name)

    class FakeClient:
        def __init__(self) -> None:
            self.files = FakeFiles()

    def fake_upload_and_activate(
        client: object,
        file_path: str,
        *,
        retry_policy: gfa.RetryPolicy,
        max_wait: int = 300,
        poll_interval: int = 5,
    ) -> object:
        assert retry_policy.max_attempts == 1
        assert max_wait == 300
        assert poll_interval == 5
        upload_calls.append(file_path)
        return type("UploadedVideo", (), {"name": "remote-video"})()

    def fake_call_gemini(
        client: object,
        video_file: object,
        model_name: str,
        *,
        system_instructions: str = "",
        user_prompt: str = "",
        response_schema: object = None,
    ) -> FakeResult:
        requested_models.append(model_name)
        assert video_file.name == "remote-video"
        assert system_instructions
        assert user_prompt
        assert response_schema is not None
        if model_name == "gemini-2.5-pro":
            raise RuntimeError("429 resource exhausted")
        return FakeResult(
            '{"fall_event":{"fall_detected":true,"fall_time_seconds":1,"staff_entry_time_seconds":2,"fall_tags":["slip"]},'
            '"staff_response_time_seconds":1,"confidence":"high","report":"ok"}'
        )

    monkeypatch.setattr(gfa, "upload_and_activate", fake_upload_and_activate)
    monkeypatch.setattr(gfa, "call_gemini", fake_call_gemini)

    records = gfa.process_video_across_models(
        client=FakeClient(),
        video_path=source,
        cached_video=artifact,
        model_names=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.1-pro-preview"],
        batch_id="20260313T190000Z",
        prompt_hash="prompt1234",
        rate_limiter=gfa.ModelRateLimiter(rpm_limits={}, fallback_sleep_seconds=0.0),
        retry_policy=gfa.RetryPolicy(max_attempts=1, base_delay_seconds=0.0),
    )

    assert upload_calls == [str(cached)]
    assert requested_models == [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3.1-pro-preview",
    ]
    assert deleted_names == ["remote-video"]
    assert [record["model"] for record in records] == requested_models
    assert "error" not in records[0]
    assert records[0]["result"]["fall_event"]["fall_time_wallclock"] == "10:00:01"
    assert records[1]["error_stage"] == "generate"
    assert "resource exhausted" in records[1]["error"]
    assert "error" not in records[2]


def test_parse_rpm_limits_accepts_slug_aliases() -> None:
    limits = gfa.parse_rpm_limits(
        ["gemini-2-5-flash=12", "gemini-3-1-pro-preview=4"],
        ["gemini-2.5-flash", "gemini-3.1-pro-preview"],
    )

    assert limits == {
        "gemini-2.5-flash": 12.0,
        "gemini-3.1-pro-preview": 4.0,
    }


def test_create_output_writers_preserves_legacy_single_model_names(tmp_path: Path) -> None:
    single = gfa.create_output_writers(
        tmp_path / "single",
        batch_id="20260313T190000Z",
        model_names=["gemini-2.5-flash"],
    )
    multi = gfa.create_output_writers(
        tmp_path / "multi",
        batch_id="20260313T190000Z",
        model_names=["gemini-2.5-flash", "gemini-2.5-pro"],
    )

    assert single["gemini-2.5-flash"].jsonl_path.name == "results_20260313T190000Z.jsonl"
    assert single["gemini-2.5-flash"].summary_path.name == "summary_20260313T190000Z.csv"
    assert (
        multi["gemini-2.5-flash"].jsonl_path.name
        == "results_20260313T190000Z_gemini-2-5-flash.jsonl"
    )
    assert (
        multi["gemini-2.5-pro"].summary_path.name
        == "summary_20260313T190000Z_gemini-2-5-pro.csv"
    )
