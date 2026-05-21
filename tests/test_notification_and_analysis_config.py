from __future__ import annotations

import json
from pathlib import Path

import pytest

import desktop_client.local_pipeline as local_pipeline
import desktop_client.uploader as uploader_module
from desktop_client import services
from desktop_client.config import DesktopSettings
from desktop_client.db import ClientJobRepository
from desktop_client.following_sync import DouyinFollowingFetcher, FollowingFetchResult, FollowingSyncError
from desktop_client.models import FollowingProfile, MonitorProfileSource
from desktop_client.models import DesktopAnalysisRequest
from desktop_client.notifications import MemoryPressureWatcher, RobotNotificationConfig, RobotNotifier
from desktop_client.services import MonitorProfileService
from desktop_client.uploader import CloudUploader
from post_live_analyst.transcriber import FasterWhisperTranscriber


def test_memory_pressure_watcher_honors_threshold_and_cooldown() -> None:
    values = iter([1024, 1024, 4096])
    watcher = MemoryPressureWatcher(get_available_memory_mb=lambda: next(values))
    config = RobotNotificationConfig(
        provider="feishu",
        webhook_url="https://example.test/webhook",
        memory_threshold_mb=2048,
        cooldown_minutes=30,
    )

    should_alert, available = watcher.should_alert(config)
    assert should_alert is True
    assert available == 1024

    should_alert, available = watcher.should_alert(config)
    assert should_alert is False
    assert available == 1024

    should_alert, available = watcher.should_alert(config)
    assert should_alert is False
    assert available == 4096


def test_robot_notifier_payload_shapes() -> None:
    feishu_payload = RobotNotifier._build_payload("feishu", "hello")
    assert feishu_payload == {
        "msg_type": "text",
        "content": {"text": "hello"},
    }

    dingtalk_payload = RobotNotifier._build_payload("dingtalk", "hello")
    assert dingtalk_payload == {
        "msgtype": "text",
        "text": {"content": "hello"},
    }


def test_analysis_request_accepts_provider_overrides() -> None:
    request = DesktopAnalysisRequest(
        replay_path=Path("replay.flv"),
        occupancy_path=Path("occupants.csv"),
        workspace_dir=Path("jobs"),
        analysis_provider="deepseek",
        analysis_base_url="https://api.deepseek.com",
        analysis_api_key="secret",
        analysis_model_id="deepseek-v4-pro",
    )
    assert request.analysis_provider == "deepseek"
    assert request.analysis_base_url == "https://api.deepseek.com"
    assert request.analysis_api_key == "secret"
    assert request.analysis_model_id == "deepseek-v4-pro"


def test_analysis_service_uses_sidecar_asr_python_when_current_env_lacks_faster_whisper(
    tmp_path,
    monkeypatch,
) -> None:
    sidecar_python = tmp_path / ".venv-asr" / "Scripts" / "python.exe"
    sidecar_python.parent.mkdir(parents=True)
    sidecar_python.write_text("", encoding="utf-8")
    settings = DesktopSettings(workspace_root=tmp_path, sqlite_path=tmp_path / "client.db")
    service = services.DesktopAnalysisService(settings, ClientJobRepository(settings.sqlite_path))

    monkeypatch.setattr(local_pipeline.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(services.DesktopAnalysisService, "_python_has_module", staticmethod(lambda *_args: True))

    resolved = service._resolve_asr_python_executable()

    assert resolved is not None
    assert Path(resolved).parts[-3:] == (".venv-asr", "Scripts", "python.exe")


def test_analysis_service_keeps_current_python_when_faster_whisper_is_installed(tmp_path, monkeypatch) -> None:
    settings = DesktopSettings(workspace_root=tmp_path, sqlite_path=tmp_path / "client.db")
    service = services.DesktopAnalysisService(settings, ClientJobRepository(settings.sqlite_path))

    monkeypatch.setattr(local_pipeline.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(services.DesktopAnalysisService, "_python_has_module", staticmethod(lambda *_args: False))

    assert service._resolve_asr_python_executable() is None


def test_analysis_service_prefers_sidecar_asr_python_over_ui_python(tmp_path, monkeypatch) -> None:
    sidecar_python = tmp_path / ".venv-asr" / "Scripts" / "python.exe"
    sidecar_python.parent.mkdir(parents=True)
    sidecar_python.write_text("", encoding="utf-8")
    settings = DesktopSettings(workspace_root=tmp_path, sqlite_path=tmp_path / "client.db")
    service = services.DesktopAnalysisService(settings, ClientJobRepository(settings.sqlite_path))

    monkeypatch.setattr(local_pipeline.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(services.DesktopAnalysisService, "_python_has_module", staticmethod(lambda *_args: True))

    assert service._resolve_asr_python_executable() == str(sidecar_python)


def test_analysis_service_finds_sidecar_asr_python_when_workspace_is_not_repo_root(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    sidecar_python = repo_root / ".venv-asr" / "Scripts" / "python.exe"
    current_python = repo_root / ".venv-py311" / "Scripts" / "python.exe"
    sidecar_python.parent.mkdir(parents=True)
    current_python.parent.mkdir(parents=True)
    sidecar_python.write_text("", encoding="utf-8")
    current_python.write_text("", encoding="utf-8")

    settings = DesktopSettings(workspace_root=tmp_path / "workdir", sqlite_path=tmp_path / "client.db")
    service = services.DesktopAnalysisService(settings, ClientJobRepository(settings.sqlite_path))

    monkeypatch.setattr(local_pipeline.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setenv("PYTHON_EXECUTABLE_FOR_ASR_DISCOVERY", str(current_python))
    monkeypatch.setattr(services.DesktopAnalysisService, "_python_has_module", staticmethod(lambda *_args: True))

    resolved = service._resolve_asr_python_executable()

    assert resolved is not None
    assert Path(resolved).parts[-3:] == (".venv-asr", "Scripts", "python.exe")


def test_transcriber_discovers_sidecar_asr_python_from_env(tmp_path, monkeypatch) -> None:
    sidecar_python = tmp_path / ".venv-asr" / "Scripts" / "python.exe"
    sidecar_python.parent.mkdir(parents=True)
    sidecar_python.write_text("", encoding="utf-8")

    monkeypatch.setenv("DSO_ASR_PYTHON", str(sidecar_python))
    monkeypatch.setattr(FasterWhisperTranscriber, "_python_has_faster_whisper", staticmethod(lambda _path: True))

    assert FasterWhisperTranscriber()._discover_sidecar_python() == str(sidecar_python)


def test_analysis_service_defaults_to_whole_asr_and_large_chunks(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DSO_ASR_MODE", raising=False)
    monkeypatch.delenv("DSO_ASR_CHUNK_SECONDS", raising=False)
    settings = DesktopSettings(workspace_root=tmp_path, sqlite_path=tmp_path / "client.db")
    service = services.DesktopAnalysisService(settings, ClientJobRepository(settings.sqlite_path))

    assert service._asr_mode() == "whole"
    assert service._asr_chunk_seconds() == 600


def test_analysis_service_auto_asr_mode_switches_by_duration(tmp_path, monkeypatch) -> None:
    settings = DesktopSettings(workspace_root=tmp_path, sqlite_path=tmp_path / "client.db")
    service = services.DesktopAnalysisService(settings, ClientJobRepository(settings.sqlite_path))

    monkeypatch.setenv("DSO_ASR_WHOLE_MAX_SECONDS", "7200")
    monkeypatch.setattr(service, "_probe_duration_seconds", lambda _path: 7201.0)
    assert service._select_asr_mode(Path("replay.flv"), "auto") == "chunked"

    monkeypatch.setattr(service, "_probe_duration_seconds", lambda _path: 300.0)
    assert service._select_asr_mode(Path("replay.flv"), "auto") == "whole"


def test_transcriber_chunk_batch_reuses_loaded_model(monkeypatch, tmp_path) -> None:
    class FakeInnerModel:
        device = "cuda"
        compute_type = "int8_float16"

    class FakeSegment:
        start = 1.0
        end = 2.0
        text = " hello "

    class FakeInfo:
        language = "zh"

    class FakeModel:
        model = FakeInnerModel()

        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, _path: str):
            self.calls += 1
            return [FakeSegment()], FakeInfo()

    fake_model = FakeModel()
    get_model_calls = 0

    def fake_get_model(self):
        nonlocal get_model_calls
        get_model_calls += 1
        return fake_model

    chunk_a = tmp_path / "chunk_00000.wav"
    chunk_b = tmp_path / "chunk_00001.wav"
    chunk_a.write_bytes(b"a")
    chunk_b.write_bytes(b"b")
    monkeypatch.setattr(FasterWhisperTranscriber, "_get_model", fake_get_model)
    monkeypatch.setattr(FasterWhisperTranscriber, "_discover_sidecar_python", lambda _self: None)

    progress: list[tuple[int, int, str]] = []
    transcriber = FasterWhisperTranscriber(device="cuda", compute_type="int8_float16")
    transcript = transcriber.transcribe_chunks(
        [chunk_a, chunk_b],
        600,
        progress_callback=lambda index, total, path: progress.append((index, total, path.name)),
    )

    assert get_model_calls == 1
    assert fake_model.calls == 2
    assert progress == [(1, 2, "chunk_00000.wav"), (2, 2, "chunk_00001.wav")]
    assert [segment.start for segment in transcript] == [1.0, 601.0]
    assert transcriber.last_used_device == "cuda"
    assert transcriber.last_used_compute_type == "int8_float16"


def test_strict_gpu_does_not_fallback_to_cpu(monkeypatch, tmp_path) -> None:
    class BrokenModel:
        def transcribe(self, _path: str):
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"a")
    monkeypatch.setenv("DSO_ASR_STRICT_GPU", "1")
    monkeypatch.setattr(FasterWhisperTranscriber, "_get_model", lambda _self: BrokenModel())
    monkeypatch.setattr(FasterWhisperTranscriber, "_discover_sidecar_python", lambda _self: None)

    transcriber = FasterWhisperTranscriber(device="cuda", compute_type="int8_float16")
    with pytest.raises(RuntimeError, match="cublas64_12"):
        transcriber.transcribe(audio_path)


def _make_analysis_request(tmp_path: Path) -> DesktopAnalysisRequest:
    replay_path = tmp_path / "replay.flv"
    occupancy_path = tmp_path / "occupants.csv"
    replay_path.write_bytes(b"fake video")
    occupancy_path.write_text(
        "elapsed_seconds,occupants\n0,100\n60,120\n120,90\n",
        encoding="utf-8",
    )
    return DesktopAnalysisRequest(
        replay_path=replay_path,
        occupancy_path=occupancy_path,
        workspace_dir=tmp_path / "analysis",
        server_base_url="https://cloud.example.test",
    )


def test_required_asr_failure_marks_job_failed_and_skips_upload(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DSO_REQUIRE_ASR", "1")
    monkeypatch.setattr(local_pipeline.FFmpegAudioExtractor, "is_available", lambda _self: True)
    monkeypatch.setattr(
        services.DesktopAnalysisService,
        "_run_asr",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ASR failed")),
    )
    monkeypatch.setattr(
        CloudUploader,
        "upload_analysis",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("upload should not be called")),
    )

    settings = DesktopSettings(workspace_root=tmp_path, sqlite_path=tmp_path / "client.db")
    repo = ClientJobRepository(settings.sqlite_path)
    result = services.DesktopAnalysisService(settings, repo).run(_make_analysis_request(tmp_path))

    row = repo.list_jobs()[0]
    manifest = json.loads((result.workspace_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert row["status"] == "failed"
    assert row["message"] == "ASR failed; upload skipped"
    assert result.uploaded is False
    assert manifest["status"] == "failed"
    assert manifest["failure_reason"] == "ASR failed; upload skipped"
    assert not (result.workspace_dir / "debug_upload_payload.json").exists()


def test_asr_failure_can_upload_when_require_asr_is_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DSO_REQUIRE_ASR", "0")
    monkeypatch.setattr(local_pipeline.FFmpegAudioExtractor, "is_available", lambda _self: True)
    monkeypatch.setattr(
        services.DesktopAnalysisService,
        "_run_asr",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ASR failed")),
    )
    uploaded_payloads: list[dict] = []

    def fake_upload(_self, payload):
        uploaded_payloads.append(payload)
        return {"job_id": "remote-1", "status": "queued", "report_url": None}

    monkeypatch.setattr(CloudUploader, "upload_analysis", fake_upload)
    monkeypatch.setattr(
        CloudUploader,
        "wait_for_analysis_report",
        lambda *_args, **_kwargs: {"job_id": "remote-1", "status": "completed", "report_url": None},
    )

    settings = DesktopSettings(workspace_root=tmp_path, sqlite_path=tmp_path / "client.db")
    repo = ClientJobRepository(settings.sqlite_path)
    result = services.DesktopAnalysisService(settings, repo).run(_make_analysis_request(tmp_path))

    row = repo.list_jobs()[0]
    assert result.uploaded is True
    assert row["status"] == "completed"
    assert row["message"] == "Uploaded without ASR by DSO_REQUIRE_ASR=0"
    assert uploaded_payloads
    assert "Uploaded without ASR by DSO_REQUIRE_ASR=0." in uploaded_payloads[0]["timeline"]["diagnostics"]


def test_cloud_uploader_debug_files_exclude_authorization_token(tmp_path, monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        text = '{"job_id":"remote-1"}'

        def json(self):
            return {"job_id": "remote-1", "status": "queued"}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(uploader_module.requests, "post", lambda *args, **kwargs: FakeResponse())

    uploader = CloudUploader("https://cloud.example.test", auth_token="secret-token", debug_dir=tmp_path)
    uploader.upload_analysis({"client_job_id": "local-1", "timeline": {"aligned_segments": []}})

    uploader.get_analysis_job = lambda _job_id: {"job_id": "remote-1", "status": "completed", "report_url": None}  # type: ignore[method-assign]
    uploader.wait_for_analysis_report("remote-1", timeout_seconds=1.0, interval_seconds=0.5)

    payload_text = (tmp_path / "debug_upload_payload.json").read_text(encoding="utf-8")
    response_text = (tmp_path / "debug_upload_response.json").read_text(encoding="utf-8")
    poll_text = (tmp_path / "debug_poll_responses.jsonl").read_text(encoding="utf-8")
    assert "secret-token" not in payload_text
    assert "Authorization" not in payload_text
    assert "secret-token" not in response_text
    assert "Authorization" not in response_text
    assert "remote-1" in poll_text


def test_following_sync_adds_updates_and_removes_only_synced_profiles(tmp_path, monkeypatch) -> None:
    settings = DesktopSettings(workspace_root=tmp_path, sqlite_path=tmp_path / "client.db")
    repo = ClientJobRepository(settings.sqlite_path)
    repo.upsert_monitor_profile(
        "https://www.douyin.com/user/manual?from_tab_name=live",
        enabled=True,
        source_type=MonitorProfileSource.MANUAL.value,
    )
    repo.upsert_monitor_profile(
        "https://www.douyin.com/user/old_synced?from_tab_name=live",
        enabled=True,
        source_type=MonitorProfileSource.FOLLOWING_SYNC.value,
        source_account_sec_uid="source",
        source_follow_sec_uid="old_synced",
    )
    repo.upsert_monitor_profile(
        "https://www.douyin.com/user/kept_synced?from_tab_name=live",
        enabled=False,
        source_type=MonitorProfileSource.FOLLOWING_SYNC.value,
        source_account_sec_uid="source",
        source_follow_sec_uid="kept_synced",
    )

    class FakeFetcher:
        def __init__(self, _settings):
            pass

        def fetch(self, _source_account):
            return FollowingFetchResult(
                source_account_sec_uid="source",
                profiles=[
                    FollowingProfile(
                        profile_url="https://www.douyin.com/user/kept_synced?from_tab_name=live",
                        sec_uid="kept_synced",
                        uid="1",
                        nickname="Kept",
                    ),
                    FollowingProfile(
                        profile_url="https://www.douyin.com/user/new_synced?from_tab_name=live",
                        sec_uid="new_synced",
                        uid="2",
                        nickname="New",
                    ),
                ],
            )

    monkeypatch.setattr(services, "DouyinFollowingFetcher", FakeFetcher)
    result = MonitorProfileService(repo, settings).sync_following_profiles("source")

    assert result.added == 1
    assert result.updated == 1
    assert result.removed == 1
    records = {record.profile_url: record for record in repo.list_monitor_profiles()}
    assert "https://www.douyin.com/user/manual?from_tab_name=live" in records
    assert "https://www.douyin.com/user/old_synced?from_tab_name=live" not in records
    assert records["https://www.douyin.com/user/kept_synced?from_tab_name=live"].enabled is False
    assert (
        records["https://www.douyin.com/user/new_synced?from_tab_name=live"].source_type
        == MonitorProfileSource.FOLLOWING_SYNC.value
    )


def test_following_sync_rejects_nonzero_api_status() -> None:
    class FailedPage:
        status_code = 2096
        status_msg = "private account"

    try:
        DouyinFollowingFetcher._raise_for_page_status(FailedPage())
    except FollowingSyncError as exc:
        assert "2096" in str(exc)
    else:
        raise AssertionError("Expected FollowingSyncError for nonzero status")
