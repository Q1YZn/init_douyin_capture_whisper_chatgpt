from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

from .config import DesktopSettings
from .db import ClientJobRepository
from .f2_runtime import F2RuntimeReport, probe_f2_runtime
from .live_capture import (
    F2LiveCaptureService,
    LiveCaptureRequest,
    LiveCaptureResult,
    MonitorEvent,
    MultiProfileLiveMonitorService,
    MultiProfileMonitorRequest,
)
from .local_pipeline import DesktopAnalysisService
from .models import DesktopAnalysisRequest, DesktopAnalysisResult, MonitorProfileRecord


class MonitorProfileService:
    """Application-facing service for monitor profile CRUD and history updates."""

    def __init__(self, repo: ClientJobRepository) -> None:
        self.repo = repo

    def list_profiles(self) -> list[MonitorProfileRecord]:
        return self.repo.list_monitor_profiles()

    def add_profile(self, profile_url: str) -> MonitorProfileRecord:
        cleaned = profile_url.strip()
        if not cleaned:
            raise ValueError("Profile URL is required.")
        existing = self.repo.get_monitor_profile(cleaned)
        if existing is not None:
            return existing
        self.repo.upsert_monitor_profile(cleaned, enabled=True)
        return self.repo.get_monitor_profile(cleaned) or MonitorProfileRecord(profile_url=cleaned, enabled=True)

    def remove_profiles(self, profile_urls: list[str]) -> int:
        removed = 0
        for profile_url in profile_urls:
            cleaned = profile_url.strip()
            if not cleaned:
                continue
            self.repo.delete_monitor_profile(cleaned)
            removed += 1
        return removed

    def save_profiles(self, profiles: list[MonitorProfileRecord]) -> None:
        self.repo.save_monitor_profiles(profiles)

    def set_enabled(self, profile_url: str, enabled: bool) -> None:
        cleaned = profile_url.strip()
        if not cleaned:
            return
        self.repo.upsert_monitor_profile(cleaned, enabled=enabled)

    def enabled_profiles(self) -> list[MonitorProfileRecord]:
        return [profile for profile in self.list_profiles() if profile.enabled]

    def apply_monitor_event(self, payload: dict) -> MonitorProfileRecord | None:
        event_type = str(payload.get("event_type", "")).strip()
        profile_url = str(payload.get("profile_url", "")).strip()
        if not profile_url:
            return None

        existing = self.repo.get_monitor_profile(profile_url) or MonitorProfileRecord(profile_url=profile_url, enabled=True)
        room_id = self._none_if_empty(payload.get("room_id"))
        workspace_dir = self._none_if_empty(payload.get("workspace_dir"))

        if event_type == "live_detected":
            self.repo.upsert_monitor_profile(
                profile_url,
                enabled=existing.enabled,
                last_live_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                last_room_id=room_id,
                last_status="Live",
            )
        elif event_type == "capture_finished":
            self.repo.upsert_monitor_profile(
                profile_url,
                enabled=existing.enabled,
                last_capture_dir=workspace_dir,
                last_room_id=room_id,
                last_status="Capture finished",
            )
        elif event_type == "capture_failed":
            self.repo.upsert_monitor_profile(
                profile_url,
                enabled=existing.enabled,
                last_room_id=room_id,
                last_status="Capture failed",
            )
        elif event_type == "offline":
            self.repo.upsert_monitor_profile(
                profile_url,
                enabled=existing.enabled,
                last_status="Offline",
            )
        elif event_type == "monitor_error":
            self.repo.upsert_monitor_profile(
                profile_url,
                enabled=existing.enabled,
                last_status="Monitor error",
            )

        return self.repo.get_monitor_profile(profile_url)

    @staticmethod
    def _none_if_empty(value: object) -> str | None:
        text = str(value).strip() if value is not None else ""
        return text or None


class DesktopWorkflowService:
    """Orchestrates capture, monitor and analysis workflows for the desktop UI."""

    def __init__(self, settings: DesktopSettings, repo: ClientJobRepository) -> None:
        self.settings = settings
        self.analysis_service = DesktopAnalysisService(settings, repo)
        self.capture_service = F2LiveCaptureService(settings)
        self.monitor_service = MultiProfileLiveMonitorService(settings)

    def resolve_asr_model_reference(self) -> str:
        return self.analysis_service._resolve_whisper_model_reference()

    def probe_f2_runtime(self) -> F2RuntimeReport:
        return probe_f2_runtime()

    def build_analysis_request(
        self,
        replay_path: str,
        occupancy_path: str,
        danmaku_path: str,
        room_id: str,
        analysis_provider: str | None = None,
        analysis_base_url: str | None = None,
        analysis_api_key: str | None = None,
        analysis_model_id: str | None = None,
    ) -> DesktopAnalysisRequest:
        replay_text = replay_path.strip()
        occupancy_text = occupancy_path.strip()
        replay = Path(replay_text)
        occupancy = Path(occupancy_text)
        danmaku_text = danmaku_path.strip()
        danmaku = Path(danmaku_text) if danmaku_text else None
        room = room_id.strip() or None

        if not replay_text or not occupancy_text or not replay.is_file() or not occupancy.is_file():
            raise ValueError("Select a replay file and occupant_history.csv first.")
        if danmaku is not None and not danmaku.is_file():
            raise ValueError("Danmaku JSONL/CSV must point to an existing file.")

        return DesktopAnalysisRequest(
            replay_path=replay,
            occupancy_path=occupancy,
            danmaku_path=danmaku,
            room_id=room,
            workspace_dir=self.settings.workspace_root / "jobs",
            analysis_provider=analysis_provider.strip() if analysis_provider else None,
            analysis_base_url=analysis_base_url.strip() if analysis_base_url else None,
            analysis_api_key=analysis_api_key.strip() if analysis_api_key else None,
            analysis_model_id=analysis_model_id.strip() if analysis_model_id else None,
        )

    def run_analysis(self, request: DesktopAnalysisRequest) -> DesktopAnalysisResult:
        return self.analysis_service.run(request)

    def build_capture_request(self, room_id_or_url: str) -> LiveCaptureRequest:
        cleaned = room_id_or_url.strip()
        if not cleaned:
            raise ValueError("Fill in a room_id, webcast_id, live URL, or profile URL first.")
        return LiveCaptureRequest(
            room_id_or_url=cleaned,
            workspace_dir=self.settings.workspace_root / "captures",
        )

    def run_capture(self, request: LiveCaptureRequest) -> LiveCaptureResult:
        return self.capture_service.run(request)

    def stop_capture(self) -> None:
        self.capture_service.request_stop()

    def build_monitor_request(self, profile_urls: list[str]) -> MultiProfileMonitorRequest:
        cleaned_urls = [url.strip() for url in profile_urls if url.strip()]
        if not cleaned_urls:
            raise ValueError("Enable at least one profile in the monitor list first.")
        return MultiProfileMonitorRequest(
            profile_urls=cleaned_urls,
            workspace_dir=self.settings.workspace_root / "captures",
            poll_interval_seconds=self.settings.monitor_poll_interval_seconds,
        )

    def run_monitor(
        self,
        request: MultiProfileMonitorRequest,
        event_callback: Callable[[MonitorEvent], None] | None,
    ) -> None:
        self.monitor_service.run(request, event_callback=event_callback)

    def stop_monitor(self) -> None:
        self.monitor_service.request_stop()
