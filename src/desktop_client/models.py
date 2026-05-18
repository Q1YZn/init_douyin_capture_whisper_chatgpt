from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class WorkflowStage(StrEnum):
    IDLE = "idle"
    CAPTURE = "capture"
    MONITOR = "monitor"
    ANALYSIS = "analysis"
    UPLOAD = "upload"
    COMPLETED = "completed"


class WorkflowStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    STOPPED = "stopped"


class MonitorEventSeverity(StrEnum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass(slots=True)
class WorkflowSnapshot:
    title: str
    stage: WorkflowStage
    status: WorkflowStatus
    message: str
    progress: int = 0
    mode_label: str = "Idle"
    outputs: dict[str, str] = field(default_factory=dict)
    summary_lines: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DesktopAnalysisRequest:
    replay_path: Path
    occupancy_path: Path
    workspace_dir: Path
    danmaku_path: Path | None = None
    room_id: str | None = None
    analysis_provider: str | None = None
    analysis_base_url: str | None = None
    analysis_api_key: str | None = None
    analysis_model_id: str | None = None


@dataclass(slots=True)
class DesktopAnalysisResult:
    job_id: str
    workspace_dir: Path
    transcript_path: Path | None
    timeline_path: Path
    summary_path: Path
    uploaded: bool
    report_url: str | None = None
    asr_model_reference: str | None = None
    asr_device: str | None = None
    asr_compute_type: str | None = None
    video_size_bytes: int | None = None
    analysis_tier: str | None = None
    payment_required: bool = False
    asset_storage: str | None = None
    asset_uri: str | None = None
    offload_required: bool = False
    summary_excerpt: str | None = None


@dataclass(slots=True)
class MonitorProfileRecord:
    profile_url: str
    enabled: bool = True
    last_live_at: str | None = None
    last_capture_dir: str | None = None
    last_room_id: str | None = None
    last_status: str | None = None
