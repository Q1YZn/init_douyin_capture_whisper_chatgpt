from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


def _default_workspace_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class DesktopSettings:
    app_name: str = "DouyinStreamOps"
    workspace_root: Path = _default_workspace_root()
    sqlite_path: Path = _default_workspace_root() / "client.db"
    ffmpeg_binary: str = os.getenv("FFMPEG_BINARY", "ffmpeg")
    server_base_url: str = os.getenv("SERVER_BASE_URL", "http://127.0.0.1:8000")
    cli_proxy_base_url: str = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8317/v1")
    cli_proxy_api_key: str = os.getenv("OPENAI_API_KEY", "your-api-key-1")
    cli_proxy_model_id: str = os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-5.4")
    douyin_cookie: str = os.getenv("DOUYIN_COOKIE", "")
    douyin_user_agent: str = os.getenv(
        "DOUYIN_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    )
    capture_poll_interval_seconds: float = float(os.getenv("CAPTURE_POLL_INTERVAL_SECONDS", "1.0"))
    capture_timeout_seconds: float = float(os.getenv("CAPTURE_TIMEOUT_SECONDS", "10"))
    monitor_poll_interval_seconds: float = float(os.getenv("MONITOR_POLL_INTERVAL_SECONDS", "30"))
    oss_offload_threshold_mb: int = int(os.getenv("OSS_OFFLOAD_THRESHOLD_MB", "512"))
    enable_paid_hotspot_analysis: bool = os.getenv("ENABLE_PAID_HOTSPOT_ANALYSIS", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    default_analysis_tier: str = os.getenv("DEFAULT_ANALYSIS_TIER", "standard")

    @classmethod
    def from_env(cls) -> "DesktopSettings":
        defaults = cls()
        workspace_root = Path(os.getenv("DSO_WORKSPACE_ROOT", str(defaults.workspace_root)))
        sqlite_path = Path(os.getenv("DSO_SQLITE_PATH", str(workspace_root / "client.db")))
        ffmpeg_binary = os.getenv("FFMPEG_BINARY") or cls._detect_ffmpeg_binary(defaults.ffmpeg_binary)
        return cls(
            workspace_root=workspace_root,
            sqlite_path=sqlite_path,
            ffmpeg_binary=ffmpeg_binary,
            server_base_url=os.getenv("SERVER_BASE_URL", defaults.server_base_url),
            cli_proxy_base_url=os.getenv("OPENAI_BASE_URL", defaults.cli_proxy_base_url),
            cli_proxy_api_key=os.getenv("OPENAI_API_KEY", defaults.cli_proxy_api_key),
            cli_proxy_model_id=os.getenv("OPENAI_CHAT_MODEL_ID", defaults.cli_proxy_model_id),
            douyin_cookie=os.getenv("DOUYIN_COOKIE", defaults.douyin_cookie),
            douyin_user_agent=os.getenv("DOUYIN_USER_AGENT", defaults.douyin_user_agent),
            capture_poll_interval_seconds=float(
                os.getenv("CAPTURE_POLL_INTERVAL_SECONDS", str(defaults.capture_poll_interval_seconds))
            ),
            capture_timeout_seconds=float(
                os.getenv("CAPTURE_TIMEOUT_SECONDS", str(defaults.capture_timeout_seconds))
            ),
            monitor_poll_interval_seconds=float(
                os.getenv("MONITOR_POLL_INTERVAL_SECONDS", str(defaults.monitor_poll_interval_seconds))
            ),
            oss_offload_threshold_mb=int(os.getenv("OSS_OFFLOAD_THRESHOLD_MB", str(defaults.oss_offload_threshold_mb))),
            enable_paid_hotspot_analysis=os.getenv(
                "ENABLE_PAID_HOTSPOT_ANALYSIS",
                "1" if defaults.enable_paid_hotspot_analysis else "0",
            ).lower()
            in {"1", "true", "yes", "on"},
            default_analysis_tier=os.getenv("DEFAULT_ANALYSIS_TIER", defaults.default_analysis_tier),
        )

    @staticmethod
    def _detect_ffmpeg_binary(default_binary: str) -> str:
        if shutil.which(default_binary):
            return default_binary

        candidates = [
            Path.home()
            / "AppData"
            / "Local"
            / "Microsoft"
            / "WinGet"
            / "Packages"
            / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
            / "ffmpeg-8.1-full_build"
            / "bin"
            / "ffmpeg.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return default_binary
