from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int_clamped(name: str, default: int, minimum: int, maximum: int) -> int:
    return min(maximum, max(minimum, _env_int(name, default)))


@dataclass(slots=True)
class ServerSettings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./cloud_api.db")
    redis_url: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    report_root: Path = Path(os.getenv("REPORT_ROOT", "./server_reports"))
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model_id: str = os.getenv("DEEPSEEK_MODEL_ID", "deepseek-chat")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_max_anchors: int = _env_int("DEEPSEEK_MAX_ANCHORS", 40)
    deepseek_timeout_seconds: int = _env_int("DEEPSEEK_TIMEOUT_SECONDS", 120)
    deepseek_anchor_concurrency: int = _env_int_clamped("DEEPSEEK_ANCHOR_CONCURRENCY", 20, 1, 50)
    report_job_timeout_seconds: int = _env_int("REPORT_JOB_TIMEOUT_SECONDS", 3600)

    @classmethod
    def from_env(cls) -> "ServerSettings":
        return cls(
            database_url=os.getenv("DATABASE_URL", "sqlite:///./cloud_api.db"),
            redis_url=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
            report_root=Path(os.getenv("REPORT_ROOT", "./server_reports")),
            public_base_url=os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            deepseek_model_id=os.getenv("DEEPSEEK_MODEL_ID", "deepseek-chat"),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_max_anchors=_env_int("DEEPSEEK_MAX_ANCHORS", 40),
            deepseek_timeout_seconds=_env_int("DEEPSEEK_TIMEOUT_SECONDS", 120),
            deepseek_anchor_concurrency=_env_int_clamped("DEEPSEEK_ANCHOR_CONCURRENCY", 20, 1, 50),
            report_job_timeout_seconds=_env_int("REPORT_JOB_TIMEOUT_SECONDS", 3600),
        )
