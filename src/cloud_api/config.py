from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ServerSettings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./cloud_api.db")
    redis_url: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    report_root: Path = Path(os.getenv("REPORT_ROOT", "./server_reports"))
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000")

    @classmethod
    def from_env(cls) -> "ServerSettings":
        return cls()
