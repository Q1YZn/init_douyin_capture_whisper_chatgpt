from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

import requests


class NotificationError(RuntimeError):
    """Raised when a robot notification cannot be delivered."""


@dataclass(slots=True)
class RobotNotificationConfig:
    provider: str = "none"
    webhook_url: str = ""
    memory_threshold_mb: int = 2048
    cooldown_minutes: int = 30

    def enabled(self) -> bool:
        return self.provider in {"feishu", "dingtalk"} and bool(self.webhook_url.strip())


class RobotNotifier:
    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    def send_text(self, config: RobotNotificationConfig, title: str, body: str) -> None:
        if not config.enabled():
            return
        message = f"{title}\n{body}".strip()
        payload = self._build_payload(config.provider, message)
        response = requests.post(
            config.webhook_url.strip(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    @staticmethod
    def _build_payload(provider: str, message: str) -> dict:
        if provider == "feishu":
            return {
                "msg_type": "text",
                "content": {
                    "text": message,
                },
            }
        if provider == "dingtalk":
            return {
                "msgtype": "text",
                "text": {
                    "content": message,
                },
            }
        raise NotificationError(f"Unsupported notification provider: {provider}")


class MemoryPressureWatcher:
    def __init__(
        self,
        get_available_memory_mb: Callable[[], int | None] | None = None,
    ) -> None:
        self.get_available_memory_mb = get_available_memory_mb or self._default_available_memory_mb
        self._last_alert_at: datetime | None = None

    def should_alert(self, config: RobotNotificationConfig) -> tuple[bool, int | None]:
        available_mb = self.get_available_memory_mb()
        if available_mb is None:
            return False, None
        if available_mb > max(config.memory_threshold_mb, 0):
            return False, available_mb
        now = datetime.now()
        if self._last_alert_at is not None:
            elapsed = now - self._last_alert_at
            if elapsed < timedelta(minutes=max(config.cooldown_minutes, 1)):
                return False, available_mb
        self._last_alert_at = now
        return True, available_mb

    @staticmethod
    def _default_available_memory_mb() -> int | None:
        try:
            import psutil
        except ImportError:
            return None
        return int(psutil.virtual_memory().available / (1024 * 1024))
