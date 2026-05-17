from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import DanmakuMessage


@dataclass(slots=True)
class DanmakuCaptureConfig:
    room_id: str
    output_path: Path
    poll_interval_seconds: float = 1.0
    flush_every: int = 20


class DanmakuSourceAdapter(Protocol):
    async def fetch_messages(self, room_id: str, cursor: str | None) -> tuple[list[DanmakuMessage], str | None]:
        """Fetch new danmaku messages after the given cursor."""


class LiveDanmakuCollector:
    """Collect live danmaku messages and persist them as JSONL for later analysis."""

    def __init__(self, adapter: DanmakuSourceAdapter, config: DanmakuCaptureConfig) -> None:
        self.adapter = adapter
        self.config = config

    async def run(self, stop_event: asyncio.Event) -> Path:
        cursor: str | None = None
        buffer: list[DanmakuMessage] = []
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

        while not stop_event.is_set():
            messages, cursor = await self.adapter.fetch_messages(self.config.room_id, cursor)
            buffer.extend(messages)
            if len(buffer) >= self.config.flush_every:
                self._flush(buffer)
                buffer.clear()
            await asyncio.sleep(self.config.poll_interval_seconds)

        if buffer:
            self._flush(buffer)
        return self.config.output_path

    def _flush(self, messages: list[DanmakuMessage]) -> None:
        with self.config.output_path.open("a", encoding="utf-8") as handle:
            for message in messages:
                handle.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")
