from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import AnchorEvent, DanmakuMessage, DetailedTimeline, TranscriptSegment


@dataclass(slots=True)
class AnalysisInputs:
    occupant_history_path: Path
    replay_video_path: Path
    workspace_dir: Path
    danmaku_path: Path | None = None
    room_id: str | None = None


class Downloader(Protocol):
    def prepare_inputs(self, inputs: AnalysisInputs) -> AnalysisInputs:
        """Prepare local artifacts before analysis."""


class Transcriber(Protocol):
    async def extract_audio_async(self, video_path: Path, audio_path: Path) -> Path:
        """Extract mono 16k WAV with ffmpeg."""

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        """Return transcript segments with start/end timestamps."""


class AgentAnalyst(Protocol):
    def analyze_anchor(self, anchor: AnchorEvent, timeline: DetailedTimeline) -> dict:
        """Classify the likely cause for an anchor event."""


class LiveDanmakuCollector(Protocol):
    async def run(self, stop_event: asyncio.Event) -> Path:
        """Persist live danmaku history for downstream analysis."""


class DanmakuSourceAdapter(Protocol):
    async def fetch_messages(self, room_id: str, cursor: str | None) -> tuple[list[DanmakuMessage], str | None]:
        """Fetch incremental danmaku messages from a live room."""
