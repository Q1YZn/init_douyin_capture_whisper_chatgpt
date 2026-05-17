from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    language: str | None = None
    speaker_id: str | None = None
    speaker_confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SpeakerTurn:
    start: float
    end: float
    speaker_id: str
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DanmakuMessage:
    timestamp: float
    text: str
    message_id: str | None = None
    user_id: str | None = None
    user_name: str | None = None
    sentiment_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OccupancyStats:
    window_start: float
    window_end: float
    sample_count: int
    mean: float | None
    peak: int | None
    minimum: int | None
    delta: float | None
    pct_change: float | None
    slope_per_sec: float | None
    slope_per_min: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DanmakuStats:
    window_start: float
    window_end: float
    message_count: int
    unique_user_count: int
    avg_sentiment: float | None
    positive_count: int
    negative_count: int
    neutral_count: int
    sample_messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnchorEvent:
    anchor_id: str
    timestamp: float
    window_start: float
    window_end: float
    direction: str
    abs_change: int
    pct_change: float
    baseline_count: int
    target_count: int
    slope_per_min: float | None = None
    reason_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AlignedSegment:
    segment_id: str
    start: float
    end: float
    text: str
    occupancy: OccupancyStats
    danmaku: DanmakuStats | None = None
    nearby_anchor_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(slots=True)
class DetailedTimeline:
    aligned_segments: list[AlignedSegment]
    anchors: list[AnchorEvent]
    coverage: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "aligned_segments": [segment.to_dict() for segment in self.aligned_segments],
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "coverage": self.coverage,
        }
