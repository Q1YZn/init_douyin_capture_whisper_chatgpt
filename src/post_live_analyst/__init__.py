"""Core package for replay post-live analysis."""

from .agent_analyst import AnchorAnalysisResult, CliProxyAgentAnalyst, CliProxyConfig
from .danmaku_collector import DanmakuCaptureConfig, LiveDanmakuCollector
from .data_aligner import AlignmentConfig, DataAligner
from .diarizer import DiarizationError, PyannoteSpeakerDiarizer, assign_speakers_to_transcript
from .downloader import F2ReplayDownloader, LocalReplayDownloader, PreparedInputs
from .models import (
    AlignedSegment,
    AnchorEvent,
    DanmakuMessage,
    DanmakuStats,
    DetailedTimeline,
    OccupancyStats,
    SpeakerTurn,
    TranscriptSegment,
)
from .transcriber import AudioExtractionError, FFmpegAudioExtractor, FasterWhisperTranscriber

__all__ = [
    "AlignedSegment",
    "AlignmentConfig",
    "AnchorAnalysisResult",
    "AnchorEvent",
    "AudioExtractionError",
    "CliProxyAgentAnalyst",
    "CliProxyConfig",
    "DanmakuCaptureConfig",
    "DanmakuMessage",
    "DanmakuStats",
    "DataAligner",
    "DiarizationError",
    "DetailedTimeline",
    "F2ReplayDownloader",
    "FFmpegAudioExtractor",
    "FasterWhisperTranscriber",
    "LocalReplayDownloader",
    "PreparedInputs",
    "LiveDanmakuCollector",
    "OccupancyStats",
    "PyannoteSpeakerDiarizer",
    "SpeakerTurn",
    "TranscriptSegment",
    "assign_speakers_to_transcript",
]
