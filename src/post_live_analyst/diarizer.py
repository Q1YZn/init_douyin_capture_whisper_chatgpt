from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import TranscriptSegment


class DiarizationError(RuntimeError):
    """Raised when diarization cannot be completed."""


@dataclass(slots=True)
class SpeakerTurn:
    start: float
    end: float
    speaker_id: str


class PyannoteSpeakerDiarizer:
    """
    pyannote.audio-based speaker diarizer.

    The model_name can be a local model path or a Hugging Face model id.
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        auth_token: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.auth_token = auth_token
        self.device = device

    def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise DiarizationError("pyannote.audio is not installed in the current environment.") from exc

        try:
            pipeline = Pipeline.from_pretrained(self.model_name, use_auth_token=self.auth_token)
        except Exception as exc:  # pragma: no cover - depends on model availability
            raise DiarizationError(f"Failed to load pyannote pipeline '{self.model_name}': {exc}") from exc

        try:
            import torch

            if self.device == "cuda" and torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
            else:
                pipeline.to(torch.device("cpu"))
        except Exception:
            # Device selection should not fail the whole diarization setup.
            pass

        try:
            diarization = pipeline(str(audio_path))
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise DiarizationError(f"pyannote diarization failed: {exc}") from exc

        turns: list[SpeakerTurn] = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(
                SpeakerTurn(
                    start=float(segment.start),
                    end=float(segment.end),
                    speaker_id=str(speaker),
                )
            )
        return turns


def assign_speakers_to_transcript(
    transcript: list[TranscriptSegment],
    turns: Iterable[SpeakerTurn],
    min_overlap_ratio: float = 0.3,
) -> list[TranscriptSegment]:
    turn_list = list(turns)
    if not transcript or not turn_list:
        return transcript

    for segment in transcript:
        seg_duration = max(segment.end - segment.start, 0.0)
        if seg_duration <= 0:
            continue

        best_turn: SpeakerTurn | None = None
        best_overlap = 0.0
        for turn in turn_list:
            overlap = _overlap_seconds(segment.start, segment.end, turn.start, turn.end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = turn

        if best_turn is None:
            continue
        if best_overlap / seg_duration < min_overlap_ratio:
            continue

        segment.speaker_id = best_turn.speaker_id
        segment.metadata.setdefault("speaker_overlap_seconds", round(best_overlap, 3))

    return transcript


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))
