from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from .models import TranscriptSegment


class AudioExtractionError(RuntimeError):
    """Raised when audio extraction cannot be completed."""


class FFmpegAudioExtractor:
    """Extract a 16kHz mono wav track from a replay video using ffmpeg."""

    def __init__(self, ffmpeg_binary: str = "ffmpeg") -> None:
        self.ffmpeg_binary = ffmpeg_binary

    def is_available(self) -> bool:
        return shutil.which(self.ffmpeg_binary) is not None

    async def extract_audio_async(self, video_path: Path, audio_path: Path) -> Path:
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        if not self.is_available():
            raise AudioExtractionError(
                f"ffmpeg binary '{self.ffmpeg_binary}' is not available in PATH."
            )
        if not video_path.exists():
            raise FileNotFoundError(f"Replay video not found: {video_path}")

        audio_path.parent.mkdir(parents=True, exist_ok=True)
        process = await asyncio.create_subprocess_exec(
            self.ffmpeg_binary,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(audio_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            message = stderr.decode("utf-8", errors="ignore").strip()
            raise AudioExtractionError(f"ffmpeg exited with code {process.returncode}: {message}")
        return audio_path


class FasterWhisperTranscriber:
    """Lazy wrapper around faster-whisper with JSON export support."""

    def __init__(self, model_size: str = "small", device: str = "auto", compute_type: str = "auto") -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.last_used_device = device
        self.last_used_compute_type = compute_type

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "faster-whisper is not installed in the current environment."
            ) from exc

        try:
            model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            segments, info = model.transcribe(str(audio_path))
        except RuntimeError as exc:
            if not self._should_fallback_to_cpu(exc):
                raise
            model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            segments, info = model.transcribe(str(audio_path))
        self.last_used_device = self._detect_runtime_device(model)
        self.last_used_compute_type = self._detect_runtime_compute_type(model)
        transcript: list[TranscriptSegment] = []
        for segment in segments:
            transcript.append(
                TranscriptSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text.strip(),
                    language=info.language,
                    speaker_id=None,  # speaker labels are filled later by diarization.
                )
            )
        return transcript

    def dump_json(self, transcript: list[TranscriptSegment], output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [segment.to_dict() for segment in transcript]
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    @staticmethod
    def _should_fallback_to_cpu(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        fallback_markers = [
            "cublas",
            "cudnn",
            "cuda",
            "libcudnn",
            "libcublas",
        ]
        return any(marker in message for marker in fallback_markers)

    @staticmethod
    def _detect_runtime_device(model: object) -> str:
        inner_model = getattr(model, "model", None)
        device = getattr(inner_model, "device", None)
        if device:
            return str(device)
        return "unknown"

    @staticmethod
    def _detect_runtime_compute_type(model: object) -> str:
        inner_model = getattr(model, "model", None)
        compute_type = getattr(inner_model, "compute_type", None)
        if compute_type:
            return str(compute_type)
        return "unknown"
