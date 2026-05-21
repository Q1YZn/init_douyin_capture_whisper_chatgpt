from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable

from .models import TranscriptSegment


class AudioExtractionError(RuntimeError):
    """Raised when audio extraction cannot be completed."""


class FFmpegAudioExtractor:
    """Extract 16kHz mono WAV audio from a replay video using ffmpeg."""

    def __init__(self, ffmpeg_binary: str = "ffmpeg") -> None:
        self.ffmpeg_binary = ffmpeg_binary

    def is_available(self) -> bool:
        return shutil.which(self.ffmpeg_binary) is not None

    async def extract_audio_async(self, video_path: Path, audio_path: Path) -> Path:
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        if not self.is_available():
            raise AudioExtractionError(f"ffmpeg binary '{self.ffmpeg_binary}' is not available in PATH.")
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

    async def segment_audio_async(self, video_path: Path, chunks_dir: Path, segment_seconds: int = 600) -> list[Path]:
        video_path = Path(video_path)
        chunks_dir = Path(chunks_dir)
        if not self.is_available():
            raise AudioExtractionError(f"ffmpeg binary '{self.ffmpeg_binary}' is not available in PATH.")
        if not video_path.exists():
            raise FileNotFoundError(f"Replay video not found: {video_path}")

        chunks_dir.mkdir(parents=True, exist_ok=True)
        for old_chunk in chunks_dir.glob("chunk_*.wav"):
            old_chunk.unlink()
        output_pattern = chunks_dir / "chunk_%05d.wav"
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
            "-f",
            "segment",
            "-segment_time",
            str(max(1, int(segment_seconds))),
            "-reset_timestamps",
            "1",
            str(output_pattern),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        chunks = sorted(chunks_dir.glob("chunk_*.wav"))
        if process.returncode != 0 or not chunks:
            message = stderr.decode("utf-8", errors="ignore").strip()
            raise AudioExtractionError(f"ffmpeg segment exited with code {process.returncode}: {message}")
        return chunks


class FasterWhisperTranscriber:
    """Lazy wrapper around faster-whisper with JSON export support."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        python_executable: str | None = None,
        debug_dir: str | Path | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.python_executable = str(python_executable).strip() if python_executable else None
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.last_used_device = device
        self.last_used_compute_type = compute_type
        self._model: object | None = None
        self.runtime_warnings: list[str] = []

    def _write_debug_json(self, file_name: str, payload: object) -> None:
        if not self.debug_dir:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        (self.debug_dir / file_name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _append_debug_jsonl(self, file_name: str, payload: object) -> None:
        if not self.debug_dir:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        with (self.debug_dir / file_name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        return self.transcribe_batch([(Path(audio_path), 0.0)])

    def transcribe_chunks(
        self,
        audio_chunks: list[Path],
        segment_seconds: int,
        progress_callback: Callable[[int, int, Path], None] | None = None,
    ) -> list[TranscriptSegment]:
        inputs = [
            (Path(chunk_path), float(index * max(1, int(segment_seconds))))
            for index, chunk_path in enumerate(audio_chunks)
        ]
        return self.transcribe_batch(inputs, progress_callback=progress_callback)

    def transcribe_batch(
        self,
        audio_inputs: Iterable[tuple[Path, float]],
        progress_callback: Callable[[int, int, Path], None] | None = None,
    ) -> list[TranscriptSegment]:
        inputs = [(Path(audio_path), float(offset_seconds)) for audio_path, offset_seconds in audio_inputs]
        if not inputs:
            return []
        if not self.python_executable:
            self.python_executable = self._discover_sidecar_python()

        if self._should_use_external_python():
            return self._transcribe_batch_external(inputs, progress_callback=progress_callback)

        model = self._get_model()
        transcript: list[TranscriptSegment] = []
        total = len(inputs)
        for index, (audio_path, offset_seconds) in enumerate(inputs, start=1):
            if progress_callback:
                progress_callback(index, total, audio_path)
            try:
                transcript.extend(self._transcribe_audio_with_model(model, audio_path, offset_seconds=offset_seconds))
            except Exception as exc:
                if not self._can_fallback_to_cpu(exc):
                    raise
                self.runtime_warnings.append(f"GPU transcription failed, fell back to CPU/int8: {exc}")
                self._model = self._load_cpu_model()
                model = self._model
                transcript.extend(self._transcribe_audio_with_model(model, audio_path, offset_seconds=offset_seconds))
        transcript.sort(key=lambda segment: (segment.start, segment.end))
        return transcript

    def _transcribe_audio(self, audio_path: Path, *, offset_seconds: float) -> list[TranscriptSegment]:
        return self.transcribe_batch([(Path(audio_path), offset_seconds)])

    def _transcribe_audio_with_model(
        self,
        model: object,
        audio_path: Path,
        *,
        offset_seconds: float,
    ) -> list[TranscriptSegment]:
        segments, info = model.transcribe(str(audio_path))  # type: ignore[attr-defined]
        self.last_used_device = self._detect_runtime_device(model)
        self.last_used_compute_type = self._detect_runtime_compute_type(model)
        transcript: list[TranscriptSegment] = []
        for segment in segments:
            metadata = {"audio_path": str(audio_path)}
            if offset_seconds:
                metadata["chunk_offset_seconds"] = offset_seconds
            transcript.append(
                TranscriptSegment(
                    start=float(segment.start) + offset_seconds,
                    end=float(segment.end) + offset_seconds,
                    text=segment.text.strip(),
                    language=info.language,
                    speaker_id=None,
                    metadata=metadata,
                )
            )
        return transcript

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "faster-whisper is not installed in the current Python environment. "
                f"Python: {sys.executable}. "
                f"Install it with: {sys.executable} -m pip install -r requirements-desktop.txt, "
                "or set DSO_ASR_PYTHON to a Python executable that has faster-whisper installed."
            ) from exc

        try:
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        except Exception as exc:
            if not self._can_fallback_to_cpu(exc):
                raise
            self.runtime_warnings.append(f"GPU model load failed, fell back to CPU/int8: {exc}")
            self._model = self._load_cpu_model()
        self.last_used_device = self._detect_runtime_device(self._model)
        self.last_used_compute_type = self._detect_runtime_compute_type(self._model)
        return self._model

    def _load_cpu_model(self) -> object:
        from faster_whisper import WhisperModel

        model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        self.last_used_device = self._detect_runtime_device(model)
        self.last_used_compute_type = self._detect_runtime_compute_type(model)
        return model

    def _discover_sidecar_python(self) -> str | None:
        explicit = os.getenv("DSO_ASR_PYTHON") or os.getenv("FASTER_WHISPER_PYTHON")
        candidates: list[Path] = []
        if explicit:
            candidates.append(Path(explicit))

        project_root = Path(__file__).resolve().parents[2]
        candidates.extend(
            [
                project_root / ".venv-asr" / "Scripts" / "python.exe",
                project_root / ".venv-asr" / "bin" / "python",
                Path.cwd() / ".venv-asr" / "Scripts" / "python.exe",
                Path.cwd() / ".venv-asr" / "bin" / "python",
            ]
        )
        executable_path = Path(sys.executable)
        candidates.extend(
            [
                executable_path.parent.parent.parent / ".venv-asr" / "Scripts" / "python.exe",
                executable_path.parent.parent.parent / ".venv-asr" / "bin" / "python",
            ]
        )

        for candidate in dict.fromkeys(candidates):
            if candidate.exists() and self._python_has_faster_whisper(candidate):
                return str(candidate)
        return None

    @staticmethod
    def _python_has_faster_whisper(python_executable: Path) -> bool:
        try:
            process = subprocess.run(
                [
                    str(python_executable),
                    "-c",
                    "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('faster_whisper') else 1)",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            return False
        return process.returncode == 0

    def _should_use_external_python(self) -> bool:
        if not self.python_executable:
            return False
        configured = Path(self.python_executable)
        if not configured.exists():
            return False
        try:
            return configured.resolve() != Path(sys.executable).resolve()
        except OSError:
            return str(configured) != sys.executable

    def _transcribe_audio_external(self, audio_path: Path, *, offset_seconds: float) -> list[TranscriptSegment]:
        return self._transcribe_batch_external([(Path(audio_path), offset_seconds)])

    def _transcribe_batch_external(
        self,
        audio_inputs: list[tuple[Path, float]],
        progress_callback: Callable[[int, int, Path], None] | None = None,
    ) -> list[TranscriptSegment]:
        assert self.python_executable is not None
        src_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(src_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONIOENCODING"] = "utf-8"
        batch_payload = {
            "inputs": [
                {"audio_path": str(audio_path), "offset_seconds": offset_seconds}
                for audio_path, offset_seconds in audio_inputs
            ]
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
            json.dump(batch_payload, handle, ensure_ascii=False)
            batch_path = Path(handle.name)

        command = [
            self.python_executable,
            "-m",
            "post_live_analyst.transcriber",
            "--transcribe-batch-json",
            "--batch-path",
            str(batch_path),
            "--model-size",
            self.model_size,
            "--device",
            self.device,
            "--compute-type",
            self.compute_type,
        ]
        self._write_debug_json(
            "debug_asr_batch_request.json",
            {
                "python": self.python_executable,
                "model_size": self.model_size,
                "device": self.device,
                "compute_type": self.compute_type,
                "command": command,
                "inputs": batch_payload["inputs"],
            },
        )
        result_payload: dict | None = None
        stderr_text = ""
        returncode = 1
        try:
            with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as stderr_handle:
                process = subprocess.Popen(
                    command,
                    cwd=str(src_root.parent),
                    env=env,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=stderr_handle,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._append_debug_jsonl("debug_asr_worker_events.jsonl", event)
                    if event.get("event") == "progress" and progress_callback:
                        progress_callback(
                            int(event.get("index") or 0),
                            int(event.get("total") or len(audio_inputs)),
                            Path(str(event.get("audio_path") or "")),
                        )
                    elif event.get("event") == "result":
                        result_payload = event
                returncode = process.wait()
                stderr_handle.seek(0)
                stderr_text = stderr_handle.read()
        finally:
            with contextlib.suppress(OSError):
                batch_path.unlink()

        if returncode != 0:
            detail = stderr_text.strip() or "no output"
            self._append_debug_jsonl(
                "debug_asr_worker_events.jsonl",
                {
                    "event": "error",
                    "returncode": returncode,
                    "stderr_tail": detail[-2000:],
                },
            )
            raise RuntimeError(
                f"External ASR batch Python failed with code {returncode}. "
                f"Python: {self.python_executable}. Detail: {detail[-2000:]}"
            )
        if result_payload is None:
            self._append_debug_jsonl(
                "debug_asr_worker_events.jsonl",
                {
                    "event": "error",
                    "returncode": returncode,
                    "message": "External ASR batch Python returned no result.",
                },
            )
            raise RuntimeError(f"External ASR batch Python returned no result. Python: {self.python_executable}.")

        self.last_used_device = str(result_payload.get("device") or self.last_used_device)
        self.last_used_compute_type = str(result_payload.get("compute_type") or self.last_used_compute_type)
        self.runtime_warnings.extend(str(item) for item in result_payload.get("warnings") or [])
        return [TranscriptSegment(**segment) for segment in result_payload.get("segments") or []]

    def dump_json(self, transcript: list[TranscriptSegment], output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [segment.to_dict() for segment in transcript]
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    def _requested_cpu(self) -> bool:
        return self.device.strip().lower() == "cpu"

    def _requested_cuda(self) -> bool:
        return self.device.strip().lower() in {"cuda", "gpu"}

    def _strict_gpu(self) -> bool:
        return os.getenv("DSO_ASR_STRICT_GPU", "0").strip().lower() in {"1", "true", "yes", "on"}

    def _can_fallback_to_cpu(self, exc: Exception) -> bool:
        if self._requested_cpu():
            return False
        if self._requested_cuda() and self._strict_gpu():
            return False
        return self._should_fallback_to_cpu(exc)

    @staticmethod
    def _should_fallback_to_cpu(exc: Exception) -> bool:
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


def _emit_json_line(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _load_batch_inputs(batch_path: Path) -> list[tuple[Path, float]]:
    payload = json.loads(batch_path.read_text(encoding="utf-8"))
    inputs = []
    for item in payload.get("inputs") or []:
        inputs.append((Path(str(item["audio_path"])), float(item.get("offset_seconds") or 0.0)))
    return inputs


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcribe-json", action="store_true")
    parser.add_argument("--transcribe-batch-json", action="store_true")
    parser.add_argument("--audio-path")
    parser.add_argument("--batch-path")
    parser.add_argument("--offset-seconds", type=float, default=0.0)
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="auto")
    args = parser.parse_args(argv)

    transcriber = FasterWhisperTranscriber(
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )

    if args.transcribe_batch_json:
        if not args.batch_path:
            parser.error("--batch-path is required with --transcribe-batch-json")
        inputs = _load_batch_inputs(Path(args.batch_path))

        def on_progress(index: int, total: int, audio_path: Path) -> None:
            _emit_json_line(
                {
                    "event": "progress",
                    "index": index,
                    "total": total,
                    "audio_path": str(audio_path),
                }
            )

        segments = transcriber.transcribe_batch(inputs, progress_callback=on_progress)
        _emit_json_line(
            {
                "event": "result",
                "segments": [segment.to_dict() for segment in segments],
                "device": transcriber.last_used_device,
                "compute_type": transcriber.last_used_compute_type,
                "warnings": transcriber.runtime_warnings,
            }
        )
        return 0

    if not args.transcribe_json:
        parser.error("--transcribe-json or --transcribe-batch-json is required")
    if not args.audio_path:
        parser.error("--audio-path is required with --transcribe-json")

    segments = transcriber._transcribe_audio(Path(args.audio_path), offset_seconds=args.offset_seconds)
    print(
        json.dumps(
            {
                "segments": [segment.to_dict() for segment in segments],
                "device": transcriber.last_used_device,
                "compute_type": transcriber.last_used_compute_type,
                "warnings": transcriber.runtime_warnings,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
