from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import uuid
from pathlib import Path
from typing import Any

from post_live_analyst import (
    AlignmentConfig,
    CliProxyAgentAnalyst,
    CliProxyConfig,
    DataAligner,
    FFmpegAudioExtractor,
    FasterWhisperTranscriber,
    LocalReplayDownloader,
)
from post_live_analyst.diarizer import (
    DiarizationError,
    PyannoteSpeakerDiarizer,
    assign_speakers_to_transcript,
)
from post_live_analyst.pipeline import AnalysisInputs

from .config import DesktopSettings
from .db import ClientJobRepository
from .models import DesktopAnalysisRequest, DesktopAnalysisResult
from .uploader import CloudUploader


class DesktopAnalysisService:
    def __init__(self, settings: DesktopSettings, repo: ClientJobRepository) -> None:
        self.settings = settings
        self.repo = repo

    def run(self, request: DesktopAnalysisRequest) -> DesktopAnalysisResult:
        job_id = uuid.uuid4().hex
        workspace_dir = request.workspace_dir / job_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        video_size_bytes = request.replay_path.stat().st_size if request.replay_path.exists() else None
        offload_required = self._should_offload_to_oss(video_size_bytes)
        payment_required = offload_required and self.settings.enable_paid_hotspot_analysis
        analysis_tier = "paid_hotspot" if payment_required else self.settings.default_analysis_tier
        asset_storage = "oss_pending" if offload_required else "inline"
        self.repo.create_job(
            {
                "job_id": job_id,
                "status": "queued",
                "replay_path": str(request.replay_path),
                "occupancy_path": str(request.occupancy_path),
                "danmaku_path": str(request.danmaku_path) if request.danmaku_path else None,
                "workspace_dir": str(workspace_dir),
                "progress": 0.0,
                "message": "Job created",
            }
        )

        downloader = LocalReplayDownloader()
        aligner = DataAligner(AlignmentConfig())
        extractor = FFmpegAudioExtractor(self.settings.ffmpeg_binary)
        model_reference = self._resolve_whisper_model_reference()
        transcriber = FasterWhisperTranscriber(model_size=model_reference)
        uploader = CloudUploader(self.settings.server_base_url)
        analyst = CliProxyAgentAnalyst(
            CliProxyConfig(
                base_url=self.settings.cli_proxy_base_url,
                api_key=self.settings.cli_proxy_api_key,
                model_id=self.settings.cli_proxy_model_id,
            )
        )

        transcript_path: Path | None = None
        transcript_segments: list[dict[str, Any]] = []
        diagnostics: list[str] = [f"ASR model reference: {model_reference}"]
        if video_size_bytes is not None:
            diagnostics.append(f"Replay size: {self._format_size(video_size_bytes)}")
        if offload_required:
            diagnostics.append(
                "Replay exceeds OSS offload threshold; asset metadata is tagged for a future paid hotspot flow."
            )

        self.repo.update_job(job_id, status="running", progress=0.1, message="Preparing local inputs")
        prepared = downloader.prepare_inputs(
            AnalysisInputs(
                occupant_history_path=request.occupancy_path,
                replay_video_path=request.replay_path,
                danmaku_path=request.danmaku_path,
                room_id=request.room_id,
                workspace_dir=workspace_dir,
            )
        )

        audio_path = workspace_dir / "voicetrack.wav"
        if extractor.is_available():
            try:
                self.repo.update_job(job_id, progress=0.25, message="Extracting audio with ffmpeg")
                asyncio.run(extractor.extract_audio_async(prepared.replay_video_path, audio_path))
            except Exception as exc:
                diagnostics.append(f"Audio extraction failed: {exc}")
                self.repo.update_job(job_id, message=f"Audio extraction failed: {exc}")
        else:
            diagnostics.append(f"ffmpeg not found, binary name is {self.settings.ffmpeg_binary!r}.")

        if audio_path.exists():
            try:
                self.repo.update_job(job_id, progress=0.45, message="Running faster-whisper")
                transcript = transcriber.transcribe(audio_path)
                if transcript:
                    diarization_model = os.getenv("PYANNOTE_DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
                    diarization_token = (
                        os.getenv("PYANNOTE_AUTH_TOKEN")
                        or os.getenv("HF_TOKEN")
                        or os.getenv("HUGGINGFACE_HUB_TOKEN")
                    )
                    diarization_device = os.getenv("PYANNOTE_DEVICE", "cpu")
                    if importlib.util.find_spec("pyannote.audio") is None:
                        diagnostics.append("pyannote.audio is not installed; speaker diarization was skipped.")
                    elif not diarization_token:
                        diagnostics.append("PYANNOTE_AUTH_TOKEN/HF_TOKEN is not set; speaker diarization was skipped.")
                    else:
                        try:
                            self.repo.update_job(job_id, progress=0.55, message="Running pyannote diarization")
                            diarizer = PyannoteSpeakerDiarizer(
                                model_name=diarization_model,
                                auth_token=diarization_token,
                                device=diarization_device,
                            )
                            speaker_turns = diarizer.diarize(audio_path)
                            transcript = assign_speakers_to_transcript(transcript, speaker_turns)
                            speaker_count = len({turn.speaker_id for turn in speaker_turns})
                            diagnostics.append(
                                f"Diarization completed: {len(speaker_turns)} turns, {speaker_count} speakers."
                            )
                        except DiarizationError as exc:
                            diagnostics.append(f"Diarization skipped: {exc}")
                        except Exception as exc:
                            diagnostics.append(f"Diarization skipped due to unexpected error: {exc}")
                transcript_path = transcriber.dump_json(transcript, workspace_dir / "transcript.json")
                transcript_segments = [segment.to_dict() for segment in transcript]
            except Exception as exc:
                diagnostics.append(f"ASR skipped: {exc}")
                self.repo.update_job(job_id, message=f"ASR skipped: {exc}")
        else:
            diagnostics.append("voicetrack.wav was not created, so ASR did not run.")

        if importlib.util.find_spec("faster_whisper") is None:
            diagnostics.append("faster-whisper is not installed in the current environment.")

        self.repo.update_job(job_id, progress=0.65, message="Building aligned timeline")
        occupancy_df = aligner.load_occupancy_history(prepared.occupant_history_path)
        danmaku_df = (
            aligner.load_danmaku_history(prepared.danmaku_path)
            if prepared.danmaku_path and prepared.danmaku_path.exists()
            else None
        )
        timeline = aligner.build_timeline(transcript_segments, occupancy_df, danmaku_df=danmaku_df)

        analyses = []
        for anchor in timeline.anchors[:3]:
            related_segments = [
                segment for segment in timeline.aligned_segments if anchor.anchor_id in segment.nearby_anchor_ids
            ]
            analyses.append(analyst.analyze_anchor(anchor, related_segments).to_dict())

        summary_path = workspace_dir / "summary.md"
        timeline_path = workspace_dir / "detailed_timeline.json"
        summary_path.write_text(self._build_markdown_summary(timeline, analyses, diagnostics), encoding="utf-8")

        payload = timeline.to_dict()
        payload["agent_analyses"] = analyses
        payload["job_id"] = job_id
        payload["diagnostics"] = diagnostics
        payload["video_size_bytes"] = video_size_bytes
        payload["asset_storage"] = asset_storage
        payload["asset_uri"] = None
        payload["analysis_tier"] = analysis_tier
        payload["payment_required"] = payment_required
        timeline_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        uploaded = False
        report_url = None
        self.repo.update_job(job_id, progress=0.85, message="Uploading structured results to cloud")
        try:
            response = uploader.upload_analysis(
                {
                    "client_job_id": job_id,
                    "summary_markdown": summary_path.read_text(encoding="utf-8"),
                    "timeline": payload,
                    "video_size_bytes": video_size_bytes,
                    "asset_storage": asset_storage,
                    "asset_uri": None,
                    "analysis_tier": analysis_tier,
                    "payment_required": payment_required,
                }
            )
            uploaded = True
            report_url = response.get("report_url")
            self.repo.update_job(
                job_id,
                remote_job_id=response.get("job_id"),
                report_url=report_url,
                message="Upload completed",
            )
        except Exception as exc:
            self.repo.update_job(job_id, message=f"Upload failed: {exc}")

        self.repo.update_job(job_id, status="completed", progress=1.0, report_url=report_url)
        return DesktopAnalysisResult(
            job_id=job_id,
            workspace_dir=workspace_dir,
            transcript_path=transcript_path,
            timeline_path=timeline_path,
            summary_path=summary_path,
            uploaded=uploaded,
            report_url=report_url,
            asr_model_reference=model_reference,
            asr_device=transcriber.last_used_device,
            asr_compute_type=transcriber.last_used_compute_type,
            video_size_bytes=video_size_bytes,
            analysis_tier=analysis_tier,
            payment_required=payment_required,
            asset_storage=asset_storage,
            asset_uri=None,
            offload_required=offload_required,
            summary_excerpt=self._build_summary_excerpt(timeline, analyses, diagnostics),
        )

    def _resolve_whisper_model_reference(self) -> str:
        explicit = os.getenv("DSO_ASR_MODEL") or os.getenv("FASTER_WHISPER_MODEL")
        if explicit:
            return explicit

        candidates = [
            self.settings.workspace_root / "models" / "faster-whisper-small",
            self.settings.workspace_root / "models" / "Systran--faster-whisper-small",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return "small"

    def _build_markdown_summary(
        self,
        timeline: Any,
        analyses: list[dict[str, Any]],
        diagnostics: list[str],
    ) -> str:
        lines = [
            "# Local Replay Summary",
            "",
            "## Overview",
            "",
            f"- Anchor count: {len(timeline.anchors)}",
            f"- Segment count: {len(timeline.aligned_segments)}",
            f"- Danmaku count: {timeline.coverage.get('danmaku_message_count')}",
            "",
            "## Anchor Analyses",
            "",
        ]
        for analysis in analyses:
            lines.append(f"### {analysis['anchor_id']}")
            lines.append("")
            lines.append(f"- Status: {analysis['status']}")
            if analysis.get("response_text"):
                lines.append(f"- Analysis: {analysis['response_text']}")
            if analysis.get("error"):
                lines.append(f"- Error: {analysis['error']}")
            lines.append("")
        lines.extend(["## Diagnostics", ""])
        if diagnostics:
            for item in diagnostics:
                lines.append(f"- {item}")
        else:
            lines.append("- None")
        lines.append("")
        return "\n".join(lines)

    def _should_offload_to_oss(self, video_size_bytes: int | None) -> bool:
        if video_size_bytes is None:
            return False
        threshold_bytes = max(self.settings.oss_offload_threshold_mb, 0) * 1024 * 1024
        return threshold_bytes > 0 and video_size_bytes >= threshold_bytes

    def _build_summary_excerpt(self, timeline: Any, analyses: list[dict[str, Any]], diagnostics: list[str]) -> str:
        lines = [
            f"Anchors {len(timeline.anchors)}",
            f"Segments {len(timeline.aligned_segments)}",
            f"Analyses {len(analyses)}",
        ]
        if diagnostics:
            lines.append(diagnostics[-1])
        return " | ".join(lines)

    def _format_size(self, size_bytes: int) -> str:
        units = ["B", "KB", "MB", "GB"]
        size = float(size_bytes)
        unit = units[0]
        for unit in units:
            if size < 1024 or unit == units[-1]:
                break
            size /= 1024
        return f"{size:.1f} {unit}"
