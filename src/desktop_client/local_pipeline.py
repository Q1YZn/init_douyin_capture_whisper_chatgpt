from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from post_live_analyst import (
    AlignmentConfig,
    DataAligner,
    FFmpegAudioExtractor,
    FasterWhisperTranscriber,
    LocalReplayDownloader,
)
from post_live_analyst.pipeline import AnalysisInputs

from .config import DesktopSettings
from .db import ClientJobRepository
from .models import DesktopAnalysisRequest, DesktopAnalysisResult
from .uploader import CloudUploader


def datetime_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


class DesktopAnalysisService:
    def __init__(self, settings: DesktopSettings, repo: ClientJobRepository) -> None:
        self.settings = settings
        self.repo = repo

    def run(
        self,
        request: DesktopAnalysisRequest,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> DesktopAnalysisResult:
        job_id = uuid.uuid4().hex
        workspace_dir = request.workspace_dir / job_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        debug_analysis = _env_flag("DSO_DEBUG_ANALYSIS")
        require_asr = _env_flag("DSO_REQUIRE_ASR", "1")

        def update_progress(progress: float, message: str, **updates: Any) -> None:
            payload = {"progress": progress, "message": message}
            payload.update(updates)
            self.repo.update_job(job_id, **payload)
            if progress_callback:
                progress_callback(progress, message)

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
        asr_python = self._resolve_asr_python_executable()
        asr_mode_config = self._asr_mode()
        asr_chunk_seconds = self._asr_chunk_seconds()
        keep_asr_audio = os.getenv("DSO_KEEP_ASR_AUDIO", os.getenv("DSO_KEEP_AUDIO_CHUNKS", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        transcriber = FasterWhisperTranscriber(
            model_size=model_reference,
            device=os.getenv("DSO_ASR_DEVICE", "auto"),
            compute_type=os.getenv("DSO_ASR_COMPUTE_TYPE", "auto"),
            python_executable=asr_python,
            debug_dir=workspace_dir if debug_analysis else None,
        )
        uploader = CloudUploader(
            request.server_base_url or self.settings.server_base_url,
            auth_token=request.server_auth_token,
            debug_dir=workspace_dir if debug_analysis else None,
        )
        transcript_path: Path | None = None
        transcript_segments: list[dict[str, Any]] = []
        diagnostics: list[str] = [
            f"ASR model reference: {model_reference}",
            f"ASR Python: {asr_python or 'current process'}",
            f"ASR requested device: {transcriber.device}",
            f"ASR requested compute_type: {transcriber.compute_type}",
            f"ASR configured mode: {asr_mode_config}",
            f"ASR required: {require_asr}",
            f"Analysis debug enabled: {debug_analysis}",
            "Client role: local audio extraction, ASR, occupancy anchor detection, timeline build and upload.",
            "Cloud role: prompt templates, provider keys, model selection, anchor explanation and clip strategy.",
        ]
        if video_size_bytes is not None:
            diagnostics.append(f"Replay file size: {self._format_size(video_size_bytes)}")
        if offload_required:
            diagnostics.append("Replay file exceeds cloud asset threshold; only structured results are uploaded.")

        update_progress(0.1, "正在准备本地录播、人数和弹幕文件", status="running")
        prepared = downloader.prepare_inputs(
            AnalysisInputs(
                occupant_history_path=request.occupancy_path,
                replay_video_path=request.replay_path,
                danmaku_path=request.danmaku_path,
                room_id=request.room_id,
                workspace_dir=workspace_dir,
            )
        )

        selected_asr_mode = self._select_asr_mode(prepared.replay_video_path, asr_mode_config)
        diagnostics.append(f"ASR selected mode: {selected_asr_mode}")
        diagnostics.append(f"ASR chunk seconds: {asr_chunk_seconds}")

        if extractor.is_available():
            try:
                transcript = self._run_asr(
                    extractor=extractor,
                    transcriber=transcriber,
                    video_path=prepared.replay_video_path,
                    workspace_dir=workspace_dir,
                    asr_mode=selected_asr_mode,
                    chunk_seconds=asr_chunk_seconds,
                    keep_asr_audio=keep_asr_audio,
                    update_progress=update_progress,
                )
                transcript_path = transcriber.dump_json(transcript, workspace_dir / "transcript.json")
                transcript_segments = [segment.to_dict() for segment in transcript]
                diagnostics.extend(transcriber.runtime_warnings)
                diagnostics.append(f"ASR completed: {len(transcript_segments)} transcript segments.")
            except Exception as exc:
                diagnostics.append(f"ASR not completed: {exc}")
                update_progress(0.45, f"ASR 未完成：{exc}")
        else:
            diagnostics.append(f"ffmpeg not found: configured binary is {self.settings.ffmpeg_binary!r}.")

        if importlib.util.find_spec("faster_whisper") is None:
            diagnostics.append("Current Python environment does not have faster-whisper installed.")

        diagnostics.append(f"ASR runtime device: {transcriber.last_used_device}")
        diagnostics.append(f"ASR runtime compute_type: {transcriber.last_used_compute_type}")

        update_progress(0.65, "正在构建 ASR 与人数曲线对齐 timeline")
        occupancy_df = aligner.load_occupancy_history(prepared.occupant_history_path)
        danmaku_df = (
            aligner.load_danmaku_history(prepared.danmaku_path)
            if prepared.danmaku_path and prepared.danmaku_path.exists()
            else None
        )
        timeline = aligner.build_timeline(transcript_segments, occupancy_df, danmaku_df=danmaku_df)

        analyses: list[dict[str, Any]] = []
        selected_anchors = self._select_anchors_for_analysis(timeline)
        diagnostics.append("Waiting for cloud analysis: the client does not run a local LLM.")

        summary_path = workspace_dir / "summary.md"
        timeline_path = workspace_dir / "detailed_timeline.json"
        manifest_path = workspace_dir / "analysis_manifest.json"

        payload = timeline.to_dict()
        payload["agent_analyses"] = analyses
        payload["analysis_selection"] = {
            "status": "client_structured_selected",
            "strategy": "dynamic_by_duration_score_distribution",
            "max_anchors": self._deepseek_max_anchors(),
            "total_anchor_count": len(timeline.anchors),
            "selected_anchor_count": len(selected_anchors),
            "weak_anchor_count": max(0, len(timeline.anchors) - len(selected_anchors)),
            "selected_anchor_ids": [anchor.anchor_id for anchor in selected_anchors],
            "weak_anchor_ids": [anchor.anchor_id for anchor in timeline.anchors if anchor not in selected_anchors],
        }
        payload["job_id"] = job_id
        payload["diagnostics"] = diagnostics
        payload["video_size_bytes"] = video_size_bytes
        payload["asset_storage"] = asset_storage
        payload["asset_uri"] = None
        payload["analysis_tier"] = analysis_tier
        payload["payment_required"] = payment_required
        payload["source_assets"] = {
            "replay_path": str(prepared.replay_video_path),
            "occupancy_path": str(prepared.occupant_history_path),
            "danmaku_path": str(prepared.danmaku_path) if prepared.danmaku_path else None,
        }
        payload["asr_runtime"] = {
            "configured_mode": asr_mode_config,
            "selected_mode": selected_asr_mode,
            "execution_mode": "batch_external_worker" if asr_python else "in_process",
            "chunk_seconds": asr_chunk_seconds,
            "model_reference": model_reference,
            "python": asr_python or sys.executable,
            "requested_device": transcriber.device,
            "requested_compute_type": transcriber.compute_type,
            "device": transcriber.last_used_device,
            "compute_type": transcriber.last_used_compute_type,
            "strict_gpu": os.getenv("DSO_ASR_STRICT_GPU", "0"),
        }

        aligned_segments = payload.get("aligned_segments") or []
        asr_available = bool(transcript_path and transcript_path.exists() and aligned_segments)
        if require_asr and not asr_available:
            failure_reason = "ASR failed; upload skipped"
            diagnostics.append(
                "ASR required but transcript.json is missing or aligned_segments is empty; cloud upload skipped."
            )
            payload["diagnostics"] = diagnostics
            summary_path.write_text(self._build_markdown_summary(timeline, analyses, diagnostics), encoding="utf-8")
            timeline_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "status": "failed",
                        "failure_reason": failure_reason,
                        "created_at": datetime_now_iso(),
                        "workspace_dir": str(workspace_dir),
                        "source_assets": payload["source_assets"],
                        "summary_path": str(summary_path),
                        "timeline_path": str(timeline_path),
                        "transcript_path": str(transcript_path) if transcript_path else None,
                        "asr_runtime": payload["asr_runtime"],
                        "keep_asr_audio": keep_asr_audio,
                        "diagnostics": diagnostics,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            update_progress(1.0, failure_reason, status="failed", report_url=None)
            return DesktopAnalysisResult(
                job_id=job_id,
                workspace_dir=workspace_dir,
                transcript_path=transcript_path,
                timeline_path=timeline_path,
                summary_path=summary_path,
                uploaded=False,
                report_url=None,
                cloud_report_path=None,
                asr_model_reference=model_reference,
                asr_device=transcriber.last_used_device,
                asr_compute_type=transcriber.last_used_compute_type,
            )

        no_asr_upload_message = "Uploaded without ASR by DSO_REQUIRE_ASR=0"
        if not asr_available:
            diagnostics.append(f"{no_asr_upload_message}.")
            payload["diagnostics"] = diagnostics

        summary_path.write_text(self._build_markdown_summary(timeline, analyses, diagnostics), encoding="utf-8")
        manifest_path.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "status": "local_completed",
                    "created_at": datetime_now_iso(),
                    "workspace_dir": str(workspace_dir),
                    "source_assets": payload["source_assets"],
                    "summary_path": str(summary_path),
                    "timeline_path": str(timeline_path),
                    "transcript_path": str(transcript_path) if transcript_path else None,
                    "asr_runtime": payload["asr_runtime"],
                    "keep_asr_audio": keep_asr_audio,
                    "diagnostics": diagnostics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        timeline_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        uploaded = False
        report_url = None
        cloud_report_path: Path | None = None
        remote_job_id = None
        remote_status = ""
        remote_result: dict[str, Any] | None = None
        cloud_error: str | None = None
        update_progress(0.85, no_asr_upload_message if not asr_available else "Uploading structured analysis to cloud")
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
            remote_result = response
            remote_job_id = response.get("job_id")
            report_url = response.get("report_url")
            remote_status = str(response.get("status") or "")
            if remote_job_id and not report_url:
                update_progress(0.9, f"等待云端报告生成：{remote_job_id}")
                remote_result = uploader.wait_for_analysis_report(str(remote_job_id), timeout_seconds=120.0)
                report_url = remote_result.get("report_url") or report_url
                remote_status = str(remote_result.get("status") or remote_status)
            if report_url:
                update_progress(0.96, "正在下载云端报告")
                cloud_report_path = workspace_dir / "cloud_report.html"
                cloud_report_path.write_text(uploader.fetch_report_html(str(report_url)), encoding="utf-8")
            update_progress(
                0.98,
                "云端报告已生成" if report_url else f"已提交云端，当前状态：{remote_status}",
                remote_job_id=remote_job_id,
                report_url=report_url,
            )
        except Exception as exc:
            cloud_error = str(exc)
            diagnostics.append(f"Cloud upload/report failed: {exc}")
            update_progress(0.92, f"提交云端失败：{exc}")

        cloud_pending_statuses = {"queued", "running", "pending_worker"}
        cloud_status_lower = remote_status.strip().lower()
        if uploaded and remote_job_id and not report_url and cloud_status_lower in cloud_pending_statuses:
            final_message = "ASR completed; cloud report pending"
        elif uploaded and remote_job_id and not report_url and cloud_status_lower == "failed":
            final_message = "ASR completed; cloud report failed"
        elif not asr_available:
            final_message = no_asr_upload_message
        else:
            final_message = "Analysis processing completed"

        self._update_manifest_cloud_state(
            manifest_path,
            uploaded=uploaded,
            remote_job_id=str(remote_job_id) if remote_job_id else None,
            cloud_status=remote_status or None,
            report_url=str(report_url) if report_url else None,
            cloud_report_path=str(cloud_report_path) if cloud_report_path else None,
            last_response=remote_result,
            error=cloud_error,
        )

        update_progress(
            1.0,
            final_message,
            status="completed",
            remote_job_id=remote_job_id,
            report_url=report_url,
        )
        return DesktopAnalysisResult(
            job_id=job_id,
            workspace_dir=workspace_dir,
            transcript_path=transcript_path,
            timeline_path=timeline_path,
            summary_path=summary_path,
            uploaded=uploaded,
            report_url=report_url,
            cloud_report_path=cloud_report_path,
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

    def _update_manifest_cloud_state(
        self,
        manifest_path: Path,
        *,
        uploaded: bool,
        remote_job_id: str | None,
        cloud_status: str | None,
        report_url: str | None,
        cloud_report_path: str | None,
        last_response: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        if not manifest_path.exists():
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return
        manifest["cloud"] = {
            "uploaded": uploaded,
            "remote_job_id": remote_job_id,
            "status": cloud_status,
            "report_url": report_url,
            "cloud_report_path": cloud_report_path,
            "last_response": last_response,
            "error": error,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _run_asr(
        self,
        extractor: FFmpegAudioExtractor,
        transcriber: FasterWhisperTranscriber,
        video_path: Path,
        workspace_dir: Path,
        asr_mode: str,
        chunk_seconds: int,
        keep_asr_audio: bool,
        update_progress: Callable[..., None],
    ) -> list[Any]:
        if asr_mode == "chunked":
            audio_chunks_dir = workspace_dir / "audio_chunks"
            update_progress(0.25, f"正在用 ffmpeg 切分音频（每段 {chunk_seconds} 秒）")
            audio_chunks = asyncio.run(
                extractor.segment_audio_async(
                    video_path,
                    audio_chunks_dir,
                    segment_seconds=chunk_seconds,
                )
            )
            update_progress(0.35, f"正在通过常驻 ASR worker 转写 {len(audio_chunks)} 个音频分片")

            def on_chunk(chunk_index: int, chunk_total: int, chunk_path: Path) -> None:
                base = 0.35
                span = 0.25
                chunk_progress = base + span * (chunk_index / max(1, chunk_total))
                update_progress(
                    chunk_progress,
                    f"正在转写音频分片 {chunk_index}/{chunk_total}：{chunk_path.name}",
                )

            transcript = transcriber.transcribe_chunks(
                audio_chunks,
                chunk_seconds,
                progress_callback=on_chunk,
            )
            if not keep_asr_audio:
                shutil.rmtree(audio_chunks_dir, ignore_errors=True)
            return transcript

        audio_path = workspace_dir / "voicetrack.wav"
        update_progress(0.25, "正在用 ffmpeg 抽取整段音频")
        asyncio.run(extractor.extract_audio_async(video_path, audio_path))
        update_progress(0.35, "正在通过常驻 ASR worker 转写整段音频")

        def on_whole(index: int, total: int, _audio_path: Path) -> None:
            update_progress(0.35 + 0.25 * (index / max(1, total)), "正在转写整段音频")

        transcript = transcriber.transcribe_batch([(audio_path, 0.0)], progress_callback=on_whole)
        if not keep_asr_audio:
            with contextlib.suppress(OSError):
                audio_path.unlink()
        return transcript

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

    def _resolve_asr_python_executable(self) -> str | None:
        explicit = os.getenv("DSO_ASR_PYTHON") or os.getenv("FASTER_WHISPER_PYTHON")
        if explicit:
            return explicit

        project_root = Path(__file__).resolve().parents[2]
        executable_path = Path(os.getenv("PYTHON_EXECUTABLE_FOR_ASR_DISCOVERY", "") or sys.executable)
        possible_roots = [
            self.settings.workspace_root,
            project_root,
            Path.cwd(),
            executable_path.parent.parent.parent,
        ]
        sidecar_candidates = [
            root / ".venv-asr" / python_path
            for root in dict.fromkeys(possible_roots)
            for python_path in (Path("Scripts") / "python.exe", Path("bin") / "python")
        ]
        for candidate in sidecar_candidates:
            if self._is_current_python(candidate):
                continue
            if candidate.exists() and self._python_has_module(candidate, "faster_whisper"):
                return str(candidate)

        if importlib.util.find_spec("faster_whisper") is not None:
            return None

        fallback_candidates = [
            root / ".venv" / python_path
            for root in dict.fromkeys(possible_roots)
            for python_path in (Path("Scripts") / "python.exe", Path("bin") / "python")
        ]
        for candidate in fallback_candidates:
            if self._is_current_python(candidate):
                continue
            if candidate.exists() and self._python_has_module(candidate, "faster_whisper"):
                return str(candidate)
        return None

    @staticmethod
    def _is_current_python(candidate: Path) -> bool:
        try:
            return candidate.resolve() == Path(sys.executable).resolve()
        except OSError:
            return str(candidate) == sys.executable

    @staticmethod
    def _python_has_module(python_executable: Path, module_name: str) -> bool:
        try:
            process = subprocess.run(
                [
                    str(python_executable),
                    "-c",
                    f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec({module_name!r}) else 1)",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            return False
        return process.returncode == 0

    def _asr_mode(self) -> str:
        mode = os.getenv("DSO_ASR_MODE", "whole").strip().lower()
        if mode not in {"whole", "chunked", "auto"}:
            return "whole"
        return mode

    def _select_asr_mode(self, video_path: Path, configured_mode: str) -> str:
        if configured_mode in {"whole", "chunked"}:
            return configured_mode
        duration = self._probe_duration_seconds(video_path)
        if duration is None:
            return "whole"
        return "chunked" if duration > self._asr_whole_max_seconds() else "whole"

    def _probe_duration_seconds(self, video_path: Path) -> float | None:
        ffprobe = self._resolve_ffprobe_binary()
        if not ffprobe:
            return None
        try:
            process = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError:
            return None
        if process.returncode != 0:
            return None
        try:
            return float(process.stdout.strip())
        except ValueError:
            return None

    def _resolve_ffprobe_binary(self) -> str | None:
        ffmpeg_path = Path(self.settings.ffmpeg_binary)
        if ffmpeg_path.name.lower().startswith("ffmpeg"):
            candidate = ffmpeg_path.with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
            if candidate.exists():
                return str(candidate)
        return shutil.which("ffprobe")

    def _asr_chunk_seconds(self) -> int:
        try:
            configured = int(os.getenv("DSO_ASR_CHUNK_SECONDS", "600"))
        except ValueError:
            configured = 600
        return min(1800, max(60, configured))

    def _asr_whole_max_seconds(self) -> int:
        try:
            configured = int(os.getenv("DSO_ASR_WHOLE_MAX_SECONDS", "7200"))
        except ValueError:
            configured = 7200
        return max(60, configured)

    def _build_markdown_summary(
        self,
        timeline: Any,
        analyses: list[dict[str, Any]],
        diagnostics: list[str],
    ) -> str:
        lines = [
            "# 本地结构化处理摘要",
            "",
            "## 基础结果",
            "",
            f"- 人数锚点数：{len(timeline.anchors)}",
            f"- ASR 对齐片段数：{len(timeline.aligned_segments)}",
            f"- 弹幕样本数：{timeline.coverage.get('danmaku_message_count')}",
            "",
            "## 云端分析状态",
            "",
            "- 客户端未运行任何本地大模型分析。",
            "- 客户端只负责音频提取、ASR、人数锚点检测、timeline 构建和上传。",
            "- 提示词模板、模型供应商密钥、模型选择、爆点解释和切片策略均由云端服务控制。",
            "- 最终面向用户的爆点结论以云端返回的报告为准。",
            "",
        ]
        lines.extend(["## 诊断信息", ""])
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

    def _select_anchors_for_analysis(self, timeline: Any) -> list[Any]:
        max_anchors = self._deepseek_max_anchors()
        duration_minutes = float(timeline.occupancy_summary.get("duration_minutes") or 0)
        high_score_count = sum(
            1
            for anchor in timeline.anchors
            if anchor.priority in {"strong", "high"} or anchor.score >= 3.0
        )
        target_count = min(
            max_anchors,
            len(timeline.anchors),
            max(15, math.ceil(duration_minutes / 5.0), high_score_count),
        )
        ranked = sorted(
            timeline.anchors,
            key=lambda anchor: (
                {"strong": 3, "high": 3, "medium": 2, "weak": 1}.get(anchor.priority, 0),
                anchor.score,
                abs(anchor.abs_change),
            ),
            reverse=True,
        )
        selected = ranked[:target_count]
        return sorted(selected, key=lambda anchor: anchor.timestamp)

    def _deepseek_max_anchors(self) -> int:
        try:
            configured = int(os.getenv("DEEPSEEK_MAX_ANCHORS", "40"))
        except ValueError:
            configured = 40
        return min(100, max(10, configured))

    def _build_summary_excerpt(self, timeline: Any, analyses: list[dict[str, Any]], diagnostics: list[str]) -> str:
        lines = [
            f"人数锚点 {len(timeline.anchors)}",
            f"ASR片段 {len(timeline.aligned_segments)}",
            "等待云端分析",
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
