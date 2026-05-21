from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields
from datetime import datetime
from typing import Any, Callable

import requests

try:  # pragma: no cover - exercised in server environments with Redis/RQ installed
    from redis import Redis
    from rq import Queue
except ImportError:  # pragma: no cover - desktop test env may not include server worker deps
    Redis = None  # type: ignore[assignment]
    Queue = None  # type: ignore[assignment]

from post_live_analyst.agent_analyst import CliProxyAgentAnalyst, CliProxyConfig
from post_live_analyst.models import AlignedSegment, AnchorEvent, DanmakuStats, OccupancyStats

from .config import ServerSettings
from .db import get_session
from .models import AnalysisJob
try:  # pragma: no cover - server dependency may be absent in desktop test env
    from .reporting import ReportBuilder
except ImportError:  # pragma: no cover
    ReportBuilder = None  # type: ignore[assignment]


settings = ServerSettings.from_env()
redis_conn = Redis.from_url(settings.redis_url) if Redis is not None else None
queue = Queue("reports", connection=redis_conn) if Queue is not None and redis_conn is not None else None


class ReportCancelled(RuntimeError):
    pass


def enqueue_report_job(job_id: str) -> str:
    if queue is None:
        raise RuntimeError("Redis/RQ worker dependencies are not installed.")
    job = queue.enqueue(generate_report, job_id, job_timeout=settings.report_job_timeout_seconds)
    return job.id


def cancel_enqueued_report_job(rq_job_id: str | None) -> bool:
    if queue is None or not rq_job_id:
        return False
    try:
        rq_job = queue.fetch_job(rq_job_id)
    except Exception:
        return False
    if rq_job is None:
        return False
    cancelled = False
    for method_name in ("cancel", "delete"):
        method = getattr(rq_job, method_name, None)
        if method is None:
            continue
        try:
            method()
            cancelled = True
        except Exception:
            continue
    return cancelled


def _raise_if_cancelled(session: Any, analysis_job: AnalysisJob, timeline: dict[str, Any] | None = None) -> None:
    try:
        session.refresh(analysis_job)
    except Exception:
        pass
    if analysis_job.status != "cancelled":
        return
    if timeline is not None:
        selection = timeline.setdefault("analysis_selection", {})
        if isinstance(selection, dict):
            selection["status"] = "cancelled"
        global_state = timeline.setdefault("global_analysis", {})
        if isinstance(global_state, dict) and global_state.get("status") == "running":
            global_state["status"] = "cancelled"
        analysis_job.timeline_json = json.dumps(timeline, ensure_ascii=False)
        session.add(analysis_job)
        session.commit()
    raise ReportCancelled("Report generation cancelled")


def generate_report(job_id: str) -> None:
    if ReportBuilder is None:
        raise RuntimeError("Report rendering dependencies are not installed.")
    report_builder = ReportBuilder(settings.report_root)
    try:
        with get_session() as session:
            analysis_job = session.get(AnalysisJob, job_id)
            if analysis_job is None:
                return
            if analysis_job.status == "cancelled":
                return
            timeline = json.loads(analysis_job.timeline_json)
            timeline["global_analysis"] = {
                "status": "running",
                "started_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
            analysis_job.status = "running"
            analysis_job.message = "Global timeline analysis running"
            analysis_job.error = None
            analysis_job.timeline_json = json.dumps(timeline, ensure_ascii=False)
            session.add(analysis_job)
            session.commit()

            timeline = _run_global_timeline_analysis(timeline)
            _raise_if_cancelled(session, analysis_job, timeline)
            selection = _select_anchors_for_deepseek(timeline, max_anchors=settings.deepseek_max_anchors)
            timeline["analysis_selection"] = {
                **selection,
                "status": "deepseek_running",
                "started_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
            analysis_job.status = "running"
            analysis_job.message = (
                "DeepSeek analysis running: "
                f"{selection.get('selected_anchor_count', 0)}/{selection.get('total_anchor_count', 0)} anchors selected"
            )
            analysis_job.error = None
            analysis_job.timeline_json = json.dumps(timeline, ensure_ascii=False)
            session.add(analysis_job)
            session.commit()
            _raise_if_cancelled(session, analysis_job, timeline)

            concurrency = settings.deepseek_anchor_concurrency

            def _record_deepseek_progress(completed: int, total: int, anchor_payload: dict[str, Any]) -> None:
                _raise_if_cancelled(session, analysis_job, timeline)
                selection_state = timeline.setdefault("analysis_selection", {})
                if isinstance(selection_state, dict):
                    selection_state["status"] = "deepseek_running"
                    selection_state["completed_anchor_count"] = completed
                    selection_state["current_anchor_total"] = total
                    selection_state["last_completed_anchor_id"] = anchor_payload.get("anchor_id")
                    selection_state["last_completed_anchor_timestamp"] = anchor_payload.get("timestamp")
                    selection_state["concurrency"] = concurrency
                analysis_job.status = "running"
                analysis_job.message = (
                    "DeepSeek analysis running: "
                    f"{completed}/{total} selected anchors, concurrency={concurrency}"
                )
                analysis_job.timeline_json = json.dumps(timeline, ensure_ascii=False)
                session.add(analysis_job)
                session.commit()

            timeline = _run_deepseek_anchor_analysis(timeline, progress_callback=_record_deepseek_progress)
            _raise_if_cancelled(session, analysis_job, timeline)
            markdown_text = report_builder.build_markdown(timeline, analysis_job.summary_markdown)
            html_text = report_builder.markdown_to_html(markdown_text)
            _, html_path, _ = report_builder.persist_report(job_id, markdown_text, html_text, timeline)
            _raise_if_cancelled(session, analysis_job, timeline)
            analysis_job.status = "completed"
            analysis_job.message = "Report completed"
            analysis_job.error = None
            analysis_job.summary_markdown = markdown_text
            analysis_job.timeline_json = json.dumps(timeline, ensure_ascii=False)
            analysis_job.report_html = html_text
            analysis_job.report_path = str(html_path)
            session.add(analysis_job)
            session.commit()
    except ReportCancelled:
        return
    except Exception as exc:
        with get_session() as session:
            analysis_job = session.get(AnalysisJob, job_id)
            if analysis_job is not None:
                timeline: dict[str, Any] | None = None
                try:
                    timeline = json.loads(analysis_job.timeline_json)
                    selection = timeline.setdefault("analysis_selection", {})
                    if isinstance(selection, dict):
                        selection["status"] = "failed"
                        selection["error"] = str(exc)[:2000]
                except Exception:
                    timeline = None
                analysis_job.status = "failed"
                analysis_job.message = "Report generation failed"
                analysis_job.error = str(exc)[:2000]
                if timeline is not None:
                    analysis_job.timeline_json = json.dumps(timeline, ensure_ascii=False)
                session.add(analysis_job)
                session.commit()
        raise


def generate_clip_candidates(
    timeline: dict[str, Any],
    *,
    summary_markdown: str | None = None,
    max_candidates: int = 20,
) -> dict[str, Any]:
    if not settings.deepseek_api_key.strip():
        raise RuntimeError("DEEPSEEK_API_KEY is not configured on the cloud server.")
    model_id = _provider_model_id(settings.deepseek_model_id, settings.deepseek_base_url)
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是直播短视频切片策划。只根据客户端上传的结构化 timeline、ASR、人数变化和已有报告，"
                    "输出适合剪辑的候选片段。不要假设存在弹幕、商品或成交数据；缺失时要降低置信度。"
                    "必须返回 JSON，格式为 {\"candidates\": [...] }。"
                ),
            },
            {
                "role": "user",
                "content": _build_clip_candidate_prompt(timeline, summary_markdown, max_candidates=max_candidates),
            },
        ],
        "temperature": 0.2,
    }
    response = requests.post(
        f"{settings.deepseek_base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.deepseek_api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code} from DeepSeek: {response.text[:1000]}")
    raw = response.json()
    content = _extract_chat_content(raw)
    parsed = _parse_json_object(content)
    candidates = parsed.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("DeepSeek clip response did not contain a candidates array.")
    return {
        "model_id": model_id,
        "raw_response": content,
        "candidates": [item for item in candidates if isinstance(item, dict)],
    }


def _build_clip_candidate_prompt(
    timeline: dict[str, Any],
    summary_markdown: str | None,
    *,
    max_candidates: int,
) -> str:
    anchors = list(timeline.get("anchors") or [])[:120]
    analyses = list(timeline.get("agent_analyses") or [])[:80]
    segments = list(timeline.get("aligned_segments") or [])
    compact_segments = []
    for segment in segments[:240]:
        if not isinstance(segment, dict):
            continue
        compact_segments.append(
            {
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": str(segment.get("text") or "")[:500],
                "nearby_anchor_ids": segment.get("nearby_anchor_ids") or [],
                "occupancy": segment.get("occupancy") or {},
            }
        )
    prompt_payload = {
        "max_candidates": min(50, max(1, int(max_candidates))),
        "occupancy_summary": timeline.get("occupancy_summary") or {},
        "coverage": timeline.get("coverage") or {},
        "anchors": anchors,
        "agent_analyses": analyses,
        "aligned_segments": compact_segments,
        "summary_markdown_excerpt": (summary_markdown or "")[:6000],
        "missing_data_notice": "如果没有弹幕、商品或成交数据，请明确写入 uncertainty。",
    }
    return (
        "请生成直播自动切片候选。每个候选必须包含："
        "start_seconds, end_seconds, title, reason, risk, confidence, source_anchor_ids。"
        "片段长度建议 20-180 秒，边界要结合 ASR 上下文，不要只取单个锚点秒数。"
        "只返回 JSON，不要输出 Markdown。\n\n"
        + json.dumps(prompt_payload, ensure_ascii=False)
    )


def _extract_chat_content(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict)).strip()
    return str(content).strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise RuntimeError("DeepSeek response was not valid JSON.")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise RuntimeError("DeepSeek response JSON root must be an object.")
    return payload


def _run_deepseek_anchor_analysis(
    timeline: dict[str, Any],
    *,
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    anchors = list(timeline.get("anchors") or [])
    selection = _select_anchors_for_deepseek(timeline, max_anchors=settings.deepseek_max_anchors)
    selected_id_order = {
        anchor_id: index
        for index, anchor_id in enumerate(selection["selected_anchor_ids"])
    }
    selected_anchors = sorted(
        [anchor for anchor in anchors if anchor.get("anchor_id") in selected_id_order],
        key=lambda anchor: selected_id_order[anchor.get("anchor_id")],
    )
    previous_selection = timeline.get("analysis_selection") if isinstance(timeline.get("analysis_selection"), dict) else {}
    timeline["analysis_selection"] = {
        **previous_selection,
        **selection,
        "status": "deepseek_running" if selected_anchors else selection.get("status", "selected"),
        "concurrency": settings.deepseek_anchor_concurrency,
    }

    existing_analyses = timeline.get("agent_analyses") or []
    if existing_analyses:
        timeline["client_agent_analyses"] = existing_analyses
    timeline["agent_analyses"] = []

    if not selected_anchors:
        timeline["analysis_selection"]["status"] = "skipped_no_anchors"
        return timeline
    if not settings.deepseek_api_key.strip():
        timeline["analysis_selection"]["status"] = "skipped_missing_deepseek_key"
        timeline["analysis_selection"]["error"] = "DEEPSEEK_API_KEY is not configured on the cloud server."
        return timeline

    analyst = CliProxyAgentAnalyst(
        CliProxyConfig(
            base_url=settings.deepseek_base_url,
            api_key=settings.deepseek_api_key,
            model_id=_provider_model_id(settings.deepseek_model_id, settings.deepseek_base_url),
            timeout_seconds=float(settings.deepseek_timeout_seconds),
        )
    )
    timeline["analysis_selection"]["model_id"] = _provider_model_id(settings.deepseek_model_id, settings.deepseek_base_url)
    occupancy_summary = _timeline_occupancy_summary(timeline)
    aligned_segments = [_aligned_segment_from_dict(segment) for segment in timeline.get("aligned_segments", [])]
    analyses_by_index: list[dict[str, Any] | None] = [None] * len(selected_anchors)

    def _analyze_anchor(index: int, anchor_payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        anchor_id = str(anchor_payload.get("anchor_id") or f"anchor_{index:04d}")
        try:
            anchor = _anchor_from_dict(anchor_payload)
            related_segments = _related_segments(anchor, aligned_segments)
            result = analyst.analyze_anchor(anchor, related_segments, occupancy_summary=occupancy_summary)
            return index, result.to_dict()
        except Exception as exc:
            return index, {
                "anchor_id": anchor_id,
                "status": "error",
                "model_id": analyst.config.model_id,
                "error": str(exc),
                "anchor": anchor_payload,
            }

    max_workers = max(1, min(settings.deepseek_anchor_concurrency, len(selected_anchors)))
    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_analyze_anchor, index, anchor_payload): (index, anchor_payload)
            for index, anchor_payload in enumerate(selected_anchors)
        }
        for future in as_completed(futures):
            index, anchor_payload = futures[future]
            try:
                result_index, result = future.result()
            except Exception as exc:  # pragma: no cover - _analyze_anchor already isolates failures
                result_index = index
                result = {
                    "anchor_id": str(anchor_payload.get("anchor_id") or f"anchor_{index:04d}"),
                    "status": "error",
                    "model_id": analyst.config.model_id,
                    "error": str(exc),
                    "anchor": anchor_payload,
                }
            analyses_by_index[result_index] = result
            completed_count += 1
            if progress_callback is not None:
                progress_callback(completed_count, len(selected_anchors), anchor_payload)

    analyses = [analysis for analysis in analyses_by_index if analysis is not None]
    ok_count = sum(1 for item in analyses if item.get("status") == "ok")
    timeline["agent_analyses"] = analyses
    timeline["analysis_selection"]["status"] = (
        "completed"
        if ok_count == len(analyses)
        else "completed_with_model_errors"
    )
    timeline["analysis_selection"]["ok_count"] = ok_count
    timeline["analysis_selection"]["error_count"] = len(analyses) - ok_count
    return timeline


def _run_global_timeline_analysis(timeline: dict[str, Any]) -> dict[str, Any]:
    global_state = timeline.setdefault("global_analysis", {})
    if not settings.deepseek_api_key.strip():
        global_state["status"] = "skipped_missing_deepseek_key"
        global_state["error"] = "DEEPSEEK_API_KEY is not configured on the cloud server."
        return timeline
    chapters = list(timeline.get("transcript_chapters") or [])
    occupancy_blocks = list(timeline.get("occupancy_timeline_blocks") or [])
    if not chapters and not occupancy_blocks:
        global_state["status"] = "skipped_no_global_inputs"
        return timeline

    model_id = _provider_model_id(settings.deepseek_model_id, settings.deepseek_base_url)
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是直播复盘分析师。请先从整场 ASR 章节和完整人数走势中判断直播节奏、"
                    "主题变化、持续高位、慢热铺垫、流失和恢复区间。弹幕不可用时不要假设观众反馈。"
                    "必须返回 JSON。"
                ),
            },
            {
                "role": "user",
                "content": _build_global_timeline_prompt(timeline),
            },
        ],
        "temperature": 0.2,
    }
    try:
        response = requests.post(
            f"{settings.deepseek_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.deepseek_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=max(float(settings.deepseek_timeout_seconds), 180.0),
        )
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code} from DeepSeek: {response.text[:1000]}")
        raw = response.json()
        content = _extract_chat_content(raw)
        parsed: dict[str, Any] | None = None
        try:
            parsed = _parse_json_object(content)
        except Exception:
            parsed = None
        global_state.update(
            {
                "status": "completed",
                "model_id": model_id,
                "response_text": content,
                "parsed": parsed,
                "candidate_intervals": _normalize_candidate_intervals(
                    (parsed or {}).get("candidate_intervals")
                    or (parsed or {}).get("hotspot_intervals")
                    or []
                ),
            }
        )
    except Exception as exc:
        global_state.update({"status": "failed", "error": str(exc)[:2000]})
    return timeline


def _build_global_timeline_prompt(timeline: dict[str, Any]) -> str:
    coverage = timeline.get("coverage") or {}
    payload = {
        "instructions": [
            "只根据 ASR 章节和人数走势分析，不要假设弹幕、商品或成交数据。",
            "请找出非突增型爆点、持续高位、慢热铺垫、话题转折、恢复和流失区间。",
            "返回 JSON：overall_summary, rhythm, candidate_intervals, uncertainty。",
            "candidate_intervals 每项包含 start, end, title, reason, confidence, evidence_chapter_ids。",
        ],
        "coverage": {
            "asr_available": coverage.get("asr_available"),
            "danmaku_available": coverage.get("danmaku_available"),
            "timeline_start": coverage.get("timeline_start"),
            "timeline_end": coverage.get("timeline_end"),
        },
        "occupancy_summary": timeline.get("occupancy_summary") or {},
        "occupancy_timeline_blocks": _compact_occupancy_blocks(
            timeline.get("occupancy_timeline_blocks") or []
        ),
        "transcript_chapters": _compact_transcript_chapters(
            timeline.get("transcript_chapters") or []
        ),
        "anchor_index": _compact_anchor_index(timeline.get("anchors") or []),
    }
    return json.dumps(payload, ensure_ascii=False)


def _compact_occupancy_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "block_id": block.get("block_id"),
            "start": block.get("start"),
            "end": block.get("end"),
            "count_start": block.get("count_start"),
            "count_end": block.get("count_end"),
            "count_min": block.get("count_min"),
            "count_max": block.get("count_max"),
            "count_mean": block.get("count_mean"),
            "delta": block.get("delta"),
            "slope_per_min": block.get("slope_per_min"),
            "relative_level": block.get("relative_level"),
            "movement": block.get("movement"),
        }
        for block in blocks
    ]


def _compact_transcript_chapters(chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for chapter in chapters:
        text = str(chapter.get("text_excerpt") or "")
        compact.append(
            {
                "chapter_id": chapter.get("chapter_id"),
                "start": chapter.get("start"),
                "end": chapter.get("end"),
                "segment_count": chapter.get("segment_count"),
                "text_excerpt": text[:1800],
                "nearby_anchor_ids": chapter.get("nearby_anchor_ids") or [],
                "occupancy": chapter.get("occupancy") or {},
            }
        )
    return compact


def _compact_anchor_index(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "anchor_id": anchor.get("anchor_id"),
            "timestamp": anchor.get("timestamp"),
            "direction": anchor.get("direction"),
            "score": anchor.get("score"),
            "priority": anchor.get("priority"),
            "abs_change": anchor.get("abs_change"),
            "baseline_count": anchor.get("baseline_count"),
            "target_count": anchor.get("target_count"),
            "reason_hint": anchor.get("reason_hint"),
        }
        for anchor in anchors[:120]
    ]


def _normalize_candidate_intervals(intervals: Any) -> list[dict[str, Any]]:
    if not isinstance(intervals, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in intervals:
        if not isinstance(item, dict):
            continue
        start = item.get("start", item.get("start_seconds"))
        end = item.get("end", item.get("end_seconds"))
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if end_f < start_f:
            start_f, end_f = end_f, start_f
        normalized.append(
            {
                "start": round(start_f, 4),
                "end": round(end_f, 4),
                "title": item.get("title"),
                "reason": item.get("reason"),
                "confidence": item.get("confidence"),
                "evidence_chapter_ids": item.get("evidence_chapter_ids") or [],
            }
        )
    return normalized


def _select_anchors_for_deepseek(timeline: dict[str, Any], max_anchors: int) -> dict[str, Any]:
    anchors = list(timeline.get("anchors") or [])
    bounded_max = min(100, max(10, int(max_anchors or 40)))
    duration_minutes = _duration_minutes(timeline)
    high_score_count = sum(
        1
        for anchor in anchors
        if str(anchor.get("priority", "")).lower() in {"strong", "high"}
        or float(anchor.get("score") or 0) >= 3.0
    )
    target_count = min(
        bounded_max,
        len(anchors),
        max(15, math.ceil(duration_minutes / 5.0), high_score_count),
    )

    ranked = sorted(
        anchors,
        key=lambda anchor: (
            _priority_weight(str(anchor.get("priority", "weak"))),
            float(anchor.get("score") or 0),
            abs(int(anchor.get("abs_change") or 0)),
        ),
        reverse=True,
    )
    candidate_intervals = _global_candidate_intervals(timeline)
    preferred = [anchor for anchor in ranked if _anchor_in_intervals(anchor, candidate_intervals)]
    selected = preferred[:target_count]
    if len(selected) < target_count:
        selected_ids_seed = {anchor.get("anchor_id") for anchor in selected}
        selected.extend(
            anchor
            for anchor in ranked
            if anchor.get("anchor_id") not in selected_ids_seed
        )
        selected = selected[:target_count]
    selected_ids = {anchor.get("anchor_id") for anchor in selected}
    weak_ids = [
        anchor.get("anchor_id")
        for anchor in anchors
        if anchor.get("anchor_id") not in selected_ids
    ]
    return {
        "status": "selected",
        "strategy": (
            "global_candidate_intervals_then_anchor_score"
            if candidate_intervals
            else "dynamic_by_duration_score_distribution"
        ),
        "max_anchors": bounded_max,
        "duration_minutes": round(duration_minutes, 4),
        "total_anchor_count": len(anchors),
        "high_score_anchor_count": high_score_count,
        "global_candidate_interval_count": len(candidate_intervals),
        "global_preferred_anchor_count": len(preferred),
        "selected_anchor_count": len(selected),
        "weak_anchor_count": len(weak_ids),
        "selected_anchor_ids": [anchor.get("anchor_id") for anchor in sorted(selected, key=lambda item: float(item.get("timestamp") or 0))],
        "weak_anchor_ids": weak_ids,
    }


def _global_candidate_intervals(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    global_analysis = timeline.get("global_analysis") or {}
    intervals = global_analysis.get("candidate_intervals") or []
    return _normalize_candidate_intervals(intervals)


def _anchor_in_intervals(anchor: dict[str, Any], intervals: list[dict[str, Any]]) -> bool:
    if not intervals:
        return False
    try:
        timestamp = float(anchor.get("timestamp"))
    except (TypeError, ValueError):
        return False
    return any(float(interval["start"]) <= timestamp <= float(interval["end"]) for interval in intervals)


def _duration_minutes(timeline: dict[str, Any]) -> float:
    summary = _timeline_occupancy_summary(timeline)
    duration = summary.get("duration_minutes")
    if duration is not None:
        return float(duration)
    coverage = timeline.get("coverage") or {}
    start = coverage.get("timeline_start")
    end = coverage.get("timeline_end")
    if start is None or end is None:
        return 0.0
    return max(0.0, (float(end) - float(start)) / 60.0)


def _timeline_occupancy_summary(timeline: dict[str, Any]) -> dict[str, Any]:
    return dict(timeline.get("occupancy_summary") or (timeline.get("coverage") or {}).get("occupancy_summary") or {})


def _priority_weight(priority: str) -> int:
    return {"strong": 3, "high": 3, "medium": 2, "weak": 1}.get(priority.lower(), 0)


def _provider_model_id(model_id: str, base_url: str) -> str:
    normalized = model_id.strip()
    if "api.deepseek.com" in base_url and normalized.startswith("deepseek/"):
        return normalized.split("/", 1)[1]
    return normalized


def _anchor_from_dict(payload: dict[str, Any]) -> AnchorEvent:
    allowed = {field.name for field in fields(AnchorEvent)}
    return AnchorEvent(**{key: value for key, value in payload.items() if key in allowed})


def _aligned_segment_from_dict(payload: dict[str, Any]) -> AlignedSegment:
    occupancy_payload = payload.get("occupancy") or {}
    danmaku_payload = payload.get("danmaku")
    return AlignedSegment(
        segment_id=str(payload.get("segment_id", "")),
        start=float(payload.get("start") or 0),
        end=float(payload.get("end") or 0),
        text=str(payload.get("text") or ""),
        occupancy=_occupancy_from_dict(occupancy_payload),
        danmaku=_danmaku_from_dict(danmaku_payload) if danmaku_payload else None,
        nearby_anchor_ids=list(payload.get("nearby_anchor_ids") or []),
        metadata=dict(payload.get("metadata") or {}),
    )


def _occupancy_from_dict(payload: dict[str, Any]) -> OccupancyStats:
    allowed = {field.name for field in fields(OccupancyStats)}
    return OccupancyStats(**{key: value for key, value in payload.items() if key in allowed})


def _danmaku_from_dict(payload: dict[str, Any]) -> DanmakuStats:
    allowed = {field.name for field in fields(DanmakuStats)}
    return DanmakuStats(**{key: value for key, value in payload.items() if key in allowed})


def _related_segments(anchor: AnchorEvent, segments: list[AlignedSegment]) -> list[AlignedSegment]:
    context_start = anchor.timestamp - 180.0
    context_end = anchor.timestamp + 180.0
    direct = [
        segment
        for segment in segments
        if anchor.anchor_id in segment.nearby_anchor_ids
        or (segment.end >= context_start and segment.start <= context_end)
    ]
    if direct:
        return sorted(direct, key=lambda segment: (segment.start, segment.end))
    return sorted(
        segments,
        key=lambda segment: min(abs(segment.start - anchor.timestamp), abs(segment.end - anchor.timestamp)),
    )[:3]
