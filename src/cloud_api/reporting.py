from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import markdown


class ReportBuilder:
    def __init__(self, report_root: Path) -> None:
        self.report_root = report_root
        self.report_root.mkdir(parents=True, exist_ok=True)

    def build_markdown(self, timeline: dict[str, Any], upstream_summary: str | None = None) -> str:
        keywords = self._extract_keywords(timeline)
        coverage = timeline.get("coverage", {})
        summary = timeline.get("occupancy_summary") or coverage.get("occupancy_summary") or {}
        selection = timeline.get("analysis_selection") or {}
        lines = [
            "# Cloud Replay Report",
            "",
            "## Overview",
            "",
            f"- All detected anchors: {len(timeline.get('anchors', []))}",
            f"- DeepSeek analyzed anchors: {selection.get('selected_anchor_count', len(timeline.get('agent_analyses', [])))}",
            f"- Unanalyzed weak anchors: {selection.get('weak_anchor_count', 0)}",
            f"- Segment count: {len(timeline.get('aligned_segments', []))}",
            f"- Danmaku count: {coverage.get('danmaku_message_count')}",
            f"- Analysis selection status: {selection.get('status', 'not_run')}",
            "",
            "## Occupancy Summary",
            "",
            f"- Duration minutes: {summary.get('duration_minutes')}",
            f"- Samples: {summary.get('sample_count')}",
            f"- Mean / Median: {summary.get('mean')} / {summary.get('median')}",
            f"- Max / Min: {summary.get('max')} / {summary.get('min')}",
            f"- P25 / P75 / P90: {summary.get('p25')} / {summary.get('p75')} / {summary.get('p90')}",
            f"- IQR / Stddev / CV: {summary.get('iqr')} / {summary.get('stddev')} / {summary.get('coefficient_of_variation')}",
            "",
            "## Anchor Selection",
            "",
            f"- Strategy: {selection.get('strategy', 'unknown')}",
            f"- Max anchors: {selection.get('max_anchors')}",
            f"- High score anchors: {selection.get('high_score_anchor_count')}",
            f"- Selected anchor IDs: {', '.join(selection.get('selected_anchor_ids') or []) or 'None'}",
            "",
            "## Missing Data Notes",
            "",
            f"- ASR available: {coverage.get('asr_available')}",
            f"- Danmaku available: {coverage.get('danmaku_available')}",
            "- Product, order and transaction data: not provided in current timeline.",
            "- Missing danmaku or transaction data can reduce confidence for attribution.",
            "",
            "## Keywords",
            "",
            *(f"- {keyword}" for keyword in keywords[:10]),
            "",
        ]
        if upstream_summary:
            lines.extend(["## Client Summary", "", upstream_summary, ""])
        global_analysis = timeline.get("global_analysis") or {}
        if global_analysis:
            lines.extend(["## Global Timeline Analysis", ""])
            lines.append(f"- Status: {global_analysis.get('status')}")
            candidate_intervals = global_analysis.get("candidate_intervals") or []
            if candidate_intervals:
                lines.append(f"- Candidate intervals: {len(candidate_intervals)}")
            if global_analysis.get("response_text"):
                lines.append("")
                lines.append(str(global_analysis.get("response_text")))
            if global_analysis.get("error"):
                lines.append(f"- Error: {global_analysis.get('error')}")
            lines.append("")
        occupancy_blocks = timeline.get("occupancy_timeline_blocks") or []
        if occupancy_blocks:
            lines.extend(["## Occupancy Timeline Blocks", ""])
            lines.append("| Time | Start -> End | Min / Max | Movement | Level |")
            lines.append("| --- | ---: | ---: | --- | --- |")
            for block in occupancy_blocks[:60]:
                lines.append(
                    "| "
                    f"{block.get('start')}s-{block.get('end')}s | "
                    f"{block.get('count_start')} -> {block.get('count_end')} | "
                    f"{block.get('count_min')} / {block.get('count_max')} | "
                    f"{block.get('movement')} | "
                    f"{block.get('relative_level')} |"
                )
            lines.append("")
        if any(
            timeline.get(key) is not None
            for key in ("video_size_bytes", "asset_storage", "asset_uri", "analysis_tier", "payment_required")
        ):
            lines.extend(["## Delivery Metadata", ""])
            if timeline.get("video_size_bytes") is not None:
                lines.append(f"- Video size (bytes): {timeline.get('video_size_bytes')}")
            if timeline.get("asset_storage"):
                lines.append(f"- Asset storage: {timeline.get('asset_storage')}")
            if timeline.get("asset_uri"):
                lines.append(f"- Asset URI: {timeline.get('asset_uri')}")
            if timeline.get("analysis_tier"):
                lines.append(f"- Analysis tier: {timeline.get('analysis_tier')}")
            if timeline.get("payment_required") is not None:
                lines.append(f"- Payment required: {timeline.get('payment_required')}")
            lines.append("")
        analyses = timeline.get("agent_analyses", [])
        if analyses:
            lines.extend(["## Cloud DeepSeek Anchor Analyses", ""])
            for analysis in analyses:
                anchor = analysis.get("anchor") or self._find_anchor(timeline, analysis.get("anchor_id"))
                lines.append(f"### {analysis.get('anchor_id')}")
                lines.append("")
                if anchor:
                    lines.append(
                        "- Anchor: "
                        f"{anchor.get('direction')} at {anchor.get('timestamp')}s, "
                        f"score={anchor.get('score')}, priority={anchor.get('priority')}, "
                        f"change={anchor.get('baseline_count')}->{anchor.get('target_count')}"
                    )
                lines.append(f"- Status: {analysis.get('status')}")
                if analysis.get("response_text"):
                    lines.append("")
                    lines.append(str(analysis.get("response_text")))
                if analysis.get("error"):
                    lines.append(f"- Error: {analysis.get('error')}")
                lines.append("")
        weak_anchors = self._weak_anchors(timeline)
        if weak_anchors:
            lines.extend(["## Unanalyzed Weak Anchors", ""])
            lines.append("| ID | Time | Type | Score | Change | Reason |")
            lines.append("| --- | ---: | --- | ---: | ---: | --- |")
            for anchor in weak_anchors[:40]:
                lines.append(
                    "| "
                    f"{anchor.get('anchor_id')} | "
                    f"{anchor.get('timestamp')}s | "
                    f"{anchor.get('direction')} | "
                    f"{anchor.get('score')} | "
                    f"{anchor.get('abs_change')} | "
                    f"{str(anchor.get('reason_hint') or '').replace('|', '/')}"
                    " |"
                )
            lines.append("")
        return "\n".join(lines)

    def markdown_to_html(self, markdown_text: str) -> str:
        return markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])

    def persist_report(self, job_id: str, markdown_text: str, html_text: str, timeline: dict[str, Any]) -> tuple[Path, Path, Path]:
        job_dir = self.report_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        md_path = job_dir / "report.md"
        html_path = job_dir / "report.html"
        timeline_path = job_dir / "timeline.json"
        md_path.write_text(markdown_text, encoding="utf-8")
        html_path.write_text(html_text, encoding="utf-8")
        timeline_path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
        return md_path, html_path, timeline_path

    def _extract_keywords(self, timeline: dict[str, Any]) -> list[str]:
        counter: Counter[str] = Counter()
        for segment in timeline.get("aligned_segments", []):
            text = str(segment.get("text", ""))
            for token in text.replace("，", " ").replace("。", " ").replace(",", " ").replace(".", " ").split():
                if len(token) >= 2:
                    counter[token] += 1
            danmaku = segment.get("danmaku") or {}
            for message in danmaku.get("sample_messages", []):
                normalized = str(message).strip()
                if len(normalized) >= 2:
                    counter[normalized] += 1
        return [token for token, _ in counter.most_common(20)]

    def _find_anchor(self, timeline: dict[str, Any], anchor_id: str | None) -> dict[str, Any] | None:
        for anchor in timeline.get("anchors", []):
            if anchor.get("anchor_id") == anchor_id:
                return anchor
        return None

    def _weak_anchors(self, timeline: dict[str, Any]) -> list[dict[str, Any]]:
        selection = timeline.get("analysis_selection") or {}
        selected_ids = set(selection.get("selected_anchor_ids") or [])
        return [
            anchor
            for anchor in timeline.get("anchors", [])
            if anchor.get("anchor_id") not in selected_ids
        ]
