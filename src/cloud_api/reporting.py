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
        lines = [
            "# Cloud Replay Report",
            "",
            "## Overview",
            "",
            f"- Anchor count: {len(timeline.get('anchors', []))}",
            f"- Segment count: {len(timeline.get('aligned_segments', []))}",
            f"- Danmaku count: {timeline.get('coverage', {}).get('danmaku_message_count')}",
            "",
            "## Keywords",
            "",
            *(f"- {keyword}" for keyword in keywords[:10]),
            "",
        ]
        if upstream_summary:
            lines.extend(["## Client Summary", "", upstream_summary, ""])
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
            lines.extend(["## Agent Analyses", ""])
            for analysis in analyses:
                lines.append(f"### {analysis.get('anchor_id')}")
                lines.append("")
                lines.append(f"- Status: {analysis.get('status')}")
                if analysis.get("response_text"):
                    lines.append(f"- Analysis: {analysis.get('response_text')}")
                if analysis.get("error"):
                    lines.append(f"- Error: {analysis.get('error')}")
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
            for token in text.replace("，", " ").replace("。", " ").split():
                if len(token) >= 2:
                    counter[token] += 1
            danmaku = segment.get("danmaku") or {}
            for message in danmaku.get("sample_messages", []):
                normalized = str(message).strip()
                if len(normalized) >= 2:
                    counter[normalized] += 1
        return [token for token, _ in counter.most_common(20)]
