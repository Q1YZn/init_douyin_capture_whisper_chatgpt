from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import requests

from .models import AlignedSegment, AnchorEvent


@dataclass(slots=True)
class CliProxyConfig:
    base_url: str
    api_key: str
    model_id: str
    timeout_seconds: float = 60.0


@dataclass(slots=True)
class AnchorAnalysisResult:
    anchor_id: str
    status: str
    model_id: str
    response_text: str | None = None
    raw_response: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CliProxyAgentAnalyst:
    """Call an OpenAI-compatible local proxy to analyze replay anchors."""

    def __init__(self, config: CliProxyConfig) -> None:
        self.config = config

    def list_models(self) -> dict[str, Any]:
        response = requests.get(
            f"{self.config.base_url.rstrip('/')}/models",
            headers=self._headers(),
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def analyze_anchor(
        self,
        anchor: AnchorEvent,
        segments: Sequence[AlignedSegment],
    ) -> AnchorAnalysisResult:
        prompt = self._build_prompt(anchor, segments)
        payload = {
            "model": self.config.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是直播复盘分析师。"
                        "请判断该爆点更像是货品给力、主播话术引导、还是投流进入导致。"
                        "如果给了弹幕信息，也请结合弹幕情绪一起判断。"
                        "请用简短中文回答，包含：结论、依据、置信度。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }

        try:
            response = requests.post(
                f"{self.config.base_url.rstrip('/')}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            raw = response.json()
            content = self._extract_content(raw)
            return AnchorAnalysisResult(
                anchor_id=anchor.anchor_id,
                status="ok",
                model_id=self.config.model_id,
                response_text=content,
                raw_response=raw,
            )
        except Exception as exc:
            return AnchorAnalysisResult(
                anchor_id=anchor.anchor_id,
                status="error",
                model_id=self.config.model_id,
                error=str(exc),
            )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _build_prompt(self, anchor: AnchorEvent, segments: Sequence[AlignedSegment]) -> str:
        segment_blocks: list[str] = []
        for segment in segments:
            danmaku = segment.danmaku
            danmaku_line = "弹幕: 无"
            if danmaku:
                danmaku_line = (
                    f"弹幕: {danmaku.message_count}条, "
                    f"情绪均值={danmaku.avg_sentiment}, "
                    f"示例={danmaku.sample_messages}"
                )
            segment_blocks.append(
                "\n".join(
                    [
                        f"时间段: {segment.start:.1f}s - {segment.end:.1f}s",
                        f"主播话术: {segment.text}",
                        (
                            "人数: "
                            f"mean={segment.occupancy.mean}, "
                            f"peak={segment.occupancy.peak}, "
                            f"slope_per_min={segment.occupancy.slope_per_min}"
                        ),
                        danmaku_line,
                    ]
                )
            )

        return "\n\n".join(
            [
                "请分析以下直播爆点：",
                (
                    f"锚点ID: {anchor.anchor_id}\n"
                    f"锚点时刻: {anchor.timestamp:.1f}s\n"
                    f"方向: {anchor.direction}\n"
                    f"变化比例: {anchor.pct_change:.1%}\n"
                    f"绝对变化: {anchor.abs_change}\n"
                    f"人数基线: {anchor.baseline_count}\n"
                    f"人数目标: {anchor.target_count}"
                ),
                "锚点上下文如下：",
                "\n\n".join(segment_blocks) if segment_blocks else "无上下文片段",
                "请判断该爆点更可能是货品给力、主播话术引导、还是投流进入，并说明理由。",
            ]
        )

    def _extract_content(self, response_payload: dict[str, Any]) -> str:
        choices = response_payload.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            return "\n".join(part.strip() for part in text_parts if part.strip())
        return str(content).strip()
