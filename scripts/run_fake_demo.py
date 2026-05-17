from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_live_analyst.agent_analyst import CliProxyAgentAnalyst, CliProxyConfig
from post_live_analyst.data_aligner import AlignmentConfig, DataAligner
from post_live_analyst.models import TranscriptSegment


def main() -> None:
    output_dir = ROOT / "demo_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    occupancy_df = pd.DataFrame(
        {
            "elapsed_seconds": list(range(0, 301, 30)),
            "occupants": [120, 126, 130, 135, 162, 188, 205, 194, 182, 176, 170],
        }
    )
    danmaku_df = pd.DataFrame(
        {
            "elapsed_seconds": [100, 118, 126, 132, 148, 165, 178],
            "text": [
                "这个价格有点猛",
                "主播讲得太到位了",
                "库存还有吗",
                "下单了下单了",
                "感觉是投流进人了",
                "福利款再讲一遍",
                "链接在哪",
            ],
            "user_id": ["u1", "u2", "u3", "u4", "u5", "u6", "u7"],
            "sentiment_score": [0.5, 0.8, 0.2, 0.9, 0.0, 0.4, 0.3],
        }
    )
    segments = [
        TranscriptSegment(start=85, end=120, text="姐妹们今天这款半价，库存不多，先抢先得。"),
        TranscriptSegment(start=120, end=150, text="现在拍立减，再送一个旅行装，老粉都知道这次价格很少见。"),
        TranscriptSegment(start=150, end=185, text="刚进来的朋友先点关注，福利链接已经挂上，今天主要冲爆款。"),
    ]

    aligner = DataAligner(
        AlignmentConfig(
            anchor_window_seconds=60,
            anchor_pct_threshold=0.2,
            anchor_abs_threshold=20,
            anchor_cooldown_seconds=30,
        )
    )
    timeline = aligner.build_timeline(
        transcript_segments=segments,
        occupancy_df=occupancy_df,
        danmaku_df=danmaku_df,
    )

    analyst = CliProxyAgentAnalyst(
        CliProxyConfig(
            base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8317/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "your-api-key-1"),
            model_id=os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-5.4"),
        )
    )

    models_response: dict | None = None
    models_error: str | None = None
    try:
        models_response = analyst.list_models()
    except Exception as exc:
        models_error = str(exc)

    analyses = []
    for anchor in timeline.anchors[:2]:
        related_segments = [
            segment
            for segment in timeline.aligned_segments
            if anchor.anchor_id in segment.nearby_anchor_ids
        ]
        analyses.append(analyst.analyze_anchor(anchor, related_segments))

    summary_path = output_dir / "summary.md"
    timeline_path = output_dir / "detailed_timeline.json"

    summary_lines = [
        "# Demo Live Replay Summary",
        "",
        "## Proxy Check",
        "",
        f"- Base URL: {analyst.config.base_url}",
        f"- Model: {analyst.config.model_id}",
        f"- Models API status: {'ok' if models_response is not None else 'error'}",
    ]
    if models_error:
        summary_lines.append(f"- Models API error: {models_error}")
    if models_response:
        summary_lines.append(f"- Models payload keys: {list(models_response.keys())}")

    summary_lines.extend(
        [
            "",
            "## Timeline Summary",
            "",
            f"- Anchor count: {len(timeline.anchors)}",
            f"- Segment count: {len(timeline.aligned_segments)}",
            f"- Danmaku count: {timeline.coverage.get('danmaku_message_count')}",
            "",
            "## Anchor Analyses",
            "",
        ]
    )

    for result in analyses:
        summary_lines.append(f"### {result.anchor_id}")
        summary_lines.append("")
        summary_lines.append(f"- Status: {result.status}")
        if result.response_text:
            summary_lines.append(f"- Analysis: {result.response_text}")
        if result.error:
            summary_lines.append(f"- Error: {result.error}")
        summary_lines.append("")

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    timeline_payload = timeline.to_dict()
    timeline_payload["agent_analyses"] = [result.to_dict() for result in analyses]
    timeline_payload["proxy_check"] = {
        "base_url": analyst.config.base_url,
        "model_id": analyst.config.model_id,
        "models_api_ok": models_response is not None,
        "models_api_error": models_error,
        "models_response": models_response,
    }
    timeline_path.write_text(json.dumps(timeline_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"summary_path={summary_path}")
    print(f"timeline_path={timeline_path}")
    print(f"models_api_ok={models_response is not None}")
    for result in analyses:
        print(f"{result.anchor_id}:{result.status}")
        if result.response_text:
            print(result.response_text)
        if result.error:
            print(result.error)


if __name__ == "__main__":
    main()
