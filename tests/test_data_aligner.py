from __future__ import annotations

import pandas as pd

from post_live_analyst.data_aligner import AlignmentConfig, DataAligner
from post_live_analyst.models import TranscriptSegment


def test_build_timeline_detects_surge_and_aligns_segment() -> None:
    occupancy_df = pd.DataFrame(
        {
            "elapsed_seconds": [0, 30, 60, 90, 120],
            "occupants": [100, 110, 140, 150, 145],
        }
    )
    segments = [
        TranscriptSegment(start=20, end=70, text="现在给大家讲一下今天的福利"),
    ]
    danmaku_df = pd.DataFrame(
        {
            "elapsed_seconds": [25, 45, 65],
            "text": ["这个价格可以", "上链接", "太划算了"],
            "user_id": ["u1", "u2", "u1"],
            "sentiment_score": [0.5, 0.1, 0.8],
        }
    )

    aligner = DataAligner(
        AlignmentConfig(
            anchor_window_seconds=60,
            anchor_pct_threshold=0.2,
            anchor_abs_threshold=10,
            anchor_cooldown_seconds=0,
        )
    )
    timeline = aligner.build_timeline(segments, occupancy_df, danmaku_df=danmaku_df)

    assert len(timeline.anchors) >= 1
    assert timeline.aligned_segments[0].occupancy.peak == 140
    assert timeline.aligned_segments[0].occupancy.mean == 125.0
    assert timeline.aligned_segments[0].danmaku is not None
    assert timeline.aligned_segments[0].danmaku.message_count == 3
    assert timeline.aligned_segments[0].danmaku.unique_user_count == 2
