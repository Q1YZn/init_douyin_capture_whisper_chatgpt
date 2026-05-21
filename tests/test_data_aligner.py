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
    assert timeline.occupancy_summary["median"] == 140.0
    assert timeline.occupancy_timeline_blocks
    assert timeline.transcript_chapters
    assert timeline.coverage["occupancy_timeline_block_count"] == len(timeline.occupancy_timeline_blocks)
    assert timeline.coverage["transcript_chapter_count"] == len(timeline.transcript_chapters)
    assert timeline.anchors[0].score > 0
    assert timeline.anchors[0].direction in {"surge", "peak", "recovery"}


def test_timeline_blocks_cover_full_occupancy_and_chapters_keep_tail() -> None:
    occupancy_df = pd.DataFrame(
        {
            "elapsed_seconds": [0, 60, 120, 180, 240, 300, 360],
            "occupants": [100, 120, 150, 150, 148, 170, 180],
        }
    )
    segments = [
        TranscriptSegment(start=10, end=40, text="opening topic"),
        TranscriptSegment(start=310, end=350, text="tail topic"),
    ]
    aligner = DataAligner(AlignmentConfig(analysis_block_seconds=180, anchor_cooldown_seconds=0))

    timeline = aligner.build_timeline(segments, occupancy_df)

    blocks = timeline.occupancy_timeline_blocks
    chapters = timeline.transcript_chapters
    assert blocks[0]["start"] == 0.0
    assert blocks[-1]["end"] == 360.0
    assert any(block["movement"] in {"rising", "stable_high", "high_plateau"} for block in blocks)
    assert len(chapters) == len(blocks)
    assert "opening topic" in chapters[0]["text_excerpt"]
    assert "tail topic" in chapters[-1]["text_excerpt"]


def test_low_viewer_small_change_is_kept_weak() -> None:
    occupancy_df = pd.DataFrame(
        {
            "elapsed_seconds": [0, 30, 60, 90, 120, 150, 180],
            "occupants": [10, 11, 12, 11, 13, 12, 11],
        }
    )
    aligner = DataAligner(
        AlignmentConfig(
            anchor_window_seconds=60,
            anchor_pct_threshold=0.1,
            anchor_cooldown_seconds=0,
        )
    )

    anchors = aligner.detect_anchor_events(occupancy_df)

    assert anchors
    assert all(anchor.priority == "weak" for anchor in anchors if abs(anchor.abs_change) <= 2)
    assert all(anchor.score < 3.0 for anchor in anchors if abs(anchor.abs_change) <= 2)


def test_high_viewer_absolute_change_gets_strong_priority() -> None:
    occupancy_df = pd.DataFrame(
        {
            "elapsed_seconds": [0, 30, 60, 90, 120, 150, 180],
            "occupants": [980, 1000, 1120, 1130, 1010, 990, 985],
        }
    )
    aligner = DataAligner(
        AlignmentConfig(
            anchor_window_seconds=60,
            anchor_pct_threshold=0.05,
            anchor_cooldown_seconds=0,
        )
    )

    anchors = aligner.detect_anchor_events(occupancy_df)

    assert anchors
    assert any(anchor.priority == "strong" and abs(anchor.abs_change) >= 100 for anchor in anchors)
