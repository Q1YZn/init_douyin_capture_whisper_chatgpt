from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from .models import (
    AlignedSegment,
    AnchorEvent,
    DanmakuMessage,
    DanmakuStats,
    DetailedTimeline,
    OccupancyStats,
    TranscriptSegment,
)


@dataclass(slots=True)
class AlignmentConfig:
    occupancy_time_column: str | None = None
    occupancy_count_column: str = "occupants"
    danmaku_time_column: str | None = None
    danmaku_text_column: str = "text"
    danmaku_user_id_column: str = "user_id"
    danmaku_user_name_column: str = "user_name"
    danmaku_sentiment_column: str = "sentiment_score"
    danmaku_sample_limit: int = 5
    anchor_window_seconds: int = 60
    anchor_pct_threshold: float = 0.2
    anchor_abs_threshold: int = 0
    anchor_cooldown_seconds: int = 30
    segment_anchor_radius_seconds: int = 30
    anchor_score_threshold: float = 1.2
    anchor_high_score_threshold: float = 3.0
    low_viewer_median_threshold: int = 20
    low_viewer_small_change: int = 2
    analysis_block_seconds: int = 300
    chapter_text_max_chars: int = 1800


class DataAligner:
    """Fuse transcript timestamps with occupancy, danmaku and anchor metrics."""

    SUPPORTED_TIME_COLUMNS = (
        "elapsed_seconds",
        "timestamp_seconds",
        "seconds",
        "time_seconds",
        "ts",
    )
    SUPPORTED_COUNT_COLUMNS = (
        "occupants",
        "occupant_count",
        "viewer_count",
        "online_count",
        "count",
    )
    SUPPORTED_DANMAKU_TEXT_COLUMNS = (
        "text",
        "content",
        "message",
        "body",
        "danmaku_text",
    )
    SUPPORTED_DANMAKU_USER_ID_COLUMNS = (
        "user_id",
        "uid",
        "author_id",
        "sender_id",
    )
    SUPPORTED_DANMAKU_USER_NAME_COLUMNS = (
        "user_name",
        "nickname",
        "author_name",
        "sender_name",
    )
    SUPPORTED_DANMAKU_SENTIMENT_COLUMNS = (
        "sentiment_score",
        "sentiment",
        "polarity",
    )

    def __init__(self, config: AlignmentConfig | None = None) -> None:
        self.config = config or AlignmentConfig()

    def load_occupancy_history(self, csv_path: str | Path) -> pd.DataFrame:
        csv_path = Path(csv_path)
        frame = pd.read_csv(csv_path)
        return self._normalize_occupancy_frame(frame)

    def load_danmaku_history(self, history_path: str | Path) -> pd.DataFrame:
        history_path = Path(history_path)
        if history_path.suffix.lower() == ".jsonl":
            rows: list[dict] = []
            with history_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            frame = pd.DataFrame(rows)
        else:
            frame = pd.read_csv(history_path)
        return self._normalize_danmaku_frame(frame)

    def align_segments(
        self,
        transcript_segments: Sequence[TranscriptSegment] | Sequence[dict],
        occupancy_df: pd.DataFrame,
        danmaku_df: pd.DataFrame | None = None,
        anchors: Sequence[AnchorEvent] | None = None,
    ) -> list[AlignedSegment]:
        normalized_segments = self._normalize_segments(transcript_segments)
        normalized_occupancy = self._normalize_occupancy_frame(occupancy_df)
        normalized_danmaku = self._normalize_danmaku_frame(danmaku_df)
        anchors = list(anchors or [])

        aligned_segments: list[AlignedSegment] = []
        for idx, segment in enumerate(normalized_segments):
            stats = self._compute_window_stats(
                normalized_occupancy,
                window_start=segment.start,
                window_end=segment.end,
            )
            danmaku_stats = self._compute_danmaku_window_stats(
                normalized_danmaku,
                window_start=segment.start,
                window_end=segment.end,
            )
            nearby_anchor_ids = [
                anchor.anchor_id
                for anchor in anchors
                if abs(anchor.timestamp - segment.start) <= self.config.segment_anchor_radius_seconds
                or abs(anchor.timestamp - segment.end) <= self.config.segment_anchor_radius_seconds
                or (segment.start <= anchor.timestamp <= segment.end)
            ]
            aligned_segments.append(
                AlignedSegment(
                    segment_id=f"seg_{idx:05d}",
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    occupancy=stats,
                    danmaku=danmaku_stats,
                    nearby_anchor_ids=nearby_anchor_ids,
                    metadata=segment.metadata.copy(),
                )
            )
        return aligned_segments

    def detect_anchor_events(
        self,
        occupancy_df: pd.DataFrame,
        config: AlignmentConfig | None = None,
    ) -> list[AnchorEvent]:
        config = config or self.config
        frame = self._normalize_occupancy_frame(occupancy_df)
        if frame.empty:
            return []

        anchors: list[AnchorEvent] = []
        last_anchor_at = -float("inf")
        summary = self._compute_occupancy_summary(frame)

        for row in frame.itertuples(index=False):
            current_ts = float(row.elapsed_seconds)
            target_ts = current_ts + config.anchor_window_seconds
            future = frame.loc[frame["elapsed_seconds"] >= target_ts]
            if future.empty:
                break

            future_row = future.iloc[0]
            baseline = int(row.occupants)
            target = int(future_row["occupants"])
            abs_change = target - baseline
            pct_change = (1.0 if target > 0 else 0.0) if baseline <= 0 else abs_change / baseline

            if abs(abs_change) < config.anchor_abs_threshold:
                continue

            window_end = float(future_row["elapsed_seconds"])
            window = frame.loc[
                (frame["elapsed_seconds"] >= current_ts)
                & (frame["elapsed_seconds"] <= window_end)
            ]
            local_range = (
                int(window["occupants"].max()) - int(window["occupants"].min())
                if not window.empty
                else 0
            )
            if abs_change == 0 and local_range < max(3, math.ceil(float(summary["iqr"] or 0))):
                continue
            slope_per_min = abs_change / max(config.anchor_window_seconds / 60.0, 1e-6)
            scores = self._score_anchor_candidate(
                baseline=baseline,
                target=target,
                abs_change=abs_change,
                pct_change=pct_change,
                slope_per_min=slope_per_min,
                window=window,
                summary=summary,
                config=config,
            )
            has_signal = (
                abs(pct_change) >= config.anchor_pct_threshold
                or abs(abs_change) >= max(config.anchor_abs_threshold, 2, math.ceil(float(summary["iqr"] or 0)))
                or scores["distribution_score"] > 0
                or scores["peak_trough_score"] > 0
            )
            if not has_signal:
                continue
            if scores["score"] < config.anchor_score_threshold and scores["peak_trough_score"] <= 0:
                continue
            if current_ts - last_anchor_at < config.anchor_cooldown_seconds:
                continue

            low_viewer_guard = self._is_low_viewer_noise(abs_change, scores, summary, config)
            if low_viewer_guard:
                scores["score"] = min(scores["score"], config.anchor_high_score_threshold - 0.01)
            priority = "weak" if low_viewer_guard else self._classify_anchor_priority(scores["score"], config)
            direction = self._classify_anchor_direction(
                baseline=baseline,
                target=target,
                abs_change=abs_change,
                pct_change=pct_change,
                window=window,
                summary=summary,
            )

            anchors.append(
                AnchorEvent(
                    anchor_id=f"anchor_{len(anchors):04d}",
                    timestamp=window_end,
                    window_start=current_ts,
                    window_end=window_end,
                    direction=direction,
                    abs_change=abs_change,
                    pct_change=round(pct_change, 6),
                    baseline_count=baseline,
                    target_count=target,
                    slope_per_min=round(slope_per_min, 4),
                    reason_hint=self._build_reason_hint(direction, pct_change, abs_change, scores["score"], priority),
                    score=round(float(scores["score"]), 4),
                    priority=priority,
                    abs_change_score=round(float(scores["abs_change_score"]), 4),
                    pct_change_score=round(float(scores["pct_change_score"]), 4),
                    distribution_score=round(float(scores["distribution_score"]), 4),
                    slope_score=round(float(scores["slope_score"]), 4),
                    peak_trough_score=round(float(scores["peak_trough_score"]), 4),
                    relative_level=self._relative_level(target, summary),
                    metadata={
                        "summary_median": summary["median"],
                        "summary_p75": summary["p75"],
                        "summary_p90": summary["p90"],
                        "low_viewer_guard": low_viewer_guard,
                    },
                )
            )
            last_anchor_at = current_ts

        return anchors

    def build_timeline(
        self,
        transcript_segments: Sequence[TranscriptSegment] | Sequence[dict] | None,
        occupancy_df: pd.DataFrame,
        danmaku_df: pd.DataFrame | None = None,
        config: AlignmentConfig | None = None,
    ) -> DetailedTimeline:
        config = config or self.config
        normalized_occupancy = self._normalize_occupancy_frame(occupancy_df)
        normalized_danmaku = self._normalize_danmaku_frame(danmaku_df)
        occupancy_summary = self._compute_occupancy_summary(normalized_occupancy)
        anchors = self.detect_anchor_events(normalized_occupancy, config=config)
        aligned_segments = self.align_segments(
            transcript_segments=transcript_segments or [],
            occupancy_df=normalized_occupancy,
            danmaku_df=normalized_danmaku,
            anchors=anchors,
        )
        occupancy_timeline_blocks = self._build_occupancy_timeline_blocks(
            normalized_occupancy,
            occupancy_summary,
            block_seconds=config.analysis_block_seconds,
        )
        transcript_chapters = self._build_transcript_chapters(
            aligned_segments,
            occupancy_timeline_blocks,
            max_text_chars=config.chapter_text_max_chars,
        )

        coverage = {
            "segment_count": len(aligned_segments),
            "anchor_count": len(anchors),
            "occupancy_timeline_block_count": len(occupancy_timeline_blocks),
            "transcript_chapter_count": len(transcript_chapters),
            "occupancy_sample_count": int(len(normalized_occupancy)),
            "danmaku_message_count": int(len(normalized_danmaku)),
            "timeline_start": float(normalized_occupancy["elapsed_seconds"].min()) if not normalized_occupancy.empty else None,
            "timeline_end": float(normalized_occupancy["elapsed_seconds"].max()) if not normalized_occupancy.empty else None,
            "asr_available": bool(transcript_segments),
            "danmaku_available": not normalized_danmaku.empty,
            "occupancy_summary": occupancy_summary,
        }
        return DetailedTimeline(
            aligned_segments=aligned_segments,
            anchors=anchors,
            coverage=coverage,
            occupancy_summary=occupancy_summary,
            occupancy_timeline_blocks=occupancy_timeline_blocks,
            transcript_chapters=transcript_chapters,
        )

    def _normalize_segments(
        self, transcript_segments: Sequence[TranscriptSegment] | Sequence[dict]
    ) -> list[TranscriptSegment]:
        normalized: list[TranscriptSegment] = []
        for segment in transcript_segments:
            if isinstance(segment, TranscriptSegment):
                normalized.append(segment)
                continue
            normalized.append(
                TranscriptSegment(
                    start=float(segment["start"]),
                    end=float(segment["end"]),
                    text=str(segment.get("text", "")).strip(),
                    language=segment.get("language"),
                    speaker_id=segment.get("speaker_id"),
                    speaker_confidence=segment.get("speaker_confidence"),
                    metadata=dict(segment.get("metadata", {})),
                )
            )
        return sorted(normalized, key=lambda item: (item.start, item.end))

    def _normalize_occupancy_frame(self, occupancy_df: pd.DataFrame) -> pd.DataFrame:
        if occupancy_df is None:
            raise ValueError("occupancy_df cannot be None")
        frame = occupancy_df.copy()
        if frame.empty:
            return pd.DataFrame(columns=["elapsed_seconds", "occupants"])

        time_col = self._resolve_time_column(frame, preferred=self.config.occupancy_time_column)
        count_col = self._resolve_count_column(frame)

        frame = frame.rename(columns={time_col: "elapsed_seconds", count_col: "occupants"})
        frame["elapsed_seconds"] = pd.to_numeric(frame["elapsed_seconds"], errors="coerce")
        frame["occupants"] = pd.to_numeric(frame["occupants"], errors="coerce")
        frame = frame.dropna(subset=["elapsed_seconds", "occupants"]).copy()
        frame["elapsed_seconds"] = frame["elapsed_seconds"].astype(float)
        frame["occupants"] = frame["occupants"].astype(int)
        frame = frame.sort_values("elapsed_seconds").drop_duplicates("elapsed_seconds", keep="last")
        frame.reset_index(drop=True, inplace=True)
        return frame

    def _normalize_danmaku_frame(
        self,
        danmaku_df: pd.DataFrame | Sequence[DanmakuMessage] | Sequence[dict] | None,
    ) -> pd.DataFrame:
        if danmaku_df is None:
            return pd.DataFrame(
                columns=[
                    "elapsed_seconds",
                    "text",
                    "user_id",
                    "user_name",
                    "sentiment_score",
                    "message_id",
                ]
            )

        if isinstance(danmaku_df, pd.DataFrame):
            frame = danmaku_df.copy()
        else:
            rows: list[dict] = []
            for message in danmaku_df:
                rows.append(message.to_dict() if isinstance(message, DanmakuMessage) else dict(message))
            frame = pd.DataFrame(rows)

        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "elapsed_seconds",
                    "text",
                    "user_id",
                    "user_name",
                    "sentiment_score",
                    "message_id",
                ]
            )

        time_col = self._resolve_time_column(frame, preferred=self.config.danmaku_time_column)
        text_col = self._resolve_danmaku_text_column(frame)
        user_id_col = self._resolve_optional_column(frame, self.config.danmaku_user_id_column, self.SUPPORTED_DANMAKU_USER_ID_COLUMNS)
        user_name_col = self._resolve_optional_column(frame, self.config.danmaku_user_name_column, self.SUPPORTED_DANMAKU_USER_NAME_COLUMNS)
        sentiment_col = self._resolve_optional_column(
            frame,
            self.config.danmaku_sentiment_column,
            self.SUPPORTED_DANMAKU_SENTIMENT_COLUMNS,
        )
        message_id_col = "message_id" if "message_id" in frame.columns else None

        rename_map = {time_col: "elapsed_seconds", text_col: "text"}
        if user_id_col:
            rename_map[user_id_col] = "user_id"
        if user_name_col:
            rename_map[user_name_col] = "user_name"
        if sentiment_col:
            rename_map[sentiment_col] = "sentiment_score"
        if message_id_col:
            rename_map[message_id_col] = "message_id"

        frame = frame.rename(columns=rename_map)
        for required_column in ("user_id", "user_name", "sentiment_score", "message_id"):
            if required_column not in frame.columns:
                frame[required_column] = None

        frame["elapsed_seconds"] = pd.to_numeric(frame["elapsed_seconds"], errors="coerce")
        frame["sentiment_score"] = pd.to_numeric(frame["sentiment_score"], errors="coerce")
        frame["text"] = frame["text"].fillna("").astype(str).str.strip()
        frame = frame.dropna(subset=["elapsed_seconds"]).copy()
        frame = frame.loc[frame["text"] != ""].copy()
        frame["elapsed_seconds"] = frame["elapsed_seconds"].astype(float)
        frame = frame.sort_values("elapsed_seconds").reset_index(drop=True)
        return frame[["elapsed_seconds", "text", "user_id", "user_name", "sentiment_score", "message_id"]]

    def _resolve_time_column(self, frame: pd.DataFrame, preferred: str | None = None) -> str:
        if preferred and preferred in frame.columns:
            return preferred
        for column in self.SUPPORTED_TIME_COLUMNS:
            if column in frame.columns:
                return column
        if isinstance(frame.index, pd.RangeIndex):
            frame["elapsed_seconds"] = frame.index.astype(float)
            return "elapsed_seconds"
        raise ValueError(
            "Unable to detect occupancy time column. "
            f"Tried: {', '.join(self.SUPPORTED_TIME_COLUMNS)}"
        )

    def _resolve_count_column(self, frame: pd.DataFrame) -> str:
        if self.config.occupancy_count_column in frame.columns:
            return self.config.occupancy_count_column
        for column in self.SUPPORTED_COUNT_COLUMNS:
            if column in frame.columns:
                return column
        raise ValueError(
            "Unable to detect occupancy count column. "
            f"Tried: {', '.join(self.SUPPORTED_COUNT_COLUMNS)}"
        )

    def _resolve_danmaku_text_column(self, frame: pd.DataFrame) -> str:
        if self.config.danmaku_text_column in frame.columns:
            return self.config.danmaku_text_column
        for column in self.SUPPORTED_DANMAKU_TEXT_COLUMNS:
            if column in frame.columns:
                return column
        raise ValueError(
            "Unable to detect danmaku text column. "
            f"Tried: {', '.join(self.SUPPORTED_DANMAKU_TEXT_COLUMNS)}"
        )

    def _resolve_optional_column(
        self,
        frame: pd.DataFrame,
        preferred: str | None,
        candidates: Sequence[str],
    ) -> str | None:
        if preferred and preferred in frame.columns:
            return preferred
        for column in candidates:
            if column in frame.columns:
                return column
        return None

    def _compute_window_stats(
        self,
        occupancy_df: pd.DataFrame,
        window_start: float,
        window_end: float,
    ) -> OccupancyStats:
        if window_end < window_start:
            window_start, window_end = window_end, window_start

        window = occupancy_df.loc[
            (occupancy_df["elapsed_seconds"] >= window_start)
            & (occupancy_df["elapsed_seconds"] <= window_end)
        ].copy()

        if window.empty:
            return OccupancyStats(
                window_start=window_start,
                window_end=window_end,
                sample_count=0,
                mean=None,
                peak=None,
                minimum=None,
                delta=None,
                pct_change=None,
                slope_per_sec=None,
                slope_per_min=None,
            )

        first_count = int(window["occupants"].iloc[0])
        last_count = int(window["occupants"].iloc[-1])
        delta = last_count - first_count
        duration = max(window_end - window_start, 1e-6)
        pct_change = None if first_count <= 0 else delta / first_count
        slope_per_sec = delta / duration

        return OccupancyStats(
            window_start=window_start,
            window_end=window_end,
            sample_count=int(len(window)),
            mean=round(float(window["occupants"].mean()), 4),
            peak=int(window["occupants"].max()),
            minimum=int(window["occupants"].min()),
            delta=round(float(delta), 4),
            pct_change=round(float(pct_change), 6) if pct_change is not None else None,
            slope_per_sec=round(float(slope_per_sec), 6),
            slope_per_min=round(float(slope_per_sec * 60.0), 6),
        )

    def _compute_danmaku_window_stats(
        self,
        danmaku_df: pd.DataFrame,
        window_start: float,
        window_end: float,
    ) -> DanmakuStats:
        if window_end < window_start:
            window_start, window_end = window_end, window_start

        window = danmaku_df.loc[
            (danmaku_df["elapsed_seconds"] >= window_start)
            & (danmaku_df["elapsed_seconds"] <= window_end)
        ].copy()

        if window.empty:
            return DanmakuStats(
                window_start=window_start,
                window_end=window_end,
                message_count=0,
                unique_user_count=0,
                avg_sentiment=None,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                sample_messages=[],
            )

        unique_user_count = int(
            window["user_id"].dropna().astype(str).nunique()
            if window["user_id"].notna().any()
            else window["user_name"].dropna().astype(str).nunique()
        )
        valid_sentiment = window["sentiment_score"].dropna()
        positive_count = int((valid_sentiment > 0.2).sum())
        negative_count = int((valid_sentiment < -0.2).sum())
        neutral_count = int(len(valid_sentiment) - positive_count - negative_count)
        sample_messages = [
            message
            for message in window["text"].dropna().astype(str).drop_duplicates().head(self.config.danmaku_sample_limit)
        ]

        return DanmakuStats(
            window_start=window_start,
            window_end=window_end,
            message_count=int(len(window)),
            unique_user_count=unique_user_count,
            avg_sentiment=round(float(valid_sentiment.mean()), 6) if not valid_sentiment.empty else None,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            sample_messages=sample_messages,
        )

    def _compute_occupancy_summary(self, occupancy_df: pd.DataFrame) -> dict[str, float | int | None]:
        if occupancy_df.empty:
            return {
                "sample_count": 0,
                "mean": None,
                "median": None,
                "max": None,
                "min": None,
                "p10": None,
                "p25": None,
                "p75": None,
                "p90": None,
                "iqr": None,
                "stddev": None,
                "duration_seconds": 0.0,
                "duration_minutes": 0.0,
                "coefficient_of_variation": None,
            }

        values = occupancy_df["occupants"].astype(float)
        mean = float(values.mean())
        stddev = float(values.std(ddof=0)) if len(values) > 1 else 0.0
        p25 = float(values.quantile(0.25))
        p75 = float(values.quantile(0.75))
        duration_seconds = float(occupancy_df["elapsed_seconds"].max() - occupancy_df["elapsed_seconds"].min())
        return {
            "sample_count": int(len(values)),
            "mean": round(mean, 4),
            "median": round(float(values.median()), 4),
            "max": int(values.max()),
            "min": int(values.min()),
            "p10": round(float(values.quantile(0.10)), 4),
            "p25": round(p25, 4),
            "p75": round(p75, 4),
            "p90": round(float(values.quantile(0.90)), 4),
            "iqr": round(float(p75 - p25), 4),
            "stddev": round(stddev, 4),
            "duration_seconds": round(duration_seconds, 4),
            "duration_minutes": round(duration_seconds / 60.0, 4),
            "coefficient_of_variation": round(stddev / mean, 6) if mean > 0 else None,
        }

    def _build_occupancy_timeline_blocks(
        self,
        occupancy_df: pd.DataFrame,
        summary: dict[str, float | int | None],
        *,
        block_seconds: int,
    ) -> list[dict[str, float | int | str | None]]:
        if occupancy_df.empty:
            return []
        block_size = max(60, int(block_seconds or 300))
        timeline_start = float(occupancy_df["elapsed_seconds"].min())
        timeline_end = float(occupancy_df["elapsed_seconds"].max())
        blocks: list[dict[str, float | int | str | None]] = []
        current = timeline_start
        while current <= timeline_end:
            block_end = min(current + block_size, timeline_end)
            window = occupancy_df.loc[
                (occupancy_df["elapsed_seconds"] >= current)
                & (occupancy_df["elapsed_seconds"] <= block_end)
            ]
            block = self._occupancy_block_from_window(
                window,
                block_start=current,
                block_end=block_end,
                summary=summary,
            )
            block["block_id"] = f"occ_{len(blocks):04d}"
            blocks.append(block)
            if block_end >= timeline_end:
                break
            current = block_end
        return blocks

    def _occupancy_block_from_window(
        self,
        window: pd.DataFrame,
        *,
        block_start: float,
        block_end: float,
        summary: dict[str, float | int | None],
    ) -> dict[str, float | int | str | None]:
        if window.empty:
            return {
                "start": round(block_start, 4),
                "end": round(block_end, 4),
                "sample_count": 0,
                "count_start": None,
                "count_end": None,
                "count_min": None,
                "count_max": None,
                "count_mean": None,
                "delta": None,
                "slope_per_min": None,
                "relative_level": "unknown",
                "movement": "missing",
            }
        first_count = int(window["occupants"].iloc[0])
        last_count = int(window["occupants"].iloc[-1])
        delta = last_count - first_count
        duration_minutes = max((block_end - block_start) / 60.0, 1e-6)
        count_mean = float(window["occupants"].mean())
        return {
            "start": round(block_start, 4),
            "end": round(block_end, 4),
            "sample_count": int(len(window)),
            "count_start": first_count,
            "count_end": last_count,
            "count_min": int(window["occupants"].min()),
            "count_max": int(window["occupants"].max()),
            "count_mean": round(count_mean, 4),
            "delta": int(delta),
            "slope_per_min": round(float(delta / duration_minutes), 6),
            "relative_level": self._relative_level(last_count, summary),
            "movement": self._classify_occupancy_block_movement(
                first_count=first_count,
                last_count=last_count,
                count_mean=count_mean,
                delta=delta,
                summary=summary,
            ),
        }

    def _classify_occupancy_block_movement(
        self,
        *,
        first_count: int,
        last_count: int,
        count_mean: float,
        delta: int,
        summary: dict[str, float | int | None],
    ) -> str:
        iqr = max(float(summary.get("iqr") or 0), 1.0)
        median = float(summary.get("median") or 0)
        p25 = float(summary.get("p25") or 0)
        p75 = float(summary.get("p75") or 0)
        p90 = float(summary.get("p90") or 0)
        small_change = max(3.0, iqr * 0.15)
        if count_mean >= p90 and abs(delta) <= small_change:
            return "high_plateau"
        if count_mean >= p75 and abs(delta) <= small_change:
            return "stable_high"
        if first_count <= p25 and last_count >= median:
            return "recovery"
        if delta >= max(3.0, iqr * 0.25):
            return "rising"
        if delta <= -max(3.0, iqr * 0.25):
            return "falling"
        return "stable"

    def _build_transcript_chapters(
        self,
        aligned_segments: Sequence[AlignedSegment],
        occupancy_blocks: Sequence[dict[str, float | int | str | None]],
        *,
        max_text_chars: int,
    ) -> list[dict[str, object]]:
        chapters: list[dict[str, object]] = []
        for index, block in enumerate(occupancy_blocks):
            start = float(block["start"] or 0)
            end = float(block["end"] or start)
            block_segments = [
                segment
                for segment in aligned_segments
                if segment.end >= start and segment.start <= end
            ]
            text = " ".join(segment.text.strip() for segment in block_segments if segment.text.strip())
            nearby_anchor_ids = sorted(
                {
                    anchor_id
                    for segment in block_segments
                    for anchor_id in segment.nearby_anchor_ids
                }
            )
            chapters.append(
                {
                    "chapter_id": f"chapter_{index:04d}",
                    "start": start,
                    "end": end,
                    "segment_count": len(block_segments),
                    "text_excerpt": self._truncate_text(text, max_text_chars),
                    "nearby_anchor_ids": nearby_anchor_ids,
                    "occupancy": dict(block),
                }
            )
        return chapters

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        normalized = " ".join(text.split())
        limit = max(200, int(max_chars or 1800))
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(0, limit - 3)].rstrip() + "..."

    def _score_anchor_candidate(
        self,
        *,
        baseline: int,
        target: int,
        abs_change: int,
        pct_change: float,
        slope_per_min: float,
        window: pd.DataFrame,
        summary: dict[str, float | int | None],
        config: AlignmentConfig,
    ) -> dict[str, float]:
        mean = float(summary["mean"] or 0)
        median = float(summary["median"] or 0)
        p25 = float(summary["p25"] or 0)
        p75 = float(summary["p75"] or 0)
        p90 = float(summary["p90"] or 0)
        p10 = float(summary["p10"] or 0)
        iqr = max(float(summary["iqr"] or 0), 1.0)
        stddev = max(float(summary["stddev"] or 0), 1.0)

        abs_scale = max(iqr, mean * 0.05, 1.0)
        abs_change_score = min(abs(abs_change) / abs_scale, 3.0)
        pct_change_score = min(abs(pct_change) / max(config.anchor_pct_threshold, 0.05), 3.0)

        distribution_score = 0.0
        if abs_change > 0:
            if baseline < median <= target:
                distribution_score += 0.35
            if target >= p75:
                distribution_score += 0.75
            if target >= p90:
                distribution_score += 0.75
        elif abs_change < 0:
            if baseline > median >= target:
                distribution_score += 0.35
            if target <= p25:
                distribution_score += 0.75
            if target <= p10:
                distribution_score += 0.75

        slope_score = min(abs(slope_per_min) / stddev, 3.0)
        peak_trough_score = 0.0
        if target >= p90 or target == summary["max"]:
            peak_trough_score += 1.0
        if target <= p10 or target == summary["min"]:
            peak_trough_score += 1.0

        if not window.empty:
            local_range = int(window["occupants"].max()) - int(window["occupants"].min())
            if local_range >= max(3, iqr):
                slope_score = min(slope_score + 0.5, 3.0)

        score = (
            abs_change_score * 0.9
            + pct_change_score * 0.65
            + distribution_score * 0.8
            + slope_score * 0.75
            + peak_trough_score * 0.65
        )

        return {
            "abs_change_score": abs_change_score,
            "pct_change_score": pct_change_score,
            "distribution_score": distribution_score,
            "slope_score": slope_score,
            "peak_trough_score": peak_trough_score,
            "score": score,
        }

    def _classify_anchor_direction(
        self,
        *,
        baseline: int,
        target: int,
        abs_change: int,
        pct_change: float,
        window: pd.DataFrame,
        summary: dict[str, float | int | None],
    ) -> str:
        median = float(summary["median"] or 0)
        p25 = float(summary["p25"] or 0)
        p90 = float(summary["p90"] or 0)
        p10 = float(summary["p10"] or 0)

        if target >= p90 and abs_change >= 0:
            return "peak"
        if target <= p10 and abs_change <= 0:
            return "trough"
        if baseline <= p25 and target >= median and abs_change > 0:
            return "recovery"
        if not window.empty:
            local_range = int(window["occupants"].max()) - int(window["occupants"].min())
            if local_range >= max(4, abs(abs_change) * 1.5) and abs(pct_change) >= 0.12:
                return "volatile"
        return "surge" if abs_change > 0 else "drop"

    def _classify_anchor_priority(self, score: float, config: AlignmentConfig) -> str:
        if score >= config.anchor_high_score_threshold:
            return "strong"
        if score >= max(2.0, config.anchor_score_threshold):
            return "medium"
        return "weak"

    def _is_low_viewer_noise(
        self,
        abs_change: int,
        scores: dict[str, float],
        summary: dict[str, float | int | None],
        config: AlignmentConfig,
    ) -> bool:
        median = float(summary["median"] or 0)
        if median > config.low_viewer_median_threshold:
            return False
        if abs(abs_change) > config.low_viewer_small_change:
            return False
        return scores["distribution_score"] < 1.8

    def _relative_level(self, target: int, summary: dict[str, float | int | None]) -> str:
        value = float(target)
        if value >= float(summary["p90"] or 0):
            return "p90_or_above"
        if value >= float(summary["p75"] or 0):
            return "p75_to_p90"
        if value <= float(summary["p10"] or 0):
            return "p10_or_below"
        if value <= float(summary["p25"] or 0):
            return "p10_to_p25"
        return "middle"

    def _build_reason_hint(
        self,
        direction: str,
        pct_change: float,
        abs_change: int,
        score: float,
        priority: str,
    ) -> str:
        direction_cn = {
            "surge": "上涨",
            "drop": "下跌",
            "peak": "接近高点",
            "trough": "接近低点",
            "recovery": "低位回升",
            "volatile": "强波动",
        }.get(direction, direction)
        return f"人数{direction_cn}，窗口变化 {pct_change:.1%}，绝对变化 {abs_change}，评分 {score:.2f}，优先级 {priority}。"
