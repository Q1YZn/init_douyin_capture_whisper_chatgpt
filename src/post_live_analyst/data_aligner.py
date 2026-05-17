from __future__ import annotations

import json
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


class DataAligner:
    """Fuse transcript timestamps with occupancy time series metrics."""

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
                    if not line:
                        continue
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

            if baseline <= 0:
                pct_change = 1.0 if target > 0 else 0.0
            else:
                pct_change = abs_change / baseline

            if abs(abs_change) < config.anchor_abs_threshold:
                continue
            if abs(pct_change) < config.anchor_pct_threshold:
                continue
            if current_ts - last_anchor_at < config.anchor_cooldown_seconds:
                continue

            direction = "surge" if pct_change > 0 else "drop"
            slope_per_min = abs_change / (config.anchor_window_seconds / 60.0)
            anchors.append(
                AnchorEvent(
                    anchor_id=f"anchor_{len(anchors):04d}",
                    timestamp=float(future_row["elapsed_seconds"]),
                    window_start=current_ts,
                    window_end=float(future_row["elapsed_seconds"]),
                    direction=direction,
                    abs_change=abs_change,
                    pct_change=round(pct_change, 6),
                    baseline_count=baseline,
                    target_count=target,
                    slope_per_min=round(slope_per_min, 4),
                    reason_hint=self._build_reason_hint(direction, pct_change, abs_change),
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
        anchors = self.detect_anchor_events(normalized_occupancy, config=config)
        aligned_segments = self.align_segments(
            transcript_segments=transcript_segments or [],
            occupancy_df=normalized_occupancy,
            danmaku_df=normalized_danmaku,
            anchors=anchors,
        )

        coverage = {
            "segment_count": len(aligned_segments),
            "anchor_count": len(anchors),
            "occupancy_sample_count": int(len(normalized_occupancy)),
            "danmaku_message_count": int(len(normalized_danmaku)),
            "timeline_start": float(normalized_occupancy["elapsed_seconds"].min()) if not normalized_occupancy.empty else None,
            "timeline_end": float(normalized_occupancy["elapsed_seconds"].max()) if not normalized_occupancy.empty else None,
            "asr_available": bool(transcript_segments),
            "danmaku_available": not normalized_danmaku.empty,
        }
        return DetailedTimeline(
            aligned_segments=aligned_segments,
            anchors=anchors,
            coverage=coverage,
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

    def _normalize_danmaku_frame(self, danmaku_df: pd.DataFrame | Sequence[DanmakuMessage] | Sequence[dict] | None) -> pd.DataFrame:
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
                if isinstance(message, DanmakuMessage):
                    rows.append(message.to_dict())
                else:
                    rows.append(dict(message))
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

    def _build_reason_hint(self, direction: str, pct_change: float, abs_change: int) -> str:
        direction_cn = "激增" if direction == "surge" else "骤降"
        return f"人数{direction_cn}，窗口变化 {pct_change:.1%}，绝对变化 {abs_change}。"
