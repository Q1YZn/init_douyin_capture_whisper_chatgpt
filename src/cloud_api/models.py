from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    client_job_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    summary_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_html: Mapped[str | None] = mapped_column(Text, nullable=True)
    timeline_json: Mapped[str] = mapped_column(Text)
    report_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_size_bytes: Mapped[int | None] = mapped_column(nullable=True)
    asset_storage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    asset_uri: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    analysis_tier: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payment_required: Mapped[bool | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
