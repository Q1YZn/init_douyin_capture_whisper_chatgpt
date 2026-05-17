from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AnalysisJobCreate(BaseModel):
    client_job_id: str
    summary_markdown: str | None = None
    timeline: dict[str, Any]
    video_size_bytes: int | None = None
    asset_storage: str | None = None
    asset_uri: str | None = None
    analysis_tier: str | None = None
    payment_required: bool | None = None


class AnalysisJobResponse(BaseModel):
    job_id: str
    status: str
    report_url: str | None = None
