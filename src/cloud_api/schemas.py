from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


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
    message: str | None = None
    error: str | None = None
    updated_at: datetime | None = None
    rq_job_id: str | None = None


class ClipCandidatesRequest(BaseModel):
    client_job_id: str
    timeline: dict[str, Any]
    summary_markdown: str | None = None
    max_candidates: int | None = 20


class ClipCandidatesResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    job_id: str
    status: str
    candidates: list[dict[str, Any]]
    model_id: str | None = None
    raw_response: str | None = None
