from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .config import ServerSettings
from .db import Base, engine, ensure_analysis_job_columns, get_session
from .models import AnalysisJob
from .schemas import AnalysisJobCreate, AnalysisJobResponse
from .tasks import enqueue_report_job


settings = ServerSettings.from_env()
app = FastAPI(title="Douyin Stream Ops Cloud API")
Base.metadata.create_all(bind=engine)
ensure_analysis_job_columns()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/analysis-jobs", response_model=AnalysisJobResponse)
def create_analysis_job(payload: AnalysisJobCreate) -> AnalysisJobResponse:
    job_id = uuid.uuid4().hex
    with get_session() as session:
        job = AnalysisJob(
            job_id=job_id,
            client_job_id=payload.client_job_id,
            status="queued",
            summary_markdown=payload.summary_markdown,
            timeline_json=json.dumps(payload.timeline, ensure_ascii=False),
            video_size_bytes=payload.video_size_bytes,
            asset_storage=payload.asset_storage,
            asset_uri=payload.asset_uri,
            analysis_tier=payload.analysis_tier,
            payment_required=payload.payment_required,
        )
        session.add(job)
        session.commit()
    try:
        enqueue_report_job(job_id)
    except Exception:
        with get_session() as session:
            job = session.get(AnalysisJob, job_id)
            if job:
                job.status = "pending_worker"
                session.add(job)
                session.commit()
    return AnalysisJobResponse(
        job_id=job_id,
        status="queued",
        report_url=f"{settings.public_base_url}/api/v1/analysis-jobs/{job_id}/report",
    )


@app.get("/api/v1/analysis-jobs/{job_id}", response_model=AnalysisJobResponse)
def get_analysis_job(job_id: str) -> AnalysisJobResponse:
    with get_session() as session:
        job = session.get(AnalysisJob, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        report_url = (
            f"{settings.public_base_url}/api/v1/analysis-jobs/{job_id}/report"
            if job.report_path
            else None
        )
        return AnalysisJobResponse(job_id=job.job_id, status=job.status, report_url=report_url)


@app.get("/api/v1/analysis-jobs/{job_id}/report")
def get_report(job_id: str) -> FileResponse:
    with get_session() as session:
        job = session.get(AnalysisJob, job_id)
        if job is None or not job.report_path:
            raise HTTPException(status_code=404, detail="report not ready")
        path = Path(job.report_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="report file missing")
        return FileResponse(path, media_type="text/html")
