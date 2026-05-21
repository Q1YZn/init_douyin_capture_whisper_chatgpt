from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .config import ServerSettings
from .db import Base, engine, ensure_analysis_job_columns, get_session
from .models import AnalysisJob
from .schemas import AnalysisJobCreate, AnalysisJobResponse, ClipCandidatesRequest, ClipCandidatesResponse
from .tasks import cancel_enqueued_report_job, enqueue_report_job, generate_clip_candidates, generate_report


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
            message="Report job queued",
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
    status = "queued"
    message = "Report job queued"
    error: str | None = None
    updated_at = None
    report_path: str | None = None
    rq_job_id: str | None = None
    try:
        rq_job_id = enqueue_report_job(job_id)
        with get_session() as session:
            job = session.get(AnalysisJob, job_id)
            if job:
                job.rq_job_id = rq_job_id
                session.add(job)
                session.commit()
    except Exception:
        try:
            generate_report(job_id)
            with get_session() as session:
                job = session.get(AnalysisJob, job_id)
                if job:
                    status = job.status
                    message = job.message or message
                    error = job.error
                    updated_at = job.updated_at
                    report_path = job.report_path
        except Exception as exc:
            status = "pending_worker"
            message = "Report worker unavailable; job is waiting for retry"
            error = str(exc)[:2000]
            with get_session() as session:
                job = session.get(AnalysisJob, job_id)
                if job:
                    job.status = status
                    job.message = message
                    job.error = error
                    session.add(job)
                    session.commit()
                    updated_at = job.updated_at
    return AnalysisJobResponse(
        job_id=job_id,
        status=status,
        message=message,
        error=error,
        updated_at=updated_at,
        report_url=f"{settings.public_base_url}/api/v1/analysis-jobs/{job_id}/report" if status == "completed" or report_path else None,
        rq_job_id=rq_job_id,
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
        return AnalysisJobResponse(
            job_id=job.job_id,
            status=job.status,
            report_url=report_url,
            message=job.message,
            error=job.error,
            updated_at=job.updated_at,
            rq_job_id=job.rq_job_id,
        )


@app.post("/api/v1/analysis-jobs/{job_id}/cancel", response_model=AnalysisJobResponse)
def cancel_analysis_job(job_id: str) -> AnalysisJobResponse:
    with get_session() as session:
        job = session.get(AnalysisJob, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status in {"completed", "failed", "cancelled"}:
            report_url = (
                f"{settings.public_base_url}/api/v1/analysis-jobs/{job_id}/report"
                if job.report_path
                else None
            )
            return AnalysisJobResponse(
                job_id=job.job_id,
                status=job.status,
                report_url=report_url,
                message=job.message,
                error=job.error,
                updated_at=job.updated_at,
                rq_job_id=job.rq_job_id,
            )
        removed = cancel_enqueued_report_job(job.rq_job_id) if job.status in {"queued", "pending_worker"} else False
        job.status = "cancelled"
        job.message = "Cloud analysis cancelled"
        job.error = None
        try:
            timeline = json.loads(job.timeline_json)
            selection = timeline.setdefault("analysis_selection", {})
            if isinstance(selection, dict):
                selection["status"] = "cancelled"
            job.timeline_json = json.dumps(timeline, ensure_ascii=False)
        except Exception:
            pass
        session.add(job)
        session.commit()
        return AnalysisJobResponse(
            job_id=job.job_id,
            status=job.status,
            report_url=None,
            message=job.message,
            error=job.error,
            updated_at=job.updated_at,
            rq_job_id=job.rq_job_id,
        )


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


@app.post("/api/v1/clip-candidates", response_model=ClipCandidatesResponse)
def create_clip_candidates(payload: ClipCandidatesRequest) -> ClipCandidatesResponse:
    try:
        result = generate_clip_candidates(
            payload.timeline,
            summary_markdown=payload.summary_markdown,
            max_candidates=payload.max_candidates or 20,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ClipCandidatesResponse(
        job_id=uuid.uuid4().hex,
        status="completed",
        candidates=result["candidates"],
        model_id=result.get("model_id"),
        raw_response=result.get("raw_response"),
    )
