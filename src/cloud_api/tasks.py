from __future__ import annotations

import json

from redis import Redis
from rq import Queue

from .config import ServerSettings
from .db import get_session
from .models import AnalysisJob
from .reporting import ReportBuilder


settings = ServerSettings.from_env()
redis_conn = Redis.from_url(settings.redis_url)
queue = Queue("reports", connection=redis_conn)


def enqueue_report_job(job_id: str) -> str:
    job = queue.enqueue(generate_report, job_id)
    return job.id


def generate_report(job_id: str) -> None:
    report_builder = ReportBuilder(settings.report_root)
    with get_session() as session:
        analysis_job = session.get(AnalysisJob, job_id)
        if analysis_job is None:
            return
        timeline = json.loads(analysis_job.timeline_json)
        markdown_text = report_builder.build_markdown(timeline, analysis_job.summary_markdown)
        html_text = report_builder.markdown_to_html(markdown_text)
        _, html_path, _ = report_builder.persist_report(job_id, markdown_text, html_text, timeline)
        analysis_job.status = "completed"
        analysis_job.summary_markdown = markdown_text
        analysis_job.report_html = html_text
        analysis_job.report_path = str(html_path)
        session.add(analysis_job)
        session.commit()
