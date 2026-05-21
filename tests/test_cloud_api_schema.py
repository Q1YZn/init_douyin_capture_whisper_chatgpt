from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

import cloud_api.tasks as cloud_tasks
from cloud_api.config import ServerSettings
from cloud_api.schemas import AnalysisJobCreate, AnalysisJobResponse


def test_analysis_job_create_accepts_delivery_metadata() -> None:
    payload = AnalysisJobCreate(
        client_job_id="job-1",
        summary_markdown="# Summary",
        timeline={"anchors": []},
        video_size_bytes=1024,
        asset_storage="oss_pending",
        asset_uri=None,
        analysis_tier="paid_hotspot",
        payment_required=True,
    )

    assert payload.video_size_bytes == 1024
    assert payload.asset_storage == "oss_pending"
    assert payload.analysis_tier == "paid_hotspot"
    assert payload.payment_required is True


def test_analysis_job_response_exposes_diagnostics() -> None:
    response = AnalysisJobResponse(
        job_id="job-1",
        status="failed",
        message="Report generation failed",
        error="DeepSeek timeout",
    )

    assert response.message == "Report generation failed"
    assert response.error == "DeepSeek timeout"


def test_server_settings_exposes_report_job_timeout(monkeypatch) -> None:
    monkeypatch.setenv("REPORT_JOB_TIMEOUT_SECONDS", "900")

    assert ServerSettings.from_env().report_job_timeout_seconds == 900


def test_server_settings_exposes_deepseek_timeout(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_TIMEOUT_SECONDS", "45")

    assert ServerSettings.from_env().deepseek_timeout_seconds == 45


def test_enqueue_report_job_uses_configured_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeQueue:
        def enqueue(self, func, job_id, **kwargs):
            captured["func"] = func
            captured["job_id"] = job_id
            captured["kwargs"] = kwargs

            class FakeJob:
                id = "rq-job-1"

            return FakeJob()

    monkeypatch.setattr(cloud_tasks, "queue", FakeQueue())
    monkeypatch.setattr(cloud_tasks.settings, "report_job_timeout_seconds", 900)

    assert cloud_tasks.enqueue_report_job("remote-1") == "rq-job-1"
    assert captured["job_id"] == "remote-1"
    assert captured["kwargs"] == {"job_timeout": 900}


class _FakeCloudJob:
    def __init__(self) -> None:
        self.job_id = "remote-1"
        self.timeline_json = json.dumps({"anchors": [], "aligned_segments": []}, ensure_ascii=False)
        self.summary_markdown = "# Summary"
        self.status = "queued"
        self.message = None
        self.error = None
        self.report_html = None
        self.report_path = None


class _FakeSession:
    def __init__(self, job: _FakeCloudJob, commits: list[str]) -> None:
        self.job = job
        self.commits = commits

    def get(self, _model, _job_id):
        return self.job

    def add(self, _job) -> None:
        return None

    def commit(self) -> None:
        self.commits.append(self.job.status)


class _FakeReportBuilder:
    def __init__(self, report_root: Path) -> None:
        self.report_root = report_root

    def build_markdown(self, _timeline, summary_markdown):
        return summary_markdown or "# Summary"

    def markdown_to_html(self, markdown_text):
        return f"<html>{markdown_text}</html>"

    def persist_report(self, job_id, _markdown_text, _html_text, _timeline):
        return (
            self.report_root / job_id / "report.md",
            self.report_root / job_id / "report.html",
            self.report_root / job_id / "timeline.json",
        )


def test_generate_report_marks_running_then_completed(monkeypatch, tmp_path) -> None:
    job = _FakeCloudJob()
    commits: list[str] = []

    @contextmanager
    def fake_get_session():
        yield _FakeSession(job, commits)

    monkeypatch.setattr(cloud_tasks, "get_session", fake_get_session)
    monkeypatch.setattr(cloud_tasks, "ReportBuilder", _FakeReportBuilder)
    monkeypatch.setattr(cloud_tasks.settings, "report_root", tmp_path)
    def complete_analysis(timeline, **_kwargs):
        return {**timeline, "analysis_selection": {"status": "completed"}}

    monkeypatch.setattr(cloud_tasks, "_run_deepseek_anchor_analysis", complete_analysis)

    cloud_tasks.generate_report(job.job_id)

    assert commits == ["running", "completed"]
    assert job.status == "completed"
    assert job.message == "Report completed"
    assert job.error is None


def test_generate_report_marks_failed_on_exception(monkeypatch, tmp_path) -> None:
    job = _FakeCloudJob()
    commits: list[str] = []

    @contextmanager
    def fake_get_session():
        yield _FakeSession(job, commits)

    def fail_analysis(_timeline, **_kwargs):
        raise RuntimeError("DeepSeek timeout")

    monkeypatch.setattr(cloud_tasks, "get_session", fake_get_session)
    monkeypatch.setattr(cloud_tasks, "ReportBuilder", _FakeReportBuilder)
    monkeypatch.setattr(cloud_tasks.settings, "report_root", tmp_path)
    monkeypatch.setattr(cloud_tasks, "_run_deepseek_anchor_analysis", fail_analysis)

    with pytest.raises(RuntimeError, match="DeepSeek timeout"):
        cloud_tasks.generate_report(job.job_id)

    assert commits == ["running", "failed"]
    assert job.status == "failed"
    assert job.message == "Report generation failed"
    assert job.error == "DeepSeek timeout"
    assert json.loads(job.timeline_json)["analysis_selection"]["status"] == "failed"
