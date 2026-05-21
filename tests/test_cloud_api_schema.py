from __future__ import annotations

import json
import threading
import time
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


def test_server_settings_clamps_anchor_concurrency(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_ANCHOR_CONCURRENCY", "20")
    assert ServerSettings.from_env().deepseek_anchor_concurrency == 20

    monkeypatch.setenv("DEEPSEEK_ANCHOR_CONCURRENCY", "bad")
    assert ServerSettings.from_env().deepseek_anchor_concurrency == 20

    monkeypatch.setenv("DEEPSEEK_ANCHOR_CONCURRENCY", "500")
    assert ServerSettings.from_env().deepseek_anchor_concurrency == 50

    monkeypatch.setenv("DEEPSEEK_ANCHOR_CONCURRENCY", "0")
    assert ServerSettings.from_env().deepseek_anchor_concurrency == 1


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
        self.rq_job_id = None


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

    def refresh(self, _job) -> None:
        return None


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

    assert commits == ["running", "running", "completed"]
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

    assert commits == ["running", "running", "failed"]
    assert job.status == "failed"
    assert job.message == "Report generation failed"
    assert job.error == "DeepSeek timeout"
    assert json.loads(job.timeline_json)["analysis_selection"]["status"] == "failed"


def test_cancel_enqueued_report_job_fetches_and_deletes_rq_job(monkeypatch) -> None:
    calls: list[str] = []

    class FakeRqJob:
        def cancel(self) -> None:
            calls.append("cancel")

        def delete(self) -> None:
            calls.append("delete")

    class FakeQueue:
        def fetch_job(self, rq_job_id):
            calls.append(f"fetch:{rq_job_id}")
            return FakeRqJob()

    monkeypatch.setattr(cloud_tasks, "queue", FakeQueue())

    assert cloud_tasks.cancel_enqueued_report_job("rq-job-1") is True
    assert calls == ["fetch:rq-job-1", "cancel", "delete"]


def test_generate_report_stops_before_report_when_job_is_cancelled(monkeypatch, tmp_path) -> None:
    job = _FakeCloudJob()
    commits: list[str] = []

    class CancellingSession(_FakeSession):
        def refresh(self, _job) -> None:
            self.job.status = "cancelled"
            self.job.message = "Cloud analysis cancelled"

    @contextmanager
    def fake_get_session():
        yield CancellingSession(job, commits)

    monkeypatch.setattr(cloud_tasks, "get_session", fake_get_session)
    monkeypatch.setattr(cloud_tasks, "ReportBuilder", _FakeReportBuilder)
    monkeypatch.setattr(cloud_tasks.settings, "report_root", tmp_path)
    monkeypatch.setattr(cloud_tasks, "_run_global_timeline_analysis", lambda timeline: timeline)
    monkeypatch.setattr(
        cloud_tasks,
        "_run_deepseek_anchor_analysis",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("anchor analysis should not run")),
    )

    cloud_tasks.generate_report(job.job_id)

    assert "completed" not in commits
    assert job.status == "cancelled"
    assert json.loads(job.timeline_json)["analysis_selection"]["status"] == "cancelled"


def _anchor_payload(anchor_id: str, timestamp: float) -> dict:
    return {
        "anchor_id": anchor_id,
        "timestamp": timestamp,
        "window_start": max(0.0, timestamp - 10),
        "window_end": timestamp + 10,
        "direction": "surge",
        "abs_change": 100,
        "pct_change": 0.1,
        "baseline_count": 1000,
        "target_count": 1100,
        "slope_per_min": 10.0,
        "reason_hint": "test",
        "score": 2.0,
        "priority": "medium",
    }


def _segment_payload(anchor_id: str, start: float) -> dict:
    return {
        "segment_id": f"segment-{anchor_id}",
        "start": start,
        "end": start + 5,
        "text": f"text for {anchor_id}",
        "occupancy": {
            "window_start": start,
            "window_end": start + 5,
            "sample_count": 1,
            "mean": 1000,
            "peak": 1100,
            "minimum": 900,
            "delta": 100,
            "pct_change": 0.1,
            "slope_per_sec": 1.0,
            "slope_per_min": 60.0,
        },
        "nearby_anchor_ids": [anchor_id],
    }


def _timeline_for_anchor_concurrency() -> dict:
    anchors = [
        _anchor_payload("anchor_0003", 30.0),
        _anchor_payload("anchor_0001", 10.0),
        _anchor_payload("anchor_0002", 20.0),
        _anchor_payload("anchor_0004", 40.0),
        _anchor_payload("anchor_0005", 50.0),
        _anchor_payload("anchor_0006", 60.0),
    ]
    return {
        "anchors": anchors,
        "aligned_segments": [
            _segment_payload(anchor["anchor_id"], float(anchor["timestamp"]) - 2)
            for anchor in anchors
        ],
        "occupancy_summary": {"duration_minutes": 30},
        "coverage": {"asr_available": True, "danmaku_available": False, "timeline_start": 0, "timeline_end": 60},
        "occupancy_timeline_blocks": [
            {
                "block_id": "occ_0000",
                "start": 0,
                "end": 60,
                "count_start": 1000,
                "count_end": 1100,
                "count_min": 900,
                "count_max": 1200,
                "count_mean": 1050,
                "delta": 100,
                "slope_per_min": 100,
                "relative_level": "p75_to_p90",
                "movement": "rising",
            }
        ],
        "transcript_chapters": [
            {
                "chapter_id": "chapter_0000",
                "start": 0,
                "end": 60,
                "segment_count": 6,
                "text_excerpt": "complete transcript view",
                "nearby_anchor_ids": ["anchor_0001"],
                "occupancy": {"movement": "rising"},
            }
        ],
    }


def test_deepseek_anchor_analysis_runs_concurrently_and_preserves_time_order(monkeypatch) -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()

    class FakeResult:
        def __init__(self, anchor_id: str) -> None:
            self.anchor_id = anchor_id

        def to_dict(self) -> dict:
            return {
                "anchor_id": self.anchor_id,
                "status": "ok",
                "model_id": "fake-model",
                "response_text": f"analysis {self.anchor_id}",
            }

    class FakeAnalyst:
        def __init__(self, config) -> None:
            self.config = config

        def analyze_anchor(self, anchor, _segments, occupancy_summary=None):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03 if anchor.anchor_id == "anchor_0001" else 0.01)
            with lock:
                active -= 1
            return FakeResult(anchor.anchor_id)

    monkeypatch.setattr(cloud_tasks.settings, "deepseek_api_key", "key")
    monkeypatch.setattr(cloud_tasks.settings, "deepseek_anchor_concurrency", 3)
    monkeypatch.setattr(cloud_tasks.settings, "deepseek_max_anchors", 40)
    monkeypatch.setattr(cloud_tasks, "CliProxyAgentAnalyst", FakeAnalyst)

    progress: list[tuple[int, int, str]] = []
    timeline = cloud_tasks._run_deepseek_anchor_analysis(
        _timeline_for_anchor_concurrency(),
        progress_callback=lambda done, total, anchor: progress.append((done, total, anchor["anchor_id"])),
    )

    assert max_active > 1
    assert [item["anchor_id"] for item in timeline["agent_analyses"]] == [
        "anchor_0001",
        "anchor_0002",
        "anchor_0003",
        "anchor_0004",
        "anchor_0005",
        "anchor_0006",
    ]
    assert progress[-1][:2] == (6, 6)
    assert timeline["analysis_selection"]["status"] == "completed"
    assert timeline["analysis_selection"]["concurrency"] == 3


def test_deepseek_anchor_analysis_keeps_report_when_one_anchor_fails(monkeypatch) -> None:
    class FakeAnalyst:
        def __init__(self, config) -> None:
            self.config = config

        def analyze_anchor(self, anchor, _segments, occupancy_summary=None):
            if anchor.anchor_id == "anchor_0002":
                raise RuntimeError("rate limited")

            class FakeResult:
                def to_dict(self_inner) -> dict:
                    return {"anchor_id": anchor.anchor_id, "status": "ok", "model_id": "fake-model"}

            return FakeResult()

    monkeypatch.setattr(cloud_tasks.settings, "deepseek_api_key", "key")
    monkeypatch.setattr(cloud_tasks.settings, "deepseek_anchor_concurrency", 20)
    monkeypatch.setattr(cloud_tasks.settings, "deepseek_max_anchors", 40)
    monkeypatch.setattr(cloud_tasks, "CliProxyAgentAnalyst", FakeAnalyst)

    timeline = cloud_tasks._run_deepseek_anchor_analysis(_timeline_for_anchor_concurrency())
    analyses = timeline["agent_analyses"]

    assert timeline["analysis_selection"]["status"] == "completed_with_model_errors"
    assert timeline["analysis_selection"]["ok_count"] == 5
    assert timeline["analysis_selection"]["error_count"] == 1
    failed = [item for item in analyses if item["status"] == "error"]
    assert failed[0]["anchor_id"] == "anchor_0002"
    assert failed[0]["error"] == "rate limited"


def test_global_timeline_analysis_uses_chapters_and_occupancy_blocks(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "overall_summary": "global view",
                                    "candidate_intervals": [
                                        {
                                            "start": 10,
                                            "end": 30,
                                            "title": "slow burn",
                                            "reason": "ASR and occupancy rose together",
                                            "confidence": 0.8,
                                            "evidence_chapter_ids": ["chapter_0000"],
                                        }
                                    ],
                                    "uncertainty": "danmaku unavailable",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    def fake_post(_url, **kwargs):
        captured["payload"] = kwargs["json"]
        return FakeResponse()

    monkeypatch.setattr(cloud_tasks.settings, "deepseek_api_key", "key")
    monkeypatch.setattr(cloud_tasks.requests, "post", fake_post)

    timeline = cloud_tasks._run_global_timeline_analysis(_timeline_for_anchor_concurrency())

    user_content = captured["payload"]["messages"][1]["content"]
    assert "complete transcript view" in user_content
    assert "occupancy_timeline_blocks" in user_content
    assert "danmaku_available" in user_content
    assert timeline["global_analysis"]["status"] == "completed"
    assert timeline["global_analysis"]["candidate_intervals"][0]["start"] == 10.0


def test_global_candidate_intervals_influence_anchor_selection() -> None:
    timeline = _timeline_for_anchor_concurrency()
    timeline["global_analysis"] = {
        "candidate_intervals": [
            {"start": 45, "end": 65, "title": "late topic"},
        ]
    }

    selection = cloud_tasks._select_anchors_for_deepseek(timeline, max_anchors=3)

    assert selection["strategy"] == "global_candidate_intervals_then_anchor_score"
    assert selection["global_candidate_interval_count"] == 1
    assert selection["global_preferred_anchor_count"] == 2
    assert {"anchor_0005", "anchor_0006"}.issubset(set(selection["selected_anchor_ids"]))


def test_global_analysis_failure_keeps_anchor_fallback(monkeypatch) -> None:
    def fake_post(_url, **_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(cloud_tasks.settings, "deepseek_api_key", "key")
    monkeypatch.setattr(cloud_tasks.requests, "post", fake_post)

    timeline = cloud_tasks._run_global_timeline_analysis(_timeline_for_anchor_concurrency())
    selection = cloud_tasks._select_anchors_for_deepseek(timeline, max_anchors=3)

    assert timeline["global_analysis"]["status"] == "failed"
    assert selection["strategy"] == "dynamic_by_duration_score_distribution"
    assert selection["selected_anchor_count"] > 0
