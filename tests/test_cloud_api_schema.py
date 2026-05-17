from __future__ import annotations

from cloud_api.schemas import AnalysisJobCreate


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
