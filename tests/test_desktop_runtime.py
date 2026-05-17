from __future__ import annotations

from desktop_client.f2_runtime import extract_viewer_count, safe_int
from desktop_client.live_capture import MonitorEvent
from desktop_client.models import MonitorEventSeverity, WorkflowStage, WorkflowStatus


def test_extract_viewer_count_supports_nested_payloads() -> None:
    payload = {"room_view_stats": {"total_user_str": "1.2w"}}
    assert extract_viewer_count(payload) == 12000


def test_safe_int_supports_wan_and_commas() -> None:
    assert safe_int("1.5w") == 15000
    assert safe_int("2万") == 20000
    assert safe_int("1,234") == 1234


def test_monitor_event_infers_ui_state() -> None:
    event = MonitorEvent("capture_finished", "https://example.test/user", "Capture finished.")
    assert event.severity == MonitorEventSeverity.SUCCESS
    assert event.stage == WorkflowStage.CAPTURE
    assert event.status == WorkflowStatus.SUCCESS
