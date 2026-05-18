from __future__ import annotations

from pathlib import Path

from desktop_client.models import DesktopAnalysisRequest
from desktop_client.notifications import MemoryPressureWatcher, RobotNotificationConfig, RobotNotifier


def test_memory_pressure_watcher_honors_threshold_and_cooldown() -> None:
    values = iter([1024, 1024, 4096])
    watcher = MemoryPressureWatcher(get_available_memory_mb=lambda: next(values))
    config = RobotNotificationConfig(
        provider="feishu",
        webhook_url="https://example.test/webhook",
        memory_threshold_mb=2048,
        cooldown_minutes=30,
    )

    should_alert, available = watcher.should_alert(config)
    assert should_alert is True
    assert available == 1024

    should_alert, available = watcher.should_alert(config)
    assert should_alert is False
    assert available == 1024

    should_alert, available = watcher.should_alert(config)
    assert should_alert is False
    assert available == 4096


def test_robot_notifier_payload_shapes() -> None:
    feishu_payload = RobotNotifier._build_payload("feishu", "hello")
    assert feishu_payload == {
        "msg_type": "text",
        "content": {"text": "hello"},
    }

    dingtalk_payload = RobotNotifier._build_payload("dingtalk", "hello")
    assert dingtalk_payload == {
        "msgtype": "text",
        "text": {"content": "hello"},
    }


def test_analysis_request_accepts_provider_overrides() -> None:
    request = DesktopAnalysisRequest(
        replay_path=Path("replay.flv"),
        occupancy_path=Path("occupants.csv"),
        workspace_dir=Path("jobs"),
        analysis_provider="deepseek",
        analysis_base_url="https://api.deepseek.com",
        analysis_api_key="secret",
        analysis_model_id="deepseek-v4-pro",
    )
    assert request.analysis_provider == "deepseek"
    assert request.analysis_base_url == "https://api.deepseek.com"
    assert request.analysis_api_key == "secret"
    assert request.analysis_model_id == "deepseek-v4-pro"
