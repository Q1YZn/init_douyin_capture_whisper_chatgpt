from __future__ import annotations

from pathlib import Path

from desktop_client import services
from desktop_client.config import DesktopSettings
from desktop_client.db import ClientJobRepository
from desktop_client.following_sync import DouyinFollowingFetcher, FollowingFetchResult, FollowingSyncError
from desktop_client.models import FollowingProfile, MonitorProfileSource
from desktop_client.models import DesktopAnalysisRequest
from desktop_client.notifications import MemoryPressureWatcher, RobotNotificationConfig, RobotNotifier
from desktop_client.services import MonitorProfileService


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


def test_following_sync_adds_updates_and_removes_only_synced_profiles(tmp_path, monkeypatch) -> None:
    settings = DesktopSettings(workspace_root=tmp_path, sqlite_path=tmp_path / "client.db")
    repo = ClientJobRepository(settings.sqlite_path)
    repo.upsert_monitor_profile(
        "https://www.douyin.com/user/manual?from_tab_name=live",
        enabled=True,
        source_type=MonitorProfileSource.MANUAL.value,
    )
    repo.upsert_monitor_profile(
        "https://www.douyin.com/user/old_synced?from_tab_name=live",
        enabled=True,
        source_type=MonitorProfileSource.FOLLOWING_SYNC.value,
        source_account_sec_uid="source",
        source_follow_sec_uid="old_synced",
    )
    repo.upsert_monitor_profile(
        "https://www.douyin.com/user/kept_synced?from_tab_name=live",
        enabled=False,
        source_type=MonitorProfileSource.FOLLOWING_SYNC.value,
        source_account_sec_uid="source",
        source_follow_sec_uid="kept_synced",
    )

    class FakeFetcher:
        def __init__(self, _settings):
            pass

        def fetch(self, _source_account):
            return FollowingFetchResult(
                source_account_sec_uid="source",
                profiles=[
                    FollowingProfile(
                        profile_url="https://www.douyin.com/user/kept_synced?from_tab_name=live",
                        sec_uid="kept_synced",
                        uid="1",
                        nickname="Kept",
                    ),
                    FollowingProfile(
                        profile_url="https://www.douyin.com/user/new_synced?from_tab_name=live",
                        sec_uid="new_synced",
                        uid="2",
                        nickname="New",
                    ),
                ],
            )

    monkeypatch.setattr(services, "DouyinFollowingFetcher", FakeFetcher)
    result = MonitorProfileService(repo, settings).sync_following_profiles("source")

    assert result.added == 1
    assert result.updated == 1
    assert result.removed == 1
    records = {record.profile_url: record for record in repo.list_monitor_profiles()}
    assert "https://www.douyin.com/user/manual?from_tab_name=live" in records
    assert "https://www.douyin.com/user/old_synced?from_tab_name=live" not in records
    assert records["https://www.douyin.com/user/kept_synced?from_tab_name=live"].enabled is False
    assert (
        records["https://www.douyin.com/user/new_synced?from_tab_name=live"].source_type
        == MonitorProfileSource.FOLLOWING_SYNC.value
    )


def test_following_sync_rejects_nonzero_api_status() -> None:
    class FailedPage:
        status_code = 2096
        status_msg = "private account"

    try:
        DouyinFollowingFetcher._raise_for_page_status(FailedPage())
    except FollowingSyncError as exc:
        assert "2096" in str(exc)
    else:
        raise AssertionError("Expected FollowingSyncError for nonzero status")
