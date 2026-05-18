from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .config import DesktopSettings
from .models import FollowingProfile


class FollowingSyncError(RuntimeError):
    """Raised when Douyin following profiles cannot be fetched."""


@dataclass(slots=True)
class FollowingFetchResult:
    source_account_sec_uid: str
    profiles: list[FollowingProfile]


class DouyinFollowingFetcher:
    def __init__(self, settings: DesktopSettings) -> None:
        self.settings = settings

    def fetch(self, source_account: str) -> FollowingFetchResult:
        return asyncio.run(self._fetch_async(source_account))

    async def _fetch_async(self, source_account: str) -> FollowingFetchResult:
        source_account = source_account.strip()
        if not source_account:
            raise FollowingSyncError("Fill in a source profile URL or sec_uid before syncing following profiles.")

        try:
            from f2.apps.douyin.handler import DouyinHandler
            from f2.apps.douyin.utils import SecUserIdFetcher
        except ImportError as exc:
            raise FollowingSyncError("f2 is not installed; cannot sync Douyin following profiles.") from exc

        source_sec_uid = await self._resolve_sec_uid(source_account, SecUserIdFetcher)
        kwargs = self._build_f2_kwargs(source_account)
        handler = DouyinHandler(kwargs)

        profiles_by_sec_uid: dict[str, FollowingProfile] = {}
        try:
            async for page in handler.fetch_user_following(sec_user_id=source_sec_uid, count=20):
                self._raise_for_page_status(page)
                for item in self._page_to_items(page):
                    profile = self._item_to_profile(item)
                    if profile is None:
                        continue
                    profiles_by_sec_uid[profile.sec_uid] = profile
        except Exception as exc:
            raise FollowingSyncError(f"Failed to fetch following profiles: {exc}") from exc

        return FollowingFetchResult(
            source_account_sec_uid=source_sec_uid,
            profiles=list(profiles_by_sec_uid.values()),
        )

    async def _resolve_sec_uid(self, source_account: str, fetcher: Any) -> str:
        if source_account.startswith(("http://", "https://")):
            try:
                return str(await fetcher.get_sec_user_id(source_account)).strip()
            except Exception as exc:
                raise FollowingSyncError(f"Failed to resolve source profile URL: {exc}") from exc
        return source_account

    def _build_f2_kwargs(self, source_account: str) -> dict[str, Any]:
        return {
            "headers": {
                "User-Agent": self.settings.douyin_user_agent,
                "Referer": "https://www.douyin.com/",
            },
            "proxies": {"http://": None, "https://": None},
            "timeout": self.settings.capture_timeout_seconds,
            "cookie": self.settings.douyin_cookie,
            "url": source_account,
        }

    @staticmethod
    def _raise_for_page_status(page: Any) -> None:
        status_code = getattr(page, "status_code", None)
        if status_code in (None, 0, "0"):
            return
        status_msg = getattr(page, "status_msg", "") or getattr(page, "msg", "")
        raise FollowingSyncError(f"Douyin following API returned status {status_code}: {status_msg}")

    @staticmethod
    def _page_to_items(page: Any) -> list[dict[str, Any]]:
        if hasattr(page, "_to_list"):
            items = page._to_list()
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        if hasattr(page, "_to_dict"):
            data = page._to_dict()
            followings = data.get("followings") if isinstance(data, dict) else None
            if isinstance(followings, list):
                return [item for item in followings if isinstance(item, dict)]
        return []

    @classmethod
    def _item_to_profile(cls, item: dict[str, Any]) -> FollowingProfile | None:
        sec_uid = str(item.get("sec_uid") or item.get("sec_user_id") or "").strip()
        if not sec_uid:
            return None
        uid = str(item.get("uid") or "").strip() or None
        nickname = str(item.get("nickname") or item.get("nickname_raw") or "").strip() or None
        return FollowingProfile(
            profile_url=cls.profile_url_for_sec_uid(sec_uid),
            sec_uid=sec_uid,
            uid=uid,
            nickname=nickname,
        )

    @staticmethod
    def profile_url_for_sec_uid(sec_uid: str) -> str:
        return f"https://www.douyin.com/user/{sec_uid}?from_tab_name=live"
