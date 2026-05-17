from __future__ import annotations

import asyncio
import contextlib
import csv
import json
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qs, urlparse

import httpx
import requests

from .config import DesktopSettings
from .f2_runtime import extract_viewer_count, safe_int
from .models import MonitorEventSeverity, WorkflowStage, WorkflowStatus


@dataclass(slots=True)
class LiveCaptureRequest:
    room_id_or_url: str
    workspace_dir: Path


@dataclass(slots=True)
class LiveCaptureResult:
    room_id: str
    webcast_id: str | None
    replay_path: Path | None
    occupancy_path: Path
    danmaku_path: Path
    metadata_path: Path
    workspace_dir: Path
    stopped_by_user: bool = False


@dataclass(slots=True)
class MultiProfileMonitorRequest:
    profile_urls: list[str]
    workspace_dir: Path
    poll_interval_seconds: float | None = None


@dataclass(slots=True)
class MonitorEvent:
    event_type: str
    profile_url: str
    message: str
    nickname: str | None = None
    room_id: str | None = None
    webcast_id: str | None = None
    replay_path: Path | None = None
    workspace_dir: Path | None = None
    severity: MonitorEventSeverity | None = None
    stage: WorkflowStage | None = None
    status: WorkflowStatus | None = None

    def __post_init__(self) -> None:
        if self.severity is None:
            self.severity = self._infer_severity(self.event_type)
        if self.stage is None:
            self.stage = self._infer_stage(self.event_type)
        if self.status is None:
            self.status = self._infer_status(self.event_type)

    @staticmethod
    def _infer_severity(event_type: str) -> MonitorEventSeverity:
        if event_type in {"capture_finished", "live_detected"}:
            return MonitorEventSeverity.SUCCESS
        if event_type in {"monitor_error", "capture_failed"}:
            return MonitorEventSeverity.ERROR
        if event_type in {"offline", "monitor_stopped"}:
            return MonitorEventSeverity.WARNING
        return MonitorEventSeverity.INFO

    @staticmethod
    def _infer_stage(event_type: str) -> WorkflowStage:
        if event_type.startswith("capture"):
            return WorkflowStage.CAPTURE
        if event_type.startswith("monitor") or event_type in {"live_detected", "offline"}:
            return WorkflowStage.MONITOR
        return WorkflowStage.IDLE

    @staticmethod
    def _infer_status(event_type: str) -> WorkflowStatus:
        if event_type == "monitor_started":
            return WorkflowStatus.RUNNING
        if event_type in {"capture_finished", "live_detected"}:
            return WorkflowStatus.SUCCESS
        if event_type in {"monitor_error", "capture_failed"}:
            return WorkflowStatus.ERROR
        if event_type in {"monitor_stopped", "offline"}:
            return WorkflowStatus.STOPPED
        return WorkflowStatus.IDLE


class F2LiveCaptureService:
    """Capture replay, occupancy history and danmaku for a live room."""

    def __init__(self, settings: DesktopSettings) -> None:
        self.settings = settings
        self._stop_requested = Event()

    def run(self, request: LiveCaptureRequest) -> LiveCaptureResult:
        self._stop_requested.clear()
        return asyncio.run(self._run_async(request))

    def request_stop(self) -> None:
        self._stop_requested.set()

    @staticmethod
    def configure_f2_console_logging() -> None:
        logger = logging.getLogger("f2")
        for handler in logger.handlers:
            if handler.__class__.__name__ != "RichHandler":
                continue
            handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )

    def build_capture_cookie(self, token_manager: Any | None = None) -> str:
        if self.settings.douyin_cookie:
            return self.settings.douyin_cookie
        if token_manager is None:
            return ""
        return (
            f"ttwid={token_manager.gen_ttwid()}; "
            '__live_version__=%221.1.2.6631%22; '
            'live_use_vvc=%22false%22;'
        )

    def build_capture_kwargs(self, capture_cookie: str) -> dict[str, Any]:
        return {
            "headers": {
                "User-Agent": self.settings.douyin_user_agent,
                "Referer": "https://www.douyin.com/",
                "Content-Type": "application/protobuffer;",
            },
            "proxies": {"http://": None, "https://": None},
            "timeout": self.settings.capture_timeout_seconds,
            "mode": "live",
            "cookie": capture_cookie,
            "naming": "{create}_{nickname}_{aweme_id}",
            "folderize": False,
        }

    def build_wss_kwargs(self) -> dict[str, Any]:
        return {
            "headers": {
                "User-Agent": self.settings.douyin_user_agent,
                "Upgrade": "websocket",
                "Connection": "Upgrade",
            },
            "proxies": {"http://": None, "https://": None},
            "timeout": self.settings.capture_timeout_seconds,
            "show_message": False,
            "cookie": "",
        }

    async def _run_async(self, request: LiveCaptureRequest) -> LiveCaptureResult:
        try:
            from f2.apps.douyin.dl import DouyinDownloader
            from f2.apps.douyin.handler import DouyinHandler
            from f2.apps.douyin.utils import TokenManager
        except ImportError as exc:
            raise RuntimeError(
                "f2 is not installed in the current environment. "
                "Please install requirements-desktop.txt before using live capture."
            ) from exc
        self.configure_f2_console_logging()

        capture_root = request.workspace_dir
        capture_root.mkdir(parents=True, exist_ok=True)

        capture_cookie = self.build_capture_cookie(TokenManager)
        kwargs = self.build_capture_kwargs(capture_cookie)
        wss_kwargs = self.build_wss_kwargs()

        handler = DouyinHandler(kwargs)
        downloader = DouyinDownloader(kwargs)
        if hasattr(handler, "enable_bark"):
            handler.enable_bark = False
        if hasattr(downloader, "enable_bark"):
            downloader.enable_bark = False

        room = await self._fetch_room(handler, request.room_id_or_url)
        room_dict = self._to_dict(room)

        resolved_ids = self._extract_ids_from_input(request.room_id_or_url)
        room_id = str(room_dict.get("room_id") or resolved_ids.get("room_id") or "")
        webcast_id = self._extract_webcast_id(room_dict) or resolved_ids.get("webcast_id")
        if not room_id and webcast_id is None:
            raise RuntimeError(
                "Unable to resolve a live room from the input. Use a live link, room_id, webcast_id, or profile URL."
            )
        if room_dict.get("live_status") != 2:
            anchor_name = self._extract_anchor_name(room_dict) or room_id or str(webcast_id or "unknown")
            raise RuntimeError(f"Anchor {anchor_name} is not live right now.")

        capture_stem = self._build_capture_stem(room_dict)
        workspace_dir = self._ensure_unique_dir(capture_root / f"{capture_stem}_live")
        workspace_dir.mkdir(parents=True, exist_ok=True)
        replay_target_stem = f"{capture_stem}_replay"
        occupancy_path = workspace_dir / f"{capture_stem}_occupants.csv"
        danmaku_path = workspace_dir / f"{capture_stem}_danmaku.jsonl"
        metadata_path = workspace_dir / f"{capture_stem}_metadata.json"
        occupancy_path.write_text("elapsed_seconds,occupants,sampled_at,room_id,webcast_id\n", encoding="utf-8")
        danmaku_path.touch()
        metadata_path.write_text(json.dumps(room_dict, ensure_ascii=False, indent=2), encoding="utf-8")

        stop_event = asyncio.Event()
        start_ts = time.time()

        download_task = asyncio.create_task(
            self._download_stream(
                downloader,
                kwargs,
                self._prepare_stream_download_payload(room_dict),
                workspace_dir,
                stop_event,
            )
        )
        occupancy_task = asyncio.create_task(
            self._poll_occupancy(handler, room_id, webcast_id, occupancy_path, start_ts, stop_event)
        )
        danmaku_task = asyncio.create_task(
            self._capture_danmaku(
                handler,
                room_id,
                webcast_id,
                danmaku_path,
                start_ts,
                capture_cookie,
                wss_kwargs,
                stop_event,
            )
        )
        stop_watch_task = asyncio.create_task(self._watch_stop_request(stop_event, download_task))

        replay_path: Path | None = None
        stopped_by_user = False
        try:
            replay_path = await download_task
            replay_path = self._normalize_replay_path(replay_path, workspace_dir, replay_target_stem)
        except asyncio.CancelledError:
            stopped_by_user = True
            replay_path = self._normalize_replay_path(
                self._find_latest_video_file(workspace_dir),
                workspace_dir,
                replay_target_stem,
            )
        finally:
            stop_event.set()
            stop_watch_task.cancel()
            await asyncio.gather(stop_watch_task, occupancy_task, danmaku_task, return_exceptions=True)
            self._cleanup_zero_byte_videos(workspace_dir)

        return LiveCaptureResult(
            room_id=room_id,
            webcast_id=str(webcast_id) if webcast_id is not None else None,
            replay_path=replay_path,
            occupancy_path=occupancy_path,
            danmaku_path=danmaku_path,
            metadata_path=metadata_path,
            workspace_dir=workspace_dir,
            stopped_by_user=stopped_by_user,
        )

    async def _fetch_room(self, handler: Any, room_id_or_url: str) -> Any:
        normalized = room_id_or_url.strip()
        if normalized.startswith("http"):
            normalized = self._resolve_share_url(normalized)
            parsed = urlparse(normalized)
            query = parse_qs(parsed.query)

            if "webcast/reflow" in normalized:
                room_id = parsed.path.rstrip("/").split("/")[-1]
                return await handler.fetch_user_live_videos_by_room_id(room_id=room_id)

            if "live.douyin.com" in normalized:
                room_id = query.get("room_id", [None])[0]
                webcast_id = parsed.path.rstrip("/").split("/")[-1]
                if room_id and str(room_id).isdigit() and len(str(room_id)) >= 15:
                    return await handler.fetch_user_live_videos_by_room_id(room_id=str(room_id))
                return await handler.fetch_user_live_videos(webcast_id=str(webcast_id))

            if self._is_profile_url(parsed):
                return await self._fetch_room_from_profile_url(handler, normalized)

        if normalized.isdigit() and len(normalized) >= 15:
            return await handler.fetch_user_live_videos_by_room_id(room_id=normalized)
        return await handler.fetch_user_live_videos(webcast_id=normalized)

    async def _fetch_room_from_profile_url(self, handler: Any, profile_url: str) -> Any:
        profile_dict = await self._fetch_profile_snapshot(handler, profile_url)
        sec_user_id = profile_dict.get("sec_user_id")
        nickname = self._extract_anchor_name(profile_dict) or str(sec_user_id or profile_url)
        room_id = profile_dict.get("room_id")
        if not room_id:
            raise RuntimeError(f"Anchor {nickname} does not expose a live room right now.")

        room = await handler.fetch_user_live_videos_by_room_id(room_id=str(room_id))
        room_dict = self._to_dict(room)
        if not room_dict.get("room_id"):
            room_dict["room_id"] = str(room_id)
        if not room_dict.get("nickname") and profile_dict.get("nickname"):
            room_dict["nickname"] = profile_dict.get("nickname")
        if room_dict.get("owner") is None and profile_dict.get("nickname"):
            room_dict["owner"] = {"nickname": profile_dict.get("nickname")}
        if room_dict.get("live_status") is None and profile_dict.get("live_status") is not None:
            room_dict["live_status"] = profile_dict.get("live_status")
        return room_dict

    async def _fetch_profile_snapshot(self, handler: Any, profile_url: str) -> dict[str, Any]:
        try:
            from f2.apps.douyin.utils import SecUserIdFetcher
        except ImportError as exc:
            raise RuntimeError("f2 profile parsing helpers are not available.") from exc

        sec_user_id = await SecUserIdFetcher.get_sec_user_id(profile_url)
        profile = await handler.fetch_user_profile(sec_user_id)
        profile_dict = self._to_dict(profile)
        if not profile_dict.get("sec_user_id"):
            profile_dict["sec_user_id"] = sec_user_id
        profile_dict["profile_url"] = profile_url
        return profile_dict

    async def _download_stream(
        self,
        downloader: Any,
        download_kwargs: dict[str, Any],
        room_dict: dict[str, Any],
        workspace_dir: Path,
        stop_event: asyncio.Event,
    ) -> Path | None:
        flv_urls = self._ordered_stream_urls(room_dict.get("flv_pull_url") or {})
        if flv_urls:
            replay_path = await self._download_flv_stream(
                downloader=downloader,
                stream_urls=flv_urls,
                workspace_dir=workspace_dir,
                room_dict=room_dict,
                stop_event=stop_event,
            )
            if replay_path is not None and replay_path.exists():
                stop_event.set()
                return replay_path

        stream_map = room_dict.get("m3u8_pull_url") or {}
        if not stream_map or not any(stream_map.values()):
            raise RuntimeError(
                "The live room did not return a downloadable stream URL. "
                "This is usually caused by missing cookies, interface limits, or the stream not being ready yet."
            )
        await downloader.create_stream_tasks(download_kwargs, room_dict, workspace_dir)
        stop_event.set()
        replay_path = self._find_latest_video_file(workspace_dir)
        if replay_path is None:
            raise RuntimeError("The HLS download finished but no replay file was created locally.")
        return replay_path

    async def _download_flv_stream(
        self,
        downloader: Any,
        stream_urls: list[str],
        workspace_dir: Path,
        room_dict: dict[str, Any],
        stop_event: asyncio.Event,
    ) -> Path | None:
        file_stem = (
            f"{self._sanitize_path_component(self._extract_anchor_name(room_dict) or str(room_dict.get('room_id') or 'live'))}_"
            f"{int(time.time())}_live"
        )
        output_path = workspace_dir / f"{file_stem}.flv"

        for stream_url in stream_urls:
            bytes_written = 0
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("wb") as handle:
                    async with downloader.aclient.stream("GET", stream_url, headers=downloader.headers) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_bytes(262144):
                            if stop_event.is_set() or self._stop_requested.is_set():
                                break
                            if not chunk:
                                continue
                            handle.write(chunk)
                            handle.flush()
                            bytes_written += len(chunk)
                if bytes_written > 0:
                    return output_path
            except asyncio.CancelledError:
                if output_path.exists():
                    with contextlib.suppress(OSError):
                        output_path.unlink()
                raise
            except httpx.HTTPError:
                pass

            if output_path.exists():
                with contextlib.suppress(OSError):
                    output_path.unlink()

        return None

    async def _watch_stop_request(self, stop_event: asyncio.Event, download_task: asyncio.Task[Path | None]) -> None:
        while not stop_event.is_set():
            if self._stop_requested.is_set():
                stop_event.set()
                download_task.cancel()
                return
            await asyncio.sleep(0.2)

    async def _poll_occupancy(
        self,
        handler: Any,
        room_id: str,
        webcast_id: Any,
        occupancy_path: Path,
        start_ts: float,
        stop_event: asyncio.Event,
    ) -> None:
        with occupancy_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["elapsed_seconds", "occupants", "sampled_at", "room_id", "webcast_id"],
            )
            writer.writeheader()
            while not stop_event.is_set():
                try:
                    room = (
                        await handler.fetch_user_live_videos_by_room_id(room_id=room_id)
                        if room_id and len(room_id) >= 15
                        else await handler.fetch_user_live_videos(webcast_id=str(webcast_id or room_id))
                    )
                    room_dict = self._to_dict(room)
                    writer.writerow(
                        {
                            "elapsed_seconds": round(time.time() - start_ts, 3),
                            "occupants": self._extract_viewer_count(room_dict),
                            "sampled_at": int(time.time()),
                            "room_id": room_dict.get("room_id") or room_id,
                            "webcast_id": self._extract_webcast_id(room_dict) or webcast_id,
                        }
                    )
                    handle.flush()
                    if room_dict.get("live_status") != 2:
                        stop_event.set()
                        break
                except Exception:
                    pass
                await asyncio.sleep(self.settings.capture_poll_interval_seconds)

    async def _capture_danmaku(
        self,
        handler: Any,
        room_id: str,
        webcast_id: Any,
        danmaku_path: Path,
        start_ts: float,
        capture_cookie: str,
        wss_kwargs: dict[str, Any],
        stop_event: asyncio.Event,
    ) -> None:
        try:
            from f2.apps.douyin.handler import DouyinHandler
        except ImportError:
            return

        guest_handler = handler
        user = None
        if self.settings.douyin_cookie:
            try:
                user = await guest_handler.fetch_query_user()
            except Exception:
                user = None

        if room_id and len(room_id) >= 15:
            room = await guest_handler.fetch_user_live_videos_by_room_id(room_id=room_id)
        else:
            live_identifier = webcast_id or room_id
            if not live_identifier:
                return
            room = await guest_handler.fetch_user_live_videos(webcast_id=str(live_identifier))

        room_dict = self._to_dict(room)
        live_room_id = room_dict.get("room_id") or room_id
        if not live_room_id:
            return

        live_im = await guest_handler.fetch_live_im(
            room_id=live_room_id,
            unique_id=getattr(user, "user_unique_id", None),
        )

        async def callback_factory(message_type: str) -> Callable[..., Awaitable[None]]:
            async def _callback(message: Any, *args: Any, **kwargs: Any) -> None:
                payload = self._to_dict(message)
                record = {
                    "elapsed_seconds": round(time.time() - start_ts, 3),
                    "message_type": message_type,
                    "room_id": live_room_id,
                    "webcast_id": self._extract_webcast_id(room_dict) or webcast_id,
                    "text": self._extract_message_text(payload),
                    "user_id": self._extract_user_id(payload),
                    "user_name": self._extract_user_name(payload),
                    "payload": payload,
                }
                with danmaku_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            return _callback

        message_types = [
            "WebcastChatMessage",
            "WebcastRoomUserSeqMessage",
            "WebcastRoomStatsMessage",
            "WebcastSocialMessage",
            "WebcastMemberMessage",
        ]
        callbacks: dict[str, Any] = {}
        for name in message_types:
            callbacks[name] = await callback_factory(name)

        wss_handler = DouyinHandler({**wss_kwargs, "cookie": capture_cookie if self.settings.douyin_cookie else ""})
        try:
            await wss_handler.fetch_live_danmaku(
                room_id=live_room_id,
                user_unique_id=getattr(user, "user_unique_id", None),
                internal_ext=getattr(live_im, "internal_ext", ""),
                cursor=getattr(live_im, "cursor", ""),
                wss_callbacks=callbacks,
            )
        except Exception:
            stop_event.set()

    def _to_dict(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if hasattr(value, "_to_dict"):
            try:
                return value._to_dict()
            except Exception:
                pass
        raw: dict[str, Any] = {}
        for attr in dir(value):
            if attr.startswith("_"):
                continue
            try:
                item = getattr(value, attr)
            except Exception:
                continue
            if callable(item):
                continue
            if isinstance(item, (str, int, float, bool, type(None), list, dict)):
                raw[attr] = item
        return raw

    def _extract_viewer_count(self, room_dict: dict[str, Any]) -> int:
        return extract_viewer_count(room_dict)

    def _extract_message_text(self, payload: dict[str, Any]) -> str:
        for key in ("content", "text", "message", "msg", "common", "chat_content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                inner = value.get("content") or value.get("text")
                if isinstance(inner, str) and inner.strip():
                    return inner.strip()
        return json.dumps(payload, ensure_ascii=False)

    def _extract_user_id(self, payload: dict[str, Any]) -> str | None:
        for key in ("user_id", "uid", "sec_uid"):
            value = payload.get(key)
            if value:
                return str(value)
        user = payload.get("user") or {}
        for key in ("id", "uid", "sec_uid"):
            value = user.get(key)
            if value:
                return str(value)
        return None

    def _extract_user_name(self, payload: dict[str, Any]) -> str | None:
        for key in ("nickname", "user_name", "display_name"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        user = payload.get("user") or {}
        for key in ("nickname", "display_name", "name"):
            value = user.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _find_latest_video_file(self, workspace_dir: Path) -> Path | None:
        candidates = [
            candidate
            for candidate in (
                list(workspace_dir.rglob("*.flv"))
                + list(workspace_dir.rglob("*.mp4"))
                + list(workspace_dir.rglob("*.mkv"))
            )
            if candidate.exists() and candidate.stat().st_size > 0
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.stat().st_mtime)

    def _cleanup_zero_byte_videos(self, workspace_dir: Path) -> None:
        for candidate in list(workspace_dir.rglob("*.flv")) + list(workspace_dir.rglob("*.mp4")) + list(
            workspace_dir.rglob("*.mkv")
        ):
            with contextlib.suppress(OSError):
                if candidate.exists() and candidate.stat().st_size == 0:
                    candidate.unlink()

    def _build_capture_stem(self, room_dict: dict[str, Any]) -> str:
        anchor_name = self._extract_anchor_name(room_dict) or str(
            room_dict.get("room_id") or room_dict.get("webcast_id") or "douyin_live"
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_anchor_name = self._sanitize_path_component(anchor_name, max_length=48)
        return f"{safe_anchor_name}_{timestamp}"

    def _extract_anchor_name(self, room_dict: dict[str, Any]) -> str | None:
        for container_key in ("owner", "user", "anchor"):
            container = room_dict.get(container_key)
            if isinstance(container, dict):
                for key in ("nickname", "display_name", "name"):
                    value = container.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        for key in ("nickname", "user_name", "display_name"):
            value = room_dict.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _sanitize_path_component(self, value: str, max_length: int = 64) -> str:
        invalid_chars = '<>:"/\\|?*'
        normalized = "".join("_" if (char in invalid_chars or ord(char) < 32) else char for char in value.strip())
        normalized = "_".join(normalized.split())
        normalized = normalized.strip(" ._")
        if not normalized:
            normalized = "douyin_live"
        return normalized[:max_length]

    def _ensure_unique_dir(self, path: Path) -> Path:
        if not path.exists():
            return path
        index = 1
        while True:
            candidate = path.parent / f"{path.name}_{index:02d}"
            if not candidate.exists():
                return candidate
            index += 1

    def _normalize_replay_path(self, replay_path: Path | None, workspace_dir: Path, target_stem: str) -> Path | None:
        if replay_path is None or not replay_path.exists():
            return replay_path
        suffix = replay_path.suffix or ".flv"
        target_path = workspace_dir / f"{target_stem}{suffix}"
        if replay_path.resolve() == target_path.resolve():
            return replay_path
        if target_path.exists():
            target_path.unlink()
        shutil.move(str(replay_path), str(target_path))
        self._cleanup_empty_parents(replay_path.parent, workspace_dir)
        return target_path

    def _cleanup_empty_parents(self, start_dir: Path, stop_dir: Path) -> None:
        current = start_dir
        stop_dir = stop_dir.resolve()
        while True:
            try:
                if current.resolve() == stop_dir:
                    break
                current.rmdir()
                current = current.parent
            except Exception:
                break

    def _safe_int(self, value: Any) -> int:
        return safe_int(value)

    def _resolve_share_url(self, url: str) -> str:
        if "v.douyin.com" not in url:
            return url
        try:
            response = requests.get(
                url,
                allow_redirects=True,
                timeout=self.settings.capture_timeout_seconds,
                headers={"User-Agent": self.settings.douyin_user_agent},
            )
            return response.url or url
        except Exception:
            return url

    def _is_profile_url(self, parsed: Any) -> bool:
        host = (parsed.hostname or "").lower()
        path = parsed.path or ""
        return "douyin.com" in host and "/user/" in path

    def _extract_ids_from_input(self, room_id_or_url: str) -> dict[str, str]:
        normalized = room_id_or_url.strip()
        identifiers: dict[str, str] = {}
        if not normalized:
            return identifiers

        if normalized.isdigit():
            if len(normalized) >= 15:
                identifiers["room_id"] = normalized
            else:
                identifiers["webcast_id"] = normalized
            return identifiers

        if not normalized.startswith("http"):
            identifiers["webcast_id"] = normalized
            return identifiers

        parsed = urlparse(normalized)
        query = parse_qs(parsed.query)
        room_id = query.get("room_id", [None])[0]
        if room_id and str(room_id).isdigit():
            identifiers["room_id"] = str(room_id)

        tail = parsed.path.rstrip("/").split("/")[-1]
        if tail.isdigit():
            if "webcast/reflow" in normalized or len(tail) >= 15:
                identifiers.setdefault("room_id", tail)
            else:
                identifiers.setdefault("webcast_id", tail)
        return identifiers

    def _extract_webcast_id(self, room_dict: dict[str, Any]) -> str | None:
        for key in ("webcast_id", "live_id", "web_rid", "stream_id"):
            value = room_dict.get(key)
            if value:
                return str(value)
        owner = room_dict.get("owner")
        if isinstance(owner, dict):
            for key in ("web_rid", "room_id"):
                value = owner.get(key)
                if value:
                    return str(value)
        return None

    def _prepare_stream_download_payload(self, room_dict: dict[str, Any]) -> dict[str, Any]:
        payload = dict(room_dict)
        owner = payload.get("owner")
        if not payload.get("nickname") and isinstance(owner, dict):
            owner_name = owner.get("nickname") or owner.get("display_name") or owner.get("name")
            if owner_name:
                payload["nickname"] = owner_name
        if not payload.get("live_title"):
            payload["live_title"] = payload.get("live_title_raw") or payload.get("title") or ""
        if not payload.get("user_id") and isinstance(owner, dict):
            owner_id = owner.get("id") or owner.get("uid")
            if owner_id:
                payload["user_id"] = owner_id

        stream_map = payload.get("m3u8_pull_url") or payload.get("hls_pull_url") or payload.get("hls_pull_url_map")
        flv_map = payload.get("flv_pull_url")
        stream_url = payload.get("stream_url")
        if stream_map is None and isinstance(stream_url, dict):
            stream_map = stream_url.get("m3u8_pull_url") or stream_url.get("hls_pull_url_map")
            flv_map = flv_map or stream_url.get("flv_pull_url")
        if isinstance(stream_map, str):
            stream_map = {"FULL_HD1": stream_map}
        elif isinstance(stream_map, dict) and "FULL_HD1" not in stream_map:
            first_url = next((value for value in stream_map.values() if value), None)
            if first_url:
                stream_map = {**stream_map, "FULL_HD1": first_url}
        if isinstance(flv_map, str):
            flv_map = {"FULL_HD1": flv_map}
        elif isinstance(flv_map, dict) and "FULL_HD1" not in flv_map:
            first_url = next((value for value in flv_map.values() if value), None)
            if first_url:
                flv_map = {**flv_map, "FULL_HD1": first_url}
        payload["m3u8_pull_url"] = stream_map or {}
        payload["flv_pull_url"] = flv_map or {}
        return payload

    def _ordered_stream_urls(self, stream_map: dict[str, Any]) -> list[str]:
        if not isinstance(stream_map, dict):
            return []
        urls: list[str] = []
        for key in ("FULL_HD1", "ORIGION", "HD1", "SD2", "SD1"):
            value = stream_map.get(key)
            if value:
                url = str(value)
                if url not in urls:
                    urls.append(url)
        for value in stream_map.values():
            if value:
                url = str(value)
                if url not in urls:
                    urls.append(url)
        return urls


class MultiProfileLiveMonitorService:
    """Monitor multiple Douyin profile URLs and auto-start capture when they go live."""

    def __init__(self, settings: DesktopSettings) -> None:
        self.settings = settings
        self._stop_requested = Event()
        self._capture_services: dict[str, F2LiveCaptureService] = {}

    def run(
        self,
        request: MultiProfileMonitorRequest,
        event_callback: Callable[[MonitorEvent], None] | None = None,
    ) -> None:
        self._stop_requested.clear()
        asyncio.run(self._run_async(request, event_callback))

    def request_stop(self) -> None:
        self._stop_requested.set()
        for service in self._capture_services.values():
            service.request_stop()

    async def _run_async(
        self,
        request: MultiProfileMonitorRequest,
        event_callback: Callable[[MonitorEvent], None] | None,
    ) -> None:
        try:
            from f2.apps.douyin.handler import DouyinHandler
            from f2.apps.douyin.utils import TokenManager
        except ImportError as exc:
            raise RuntimeError(
                "f2 is not installed in the current environment. "
                "Please install requirements-desktop.txt before using monitor mode."
            ) from exc
        helper = F2LiveCaptureService(self.settings)
        helper.configure_f2_console_logging()

        request.workspace_dir.mkdir(parents=True, exist_ok=True)
        capture_cookie = helper.build_capture_cookie(TokenManager)
        kwargs = helper.build_capture_kwargs(capture_cookie)
        handler = DouyinHandler(kwargs)
        if hasattr(handler, "enable_bark"):
            handler.enable_bark = False

        poll_interval = request.poll_interval_seconds or self.settings.monitor_poll_interval_seconds
        active_tasks: dict[str, asyncio.Task[None]] = {}
        captured_room_ids: dict[str, str] = {}
        last_known_room_ids: dict[str, str] = {}
        offline_streaks: dict[str, int] = {}
        last_capture_end_at: dict[str, float] = {}
        last_status: dict[str, str] = {}
        offline_confirmations = 2
        rearm_same_room_seconds = max(5.0, poll_interval)

        normalized_urls = [url.strip() for url in request.profile_urls if url.strip()]
        if not normalized_urls:
            raise RuntimeError("No profile URLs were provided for monitoring.")

        self._emit(event_callback, MonitorEvent("monitor_started", "", f"Started monitoring {len(normalized_urls)} profiles."))
        try:
            while not self._stop_requested.is_set():
                for profile_url in normalized_urls:
                    if self._stop_requested.is_set():
                        break

                    active_task = active_tasks.get(profile_url)
                    if active_task is not None and not active_task.done():
                        continue
                    if active_task is not None and active_task.done():
                        active_tasks.pop(profile_url, None)
                        last_capture_end_at[profile_url] = time.time()

                    try:
                        snapshot = await helper._fetch_profile_snapshot(handler, profile_url)
                        nickname = helper._extract_anchor_name(snapshot) or profile_url
                        room_id = str(snapshot.get("room_id") or "")
                        if room_id:
                            last_known_room_ids[profile_url] = room_id
                        live_status = self._safe_live_status(snapshot.get("live_status"))
                        if room_id:
                            room_id, live_status, nickname = await self._refresh_room_status(
                                handler,
                                helper,
                                room_id,
                                nickname,
                                live_status,
                            )
                            if room_id:
                                last_known_room_ids[profile_url] = room_id
                        else:
                            cached_room_id = last_known_room_ids.get(profile_url)
                            if cached_room_id:
                                room_id, live_status, nickname = await self._refresh_room_status(
                                    handler,
                                    helper,
                                    cached_room_id,
                                    nickname,
                                    live_status,
                                )
                                if room_id:
                                    last_known_room_ids[profile_url] = room_id
                            elif live_status == 2:
                                pending_key = "pending_room_id"
                                if last_status.get(profile_url) != pending_key:
                                    self._emit(
                                        event_callback,
                                        MonitorEvent(
                                            "monitor_error",
                                            profile_url,
                                            f"{nickname} looks live, but no room_id is exposed yet. Retrying soon.",
                                            nickname=nickname,
                                        ),
                                    )
                                    last_status[profile_url] = pending_key
                                continue
                    except Exception as exc:
                        error_key = f"error:{exc}"
                        if last_status.get(profile_url) != error_key:
                            self._emit(
                                event_callback,
                                MonitorEvent("monitor_error", profile_url, f"Monitor check failed: {exc}", nickname=profile_url),
                            )
                            last_status[profile_url] = error_key
                        continue

                    if live_status == 2 and room_id:
                        offline_streaks[profile_url] = 0
                        previous_room_id = captured_room_ids.get(profile_url)
                        if previous_room_id == room_id:
                            last_finished = last_capture_end_at.get(profile_url, 0.0)
                            if time.time() - last_finished < rearm_same_room_seconds:
                                continue
                        if previous_room_id == room_id and profile_url not in last_capture_end_at:
                            continue
                        last_status[profile_url] = "live"
                        captured_room_ids[profile_url] = room_id
                        self._emit(
                            event_callback,
                            MonitorEvent(
                                "live_detected",
                                profile_url,
                                f"{nickname} is live. Starting capture.",
                                nickname=nickname,
                                room_id=room_id,
                            ),
                        )
                        capture_service = F2LiveCaptureService(self.settings)
                        self._capture_services[profile_url] = capture_service
                        active_tasks[profile_url] = asyncio.create_task(
                            self._run_capture_task(
                                capture_service,
                                profile_url,
                                room_id,
                                request.workspace_dir,
                                event_callback,
                            )
                        )
                    else:
                        streak = offline_streaks.get(profile_url, 0) + 1
                        offline_streaks[profile_url] = streak
                        if streak < offline_confirmations:
                            continue
                        if last_status.get(profile_url) != "offline":
                            self._emit(
                                event_callback,
                                MonitorEvent(
                                    "offline",
                                    profile_url,
                                    f"{nickname} is offline. Waiting for the next live session.",
                                    nickname=nickname,
                                ),
                            )
                            last_status[profile_url] = "offline"
                        captured_room_ids.pop(profile_url, None)
                        last_known_room_ids.pop(profile_url, None)
                        last_capture_end_at.pop(profile_url, None)

                await asyncio.sleep(poll_interval)
        finally:
            self.request_stop()
            if active_tasks:
                await asyncio.gather(*active_tasks.values(), return_exceptions=True)
            self._emit(event_callback, MonitorEvent("monitor_stopped", "", "Monitoring stopped."))

    async def _refresh_room_status(
        self,
        handler: Any,
        helper: F2LiveCaptureService,
        room_id: str,
        fallback_nickname: str,
        fallback_live_status: int,
    ) -> tuple[str, int, str]:
        room = await handler.fetch_user_live_videos_by_room_id(room_id=room_id)
        room_dict = helper._to_dict(room)
        nickname = helper._extract_anchor_name(room_dict) or fallback_nickname
        resolved_room_id = str(room_dict.get("room_id") or room_id)
        live_status = self._safe_live_status(room_dict.get("live_status"), fallback_live_status)
        return resolved_room_id, live_status, nickname

    @staticmethod
    def _safe_live_status(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default or 0)

    async def _run_capture_task(
        self,
        capture_service: F2LiveCaptureService,
        profile_url: str,
        room_id: str,
        workspace_dir: Path,
        event_callback: Callable[[MonitorEvent], None] | None,
    ) -> None:
        try:
            result = await asyncio.to_thread(
                capture_service.run,
                LiveCaptureRequest(room_id_or_url=room_id, workspace_dir=workspace_dir),
            )
            self._emit(
                event_callback,
                MonitorEvent(
                    "capture_finished",
                    profile_url,
                    f"Capture finished for room {result.room_id}.",
                    nickname=result.workspace_dir.name,
                    room_id=result.room_id,
                    webcast_id=result.webcast_id,
                    replay_path=result.replay_path,
                    workspace_dir=result.workspace_dir,
                ),
            )
        except Exception as exc:
            self._emit(event_callback, MonitorEvent("capture_failed", profile_url, f"Capture failed: {exc}"))
        finally:
            self._capture_services.pop(profile_url, None)

    def _emit(
        self,
        event_callback: Callable[[MonitorEvent], None] | None,
        event: MonitorEvent,
    ) -> None:
        if event_callback is not None:
            event_callback(event)
