from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .models import MonitorProfileRecord, MonitorProfileSource


class ClientJobRepository:
    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    replay_path TEXT NOT NULL,
                    occupancy_path TEXT NOT NULL,
                    danmaku_path TEXT,
                    workspace_dir TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    message TEXT,
                    report_url TEXT,
                    remote_job_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS monitor_profiles (
                    profile_url TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    last_live_at TEXT,
                    last_capture_dir TEXT,
                    last_room_id TEXT,
                    last_status TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        self._ensure_column("monitor_profiles", "last_status", "TEXT")
        self._ensure_column("monitor_profiles", "source_type", "TEXT")
        self._ensure_column("monitor_profiles", "source_account_sec_uid", "TEXT")
        self._ensure_column("monitor_profiles", "source_follow_sec_uid", "TEXT")
        self._ensure_column("monitor_profiles", "source_follow_uid", "TEXT")
        self._ensure_column("monitor_profiles", "source_nickname", "TEXT")
        self._backfill_monitor_profile_sources()
        self._migrate_legacy_monitor_profiles()

    def create_job(self, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO analysis_jobs (
                    job_id, status, replay_path, occupancy_path, danmaku_path,
                    workspace_dir, progress, message, report_url, remote_job_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["job_id"],
                    payload["status"],
                    payload["replay_path"],
                    payload["occupancy_path"],
                    payload.get("danmaku_path"),
                    payload["workspace_dir"],
                    payload.get("progress", 0.0),
                    payload.get("message"),
                    payload.get("report_url"),
                    payload.get("remote_job_id"),
                ),
            )
            conn.commit()

    def update_job(self, job_id: str, **updates: Any) -> None:
        if not updates:
            return
        assignments = ", ".join(f"{key} = ?" for key in updates.keys())
        values = list(updates.values()) + [job_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE analysis_jobs SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE job_id = ?",
                values,
            )
            conn.commit()

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM analysis_jobs ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return row["value"]

    def set_setting(self, key: str, value: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (key, value),
            )
            conn.commit()

    def delete_setting(self, key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM app_settings WHERE key = ?", (key,))
            conn.commit()

    def list_monitor_profiles(self) -> list[MonitorProfileRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT profile_url, enabled, last_live_at, last_capture_dir, last_room_id,
                    last_status, source_type, source_account_sec_uid, source_follow_sec_uid,
                    source_follow_uid, source_nickname
                FROM monitor_profiles
                ORDER BY updated_at DESC, profile_url ASC
                """
            ).fetchall()
        return [
            MonitorProfileRecord(
                profile_url=row["profile_url"],
                enabled=bool(row["enabled"]),
                last_live_at=row["last_live_at"],
                last_capture_dir=row["last_capture_dir"],
                last_room_id=row["last_room_id"],
                last_status=row["last_status"],
                source_type=row["source_type"] or MonitorProfileSource.MANUAL.value,
                source_account_sec_uid=row["source_account_sec_uid"],
                source_follow_sec_uid=row["source_follow_sec_uid"],
                source_follow_uid=row["source_follow_uid"],
                source_nickname=row["source_nickname"],
            )
            for row in rows
        ]

    def save_monitor_profiles(self, profiles: list[MonitorProfileRecord]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM monitor_profiles")
            conn.executemany(
                """
                INSERT INTO monitor_profiles (
                    profile_url, enabled, last_live_at, last_capture_dir, last_room_id, last_status,
                    source_type, source_account_sec_uid, source_follow_sec_uid, source_follow_uid,
                    source_nickname, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [
                    (
                        profile.profile_url,
                        1 if profile.enabled else 0,
                        profile.last_live_at,
                        profile.last_capture_dir,
                        profile.last_room_id,
                        profile.last_status,
                        profile.source_type,
                        profile.source_account_sec_uid,
                        profile.source_follow_sec_uid,
                        profile.source_follow_uid,
                        profile.source_nickname,
                    )
                    for profile in profiles
                ],
            )
            conn.commit()

    def upsert_monitor_profile(
        self,
        profile_url: str,
        *,
        enabled: bool | None = None,
        last_live_at: str | None = None,
        last_capture_dir: str | None = None,
        last_room_id: str | None = None,
        last_status: str | None = None,
        source_type: str | None = None,
        source_account_sec_uid: str | None = None,
        source_follow_sec_uid: str | None = None,
        source_follow_uid: str | None = None,
        source_nickname: str | None = None,
    ) -> None:
        existing = self.get_monitor_profile(profile_url)
        if existing is None:
            existing = MonitorProfileRecord(profile_url=profile_url)
        record = MonitorProfileRecord(
            profile_url=profile_url,
            enabled=existing.enabled if enabled is None else enabled,
            last_live_at=last_live_at if last_live_at is not None else existing.last_live_at,
            last_capture_dir=last_capture_dir if last_capture_dir is not None else existing.last_capture_dir,
            last_room_id=last_room_id if last_room_id is not None else existing.last_room_id,
            last_status=last_status if last_status is not None else existing.last_status,
            source_type=source_type if source_type is not None else existing.source_type,
            source_account_sec_uid=source_account_sec_uid
            if source_account_sec_uid is not None
            else existing.source_account_sec_uid,
            source_follow_sec_uid=source_follow_sec_uid
            if source_follow_sec_uid is not None
            else existing.source_follow_sec_uid,
            source_follow_uid=source_follow_uid if source_follow_uid is not None else existing.source_follow_uid,
            source_nickname=source_nickname if source_nickname is not None else existing.source_nickname,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO monitor_profiles (
                    profile_url, enabled, last_live_at, last_capture_dir, last_room_id, last_status,
                    source_type, source_account_sec_uid, source_follow_sec_uid, source_follow_uid,
                    source_nickname, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(profile_url) DO UPDATE SET
                    enabled = excluded.enabled,
                    last_live_at = excluded.last_live_at,
                    last_capture_dir = excluded.last_capture_dir,
                    last_room_id = excluded.last_room_id,
                    last_status = excluded.last_status,
                    source_type = excluded.source_type,
                    source_account_sec_uid = excluded.source_account_sec_uid,
                    source_follow_sec_uid = excluded.source_follow_sec_uid,
                    source_follow_uid = excluded.source_follow_uid,
                    source_nickname = excluded.source_nickname,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    record.profile_url,
                    1 if record.enabled else 0,
                    record.last_live_at,
                    record.last_capture_dir,
                    record.last_room_id,
                    record.last_status,
                    record.source_type,
                    record.source_account_sec_uid,
                    record.source_follow_sec_uid,
                    record.source_follow_uid,
                    record.source_nickname,
                ),
            )
            conn.commit()

    def get_monitor_profile(self, profile_url: str) -> MonitorProfileRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT profile_url, enabled, last_live_at, last_capture_dir, last_room_id,
                    last_status, source_type, source_account_sec_uid, source_follow_sec_uid,
                    source_follow_uid, source_nickname
                FROM monitor_profiles
                WHERE profile_url = ?
                """,
                (profile_url,),
            ).fetchone()
        if row is None:
            return None
        return MonitorProfileRecord(
            profile_url=row["profile_url"],
            enabled=bool(row["enabled"]),
            last_live_at=row["last_live_at"],
            last_capture_dir=row["last_capture_dir"],
            last_room_id=row["last_room_id"],
            last_status=row["last_status"],
            source_type=row["source_type"] or MonitorProfileSource.MANUAL.value,
            source_account_sec_uid=row["source_account_sec_uid"],
            source_follow_sec_uid=row["source_follow_sec_uid"],
            source_follow_uid=row["source_follow_uid"],
            source_nickname=row["source_nickname"],
        )

    def delete_following_sync_profiles(self, source_account_sec_uid: str, keep_sec_uids: set[str]) -> int:
        placeholders = ",".join("?" for _ in keep_sec_uids)
        params: list[Any] = [
            MonitorProfileSource.FOLLOWING_SYNC.value,
            source_account_sec_uid,
        ]
        sql = """
            DELETE FROM monitor_profiles
            WHERE source_type = ?
              AND source_account_sec_uid = ?
        """
        if keep_sec_uids:
            sql += f" AND (source_follow_sec_uid IS NULL OR source_follow_sec_uid NOT IN ({placeholders}))"
            params.extend(sorted(keep_sec_uids))
        with self._connect() as conn:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.rowcount if cursor.rowcount is not None else 0

    def delete_monitor_profile(self, profile_url: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM monitor_profiles WHERE profile_url = ?", (profile_url,))
            conn.commit()

    def _ensure_column(self, table_name: str, column_name: str, definition: str) -> None:
        with self._connect() as conn:
            existing = {
                row["name"]
                for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            }
            if column_name in existing:
                return
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")
            conn.commit()

    def _backfill_monitor_profile_sources(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE monitor_profiles
                SET source_type = ?
                WHERE source_type IS NULL OR source_type = ''
                """,
                (MonitorProfileSource.MANUAL.value,),
            )
            conn.commit()

    def _migrate_legacy_monitor_profiles(self) -> None:
        raw = self.get_setting("monitor_profile_urls")
        if not raw:
            return
        try:
            import json

            profile_urls = json.loads(raw)
        except Exception:
            return
        if not isinstance(profile_urls, list):
            return
        existing_urls = {profile.profile_url for profile in self.list_monitor_profiles()}
        migrated = False
        for profile_url in profile_urls:
            cleaned = str(profile_url).strip()
            if cleaned and cleaned not in existing_urls:
                self.upsert_monitor_profile(cleaned, enabled=True)
                migrated = True
        if migrated or profile_urls == []:
            self.delete_setting("monitor_profile_urls")
