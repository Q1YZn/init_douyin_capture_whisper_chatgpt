from __future__ import annotations

from typing import Any
from datetime import datetime, timezone
from pathlib import Path
import json
import time

import requests


class CloudUploader:
    def __init__(
        self,
        server_base_url: str,
        timeout_seconds: float = 60.0,
        auth_token: str | None = None,
        debug_dir: str | Path | None = None,
    ) -> None:
        self.server_base_url = server_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.auth_token = (auth_token or "").strip()
        self.debug_dir = Path(debug_dir) if debug_dir else None

    def _headers(self) -> dict[str, str] | None:
        return {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else None

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write_debug_json(self, file_name: str, payload: Any) -> None:
        if not self.debug_dir:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        (self.debug_dir / file_name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _append_debug_jsonl(self, file_name: str, payload: Any) -> None:
        if not self.debug_dir:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        with (self.debug_dir / file_name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def upload_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._write_debug_json("debug_upload_payload.json", payload)
        response = requests.post(
            f"{self.server_base_url}/api/v1/analysis-jobs",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response_debug: dict[str, Any] = {
            "at": self._now_iso(),
            "url": f"{self.server_base_url}/api/v1/analysis-jobs",
            "status_code": response.status_code,
        }
        try:
            response_debug["body"] = response.json()
        except ValueError:
            response_debug["body_text"] = response.text
        self._write_debug_json("debug_upload_response.json", response_debug)
        response.raise_for_status()
        return response.json()

    def get_analysis_job(self, job_id: str) -> dict[str, Any]:
        response = requests.get(
            f"{self.server_base_url}/api/v1/analysis-jobs/{job_id}",
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def wait_for_analysis_report(
        self,
        job_id: str,
        *,
        timeout_seconds: float = 120.0,
        interval_seconds: float = 2.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + max(1.0, timeout_seconds)
        latest: dict[str, Any] = {"job_id": job_id, "status": "unknown", "report_url": None}
        while time.monotonic() < deadline:
            latest = self.get_analysis_job(job_id)
            self._append_debug_jsonl(
                "debug_poll_responses.jsonl",
                {
                    "at": self._now_iso(),
                    "job_id": job_id,
                    "status": latest.get("status"),
                    "report_url": latest.get("report_url"),
                    "response": latest,
                },
            )
            if latest.get("report_url") or str(latest.get("status") or "").lower() == "completed":
                return latest
            time.sleep(max(0.5, interval_seconds))
        return latest

    def fetch_report_html(self, report_url: str) -> str:
        response = requests.get(report_url, headers=self._headers(), timeout=max(self.timeout_seconds, 120.0))
        response.raise_for_status()
        return response.text

    def request_clip_candidates(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.server_base_url}/api/v1/clip-candidates",
            json=payload,
            headers=self._headers(),
            timeout=max(self.timeout_seconds, 120.0),
        )
        response.raise_for_status()
        return response.json()
