from __future__ import annotations

from typing import Any

import requests


class CloudUploader:
    def __init__(self, server_base_url: str, timeout_seconds: float = 60.0) -> None:
        self.server_base_url = server_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def upload_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.server_base_url}/api/v1/analysis-jobs",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
