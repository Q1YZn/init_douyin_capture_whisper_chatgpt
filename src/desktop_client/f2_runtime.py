from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any


REQUIRED_F2_METHODS = (
    "fetch_user_live_videos_by_room_id",
    "fetch_user_live_videos",
    "fetch_live_im",
)

EXPECTED_VIEWER_KEYS = (
    "user_count_str",
    "user_count",
    "room_view_stats",
    "total_user_str",
    "current_user_count",
    "online_count",
)


@dataclass(slots=True)
class F2RuntimeReport:
    available: bool
    version: str
    method_support: dict[str, bool]
    viewer_key_support: tuple[str, ...]
    warnings: list[str]

    @property
    def ready(self) -> bool:
        return self.available and all(self.method_support.values())


def probe_f2_runtime() -> F2RuntimeReport:
    try:
        import f2
        from f2.apps.douyin.handler import DouyinHandler
    except Exception as exc:
        return F2RuntimeReport(
            available=False,
            version="missing",
            method_support={name: False for name in REQUIRED_F2_METHODS},
            viewer_key_support=EXPECTED_VIEWER_KEYS,
            warnings=[f"f2 runtime unavailable: {exc}"],
        )

    version = getattr(f2, "__version__", "") or _detect_package_version("f2") or "unknown"
    method_support = {name: callable(getattr(DouyinHandler, name, None)) for name in REQUIRED_F2_METHODS}
    warnings: list[str] = []
    if not all(method_support.values()):
        missing = [name for name, ok in method_support.items() if not ok]
        warnings.append(f"Missing DouyinHandler methods: {', '.join(missing)}")
    if version != "0.0.1.7":
        warnings.append(f"Expected f2 version 0.0.1.7, detected {version}.")

    return F2RuntimeReport(
        available=True,
        version=version,
        method_support=method_support,
        viewer_key_support=EXPECTED_VIEWER_KEYS,
        warnings=warnings,
    )


def extract_viewer_count(payload: dict[str, Any]) -> int:
    for key in EXPECTED_VIEWER_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            nested = value.get("total_user_str") or value.get("total_user") or value.get("user_count")
            if nested is not None:
                return safe_int(nested)
        if value is not None:
            return safe_int(value)
    return 0


def safe_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().replace(",", "")
    if "万" in text:
        try:
            return int(float(text.replace("万", "")) * 10000)
        except ValueError:
            return 0
    if "w" in text.lower():
        try:
            return int(float(text[:-1]) * 10000)
        except ValueError:
            return 0
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else 0


def _detect_package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None
