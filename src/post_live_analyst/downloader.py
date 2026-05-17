from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .pipeline import AnalysisInputs


@dataclass(slots=True)
class PreparedInputs:
    occupant_history_path: Path
    replay_video_path: Path
    workspace_dir: Path
    danmaku_path: Path | None = None
    room_id: str | None = None


class LocalReplayDownloader:
    """Prepare already-downloaded local assets inside the analysis workspace."""

    def prepare_inputs(self, inputs: AnalysisInputs) -> PreparedInputs:
        workspace_dir = inputs.workspace_dir
        workspace_dir.mkdir(parents=True, exist_ok=True)

        occupant_history_path = self._copy_if_needed(inputs.occupant_history_path, workspace_dir / "occupant_history.csv")
        replay_video_path = self._copy_if_needed(inputs.replay_video_path, workspace_dir / inputs.replay_video_path.name)
        danmaku_path = None
        if inputs.danmaku_path:
            danmaku_path = self._copy_if_needed(inputs.danmaku_path, workspace_dir / inputs.danmaku_path.name)

        return PreparedInputs(
            occupant_history_path=occupant_history_path,
            replay_video_path=replay_video_path,
            workspace_dir=workspace_dir,
            danmaku_path=danmaku_path,
            room_id=inputs.room_id,
        )

    def _copy_if_needed(self, source: Path, destination: Path) -> Path:
        source = Path(source)
        destination = Path(destination)
        if not source.exists():
            raise FileNotFoundError(f"Input asset not found: {source}")
        if source.resolve() == destination.resolve():
            return source
        shutil.copy2(source, destination)
        return destination


class F2ReplayDownloader:
    """Optional downloader placeholder for future f2 integration."""

    def __init__(self) -> None:
        try:
            import f2  # type: ignore  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "f2 is not installed in the current environment. "
                "Use LocalReplayDownloader for local assets or install f2 before enabling remote download."
            ) from exc

    def prepare_inputs(self, inputs: AnalysisInputs) -> PreparedInputs:
        raise NotImplementedError("F2ReplayDownloader needs a concrete f2-based fetch implementation.")
