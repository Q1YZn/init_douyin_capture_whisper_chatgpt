from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def build_repo_id(model_name: str) -> str:
    if "/" in model_name:
        return model_name
    return f"Systran/faster-whisper-{model_name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-download a faster-whisper model with HF_TOKEN to avoid runtime downloads."
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Model name such as tiny, base, small, medium, large-v3, or a full HF repo id.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN from the environment.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("HF_HOME"),
        help="Optional Hugging Face cache directory. If omitted, use the default HF cache.",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Optional explicit output directory. If omitted, files are only warmed into the HF cache.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.token:
        raise SystemExit("HF_TOKEN is required. Pass --token or set HF_TOKEN in the environment.")

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    repo_id = build_repo_id(args.model)

    download_kwargs: dict[str, object] = {
        "repo_id": repo_id,
        "token": args.token,
    }
    if args.cache_dir:
        download_kwargs["cache_dir"] = str(Path(args.cache_dir))
    if args.local_dir:
        download_kwargs["local_dir"] = str(Path(args.local_dir))
        download_kwargs["local_dir_use_symlinks"] = False

    local_path = snapshot_download(**download_kwargs)
    print(f"repo_id={repo_id}")
    print(f"local_path={local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
