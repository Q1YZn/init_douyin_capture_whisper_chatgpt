from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from desktop_client.main import main


if __name__ == "__main__":
    raise SystemExit(main(app_mode="capture", auto_start_monitor=True))
