from pathlib import Path
import sys

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


if __name__ == "__main__":
    uvicorn.run("cloud_api.app:app", host="0.0.0.0", port=8000, reload=False)
