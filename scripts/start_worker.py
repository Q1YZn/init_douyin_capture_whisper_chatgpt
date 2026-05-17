from pathlib import Path
import sys

from rq import Worker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cloud_api.tasks import redis_conn


if __name__ == "__main__":
    worker = Worker(["reports"], connection=redis_conn)
    worker.work()
