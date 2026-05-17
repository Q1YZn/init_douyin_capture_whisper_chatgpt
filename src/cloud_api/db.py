from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import ServerSettings


class Base(DeclarativeBase):
    pass


settings = ServerSettings.from_env()
engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


@contextmanager
def get_session() -> Session:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def ensure_analysis_job_columns() -> None:
    required_columns = {
        "video_size_bytes": _dialect_type("BIGINT", "INTEGER"),
        "asset_storage": "VARCHAR(64)",
        "asset_uri": "VARCHAR(2048)",
        "analysis_tier": "VARCHAR(64)",
        "payment_required": _dialect_type("BOOLEAN", "INTEGER"),
    }
    inspector = inspect(engine)
    existing = {column["name"] for column in inspector.get_columns("analysis_jobs")}
    with engine.begin() as conn:
        for column_name, definition in required_columns.items():
            if column_name in existing:
                continue
            conn.execute(text(f"ALTER TABLE analysis_jobs ADD COLUMN {column_name} {definition}"))


def _dialect_type(default_type: str, sqlite_type: str) -> str:
    return sqlite_type if engine.dialect.name == "sqlite" else default_type
