"""
test_pipeline_session_failure.py
Regression test: a query session whose pipeline raises a database error must
end in status="failed" — never stay "running".

A session stuck in "running" blocks every later query (POST /api/queries
returns 409 while any session is running). The original handler tried to
commit status="failed" on the same SQLAlchemy session that had just failed a
flush, which raised PendingRollbackError and left the row "running".

These tests:
  - Use a temporary SQLite database (never touches db/trace.db).
  - Do NOT require GPU, model weights, or a running server.
  - Need Python >= 3.9 with aiosqlite installed.

Run:
    pytest ai_pipeline/reid/test_pipeline_session_failure.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("aiosqlite")

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import backend.app.services.pipeline_service as ps_module
from backend.app.database import Base
from backend.app.models.orm import QuerySession, Sighting
from backend.app.services.pipeline_service import PipelineService


class _FakeWs:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def broadcast(self, data: dict) -> None:
        self.messages.append(data)


async def _run(tmp_path: Path, execute) -> tuple[QuerySession, _FakeWs]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with maker() as db:
        qs = QuerySession(status="pending", progress_pct=0)
        db.add(qs)
        await db.commit()
        session_id = qs.id

    ws = _FakeWs()
    svc = PipelineService(ws)  # type: ignore[arg-type]
    svc._execute = execute.__get__(svc)  # type: ignore[method-assign]

    original = ps_module.async_session
    ps_module.async_session = maker
    try:
        await svc.run_session(session_id)
    finally:
        ps_module.async_session = original

    async with maker() as db:
        final = await db.get(QuerySession, session_id)
    await engine.dispose()
    return final, ws


async def _execute_flush_failure(self, session, db):
    # Reproduces the real failure: persisting a sighting violates a
    # NOT NULL constraint during flush, leaving `db` in a failed state.
    db.add(Sighting(session_id=session.id, camera_id="C01", track_id=None))
    await db.flush()


async def _execute_plain_error(self, session, db):
    raise RuntimeError("boom")


class TestSessionFailureHandling:

    def test_db_flush_failure_marks_session_failed(self, tmp_path):
        final, ws = asyncio.run(_run(tmp_path, _execute_flush_failure))
        assert final.status == "failed"
        assert "NOT NULL" in (final.error_message or "")
        assert final.completed_at is not None
        assert ws.messages and ws.messages[-1]["session_id"] == final.id

    def test_plain_exception_marks_session_failed(self, tmp_path):
        final, ws = asyncio.run(_run(tmp_path, _execute_plain_error))
        assert final.status == "failed"
        assert final.error_message == "boom"
        assert ws.messages[-1]["message"] == "boom"
