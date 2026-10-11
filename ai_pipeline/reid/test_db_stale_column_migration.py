"""
test_db_stale_column_migration.py
Startup migration must make NOT NULL columns that the ORM no longer defines
nullable, so inserts from current code succeed on databases created by older
code (e.g. a stale ``sightings.match_tier TEXT NOT NULL`` made every query
fail while saving sightings).

These tests use a temporary SQLite file (never touches db/trace.db).

Run:
    pytest ai_pipeline/reid/test_db_stale_column_migration.py -v
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("aiosqlite")

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.database import Base
from backend.app.main import _migrate_sightings_schema
from backend.app.models.orm import QuerySession, Sighting


def _make_stale_db(path: Path) -> None:
    """Current schema, but sightings has an extra stale NOT NULL column + an index + a row."""
    async def build():
        engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()
    asyncio.run(build())

    c = sqlite3.connect(path)
    c.execute("ALTER TABLE sightings RENAME TO sightings_old")
    old_sql = c.execute("SELECT sql FROM sqlite_master WHERE name='sightings_old'").fetchone()[0]
    stale_sql = old_sql.replace("sightings_old", "sightings", 1).replace(
        "track_id INTEGER NOT NULL,", "track_id INTEGER NOT NULL, \n\tmatch_tier TEXT NOT NULL,", 1)
    assert "match_tier TEXT NOT NULL" in stale_sql
    c.execute(stale_sql)
    c.execute("DROP TABLE sightings_old")
    c.execute("CREATE INDEX ix_sightings_camera ON sightings (camera_id)")
    c.execute("INSERT INTO query_sessions (id, status, progress_pct, match_decision) VALUES (1, 'done', 100, 'NO_CONFIDENT_MATCH')")
    required = {   # NOT NULL columns whose defaults the ORM normally supplies
        "best_confidence": 0.0, "mean_confidence": 0.0, "appearance_score": 0.0,
        "spatial_score": 0.0, "temporal_score": 0.0, "fusion_score": 0.0,
        "match_status": "CONFIDENT_MATCH", "face_veto_applied": 0, "fused_score": 0.0,
        "active_backbone": "OSNet x1_0", "embedding_dim": 512, "matching_mode": "body",
        "face_used": 0, "body_similarity": 0.0, "identity_score": 0.0,
    }
    row = {"id": 1, "session_id": 1, "camera_id": "C01", "track_id": 7, "match_tier": "HIGH", **required}
    c.execute(f"INSERT INTO sightings ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
              list(row.values()))
    c.commit()
    c.close()


async def _migrate_and_insert(path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _migrate_sightings_schema(conn)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with maker() as db:
        db.add(Sighting(session_id=1, camera_id="C02", track_id=9))   # ORM never sets match_tier
        await db.commit()
    await engine.dispose()


def _col(path: Path, name: str):
    c = sqlite3.connect(path)
    row = [r for r in c.execute("PRAGMA table_info(sightings)") if r[1] == name]
    c.close()
    return row[0] if row else None


class TestStaleColumnMigration:

    def test_insert_fails_before_migration(self, tmp_path):
        db = tmp_path / "stale.db"
        _make_stale_db(db)

        async def insert_only():
            engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
            maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
            async with maker() as s:
                s.add(Sighting(session_id=1, camera_id="C02", track_id=9))
                await s.commit()
        with pytest.raises(Exception, match="match_tier"):
            asyncio.run(insert_only())

    def test_migration_makes_stale_column_nullable_and_insert_succeeds(self, tmp_path):
        db = tmp_path / "stale.db"
        _make_stale_db(db)
        asyncio.run(_migrate_and_insert(db))

        assert _col(db, "match_tier")[3] == 0          # notnull flag cleared
        c = sqlite3.connect(db)
        rows = c.execute("SELECT id, camera_id, track_id, match_tier FROM sightings ORDER BY id").fetchall()
        indexes = [r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='sightings'")]
        c.close()
        assert rows == [(1, "C01", 7, "HIGH"), (2, "C02", 9, None)]   # old row kept, new row inserted
        assert "ix_sightings_camera" in indexes

    def test_current_columns_keep_their_constraints(self, tmp_path):
        db = tmp_path / "stale.db"
        _make_stale_db(db)
        asyncio.run(_migrate_and_insert(db))
        assert _col(db, "track_id")[3] == 1
        assert _col(db, "camera_id")[3] == 1

    def test_migration_is_idempotent(self, tmp_path):
        db = tmp_path / "stale.db"
        _make_stale_db(db)
        asyncio.run(_migrate_and_insert(db))
        asyncio.run(_migrate_and_insert(db))
        c = sqlite3.connect(db)
        assert c.execute("SELECT COUNT(*) FROM sightings").fetchone()[0] == 3
        c.close()

    def test_clean_database_untouched(self, tmp_path):
        db = tmp_path / "clean.db"
        async def build():
            engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            await engine.dispose()
        asyncio.run(build())
        before = sqlite3.connect(db).execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
        c = sqlite3.connect(db)
        c.execute("INSERT INTO query_sessions (id, status, progress_pct, match_decision) VALUES (1, 'done', 100, 'NO_CONFIDENT_MATCH')")
        c.commit(); c.close()
        asyncio.run(_migrate_and_insert(db))
        after = sqlite3.connect(db).execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
        assert before == after


# ===========================================================================
# Interrupted sessions are failed at startup
# ===========================================================================

class TestInterruptedSessions:

    def test_running_and_pending_sessions_marked_failed(self, tmp_path):
        from backend.app.main import _fail_interrupted_sessions
        db = tmp_path / "s.db"

        async def go():
            engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                for sid, status in [(1, "done"), (2, "running"), (3, "pending"), (4, "failed")]:
                    await conn.execute(text(
                        "INSERT INTO query_sessions (id, status, progress_pct, match_decision) "
                        "VALUES (:i, :s, 0, 'NO_CONFIDENT_MATCH')"), {"i": sid, "s": status})
            async with engine.begin() as conn:
                await _fail_interrupted_sessions(conn)
            await engine.dispose()
        asyncio.run(go())

        c = sqlite3.connect(db)
        rows = dict(c.execute("SELECT id, status FROM query_sessions").fetchall())
        msgs = dict(c.execute("SELECT id, error_message FROM query_sessions").fetchall())
        done_at = dict(c.execute("SELECT id, completed_at FROM query_sessions").fetchall())
        c.close()
        assert rows == {1: "done", 2: "failed", 3: "failed", 4: "failed"}
        assert "Interrupted" in msgs[2] and "Interrupted" in msgs[3]
        assert msgs[1] is None and msgs[4] is None
        assert done_at[2] is not None
