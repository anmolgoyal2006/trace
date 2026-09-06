"""
database.py — SQLAlchemy async engine + session factory for Trace.

Usage:
    from backend.app.database import async_session, engine, Base

    # In FastAPI lifespan:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # In a route / service:
    async with async_session() as session:
        result = await session.execute(select(Person))
"""

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from backend.app.config import settings

# Ensure the db directory exists before SQLite tries to open the file
settings.db_path.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite+aiosqlite:///{settings.db_path}"

engine = create_async_engine(
    DATABASE_URL,
    echo=settings.debug,
    connect_args={"check_same_thread": False},
)

async_session: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields a database session per request."""
    async with async_session() as session:
        yield session
