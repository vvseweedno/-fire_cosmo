"""Database connection and initialization."""

import logging
from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..settings import get_settings
from .models import Base

logger = logging.getLogger(__name__)


def get_db_path() -> str:
    """Get the database file path."""
    base_dir = Path(__file__).parent.parent.parent
    db_path = base_dir / "data" / "wildfire.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{db_path}"


# Global engine and session factory
_engine = None
_session_factory = None


def get_engine():
    """Get or create the async engine."""
    global _engine
    if _engine is None:
        db_url = get_db_path()
        logger.info(f"Creating database engine for {db_url}")
        _engine = create_async_engine(
            db_url,
            echo=False,  # Set to True for SQL debugging
            future=True,
        )
    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def init_db() -> None:
    """Initialize the database (create tables)."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized successfully")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
