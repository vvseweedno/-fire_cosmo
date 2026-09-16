"""
Database connection and session management.
"""

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from .models import Base


# SQLite database path
DB_PATH = Path(__file__).parent.parent.parent / "data" / "nexus.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Create engine with SQLite-specific optimizations
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    echo=False,
)

# Session factory
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """
    Initialize database tables.
    
    Creates all tables defined in models.py if they don't exist.
    """
    Base.metadata.create_all(bind=engine)


def get_session():
    """
    Get database session for dependency injection.
    
    Yields:
        Session: SQLAlchemy session
        
    Usage:
        with get_session() as session:
            # use session
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_session_sync():
    """
    Get database session for synchronous usage.
    
    Returns:
        Session: SQLAlchemy session
        
    Usage:
        session = get_session_sync()
        try:
            # use session
        finally:
            session.close()
    """
    return SessionLocal()
