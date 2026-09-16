"""
Simple database migrations without Alembic.
"""

import logging

from sqlalchemy import inspect

from .connection import engine
from .models import Base

logger = logging.getLogger(__name__)


def check_and_migrate() -> None:
    """
    Check database schema and apply simple migrations if needed.
    
    This is a simplified migration approach for the 48-hour implementation.
    For production use, consider using Alembic.
    """
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    # Create all tables (this is safe - won't drop existing)
    Base.metadata.create_all(bind=engine)
    
    logger.info(f"Database initialized with tables: {existing_tables}")


def run_migrations() -> None:
    """
    Run any pending migrations.
    
    Currently just ensures tables exist. Can be extended for schema changes.
    """
    check_and_migrate()
    logger.info("Migrations completed successfully")
