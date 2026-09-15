"""Database migrations (simple, no Alembic)."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def run_migrations() -> None:
    """Run database migrations if needed.
    
    For the 48h sprint, we use simple auto-migration via SQLAlchemy.
    This module is a placeholder for future migration logic.
    """
    logger.info("Migrations module initialized (using auto-migration)")
