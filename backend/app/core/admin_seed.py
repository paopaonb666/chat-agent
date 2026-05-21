import os
import logging

from app.core.security import get_password_hash

logger = logging.getLogger(__name__)


def ensure_admin_user(db_session):
    """Create an admin user if none exists."""
    from app.models import User

    existing = db_session.query(User).filter(User.role == "admin").first()
    if existing:
        return

    username = os.getenv("ADMIN_USERNAME", "admin")
    password = os.getenv("ADMIN_PASSWORD", "admin123")

    admin = User(
        username=username,
        password_hash=get_password_hash(password),
        role="admin",
    )
    db_session.add(admin)
    db_session.commit()
    logger.info("Admin user '%s' created (password from ADMIN_PASSWORD env)", username)
