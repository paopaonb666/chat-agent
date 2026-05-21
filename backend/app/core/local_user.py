from sqlalchemy.orm import Session
from app.models import User
from app.core.security import get_password_hash

_LOCAL_USER_CACHE: dict[str, int | None] = {"id": None}


def get_local_user(db: Session) -> User:
    """Get or create the default local user for single-user mode."""
    if _LOCAL_USER_CACHE["id"] is not None:
        user = db.query(User).filter(User.id == _LOCAL_USER_CACHE["id"]).first()
        if user:
            return user
    user = db.query(User).order_by(User.id).first()
    if user:
        _LOCAL_USER_CACHE["id"] = user.id
        return user
    user = User(username="local", password_hash=get_password_hash("local"))
    db.add(user)
    db.commit()
    db.refresh(user)
    _LOCAL_USER_CACHE["id"] = user.id
    return user
