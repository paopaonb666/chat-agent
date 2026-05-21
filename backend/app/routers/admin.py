from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User, Conversation, Message, UploadedFile, UserMemory
from app.deps import get_admin_user
from app.core.security import get_password_hash

router = APIRouter(prefix="/admin", tags=["admin"])


# ── Request/Response models ──────────────────────────────────────────────

class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"  # "user" | "admin"


class UpdateUserRequest(BaseModel):
    username: str | None = None
    password: str | None = None
    role: str | None = None


class UserItem(BaseModel):
    id: int
    username: str
    role: str
    created_at: str

    @classmethod
    def from_user(cls, u: User) -> "UserItem":
        return cls(
            id=u.id,
            username=u.username,
            role=u.role,
            created_at=u.created_at.isoformat(),
        )


class ConversationItem(BaseModel):
    id: str
    title: str
    model: str
    username: str | None
    message_count: int
    created_at: str
    updated_at: str

    @classmethod
    def from_conv(cls, c: Conversation) -> "ConversationItem":
        return cls(
            id=c.id,
            title=c.title,
            model=c.model,
            username=c.owner.username if c.owner else None,
            message_count=len(c.messages),
            created_at=c.created_at.isoformat(),
            updated_at=c.updated_at.isoformat(),
        )


# ── Stats ─────────────────────────────────────────────────────────────────

@router.get("/stats", summary="获取统计数据", description="返回平台统计数据：用户数、对话数、消息数、文件数、记忆数、7日活跃用户")
def get_stats(db: Session = Depends(get_db), _admin: User = Depends(get_admin_user)):
    total_users = db.query(func.count(User.id)).scalar()
    total_conversations = db.query(func.count(Conversation.id)).scalar()
    total_messages = db.query(func.count(Message.id)).scalar()
    total_files = db.query(func.count(UploadedFile.id)).scalar()
    total_memories = db.query(func.count(UserMemory.id)).scalar()

    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    active_users_7d = (
        db.query(func.count(func.distinct(Conversation.user_id)))
        .filter(Conversation.updated_at >= seven_days_ago)
        .scalar()
    )

    return {
        "total_users": total_users,
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "total_files": total_files,
        "total_memories": total_memories,
        "active_users_7d": active_users_7d,
    }


# ── User management ───────────────────────────────────────────────────────

@router.get("/users", summary="获取用户列表", description="获取所有注册用户列表，按注册时间降序排列")
def list_users(db: Session = Depends(get_db), _admin: User = Depends(get_admin_user)):
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [UserItem.from_user(u) for u in users]


@router.post("/users", summary="创建用户", description="管理员创建新用户，可指定角色为 user 或 admin", responses={400: {"description": "用户名已存在或角色无效"}})
def create_user(
    payload: CreateUserRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_admin_user),
):
    existing = db.query(User).filter(User.username == payload.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    if payload.role not in ("user", "admin"):
        raise HTTPException(status_code=400, detail="role must be 'user' or 'admin'")
    user = User(
        username=payload.username,
        password_hash=get_password_hash(payload.password),
        role=payload.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserItem.from_user(user)


@router.put("/users/{user_id}", summary="更新用户", description="更新指定用户的用户名、密码或角色", responses={400: {"description": "用户名已被占用或角色无效"}, 404: {"description": "用户不存在"}})
def update_user(
    user_id: int,
    payload: UpdateUserRequest,
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_admin_user),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if payload.username is not None:
        existing = db.query(User).filter(User.username == payload.username, User.id != user_id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Username already taken")
        user.username = payload.username
    if payload.password is not None:
        user.password_hash = get_password_hash(payload.password)
    if payload.role is not None:
        if payload.role not in ("user", "admin"):
            raise HTTPException(status_code=400, detail="role must be 'user' or 'admin'")
        user.role = payload.role
    db.commit()
    db.refresh(user)
    return UserItem.from_user(user)


@router.delete("/users/{user_id}", summary="删除用户", description="删除指定用户（不能删除自己）", responses={400: {"description": "不能删除自己"}, 404: {"description": "用户不存在"}})
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_admin_user),
):
    if user_id == admin_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {"detail": "User deleted"}


# ── Conversation management ───────────────────────────────────────────────

@router.get("/conversations", summary="获取所有对话", description="获取全平台所有用户的对话列表，按更新时间降序")
def list_all_conversations(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_admin_user),
):
    convs = db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
    return [ConversationItem.from_conv(c) for c in convs]


@router.get("/conversations/{conv_id}", summary="获取对话详情", description="获取指定对话的完整信息，包括所有消息", responses={404: {"description": "对话不存在"}})
def get_conversation_detail(
    conv_id: str,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_admin_user),
):
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "id": conv.id,
        "title": conv.title,
        "model": conv.model,
        "username": conv.owner.username if conv.owner else None,
        "created_at": conv.created_at.isoformat(),
        "updated_at": conv.updated_at.isoformat(),
        "messages": [{"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in conv.messages],
    }


@router.delete("/conversations/{conv_id}", summary="删除对话", description="删除指定对话及其所有消息", responses={404: {"description": "对话不存在"}})
def delete_conversation(
    conv_id: str,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_admin_user),
):
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conv)
    db.commit()
    return {"detail": "Conversation deleted"}
