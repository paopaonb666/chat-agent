import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio
from sqlalchemy.orm import Session
from app.core.config import settings
from app.db import get_db
from app.models import Conversation, Message, UserMemory
from app.services.rag_pipeline import run_rag
from app.core.sse import step_line
from app.core.metrics import get_memory_op_counter
from app.services.memory_client import get_memory
from app.services.intent import should_extract_memory
from app.services.prompts import MEMORY_GUARDRAIL_INSTRUCTION
from app.deps import get_current_user
from app.services.loop_agent import LoopAgent
from app.services.indexing import index_message
from app.services.title_generator import auto_title_on_first_exchange
from app.langgraph_agent.agent import LangGraphAgent
from langchain.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


MODEL_CONFIG = {
    "deepseek-chat": {
        "api_key": settings.deepseek_api_key,
        "base_url": settings.deepseek_base_url,
        "model": settings.deepseek_model,
    },
    "glm-4": {
        "api_key": settings.zhipu_api_key,
        "base_url": settings.zhipu_base_url,
        "model": settings.zhipu_model,
    },
}


class CreateConversationRequest(BaseModel):
    title: str = Field("New Conversation", description="对话标题")
    model: str = Field("deepseek-chat", description="选择模型: deepseek-chat 或 glm-4")


class ChatCompletionsRequest(BaseModel):
    conversation_id: str = Field(..., description="会话 ID")
    message: str = Field("", description="用户消息内容")
    enable_web_search: bool = Field(False, description="是否启用联网搜索")


class UpdateConversationRequest(BaseModel):
    title: str | None = Field(None, description="新的对话标题，不传则不修改")


def _save_assistant_message(db: Session, conv_id: str, content: str, sources: str = "", reasoning: str = "") -> int:
    msg = Message(conversation_id=conv_id, role="assistant", content=content, sources=sources, reasoning=reasoning)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg.id


@router.post("/conversations", summary="创建新对话", description="为用户创建一个新的对话，可选指定标题和模型", responses={400: {"description": "参数校验失败"}})
def create_conversation(payload: CreateConversationRequest, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    conv = Conversation(title=payload.title, model=payload.model, user_id=current_user.id)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return {"id": conv.id, "title": conv.title, "model": conv.model, "messages": []}


@router.get("/conversations", summary="获取对话列表", description="返回当前用户的所有对话，按更新时间降序排列")
def list_conversations(db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    convs = db.query(Conversation).filter(Conversation.user_id == current_user.id).order_by(Conversation.updated_at.desc()).all()
    return [
        {
            "id": c.id,
            "title": c.title,
            "model": c.model,
            "messages": [{"role": m.role, "content": m.content, "sources": json.loads(m.sources) if m.sources else [], "reasoning": m.reasoning or ""} for m in c.messages],
        }
        for c in convs
    ]


@router.get("/conversations/{conv_id}", summary="获取对话详情", description="返回指定对话的完整信息，包括所有消息历史", responses={404: {"description": "对话不存在"}})
def get_conversation(conv_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    conv = db.query(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == current_user.id).first()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "id": conv.id,
        "title": conv.title,
        "model": conv.model,
        "messages": [{"role": m.role, "content": m.content, "sources": json.loads(m.sources) if m.sources else [], "reasoning": m.reasoning or ""} for m in conv.messages],
    }


@router.patch("/conversations/{conv_id}", summary="更新对话", description="更新对话标题", responses={404: {"description": "对话不存在"}})
def update_conversation(conv_id: str, payload: UpdateConversationRequest, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    conv = db.query(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == current_user.id).first()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if payload.title is not None:
        conv.title = payload.title
    db.commit()
    return {"id": conv.id, "title": conv.title}


@router.post("/chat/completions", summary="发送消息（流式）", description="向指定对话发送消息并获取 AI 流式回复。支持联网搜索、记忆检索和知识库检索。返回 SSE 事件流。", responses={400: {"description": "缺少 conversation_id"}, 404: {"description": "对话不存在"}})
async def chat_completions(payload: ChatCompletionsRequest, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    conv_id = payload.conversation_id
    message = payload.message
    if not conv_id:
        raise HTTPException(status_code=400, detail="conversation_id required")

    conv = db.query(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == current_user.id).first()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    new_msg = Message(conversation_id=conv_id, role="user", content=message)
    db.add(new_msg)
    db.commit()
    db.refresh(new_msg)
    db.refresh(conv)
    messages = [{"role": m.role, "content": m.content} for m in conv.messages]

    asyncio.create_task(
        index_message(conv_id, conv.user_id, "user", message, new_msg.id)
    )

    cfg = MODEL_CONFIG.get(conv.model, MODEL_CONFIG["deepseek-chat"])
    api_key = cfg["api_key"]
    base_url = cfg["base_url"]
    model_name = cfg["model"]

    async def event_stream():
        # ── Phase 1: Memory self-check ───────────────────────────────
        yield step_line("memory_check", "running", "记忆检索", "正在搜索相关记忆...")
        memory_context = ""
        try:
            uid = str(conv.user_id) if conv.user_id else str(current_user.id)
            logger.info("Phase 1 (memory): searching for user_id=%s query=%.50s", uid, message)
            memory = get_memory()
            results = memory.search(
                query=message,
                filters={"user_id": uid},
                top_k=3,
            )
            items = results.get("results", []) if isinstance(results, dict) else []
            get_memory_op_counter().labels(operation="search", status="success").inc()
            if items:
                lines = [f"- {r['memory']}" for r in items]
                memory_context = (
                    MEMORY_GUARDRAIL_INSTRUCTION + "\n\n"
                    + "以下是该用户的长期记忆中存储的相关信息：\n"
                    + "\n".join(lines)
                )
                yield step_line("memory_check", "completed", "记忆检索", f"找到 {len(items)} 条相关记忆")
            else:
                yield step_line("memory_check", "completed", "记忆检索", "未找到相关记忆")
        except Exception:
            logger.exception("Memory search failed")
            get_memory_op_counter().labels(operation="search", status="error").inc()
            yield step_line("memory_check", "error", "记忆检索", "检索失败")

        if settings.use_langgraph_agent:
            # ── LangGraph Agent path ──────────────────────────────
            uid = str(conv.user_id) if conv.user_id else str(current_user.id)
            initial_state = {
                "messages": [
                    HumanMessage(content=m["content"]) if m["role"] == "user"
                    else AIMessage(content=m["content"])
                    for m in messages
                ],
                "user_message": message,
                "enable_web_search": payload.enable_web_search,
                "api_key": api_key,
                "base_url": base_url,
                "model_name": model_name,
                "user_id": uid,
                "conversation_id": conv_id,
                "memory_context": memory_context,
                "rag_context": "",
                "web_sources": [],
                "iteration_count": 0,
                "last_failure_reason": "",
                "same_failure_count": 0,
                "final_content": "",
                "continue_loop": False,
            }
            langgraph_agent = LangGraphAgent()
            event_stream_gen, holder = await langgraph_agent.run(initial_state)
            web_sources = []
            async for event in event_stream_gen:
                if event.startswith("data: "):
                    try:
                        parsed = json.loads(event[6:])
                        if parsed.get("type") == "sources":
                            web_sources = parsed.get("sources", [])
                    except Exception:
                        logger.exception("LangGraph SSE event parse failed")
                yield event
            assistant_content = holder["final_content"]

        else:
            # ── Phase 2: RAG ────────────────────────────────────────
            yield step_line("rag_retrieval", "running", "知识库检索", "正在检索历史对话...")
            rag_context = ""
            try:
                rag_context = await asyncio.wait_for(
                    run_rag(db, message, conv_id, conv.user_id, messages, query_override=message),
                    timeout=10.0,
                )
                if rag_context:
                    yield step_line("rag_retrieval", "completed", "知识库检索", "检索到相关内容")
                else:
                    yield step_line("rag_retrieval", "completed", "知识库检索", "未检索到相关内容")
            except Exception:
                logger.exception("RAG retrieval failed")
                yield step_line("rag_retrieval", "error", "知识库检索", "检索超时")

            # ── Phase 3: Build context ─────────────────────────────────
            combined_parts = [p for p in [memory_context, rag_context] if p]

            llm_messages = messages
            if combined_parts:
                system_content = "\n\n".join(combined_parts)
                llm_messages = [{"role": "system", "content": system_content}] + messages

            if payload.enable_web_search:
                search_instruction = (
                    "用户要求启用联网搜索。你必须先调用 web_search 工具搜索相关信息，"
                    "再根据搜索结果回答。搜索关键词应保留完整的人名、地名和专有名词。"
                )
                llm_messages = [{"role": "system", "content": search_instruction}] + llm_messages

            # ── Phase 4: Loop Agent ──────────────────────────────────
            agent = LoopAgent(max_iterations=50)
            web_sources = []
            async for event in agent.run(
                llm_messages, message, payload.enable_web_search,
                api_key, base_url, model_name,
            ):
                if event.startswith("data: "):
                    try:
                        parsed = json.loads(event[6:])
                        if parsed.get("type") == "sources":
                            web_sources = parsed.get("sources", [])
                    except Exception:
                        pass
                yield event
            assistant_content = agent.final_content

        # ── Phase 5: Save assistant message ──────────────────────────
        if assistant_content:
            sources_json = json.dumps(web_sources) if web_sources else ""
            msg_id = await asyncio.to_thread(
                _save_assistant_message, db, conv_id, assistant_content, sources_json, ""
            )
            asyncio.create_task(
                index_message(conv_id, conv.user_id, "assistant", assistant_content, msg_id)
            )

        # ── Phase 6: Auto-generate title ─────────────────────────────
        if conv.title == "新对话":
            first_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
            if first_msg and assistant_content:
                asyncio.create_task(
                    auto_title_on_first_exchange(conv_id, first_msg, assistant_content, cfg, db)
                )

        # ── Phase 7: Auto-extract memories via mem0 ──────────────────
        logger.info(
            "Phase 7 (memory): entering — assistant_content len=%d, enable_memory_intent_filter=%s",
            len(assistant_content) if assistant_content else 0,
            settings.enable_memory_intent_filter,
        )
        if assistant_content:
            try:
                uid = str(conv.user_id) if conv.user_id else str(current_user.id)
                if settings.enable_memory_intent_filter:
                    logger.info("Phase 7 (memory): calling should_extract_memory...")
                    needs_memory, memory_content = await should_extract_memory([
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": assistant_content},
                    ])
                    logger.info(
                        "Phase 7 (memory): intent result — needs=%s, content_len=%d",
                        needs_memory,
                        len(memory_content) if memory_content else 0,
                    )
                    if needs_memory and memory_content:
                        logger.info("Memory intent: storing — %s", memory_content[:80])
                        memory = get_memory()
                        memory.add(
                            [{"role": "user", "content": memory_content}],
                            user_id=uid,
                        )
                        get_memory_op_counter().labels(operation="add", status="success").inc()
                        # Sync to PostgreSQL
                        try:
                            pg_mem = UserMemory(user_id=int(uid), content=memory_content, source="auto_extracted")
                            db.add(pg_mem)
                            db.commit()
                        except Exception:
                            logger.warning("Failed to write memory to PostgreSQL", exc_info=True)
                    else:
                        logger.debug("Memory intent: skipped (no personal info detected)")
                else:
                    memory = get_memory()
                    memory.add(
                        [
                            {"role": "user", "content": message},
                            {"role": "assistant", "content": assistant_content},
                        ],
                        user_id=uid,
                    )
                    get_memory_op_counter().labels(operation="add", status="success").inc()
                    # Sync to PostgreSQL (store user message as the memory content since mem0 extracts internally)
                    try:
                        pg_mem = UserMemory(user_id=int(uid), content=message, source="auto_extracted")
                        db.add(pg_mem)
                        db.commit()
                    except Exception:
                        logger.warning("Failed to write memory to PostgreSQL", exc_info=True)
            except Exception:
                logger.exception("mem0 memory extraction failed")
                get_memory_op_counter().labels(operation="add", status="error").inc()
        else:
            logger.warning("Phase 7 (memory): SKIPPED — assistant_content is empty or None")

    return StreamingResponse(event_stream(), media_type="text/event-stream")
