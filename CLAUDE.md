# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Full-stack Chat Agent web app (Chinese UI). React 18 + Vite + Tailwind CSS frontend, FastAPI + SQLAlchemy backend. Supports multi-turn streaming chat, file upload with text extraction, multi-model switching (DeepSeek, Zhipu GLM-4), web search, RAG knowledge retrieval, long-term memory, and dual agent architectures (LangGraph / Loop Agent).

## Common Commands

**Frontend (dev server)**
```bash
cd frontend && npm run start     # vite --port 5173 --host
cd frontend && npm run build     # tsc && vite build
```

**Backend (dev server)**
```bash
cd backend && run.bat            # Windows .bat launcher
cd backend && venv/Scripts/python run.py   # uvicorn with reload
```

**Backend tests**
```bash
cd backend && venv/Scripts/python -m pytest tests/ -v
# single test
cd backend && venv/Scripts/python -m pytest tests/test_chat_router.py::test_chat_stream -v
```

## Architecture

### Backend (`backend/`)

**Entry point wiring** (`app/main.py`)
- `load_dotenv()` executes at the very top (before imports) with an absolute path to `backend/.env`. This is critical: module-level `os.getenv()` calls in routers depend on it.
- `Base.metadata.create_all(bind=engine)` runs at module import time. Any test that imports `app.main` will trigger this against the production engine, so test fixtures **must** override `get_db`.
- Routers are registered under `/api/v1` prefix: `chat`, `auth`, `files`, `memory`, `admin`.
- `/health` checks database and milvus status; `/metrics` exposes Prometheus metrics.

**Configuration layer** (`app/core/config.py`)
- Uses `pydantic_settings.BaseSettings`, reads all config from `.env`.
- Supported env vars: `DATABASE_URL`, `DEEPSEEK_*`, `ZHIPU_*`, `MILVUS_URI`, `OLLAMA_*`, `TAVILY_API_KEY`, `SILICONFLOW_*`.
- Feature flags: `use_langgraph_agent` (default `False`), `enable_memory_intent_filter` (default `True`).

**Database layer** (`app/db.py`, `app/models.py`)
- SQLAlchemy 2.0 declarative style with `Mapped` / `mapped_column`.
- Production uses PostgreSQL (`DATABASE_URL` in `.env`). Default fallback is `sqlite:///./chat_agent.db`.
- `Conversation.id` is a **UUID string** primary key (`default=lambda: str(uuid.uuid4())`). `Message` and `User` use integer auto-increment PKs.
- `Conversation` has a `model: Mapped[str]` field that drives which LLM API is called.
- `UploadedFile` stores extracted text in `extracted_text` (Text column) and physical files in `uploads/{conversation_id}/`.
- `User` has a `role` field (`"user" | "admin"`). `UserMemory` stores auto-extracted long-term memories.

**Auth & Authorization** (`app/routers/auth.py`, `app/core/security.py`, `app/deps.py`)
- JWT via `python-jose`, bcrypt via `passlib`.
- Login uses `OAuth2PasswordRequestForm` (form data, not JSON).
- `get_current_user` dependency decodes the Bearer token and queries the DB.
- `get_admin_user` dependency requires `role == "admin"`.

**Chat flow** (`app/routers/chat.py`)
- `MODEL_CONFIG` dict maps `conv.model` values to `{api_key, base_url, model}`.
- To add a new model provider, add env vars to `.env` and a new entry to `MODEL_CONFIG`.
- Every message first runs **memory retrieval** (mem0), then **RAG retrieval** (Milvus), then enters the Agent path.
- Two mutually exclusive Agent paths, controlled by `settings.use_langgraph_agent`:
  - **LangGraph Agent** (`app/langgraph_agent/`): based on `StateGraph`, nodes run memory + RAG in parallel, then context → LLM → tool call / evaluation → loop or end.
  - **Loop Agent** (`app/services/loop_agent.py`): iterative LLM calls with tool_call support (web search). After each iteration a non-streaming LLM evaluates response quality; if it fails, the agent auto-corrects, up to 50 iterations.
- Both agents return the same SSE event types: `step` (phase status), `content` (streaming text), `sources` (web search results), `tool_call` / `tool_result` (LangGraph tool calls).
- Assistant messages are saved inside the async generator via `asyncio.to_thread(...)` because SQLAlchemy sync sessions cannot be used directly in async generators.
- Auto-generates a title (3–8 Chinese characters) after the first exchange.
- Auto-extracts long-term memory (mem0) after a conversation ends. When `enable_memory_intent_filter` is on, SiliconFlow is used to judge whether the dialogue contains personal information worth storing.

**RAG retrieval** (`app/services/rag_pipeline.py`)
- `run_rag()` chain: `get_dense_embedding` (Ollama) → `hybrid_search` (Milvus vector search + keyword BM25 hybrid) → `rerank_passages` (Ollama reranker).
- Parameters: `top_k_hybrid=15`, `top_n_rerank=5`. Skips rerank when fewer than 3 results.

**Vector store** (`app/core/milvus.py`, `app/services/milvus_store.py`)
- Milvus client via `get_milvus_client()`, URI controlled by `MILVUS_URI`.
- `ensure_collection()` creates the collection; `insert_message()` writes message dense embeddings.
- Every user / assistant message is indexed to Milvus asynchronously after being sent.

**Web search** (`app/services/web_search.py`, `app/services/search_engines/`)
- Supports multiple engines: Tavily, DuckDuckGo, SearXNG.
- Priority controlled by `search_engine_priority` (comma-separated), e.g. `"tavily,duckduckgo"`.
- `app/langgraph_agent/tools/web_search.py` provides `web_search_tool` for the LangGraph Agent.

**Long-term memory** (`app/services/memory_client.py`, `app/routers/memory.py`)
- Uses mem0 as the underlying memory engine, accessed via `get_memory()`.
- Auto-extraction: after a conversation ends, stores user message + assistant reply into mem0, isolated by `user_id`.
- Intent filter (`app/services/intent.py`): when `enable_memory_intent_filter=True`, uses a SiliconFlow model to judge whether the dialogue contains personal info; only stores when valuable.
- `memory.py` router provides manual CRUD, search, paginated list, and `/memory/store` SSE streaming storage endpoint.

**File upload** (`app/routers/files.py`)
- Supported MIME types: `text/plain`, `application/pdf`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document`.
- PDF uses `pypdf` (not `PyPDF2`), DOCX uses `python-docx`.
- Max 10 MB. Files stored in `uploads/{conversation_id}/` with UUID suffix filenames.

**Admin panel** (`app/routers/admin.py`)
- Prefix `/api/v1/admin`, requires `admin` role.
- Provides stats (users, conversations, messages, files, memories, 7-day active users), user management (CRUD), and conversation management (list, detail, delete).

### Frontend (`frontend/src/`)

**Pages & routing**
- `App.tsx` is a single-page app without React Router; `currentPage` state switches between `chat` | `memories` | `admin`.
- Page components: `LoginPage`, `AdminPage`, `MemoriesPage`.

**State management**
- `useChatStream` (custom hook) is the single source of truth. Holds `conversations`, `currentConvId`, `currentModel`, `isStreaming`, `toolSteps`, `pendingMemory`, `memoryStore`, `enableWebSearch`.
- SSE parsing is done manually with `fetch()` + `ReadableStream.getReader()` + `TextDecoder`. Supported event types:
  - `step` — displays Agent execution phases (memory check, RAG, web search, etc.).
  - `content` — streaming text delta.
  - `sources` — web search results, rendered in the `SearchSources` component below the message.
  - `tool_call` / `tool_result` — LangGraph tool calls and results, rendered in the `ToolCallBox` component.
  - `memory_prompt` — prompts the user to save a long-term memory.
- `reasoningBufferRef` collects all `step` events; after streaming ends it is written into `message.reasoning` and displayed by `ReasoningPanel`.
- File upload happens **before** sending the chat message. Extracted text from uploaded files is prefixed to the user message as `【文件：{filename}】\n{extracted_text}`.

**Component flow**
- `App.tsx` -> `Sidebar` (conversation list + page nav) + `ChatArea` (header with model selector + messages + `InputBox`).
- `InputBox` handles file selection, web search toggle, and message sending.
- `MessageItem` renders normal text, source cards, reasoning panel, and tool call cards.
- `ChatArea` header has a `<select>` for model switching (`deepseek-chat` / `glm-4`).

## Environment Variables

All backend config lives in `backend/.env` (loaded at `main.py` import time):

- `DATABASE_URL` — PostgreSQL connection string (production) or SQLite path.
- `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` — DeepSeek config.
- `ZHIPU_API_KEY` / `ZHIPU_BASE_URL` / `ZHIPU_MODEL` — Zhipu config.
- `MILVUS_URI` — Milvus server address (default `http://localhost:19530`).
- `OLLAMA_BASE_URL` — Ollama address (default `http://localhost:11434`).
- `EMBEDDING_MODEL` / `RERANK_MODEL` / `OLLAMA_LARGE_MODEL` — Ollama model names.
- `TAVILY_API_KEY` — Tavily search API key.
- `SEARCH_ENGINE_PRIORITY` — Search engine priority, comma-separated (default `tavily,duckduckgo`).
- `SILICONFLOW_API_KEY` / `SILICONFLOW_BASE_URL` / `SILICONFLOW_MODEL` — Memory intent filter model config.
- `ENABLE_MEMORY_INTENT_FILTER` — Whether to enable memory intent filtering (default `true`).
- `USE_LANGGRAPH_AGENT` — Whether to use the LangGraph Agent (default `false`).
- `SECRET_KEY` — JWT signing key.
- `ENV` — `development` or `production`.
- `CORS_ORIGINS` — Comma-separated allowed origins (default `http://localhost:5173`).

## Testing Patterns

- Every test file that hits the DB must:
  1. Import `app.models` to register tables with `Base.metadata`.
  2. Create its own SQLite `:memory:` engine with `StaticPool`.
  3. Set `app.dependency_overrides[get_db] = override_get_db` **inside** an `autouse` fixture, and clear it in teardown (`app.dependency_overrides.clear()`). Failure to do this causes cross-test engine conflicts (e.g. "no such table" errors).
- Streaming tests mock `httpx.AsyncClient` with `FakeAsyncClient` / `FakeStreamResponse`. `FakeAsyncClient` records the last `stream()` call in `last_stream_call` so tests can assert on the URL, headers, and request body.

## Notes

- No existing `README.md`, `.cursorrules`, or Copilot instructions at project root.
- `backend/.env.example` exists but is outdated (only lists old vars; missing Zhipu, Milvus, Ollama, Tavily, SiliconFlow, etc.).
- `PyPDF2` is deprecated; the project has migrated to `pypdf`.
- The backend `uploads/` directory is created on-demand by the file upload router.
- Milvus connection failures do not block the chat flow; they are logged and the pipeline continues.
