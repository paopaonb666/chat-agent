# CLAUDE.md（中文版）

本文件为 Claude Code（claude.ai/code）在处理此仓库代码时提供指导。

## 项目概述

全栈聊天代理 Web 应用（中文界面）。前端使用 React 18 + Vite + Tailwind CSS，后端使用 FastAPI + SQLAlchemy。支持多轮流式聊天、文件上传与文本提取、多模型切换（DeepSeek、智谱 GLM-4）、联网搜索、RAG 知识库检索、长期记忆、LangGraph / Loop Agent 双架构。

## 常用命令

**前端（开发服务器）**
```bash
cd frontend && npm run start     # vite --port 5173 --host
cd frontend && npm run build     # tsc && vite build
```

**后端（开发服务器）**
```bash
cd backend && run.bat            # Windows .bat 启动器
cd backend && venv/Scripts/python run.py   # uvicorn 热重载模式
```

**后端测试**
```bash
cd backend && venv/Scripts/python -m pytest tests/ -v
# 单个测试
cd backend && venv/Scripts/python -m pytest tests/test_chat_router.py::test_chat_stream -v
```

## 架构

### 后端（`backend/`）

**入口点配置**（`app/main.py`）
- `load_dotenv()` 在最顶部（import 之前）执行，使用绝对路径加载 `backend/.env`。这非常关键：路由中的模块级 `os.getenv()` 调用依赖于此。
- `Base.metadata.create_all(bind=engine)` 在模块导入时运行。任何导入 `app.main` 的测试都会触发生产引擎，因此测试夹具**必须**重写 `get_db`。
- 路由注册在 `/api/v1` 前缀下：`chat`、`auth`、`files`、`memory`、`admin`。
- `/health` 检查 database 和 milvus 状态；`/metrics` 暴露 Prometheus 指标。

**配置层**（`app/core/config.py`）
- 使用 `pydantic_settings.BaseSettings`，从 `.env` 读取全部配置。
- 支持的环境变量：`DATABASE_URL`、`DEEPSEEK_*`、`ZHIPU_*`、`MILVUS_URI`、`OLLAMA_*`、`TAVILY_API_KEY`、`SILICONFLOW_*`。
- 功能开关：`use_langgraph_agent`（默认 `False`）、`enable_memory_intent_filter`（默认 `True`）。

**数据库层**（`app/db.py`、`app/models.py`）
- SQLAlchemy 2.0 声明式风格，使用 `Mapped` / `mapped_column`。
- 生产环境使用 PostgreSQL（`.env` 中的 `DATABASE_URL`）。默认回退为 `sqlite:///./chat_agent.db`。
- `Conversation.id` 是 **UUID 字符串**主键（`default=lambda: str(uuid.uuid4())`）。`Message` 和 `User` 使用整数自增主键。
- `Conversation` 有一个 `model: Mapped[str]` 字段，用于决定调用哪个 LLM API。
- `UploadedFile` 将提取的文本存储在 `extracted_text`（Text 列）中，物理文件存储在 `uploads/{conversation_id}/`。
- `User` 有 `role` 字段（`"user" | "admin"`），`UserMemory` 存储自动提取的长期记忆。

**认证与授权**（`app/routers/auth.py`、`app/core/security.py`、`app/deps.py`）
- 使用 `python-jose` 实现 JWT，使用 `passlib` 实现 bcrypt。
- 登录使用 `OAuth2PasswordRequestForm`（表单数据，非 JSON）。
- `get_current_user` 依赖解码 Bearer token 并查询数据库。
- `get_admin_user` 依赖要求 `role == "admin"`。

**聊天流**（`app/routers/chat.py`）
- `MODEL_CONFIG` 字典将 `conv.model` 值映射到 `{api_key, base_url, model}`。
- 要添加新的模型提供商，在 `.env` 中添加环境变量，并在 `MODEL_CONFIG` 中添加新条目。
- 每条消息发送时，先进行 **记忆检索**（mem0），然后进行 **RAG 检索**（Milvus），最后进入 Agent 路径。
- 有两个互斥的 Agent 路径，由 `settings.use_langgraph_agent` 控制：
  - **LangGraph Agent**（`app/langgraph_agent/`）：基于 `StateGraph`，节点并行执行 memory + RAG，然后 context → LLM → 工具调用/评估 → 循环或结束。
  - **Loop Agent**（`app/services/loop_agent.py`）：迭代式 LLM 调用，支持 tool_call（联网搜索），每次迭代后通过非流式 LLM 评估响应质量，未通过则自动修正，最多 50 次迭代。
- 两个 Agent 都通过 SSE 返回相同格式的事件：`step`（阶段状态）、`content`（流式文本）、`sources`（联网搜索结果）、`tool_call` / `tool_result`（LangGraph 工具调用）。
- 助手消息通过 `asyncio.to_thread(...)` 在异步生成器内保存，因为 SQLAlchemy 同步会话不能直接在异步生成器中使用。
- 首次对话后自动调用 LLM 生成标题（3–8 字）。
- 对话结束后自动提取长期记忆（mem0），受 `enable_memory_intent_filter` 控制时通过 SiliconFlow 判断是否为个人相关信息。

**RAG 检索**（`app/services/rag_pipeline.py`）
- `run_rag()` 调用链路：`get_dense_embedding`（Ollama）→ `hybrid_search`（Milvus 向量搜索 + 关键词 BM25 混合）→ `rerank_passages`（Ollama reranker）。
- 参数：`top_k_hybrid=15`、`top_n_rerank=5`。结果少于 3 条时跳过 rerank。

**向量存储**（`app/core/milvus.py`、`app/services/milvus_store.py`）
- Milvus 客户端通过 `get_milvus_client()` 获取，URI 由 `MILVUS_URI` 控制。
- `ensure_collection()` 创建 collection，`insert_message()` 写入 message 的 dense embedding。
- 每次用户/助手消息发出后都会异步索引到 Milvus。

**联网搜索**（`app/services/web_search.py`、`app/services/search_engines/`）
- 支持多个搜索引擎：Tavily、DuckDuckGo、SearXNG。
- 优先级由 `search_engine_priority`（逗号分隔）控制，例如 `"tavily,duckduckgo"`。
- `app/langgraph_agent/tools/web_search.py` 为 LangGraph Agent 提供 `web_search_tool`。

**长期记忆**（`app/services/memory_client.py`、`app/routers/memory.py`）
- 使用 mem0 作为底层记忆引擎，通过 `get_memory()` 获取客户端。
- 自动提取：对话结束后将用户消息和助手回复存入 mem0，按 `user_id` 隔离。
- 意图过滤（`app/services/intent.py`）：若 `enable_memory_intent_filter=True`，用 SiliconFlow 模型判断对话是否包含个人信息，仅在有价值时才存储。
- `memory.py` 路由提供手动增删改查、搜索、分页列表，以及 `/memory/store` SSE 流式存储接口。

**文件上传**（`app/routers/files.py`）
- 支持 MIME 类型：`text/plain`、`application/pdf`、`application/vnd.openxmlformats-officedocument.wordprocessingml.document`。
- PDF 使用 `pypdf`（不是 `PyPDF2`），DOCX 使用 `python-docx`。
- 最大 10 MB，文件存储在 `uploads/{conversation_id}/`，文件名使用 UUID 后缀。

**后台管理**（`app/routers/admin.py`）
- 前缀 `/api/v1/admin`，需要 `admin` 角色。
- 提供统计（用户数、对话数、消息数、文件数、记忆数、7 日活跃用户）、用户管理（增删改查）、对话管理（列表、详情、删除）。

### 前端（`frontend/src/`）

**页面与路由**
- `App.tsx` 是单页应用，无 React Router，通过 `currentPage` state 切换：`chat` | `memories` | `admin`。
- 页面组件：`LoginPage`、`AdminPage`、`MemoriesPage`。

**状态管理**
- `useChatStream`（自定义 hook）是单一数据源。它保存 `conversations`、`currentConvId`、`currentModel`、`isStreaming`、`toolSteps`、`pendingMemory`、`memoryStore`、`enableWebSearch`。
- SSE 解析通过 `fetch()` + `ReadableStream.getReader()` + `TextDecoder` 手动完成。支持的事件类型：
  - `step` — 显示 Agent 执行阶段（记忆检索、RAG、联网搜索等）。
  - `content` — 流式文本增量。
  - `sources` — 联网搜索结果，渲染在消息下方的 `SearchSources` 组件。
  - `tool_call` / `tool_result` — LangGraph 工具调用与结果，渲染在 `ToolCallBox` 组件。
  - `memory_prompt` — 提示用户保存长期记忆。
- `reasoningBufferRef` 收集所有 `step` 事件，流结束后写入 `message.reasoning`，由 `ReasoningPanel` 展示。
- 文件上传在发送聊天消息**之前**进行。从上传文件中提取的文本作为前缀添加到用户消息中，格式为 `【文件：{filename}】\n{extracted_text}`。

**组件流程**
- `App.tsx` -> `Sidebar`（对话列表 + 页面导航）+ `ChatArea`（头部模型选择器 + 消息列表 + `InputBox`）。
- `InputBox` 处理文件选择、联网搜索开关、发送消息。
- `MessageItem` 渲染普通文本、来源卡片、思考过程面板、工具调用卡片。
- `ChatArea` 头部有 `<select>` 用于模型切换（`deepseek-chat` / `glm-4`）。

## 环境变量

所有后端配置存储在 `backend/.env`（在 `main.py` 导入时加载）：

- `DATABASE_URL` — PostgreSQL 连接字符串（生产环境）或 SQLite 路径。
- `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` — DeepSeek 配置。
- `ZHIPU_API_KEY` / `ZHIPU_BASE_URL` / `ZHIPU_MODEL` — 智谱配置。
- `MILVUS_URI` — Milvus 服务器地址（默认 `http://localhost:19530`）。
- `OLLAMA_BASE_URL` — Ollama 地址（默认 `http://localhost:11434`）。
- `EMBEDDING_MODEL` / `RERANK_MODEL` / `OLLAMA_LARGE_MODEL` — Ollama 模型名称。
- `TAVILY_API_KEY` — Tavily 搜索 API Key。
- `SEARCH_ENGINE_PRIORITY` — 搜索引擎优先级，逗号分隔（默认 `tavily,duckduckgo`）。
- `SILICONFLOW_API_KEY` / `SILICONFLOW_BASE_URL` / `SILICONFLOW_MODEL` — 记忆意图过滤模型配置。
- `ENABLE_MEMORY_INTENT_FILTER` — 是否开启记忆意图过滤（默认 `true`）。
- `USE_LANGGRAPH_AGENT` — 是否使用 LangGraph Agent（默认 `false`）。
- `SECRET_KEY` — JWT 签名密钥。
- `ENV` — `development` 或 `production`。
- `CORS_ORIGINS` — 逗号分隔的允许源（默认 `http://localhost:5173`）。

## 测试模式

- 每个访问数据库的测试文件必须：
  1. 导入 `app.models` 以向 `Base.metadata` 注册表。
  2. 使用 `StaticPool` 创建自己的 SQLite `:memory:` 引擎。
  3. 在 `autouse` 夹具**内部**设置 `app.dependency_overrides[get_db] = override_get_db`，并在 teardown 时清除（`app.dependency_overrides.clear()`）。不这样做会导致测试间引擎冲突（例如 "no such table" 错误）。
- 流式测试使用 `FakeAsyncClient` / `FakeStreamResponse` 模拟 `httpx.AsyncClient`。`FakeAsyncClient` 将最后一次 `stream()` 调用记录在 `last_stream_call` 中，以便测试可以断言 URL、请求头和请求体。

## 注意事项

- 项目根目录没有现有的 `README.md`、`.cursorrules` 或 Copilot 指令。
- `backend/.env.example` 存在但已过时（仅列出了旧变量；缺少 Zhipu、Milvus、Ollama、Tavily、SiliconFlow 等）。
- `PyPDF2` 已弃用；项目已迁移到 `pypdf`。
- 后端 `uploads/` 目录由文件上传路由按需创建。
- Milvus 连接失败不会阻塞聊天流程，会在日志中记录异常并继续。
