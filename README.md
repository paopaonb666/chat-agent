# Chat Agent

全栈智能对话代理，支持多轮流式对话、RAG 知识检索、Web 搜索、长期记忆、文件上传与双 Agent 架构。

## 功能特性

- **多轮流式对话** — SSE 实时流式输出，支持推理过程展示
- **多模型切换** — DeepSeek Chat / Zhipu GLM-4
- **RAG 知识检索** — Ollama 向量嵌入 → Milvus 混合搜索（向量 + BM25）→ 重排序
- **Web 搜索** — Tavily / DuckDuckGo / SearXNG 多引擎支持
- **长期记忆** — mem0 自动提取与持久化，支持意图过滤
- **文件上传** — PDF / DOCX / TXT 文本提取并注入对话上下文
- **双 Agent 架构** — LangGraph StateGraph 与 Loop Auto-Correction 两种模式
- **用户认证** — JWT 登录与基于角色的鉴权（user / admin）
- **管理面板** — 用户管理、对话管理、系统统计
- **自动标题** — 首轮对话后自动生成 3-8 字对话标题

## 技术栈

| 层 | 技术 |
|---|---|
| 前端 | React 18 + TypeScript + Vite + Tailwind CSS |
| 后端 | Python 3.11+ / FastAPI + SQLAlchemy 2.0 |
| 数据库 | PostgreSQL（生产）/ SQLite（开发） |
| 向量库 | Milvus |
| 向量模型 | Ollama（qwen3-embedding:0.6b） |
| 重排序 | Ollama（qwen3-reranker-0.6b） |
| 记忆引擎 | mem0 |
| Agent | LangGraph / Loop Agent |
| 认证 | JWT（python-jose）+ bcrypt（passlib） |
| 搜索引擎 | Tavily API / DuckDuckGo / SearXNG |

## 快速开始

### 前置依赖

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com/)（本地向量嵌入与重排序）
- [Milvus](https://milvus.io/)（向量数据库，推荐 Docker 部署）
- PostgreSQL（可选，开发环境可用 SQLite）

### 后端

```bash
# 1. 进入后端目录
cd backend

# 2. 创建虚拟环境
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
# 或 venv\Scripts\activate    # Windows CMD

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key 等配置

# 5. 启动服务
python run.py
# 或 run.bat
# 服务运行在 http://localhost:8000
```

### 前端

```bash
# 1. 进入前端目录
cd frontend

# 2. 安装依赖
npm install

# 3. 启动开发服务器
npm run start
# 或 npm run dev
# 服务运行在 http://localhost:5173
```

### 初始用户

启动后端后，首次注册的用户自动为管理员。或通过 API 创建：

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

## 配置说明

所有后端配置通过 `backend/.env` 文件设置：

### 必需配置

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 |
| `SECRET_KEY` | JWT 签名密钥（生产环境请更换） |

### 模型配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | DeepSeek API 地址 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek 模型名 |
| `ZHIPU_API_KEY` | — | Zhipu API 密钥 |
| `ZHIPU_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4/` | Zhipu API 地址 |
| `ZHIPU_MODEL` | `glm-4` | Zhipu 模型名 |

### 向量与 RAG

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MILVUS_URI` | `http://localhost:19530` | Milvus 服务地址 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 地址 |
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | 向量嵌入模型 |
| `RERANK_MODEL` | `pdurugyan/qwen3-reranker-0.6b-q8_0` | 重排序模型 |
| `OLLAMA_LARGE_MODEL` | `qwen2.5:1.5b` | 评估用模型 |

### 搜索引擎

| 变量 | 默认值 | 说明 |
|---|---|---|
| `TAVILY_API_KEY` | — | Tavily 搜索 API 密钥 |
| `SEARCH_ENGINE_PRIORITY` | `tavily,duckduckgo` | 引擎优先级，逗号分隔 |

### 记忆与 AI 过滤

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SILICONFLOW_API_KEY` | — | SiliconFlow API 密钥（记忆意图过滤） |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.cn/v1` | SiliconFlow API 地址 |
| `SILICONFLOW_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | 意图过滤模型 |
| `ENABLE_MEMORY_INTENT_FILTER` | `true` | 是否启用记忆意图过滤 |

### 特性开关

| 变量 | 默认值 | 说明 |
|---|---|---|
| `USE_LANGGRAPH_AGENT` | `false` | 使用 LangGraph Agent（否则使用 Loop Agent） |

### 其他

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./chat_agent.db` | 数据库连接串 |
| `ENV` | `development` | 运行环境 |
| `CORS_ORIGINS` | `http://localhost:5173` | 允许的跨域来源 |

## 项目结构

```
chat-agent/
├── backend/
│   ├── app/
│   │   ├── core/              # 配置、安全、DB、Milvus、SSE
│   │   ├── langgraph_agent/   # LangGraph Agent 实现
│   │   ├── routers/           # API 路由（chat/auth/files/memory/admin）
│   │   └── services/          # 业务逻辑（RAG/搜索/记忆/嵌入）
│   ├── tests/                 # 后端测试
│   ├── alembic/               # 数据库迁移
│   ├── scripts/               # 工具脚本
│   └── .env                   # 环境配置（不提交）
│
├── frontend/
│   └── src/
│       ├── components/        # UI 组件
│       ├── hooks/             # 自定义 Hooks
│       ├── pages/             # 页面组件
│       ├── services/          # API 调用
│       └── types/             # TypeScript 类型定义
│
├── docs/                      # 设计文档
└── .gitignore
```

## Agent 架构

项目提供两种 Agent 实现，通过 `USE_LANGGRAPH_AGENT` 切换：

### LangGraph Agent（有状态图）

基于 `StateGraph` 的有状态 Agent，节点包括：
- **memory_node** — 并行检索长期记忆
- **rag_node** — 并行执行 RAG 知识检索
- **context_node** — 整合上下文供 LLM 生成
- **llm_node** — 调用 LLM 并支持工具调用
- **evaluation_node** — 评估输出质量，决定是否重试

### Loop Agent（循环自校正）

迭代式 Agent：
1. 调用 LLM 生成回复（支持工具调用）
2. 用非流式 LLM 评估输出质量
3. 不通过则自动修正，最多 50 次迭代

## API 概览

所有接口挂载在 `/api/v1` 前缀下：

| 路径 | 说明 |
|---|---|
| `POST /auth/register` | 用户注册 |
| `POST /auth/login` | 登录获取 JWT |
| `POST /chat/completions` | 流式对话（SSE） |
| `POST /chat/title/{conv_id}` | 生成/更新对话标题 |
| `POST /files/upload` | 上传文件（PDF/DOCX/TXT） |
| `GET /memory` | 查询长期记忆 |
| `GET /admin/stats` | 系统统计（需 admin） |
| `GET /health` | 健康检查 |
| `GET /metrics` | Prometheus 指标 |

## 测试

```bash
cd backend
venv/Scripts/python -m pytest tests/ -v

# 运行单个测试
venv/Scripts/python -m pytest tests/test_chat_router.py -v
```

测试覆盖：对话流式接口、RAG 流水线、向量搜索、Web 搜索、记忆引擎、认证授权、LangGraph Agent、Loop Agent。

## 许可

MIT
