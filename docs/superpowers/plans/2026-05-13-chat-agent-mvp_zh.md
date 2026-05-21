# Chat Agent MVP 实现计划

> **面向智能体工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 来逐个任务实现此计划。步骤使用复选框（`- [ ]`）语法进行跟踪。

**目标：** 构建一个最小化的 Web 聊天代理，包含 React 前端和 FastAPI 后端，支持通过 Moonshot API 进行流式 SSE 对话，使用内存会话管理。

**架构：** FastAPI 后端暴露 REST + SSE 端点用于聊天和会话管理。React 前端使用 EventSource 进行流式传输，并使用简单的侧边栏进行会话切换。MVP 阶段不使用数据库——会话存储在内存中。

**技术栈：** React 18 + TypeScript + Vite + Tailwind CSS（前端），Python 3.11 + FastAPI + uvicorn（后端），Moonshot OpenAI 兼容 API。

---

## 文件结构

```
chat-agent/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              — FastAPI 应用入口，CORS，路由
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   └── chat.py          — /chat/completions SSE 端点
│   │   └── services/
│   │       ├── __init__.py
│   │       └── conversation.py  — InMemoryConversationStore
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_conversation.py
│   │   └── test_chat_router.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── Sidebar.tsx
│   │   │   ├── ChatArea.tsx
│   │   │   ├── MessageList.tsx
│   │   │   ├── MessageItem.tsx
│   │   │   └── InputBox.tsx
│   │   ├── hooks/
│   │   │   └── useChatStream.ts
│   │   └── types/
│   │       └── index.ts
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── tailwind.config.js
└── docs/superpowers/plans/2026-05-13-chat-agent-mvp.md
```

---

## 任务 1：初始化后端项目

**文件：**
- 创建：`backend/requirements.txt`
- 创建：`backend/.env.example`
- 创建：`backend/app/__init__.py`
- 创建：`backend/app/main.py`

脚手架搭建不使用 TDD。只需设置项目骨架。

- [ ] **步骤 1：创建后端目录结构**

```bash
mkdir -p backend/app/routers backend/app/services backend/tests
```

- [ ] **步骤 2：编写 requirements.txt**

```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
httpx>=0.27.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

- [ ] **步骤 3：编写 .env.example**

```
MOONSHOT_API_KEY=your_api_key_here
MOONSHOT_BASE_URL=https://api.moonshot.cn/v1
MOONSHOT_MODEL=moonshot-v1-8k
```

- [ ] **步骤 4：编写 main.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat

app = FastAPI(title="Chat Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **步骤 5：创建空的 __init__.py 文件**

```bash
touch backend/app/__init__.py backend/app/routers/__init__.py backend/app/services/__init__.py backend/tests/__init__.py
```

---

## 任务 2：InMemoryConversationStore（TDD）

**文件：**
- 创建：`backend/app/services/conversation.py`
- 创建：`backend/tests/test_conversation.py`

- [ ] **步骤 1：编写失败测试**

```python
import pytest
from app.services.conversation import InMemoryConversationStore

@pytest.fixture
def store():
    return InMemoryConversationStore()

@pytest.mark.asyncio
async def test_create_conversation(store):
    conv = await store.create(title="Test")
    assert conv["id"] is not None
    assert conv["title"] == "Test"
    assert conv["messages"] == []

@pytest.mark.asyncio
async def test_get_conversation(store):
    conv = await store.create(title="Test")
    fetched = await store.get(conv["id"])
    assert fetched["id"] == conv["id"]

@pytest.mark.asyncio
async def test_get_nonexistent(store):
    fetched = await store.get("nonexistent")
    assert fetched is None

@pytest.mark.asyncio
async def test_add_message(store):
    conv = await store.create(title="Test")
    await store.add_message(conv["id"], "user", "hello")
    fetched = await store.get(conv["id"])
    assert len(fetched["messages"]) == 1
    assert fetched["messages"][0]["role"] == "user"
    assert fetched["messages"][0]["content"] == "hello"

@pytest.mark.asyncio
async def test_list_conversations(store):
    await store.create(title="A")
    await store.create(title="B")
    convs = await store.list_all()
    assert len(convs) == 2
```

- [ ] **步骤 2：运行测试验证失败**

```bash
cd backend && python -m pytest tests/test_conversation.py -v
```

预期结果：FAIL（ImportError 或类未定义）。

- [ ] **步骤 3：编写最小实现**

```python
import uuid
from typing import Optional

class InMemoryConversationStore:
    def __init__(self):
        self._conversations: dict[str, dict] = {}

    async def create(self, title: str) -> dict:
        conv_id = str(uuid.uuid4())
        conv = {"id": conv_id, "title": title, "messages": []}
        self._conversations[conv_id] = conv
        return conv

    async def get(self, conv_id: str) -> Optional[dict]:
        conv = self._conversations.get(conv_id)
        if conv is None:
            return None
        return {"id": conv["id"], "title": conv["title"], "messages": list(conv["messages"])}

    async def add_message(self, conv_id: str, role: str, content: str) -> None:
        conv = self._conversations.get(conv_id)
        if conv is None:
            raise ValueError(f"Conversation {conv_id} not found")
        conv["messages"].append({"role": role, "content": content})

    async def list_all(self) -> list[dict]:
        return [{"id": c["id"], "title": c["title"], "messages": list(c["messages"])} for c in self._conversations.values()]
```

- [ ] **步骤 4：运行测试验证通过**

```bash
cd backend && python -m pytest tests/test_conversation.py -v
```

预期结果：PASS

---

## 任务 3：带 SSE 流式传输的聊天路由（TDD）

**文件：**
- 创建：`backend/app/routers/chat.py`
- 创建：`backend/tests/test_chat_router.py`
- 修改：`backend/app/main.py`（导入 chat 路由——任务 1 中已完成）

- [ ] **步骤 1：编写失败测试**

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_chat_stream():
    # 先创建会话
    resp = client.post("/api/v1/conversations", json={"title": "Test"})
    assert resp.status_code == 200
    conv_id = resp.json()["id"]

    # 此测试仅验证端点结构。
    # 实际流式传输需要真实的 API key，因此我们在集成级别进行模拟。
    resp = client.post("/api/v1/chat/completions", json={
        "conversation_id": conv_id,
        "message": "hi",
    })
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
```

- [ ] **步骤 2：运行测试验证失败**

```bash
cd backend && python -m pytest tests/test_chat_router.py -v
```

预期结果：FAIL（404 或端点未找到）。

- [ ] **步骤 3：编写最小实现**

```python
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import httpx
from app.services.conversation import InMemoryConversationStore

router = APIRouter()
store = InMemoryConversationStore()

API_KEY = os.getenv("MOONSHOT_API_KEY", "")
BASE_URL = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
MODEL = os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k")

@router.post("/conversations")
async def create_conversation(payload: dict):
    title = payload.get("title", "New Conversation")
    conv = await store.create(title)
    return conv

@router.get("/conversations")
async def list_conversations():
    return await store.list_all()

@router.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    conv = await store.get(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv

@router.post("/chat/completions")
async def chat_completions(payload: dict):
    conv_id = payload.get("conversation_id")
    message = payload.get("message", "")
    if not conv_id:
        raise HTTPException(status_code=400, detail="conversation_id required")

    conv = await store.get(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await store.add_message(conv_id, "user", message)

    messages = [{"role": m["role"], "content": m["content"]} for m in conv["messages"]]
    messages.append({"role": "user", "content": message})

    async def event_stream():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": messages, "stream": True},
                timeout=60.0,
            ) as response:
                assistant_content = ""
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            await store.add_message(conv_id, "assistant", assistant_content)
                            yield "data: [DONE]\n\n"
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            assistant_content += delta
                            yield f"data: {json.dumps({'content': delta})}\n\n"
                        except Exception:
                            continue

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

- [ ] **步骤 4：运行测试验证通过**

```bash
cd backend && python -m pytest tests/test_chat_router.py -v
```

预期结果：PASS（仅结构；实际 SSE 流需要运行时的 API key）。

---

## 任务 4：初始化前端项目

脚手架搭建不使用 TDD。

**文件：**
- 创建：`frontend/package.json`
- 创建：`frontend/tsconfig.json`
- 创建：`frontend/vite.config.ts`
- 创建：`frontend/tailwind.config.js`
- 创建：`frontend/index.html`
- 创建：`frontend/src/main.tsx`
- 创建：`frontend/src/App.tsx`
- 创建：`frontend/src/index.css`
- 创建：`frontend/src/types/index.ts`

- [ ] **步骤 1：创建前端目录结构**

```bash
mkdir -p frontend/src/components frontend/src/hooks frontend/src/types
```

- [ ] **步骤 2：编写 package.json**

```json
{
  "name": "chat-agent-frontend",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-markdown": "^9.0.1",
    "remark-gfm": "^4.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.55",
    "@types/react-dom": "^18.2.19",
    "@vitejs/plugin-react": "^4.2.1",
    "autoprefixer": "^10.4.17",
    "postcss": "^8.4.35",
    "tailwindcss": "^3.4.1",
    "typescript": "^5.3.3",
    "vite": "^5.1.0"
  }
}
```

- [ ] **步骤 3：编写 tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

- [ ] **步骤 4：编写 vite.config.ts**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

- [ ] **步骤 5：编写 tailwind.config.js**

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

- [ ] **步骤 6：编写 index.html**

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Chat Agent</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **步骤 7：编写 src/index.css**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

- [ ] **步骤 8：编写 src/main.tsx**

```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

- [ ] **步骤 9：编写 src/types/index.ts**

```typescript
export interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
}
```

---

## 任务 5：前端组件

UI 组件不使用 TDD（手动浏览器测试）。

**文件：**
- 创建：`frontend/src/components/Sidebar.tsx`
- 创建：`frontend/src/components/ChatArea.tsx`
- 创建：`frontend/src/components/MessageList.tsx`
- 创建：`frontend/src/components/MessageItem.tsx`
- 创建：`frontend/src/components/InputBox.tsx`
- 创建：`frontend/src/hooks/useChatStream.ts`
- 修改：`frontend/src/App.tsx`

- [ ] **步骤 1：编写 useChatStream.ts**

```typescript
import { useState, useCallback, useRef } from 'react';
import type { Message, Conversation } from '../types';

export function useChatStream() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConvId, setCurrentConvId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const createConversation = useCallback(async () => {
    const resp = await fetch('/api/v1/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: 'New Conversation' }),
    });
    const conv: Conversation = await resp.json();
    setConversations((prev) => [conv, ...prev]);
    setCurrentConvId(conv.id);
    return conv.id;
  }, []);

  const selectConversation = useCallback((id: string) => {
    setCurrentConvId(id);
  }, []);

  const sendMessage = useCallback(async (message: string) => {
    let convId = currentConvId;
    if (!convId) {
      convId = await createConversation();
    }

    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId
          ? { ...c, messages: [...c.messages, { role: 'user', content: message }] }
          : c
      )
    );

    setIsStreaming(true);
    const assistantMsg: Message = { role: 'assistant', content: '' };

    setConversations((prev) =>
      prev.map((c) => (c.id === convId ? { ...c, messages: [...c.messages, assistantMsg] } : c))
    );

    abortRef.current = new AbortController();

    try {
      const resp = await fetch('/api/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation_id: convId, message }),
        signal: abortRef.current.signal,
      });

      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;
            try {
              const parsed = JSON.parse(data);
              const delta = parsed.content || '';
              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== convId) return c;
                  const msgs = [...c.messages];
                  const last = msgs[msgs.length - 1];
                  if (last && last.role === 'assistant') {
                    msgs[msgs.length - 1] = { ...last, content: last.content + delta };
                  }
                  return { ...c, messages: msgs };
                })
              );
            } catch {
              // 忽略格式错误的行
            }
          }
        }
      }
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
    }
  }, [currentConvId, createConversation]);

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const currentMessages = conversations.find((c) => c.id === currentConvId)?.messages || [];

  return {
    conversations,
    currentConvId,
    currentMessages,
    isStreaming,
    createConversation,
    selectConversation,
    sendMessage,
    stopGeneration,
  };
}
```

- [ ] **步骤 2：编写 Sidebar.tsx**

```tsx
import type { Conversation } from '../types';

interface SidebarProps {
  conversations: Conversation[];
  currentConvId: string | null;
  onSelect: (id: string) => void;
  onCreate: () => void;
}

export default function Sidebar({ conversations, currentConvId, onSelect, onCreate }: SidebarProps) {
  return (
    <aside className="w-64 bg-gray-100 border-r flex flex-col h-full">
      <div className="p-4 border-b">
        <button
          onClick={onCreate}
          className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 transition"
        >
          + New Chat
        </button>
      </div>
      <ul className="flex-1 overflow-y-auto p-2 space-y-1">
        {conversations.map((conv) => (
          <li
            key={conv.id}
            onClick={() => onSelect(conv.id)}
            className={`cursor-pointer px-3 py-2 rounded truncate ${
              conv.id === currentConvId ? 'bg-blue-100 text-blue-700' : 'hover:bg-gray-200'
            }`}
          >
            {conv.title}
          </li>
        ))}
      </ul>
    </aside>
  );
}
```

- [ ] **步骤 3：编写 MessageItem.tsx**

```tsx
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Message } from '../types';

interface MessageItemProps {
  message: Message;
}

export default function MessageItem({ message }: MessageItemProps) {
  const isUser = message.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[80%] px-4 py-2 rounded-lg ${
          isUser ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-900'
        }`}
      >
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
      </div>
    </div>
  );
}
```

- [ ] **步骤 4：编写 InputBox.tsx**

```tsx
import { useState } from 'react';

interface InputBoxProps {
  onSend: (message: string) => void;
  onStop: () => void;
  isStreaming: boolean;
}

export default function InputBox({ onSend, onStop, isStreaming }: InputBoxProps) {
  const [text, setText] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim() || isStreaming) return;
    onSend(text.trim());
    setText('');
  };

  return (
    <form onSubmit={handleSubmit} className="p-4 border-t flex gap-2">
      <input
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type a message..."
        className="flex-1 border rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
      {isStreaming ? (
        <button
          type="button"
          onClick={onStop}
          className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition"
        >
          Stop
        </button>
      ) : (
        <button
          type="submit"
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
        >
          Send
        </button>
      )}
    </form>
  );
}
```

- [ ] **步骤 5：编写 ChatArea.tsx**

```tsx
import MessageItem from './MessageItem';
import InputBox from './InputBox';
import type { Message } from '../types';

interface ChatAreaProps {
  messages: Message[];
  isStreaming: boolean;
  onSend: (message: string) => void;
  onStop: () => void;
}

export default function ChatArea({ messages, isStreaming, onSend, onStop }: ChatAreaProps) {
  return (
    <div className="flex-1 flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 && (
          <div className="h-full flex items-center justify-center text-gray-400">
            Start a new conversation
          </div>
        )}
        {messages.map((msg, idx) => (
          <MessageItem key={idx} message={msg} />
        ))}
      </div>
      <InputBox onSend={onSend} onStop={onStop} isStreaming={isStreaming} />
    </div>
  );
}
```

- [ ] **步骤 6：编写 App.tsx**

```tsx
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import { useChatStream } from './hooks/useChatStream';

export default function App() {
  const {
    conversations,
    currentConvId,
    currentMessages,
    isStreaming,
    createConversation,
    selectConversation,
    sendMessage,
    stopGeneration,
  } = useChatStream();

  return (
    <div className="h-screen flex">
      <Sidebar
        conversations={conversations}
        currentConvId={currentConvId}
        onSelect={selectConversation}
        onCreate={createConversation}
      />
      <ChatArea
        messages={currentMessages}
        isStreaming={isStreaming}
        onSend={sendMessage}
        onStop={stopGeneration}
      />
    </div>
  );
}
```

---

## 任务 6：端到端验证

- [ ] **步骤 1：安装后端依赖**

```bash
cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

- [ ] **步骤 2：运行后端测试**

```bash
cd backend && python -m pytest tests/ -v
```

预期结果：所有测试通过。

- [ ] **步骤 3：启动后端服务器**

```bash
cd backend && uvicorn app.main:app --reload --port 8000
```

- [ ] **步骤 4：安装前端依赖**

```bash
cd frontend && npm install
```

- [ ] **步骤 5：启动前端开发服务器**

```bash
cd frontend && npm run dev
```

- [ ] **步骤 6：手动浏览器验证**

1. 打开 `http://localhost:5173`
2. 点击 "New Chat"
3. 输入消息并发送
4. 验证流式响应逐字显示
5. 验证侧边栏显示会话标题
6. 验证会话切换正常工作

---

## 规格覆盖检查

| 需求 | 任务 |
|-------------|------|
| FastAPI 后端骨架 | 任务 1 |
| 内存会话管理 | 任务 2 |
| SSE 流式聊天端点 | 任务 3 |
| React 前端脚手架 | 任务 4 |
| 侧边栏 + 聊天布局 | 任务 5 |
| Markdown 渲染 | 任务 5（MessageItem） |
| 停止生成 | 任务 5（InputBox + useChatStream） |
| 端到端验证 | 任务 6 |

**待改进项：**
- MVP 中没有代码块复制按钮（可后续添加）
- MVP 中没有文件上传（第二阶段）
- MVP 中没有用户认证（第二阶段）
- MVP 中没有深色模式（第三阶段）
