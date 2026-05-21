# Chat Agent MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal web Chat Agent with React frontend and FastAPI backend, supporting streaming SSE conversation via Moonshot API, in-memory session management.

**Architecture:** FastAPI backend exposes REST + SSE endpoints for chat and conversation management. React frontend uses EventSource for streaming and a simple sidebar for session switching. No database in MVP — sessions stored in-memory.

**Tech Stack:** React 18 + TypeScript + Vite + Tailwind CSS (frontend), Python 3.11 + FastAPI + uvicorn (backend), Moonshot OpenAI-compatible API.

---

## File Structure

```
chat-agent/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              — FastAPI app entry, CORS, routers
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   └── chat.py          — /chat/completions SSE endpoint
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

## Task 1: Initialize Backend Project

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/.env.example`
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`

No TDD for scaffolding. Just set up project skeleton.

- [ ] **Step 1: Create backend directory structure**

```bash
mkdir -p backend/app/routers backend/app/services backend/tests
```

- [ ] **Step 2: Write requirements.txt**

```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
httpx>=0.27.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 3: Write .env.example**

```
MOONSHOT_API_KEY=your_api_key_here
MOONSHOT_BASE_URL=https://api.moonshot.cn/v1
MOONSHOT_MODEL=moonshot-v1-8k
```

- [ ] **Step 4: Write main.py**

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

- [ ] **Step 5: Create empty __init__.py files**

```bash
touch backend/app/__init__.py backend/app/routers/__init__.py backend/app/services/__init__.py backend/tests/__init__.py
```

---

## Task 2: InMemoryConversationStore (TDD)

**Files:**
- Create: `backend/app/services/conversation.py`
- Create: `backend/tests/test_conversation.py`

- [ ] **Step 1: Write failing test**

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

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && python -m pytest tests/test_conversation.py -v
```

Expected: FAIL with ImportError or class not defined.

- [ ] **Step 3: Write minimal implementation**

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

- [ ] **Step 4: Run test to verify it passes**

```bash
cd backend && python -m pytest tests/test_conversation.py -v
```

Expected: PASS

---

## Task 3: Chat Router with SSE Streaming (TDD)

**Files:**
- Create: `backend/app/routers/chat.py`
- Create: `backend/tests/test_chat_router.py`
- Modify: `backend/app/main.py` (import chat router — already done in Task 1)

- [ ] **Step 1: Write failing test**

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_chat_stream():
    # Create conversation first
    resp = client.post("/api/v1/conversations", json={"title": "Test"})
    assert resp.status_code == 200
    conv_id = resp.json()["id"]

    # This test just verifies the endpoint structure.
    # Actual streaming requires a real API key, so we mock at integration level.
    resp = client.post("/api/v1/chat/completions", json={
        "conversation_id": conv_id,
        "message": "hi",
    })
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && python -m pytest tests/test_chat_router.py -v
```

Expected: FAIL (404 or endpoint not found).

- [ ] **Step 3: Write minimal implementation**

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

- [ ] **Step 4: Run test to verify it passes**

```bash
cd backend && python -m pytest tests/test_chat_router.py -v
```

Expected: PASS (structure only; actual SSE stream requires API key at runtime).

---

## Task 4: Initialize Frontend Project

No TDD for scaffolding.

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tailwind.config.js`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/index.css`
- Create: `frontend/src/types/index.ts`

- [ ] **Step 1: Create frontend directory structure**

```bash
mkdir -p frontend/src/components frontend/src/hooks frontend/src/types
```

- [ ] **Step 2: Write package.json**

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

- [ ] **Step 3: Write tsconfig.json**

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

- [ ] **Step 4: Write vite.config.ts**

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

- [ ] **Step 5: Write tailwind.config.js**

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

- [ ] **Step 6: Write index.html**

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

- [ ] **Step 7: Write src/index.css**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

- [ ] **Step 8: Write src/main.tsx**

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

- [ ] **Step 9: Write src/types/index.ts**

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

## Task 5: Frontend Components

No TDD for UI components (manual browser testing).

**Files:**
- Create: `frontend/src/components/Sidebar.tsx`
- Create: `frontend/src/components/ChatArea.tsx`
- Create: `frontend/src/components/MessageList.tsx`
- Create: `frontend/src/components/MessageItem.tsx`
- Create: `frontend/src/components/InputBox.tsx`
- Create: `frontend/src/hooks/useChatStream.ts`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Write useChatStream.ts**

```typescript
import { useState, useCallback } from 'react';
import type { Message, Conversation } from '../types';

export function useChatStream() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConvId, setCurrentConvId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);

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

    const eventSource = new EventSource(
      `/api/v1/chat/completions?conversation_id=${convId}&message=${encodeURIComponent(message)}`
    );
    // NOTE: EventSource does not support POST. We will switch to fetch + ReadableStream.
  }, [currentConvId, createConversation]);

  return { conversations, currentConvId, isStreaming, createConversation, selectConversation, sendMessage };
}
```

Wait — EventSource only supports GET, but our endpoint is POST. We need `fetch` + `ReadableStream`. Let's write it properly.

**Revised useChatStream.ts:**

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
              // ignore malformed lines
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

- [ ] **Step 2: Write Sidebar.tsx**

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

- [ ] **Step 3: Write MessageItem.tsx**

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

- [ ] **Step 4: Write InputBox.tsx**

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

- [ ] **Step 5: Write ChatArea.tsx**

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

- [ ] **Step 6: Write App.tsx**

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

## Task 6: Verify End-to-End

- [ ] **Step 1: Install backend dependencies**

```bash
cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

- [ ] **Step 2: Run backend tests**

```bash
cd backend && python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 3: Start backend server**

```bash
cd backend && uvicorn app.main:app --reload --port 8000
```

- [ ] **Step 4: Install frontend dependencies**

```bash
cd frontend && npm install
```

- [ ] **Step 5: Start frontend dev server**

```bash
cd frontend && npm run dev
```

- [ ] **Step 6: Manual browser verification**

1. Open `http://localhost:5173`
2. Click "New Chat"
3. Type a message and send
4. Verify streaming response appears word-by-word
5. Verify sidebar shows conversation title
6. Verify switching conversations works

---

## Spec Coverage Check

| Requirement | Task |
|-------------|------|
| FastAPI backend skeleton | Task 1 |
| In-memory session management | Task 2 |
| SSE streaming chat endpoint | Task 3 |
| React frontend scaffolding | Task 4 |
| Sidebar + chat layout | Task 5 |
| Markdown rendering | Task 5 (MessageItem) |
| Stop generation | Task 5 (InputBox + useChatStream) |
| End-to-end verification | Task 6 |

**Gaps:**
- No code-block copy button in MVP (can add later)
- No file upload in MVP (Phase 2)
- No user auth in MVP (Phase 2)
- No dark mode in MVP (Phase 3)
