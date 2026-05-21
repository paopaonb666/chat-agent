import { useState, useCallback, useRef, useEffect } from 'react';
import type { Message, Conversation, MemoryPrompt, MemoryStoreState, StepStatus, ToolStep } from '../types';
import { fetchConversations, createConversation as apiCreate, renameConversation as apiRename } from '../services/api';
import { useFileUpload } from './useFileUpload';

function getToken(): string | null {
  return localStorage.getItem('chat_agent_token');
}

function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

const MEMORY_STEPS: { id: string; label: string }[] = [
  { id: 'prepare', label: '准备分析' },
  { id: 'embed', label: '生成向量' },
  { id: 'dedup', label: '去重检查' },
  { id: 'save', label: '存入数据库' },
  { id: 'index', label: '存入记忆库' },
];

export function useChatStream() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConvId, setCurrentConvId] = useState<string | null>(null);
  const [currentModel, setCurrentModel] = useState<string>('deepseek-chat');
  const [isStreaming, setIsStreaming] = useState(false);
  const [pendingMemory, setPendingMemory] = useState<MemoryPrompt | null>(null);
  const [memoryStore, setMemoryStore] = useState<MemoryStoreState | null>(null);
  const [enableWebSearch, setEnableWebSearch] = useState(true);
  const [toolSteps, setToolSteps] = useState<ToolStep[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const reasoningBufferRef = useRef('');
  const { uploadFiles } = useFileUpload();

  const loadConversations = useCallback(() => {
    fetchConversations()
      .then((data) => {
        setConversations(data);
        if (data.length > 0) setCurrentConvId(data[0].id);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    loadConversations();
    const handler = () => loadConversations();
    window.addEventListener('auth:login', handler);
    return () => window.removeEventListener('auth:login', handler);
  }, [loadConversations]);

  const createConversation = useCallback(async () => {
    const conv = await apiCreate(currentModel);
    setConversations((prev) => [conv, ...prev]);
    setCurrentConvId(conv.id);
    return conv.id;
  }, [currentModel]);

  const selectConversation = useCallback((id: string) => {
    setCurrentConvId(id);
    const conv = conversations.find((c) => c.id === id);
    if (conv) setCurrentModel(conv.model);
  }, [conversations]);

  const sendMessage = useCallback(async (message: string, files?: FileList) => {
    let convId = currentConvId;
    if (!convId) {
      convId = await createConversation();
    }

    let finalMessage = message;
    if (files && files.length > 0) {
      const uploaded = await uploadFiles(convId, files);
      const fileContexts = uploaded
        .filter((f) => f.extracted_text)
        .map((f) => `【文件：${f.filename}】\n${f.extracted_text}`)
        .join('\n\n');
      if (fileContexts) {
        finalMessage = `${fileContexts}\n\n${message}`;
      }
    }

    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId
          ? { ...c, messages: [...c.messages, { role: 'user', content: message }] }
          : c
      )
    );

    setIsStreaming(true);
    setToolSteps([]);
    reasoningBufferRef.current = '';
    const assistantMsg: Message = { role: 'assistant', content: '' };

    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId ? { ...c, messages: [...c.messages, assistantMsg] } : c
      )
    );

    abortRef.current = new AbortController();

    try {
      const resp = await fetch('/api/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({
          conversation_id: convId,
          message: finalMessage,
          enable_web_search: enableWebSearch,
        }),
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

              // Tool step events (memory_check, rag_retrieval, web_search, etc.)
              if (parsed.type === 'step') {
                setToolSteps((prev) => {
                  const idx = prev.findIndex(s => s.name === parsed.name);
                  const step: ToolStep = { name: parsed.name, label: parsed.label, status: parsed.status, detail: parsed.detail || '' };
                  if (idx >= 0) {
                    const updated = [...prev];
                    updated[idx] = step;
                    return updated;
                  }
                  return [...prev, step];
                });
                const icon = parsed.status === 'completed' ? '✅'
                           : parsed.status === 'error' ? '❌'
                           : '⏳';
                reasoningBufferRef.current += `${icon} ${parsed.label}：${parsed.detail || ''}\n`;
                continue;
              }

              // Web search sources
              if (parsed.type === 'sources') {
                setConversations((prev) =>
                  prev.map((c) => {
                    if (c.id !== convId) return c;
                    const msgs = [...c.messages];
                    const last = msgs[msgs.length - 1];
                    if (last && last.role === 'assistant') {
                      msgs[msgs.length - 1] = { ...last, sources: parsed.sources };
                    }
                    return { ...c, messages: msgs };
                  })
                );
                continue;
              }

              // LangGraph tool_call events (AIMessage.tool_calls)
              if (parsed.type === 'tool_call' && parsed.tool_calls) {
                setConversations((prev) =>
                  prev.map((c) => {
                    if (c.id !== convId) return c;
                    const msgs = [...c.messages];
                    const last = msgs[msgs.length - 1];
                    if (last && last.role === 'assistant') {
                      const existing = last.toolCalls || [];
                      const newCalls = parsed.tool_calls.map((tc: any) => ({
                        id: tc.id,
                        name: tc.name,
                        args: tc.args,
                        status: 'running' as const,
                      }));
                      msgs[msgs.length - 1] = { ...last, toolCalls: [...existing, ...newCalls] };
                    }
                    return { ...c, messages: msgs };
                  })
                );
                continue;
              }

              // LangGraph tool_result events (ToolMessage)
              if (parsed.type === 'tool_result' && parsed.tool_call_id) {
                setConversations((prev) =>
                  prev.map((c) => {
                    if (c.id !== convId) return c;
                    const msgs = [...c.messages];
                    const last = msgs[msgs.length - 1];
                    if (last && last.role === 'assistant' && last.toolCalls) {
                      const updatedCalls = last.toolCalls.map((tc) =>
                        tc.id === parsed.tool_call_id
                          ? { ...tc, result: parsed.content, status: 'completed' as const }
                          : tc
                      );
                      msgs[msgs.length - 1] = { ...last, toolCalls: updatedCalls };
                    }
                    return { ...c, messages: msgs };
                  })
                );
                continue;
              }

              if (parsed.type === 'memory_prompt') {
                setPendingMemory({
                  memory_content: parsed.memory_content,
                  conversation_id: parsed.conversation_id,
                });
                continue;
              }

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
      if (reasoningBufferRef.current && convId) {
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== convId) return c;
            const msgs = [...c.messages];
            const last = msgs[msgs.length - 1];
            if (last && last.role === 'assistant') {
              msgs[msgs.length - 1] = { ...last, reasoning: reasoningBufferRef.current };
            }
            return { ...c, messages: msgs };
          })
        );
        reasoningBufferRef.current = '';
      }
      setIsStreaming(false);
      abortRef.current = null;
    }
  }, [currentConvId, createConversation, uploadFiles, enableWebSearch]);

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const storeMemory = useCallback(async (content: string, convId: string) => {
    const steps = MEMORY_STEPS.map((s) => ({ id: s.id, label: s.label, status: 'pending' as StepStatus }));
    setMemoryStore({ steps, message: '正在准备...', done: false });

    try {
      const resp = await fetch('/api/v1/memory/store', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({ content, conversation_id: convId }),
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
          if (!line.startsWith('data: ')) continue;
          const data = JSON.parse(line.slice(6));

          if (data.type === 'step') {
            setMemoryStore((prev) => {
              if (!prev) return null;
              const newSteps = prev.steps.map((s) => {
                if (s.id === data.step) {
                  return { ...s, status: data.status as StepStatus };
                }
                return s;
              });
              return { ...prev, steps: newSteps, message: data.status === 'running' ? data.label : prev.message };
            });
          } else if (data.type === 'done') {
            setPendingMemory(null);
            setMemoryStore((prev) => prev ? { ...prev, message: data.message, done: true } : null);
            setTimeout(() => setMemoryStore(null), 4000);
          }
        }
      }
    } catch {
      setMemoryStore((prev) => {
        if (!prev) return null;
        const newSteps = prev.steps.map((s) => s.status === 'running' ? { ...s, status: 'error' as StepStatus } : s);
        return { ...prev, steps: newSteps, message: '存储失败', done: true };
      });
    }
  }, []);

  const dismissMemory = useCallback(() => {
    setPendingMemory(null);
  }, []);

  const renameConversation = useCallback(async (convId: string, title: string) => {
    try {
      await apiRename(convId, title);
      setConversations((prev) =>
        prev.map((c) => (c.id === convId ? { ...c, title } : c))
      );
    } catch {
      // ignore
    }
  }, []);

  const currentMessages = conversations.find((c) => c.id === currentConvId)?.messages || [];

  return {
    conversations,
    currentConvId,
    currentMessages,
    currentModel,
    isStreaming,
    pendingMemory,
    memoryStore,
    enableWebSearch,
    toolSteps,
    createConversation,
    selectConversation,
    setCurrentModel,
    setEnableWebSearch,
    sendMessage,
    stopGeneration,
    renameConversation,
    storeMemory,
    dismissMemory,
  };
}
