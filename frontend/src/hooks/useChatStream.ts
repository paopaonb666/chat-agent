import { useState, useCallback, useRef } from 'react';
import type { Message, Conversation, UploadedFile } from '../types';

export function useChatStream() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConvId, setCurrentConvId] = useState<string | null>(null);
  const [currentModel, setCurrentModel] = useState<string>('deepseek-chat');
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const createConversation = useCallback(async () => {
    const resp = await fetch('/api/v1/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: '新对话', model: currentModel }),
    });
    const conv: Conversation = await resp.json();
    setConversations((prev) => [conv, ...prev]);
    setCurrentConvId(conv.id);
    return conv.id;
  }, [currentModel]);

  const selectConversation = useCallback((id: string) => {
    setCurrentConvId(id);
    const conv = conversations.find((c) => c.id === id);
    if (conv) setCurrentModel(conv.model);
  }, [conversations]);

  const uploadFiles = useCallback(async (convId: string, files: FileList): Promise<UploadedFile[]> => {
    const uploaded: UploadedFile[] = [];
    for (const file of Array.from(files)) {
      const form = new FormData();
      form.append('conversation_id', convId);
      form.append('file', file);
      const resp = await fetch('/api/v1/files/upload', { method: 'POST', body: form });
      if (resp.ok) {
        uploaded.push(await resp.json());
      }
    }
    return uploaded;
  }, []);

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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation_id: convId, message: finalMessage }),
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
  }, [currentConvId, createConversation, uploadFiles]);

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const currentMessages = conversations.find((c) => c.id === currentConvId)?.messages || [];

  return {
    conversations,
    currentConvId,
    currentMessages,
    currentModel,
    isStreaming,
    createConversation,
    selectConversation,
    setCurrentModel,
    sendMessage,
    stopGeneration,
  };
}
