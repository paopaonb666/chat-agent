import { useEffect, useRef } from 'react';
import { Bot, Loader2 } from 'lucide-react';
import MessageItem from './MessageItem';
import InputBox from './InputBox';
import MemoryPromptPanel from './MemoryPrompt';
import type { Message, MemoryPrompt, MemoryStoreState, ToolStep, ToolApprovalAction } from '../types';

interface ChatAreaProps {
  messages: Message[];
  isStreaming: boolean;
  currentModel: string;
  currentTitle: string;
  pendingMemory: MemoryPrompt | null;
  memoryStore: MemoryStoreState | null;
  enableWebSearch: boolean;
  toolSteps: ToolStep[];
  onModelChange: (model: string) => void;
  onWebSearchToggle: (enabled: boolean) => void;
  onSend: (message: string, files?: FileList) => void;
  onStop: () => void;
  onStoreMemory: (content: string, convId: string) => void;
  onDismissMemory: () => void;
  onToolAction?: (action: ToolApprovalAction) => void;
}

const MODEL_OPTIONS = [
  { value: 'deepseek-chat', label: 'DeepSeek' },
  { value: 'glm-4', label: '智谱 GLM-4' },
];

export default function ChatArea({
  messages,
  isStreaming,
  currentModel,
  currentTitle,
  pendingMemory,
  memoryStore,
  enableWebSearch,
  toolSteps,
  onModelChange,
  onWebSearchToggle,
  onSend,
  onStop,
  onStoreMemory,
  onDismissMemory,
  onToolAction,
}: ChatAreaProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex-1 flex flex-col h-full min-w-0">
      {/* Header */}
      <div className="h-14 border-b border-gray-200 bg-white flex items-center justify-between px-6 shrink-0">
        <h1 className="text-gray-800 font-semibold text-lg truncate max-w-md">{currentTitle}</h1>
        <select
          value={currentModel}
          onChange={(e) => onModelChange(e.target.value)}
          className="text-sm bg-gray-100 border border-gray-200 rounded-lg px-3 py-1.5 text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
        >
          {MODEL_OPTIONS.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-gray-400">
            <Bot size={48} className="mb-4 text-gray-300" />
            <p className="text-lg font-medium text-gray-500">开始你的第一次对话</p>
            <p className="text-sm mt-2">在下方输入框发送消息，AI 将实时回复</p>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto px-4 py-6 space-y-2">
            {/* Running tool steps as inline cards */}
            {toolSteps.filter(s => s.status === 'running').map(s => (
              <div key={s.name} className="flex items-center gap-2.5 text-sm text-blue-700 bg-blue-50 border border-blue-100 rounded-lg px-4 py-2.5">
                <Loader2 size={14} className="animate-spin shrink-0" />
                <span>{s.detail}</span>
              </div>
            ))}
            {messages.map((msg, idx) => (
              <MessageItem key={idx} message={msg} onToolAction={onToolAction} />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <InputBox
        onSend={onSend}
        onStop={onStop}
        isStreaming={isStreaming}
        enableWebSearch={enableWebSearch}
        onWebSearchToggle={onWebSearchToggle}
      />
      <MemoryPromptPanel
        prompt={pendingMemory}
        storeState={memoryStore}
        onConfirm={onStoreMemory}
        onDismiss={onDismissMemory}
      />
    </div>
  );
}
