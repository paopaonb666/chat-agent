import { useState } from 'react';
import type { MemoryPrompt, MemoryStoreState, StepStatus } from '../types';

interface MemoryPromptPanelProps {
  prompt: MemoryPrompt | null;
  storeState: MemoryStoreState | null;
  onConfirm: (content: string, convId: string) => void;
  onDismiss: () => void;
}

function StepIcon({ status }: { status: StepStatus }) {
  if (status === 'completed') {
    return (
      <span className="w-5 h-5 rounded-full bg-emerald-500 flex items-center justify-center shrink-0">
        <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
        </svg>
      </span>
    );
  }
  if (status === 'running') {
    return (
      <span className="w-5 h-5 rounded-full border-2 border-amber-400 border-t-transparent animate-spin shrink-0" />
    );
  }
  if (status === 'error') {
    return (
      <span className="w-5 h-5 rounded-full bg-red-500 flex items-center justify-center shrink-0">
        <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </span>
    );
  }
  return <span className="w-5 h-5 rounded-full border-2 border-slate-300 shrink-0" />;
}

export default function MemoryPromptPanel({ prompt, storeState, onConfirm, onDismiss }: MemoryPromptPanelProps) {
  const [editContent, setEditContent] = useState('');

  // Show step chain during storage
  if (storeState) {
    return (
      <div className="border-t border-slate-100 bg-white px-4 py-4 shrink-0">
        <div className="max-w-3xl mx-auto">
          <div className="bg-white border border-slate-200 rounded-xl px-5 py-4">
            <p className="text-xs font-medium text-slate-500 mb-3 uppercase tracking-wide">存储到长期记忆</p>
            <div className="space-y-0">
              {storeState.steps.map((step, idx) => (
                <div key={step.id} className="flex items-start gap-3">
                  <div className="flex flex-col items-center">
                    <StepIcon status={step.status} />
                    {idx < storeState.steps.length - 1 && (
                      <div className={`w-px h-6 ${step.status === 'completed' ? 'bg-emerald-300' : 'bg-slate-200'}`} />
                    )}
                  </div>
                  <span className={`text-sm py-0.5 ${
                    step.status === 'completed' ? 'text-emerald-700' :
                    step.status === 'running' ? 'text-amber-700 font-medium' :
                    step.status === 'error' ? 'text-red-600' :
                    'text-slate-400'
                  }`}>
                    {step.label}
                  </span>
                </div>
              ))}
            </div>
            {storeState.done && (
              <p className="text-xs text-emerald-600 mt-3 text-center">{storeState.message}</p>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (!prompt) return null;

  // Show editable confirmation prompt
  const displayContent = editContent || prompt.memory_content;

  return (
    <div className="border-t border-slate-100 bg-white px-4 py-4 shrink-0">
      <div className="max-w-3xl mx-auto">
        <div className="bg-amber-50 border border-amber-200 rounded-xl px-4 py-3">
          <p className="text-sm font-medium text-amber-800 mb-2">检测到可记忆的信息</p>
          <textarea
            value={displayContent}
            onChange={(e) => setEditContent(e.target.value)}
            rows={2}
            className="w-full px-3 py-2 rounded-lg border border-amber-300 bg-white text-sm text-slate-800 resize-none focus:outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500"
            placeholder="编辑记忆内容..."
          />
          <div className="flex justify-end gap-2 mt-2">
            <button
              onClick={onDismiss}
              className="text-xs px-3 py-1.5 rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50 transition-colors"
            >
              忽略
            </button>
            <button
              onClick={() => onConfirm(displayContent, prompt.conversation_id)}
              className="text-xs px-3 py-1.5 rounded-lg bg-amber-600 text-white hover:bg-amber-700 transition-colors"
            >
              存入记忆
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
