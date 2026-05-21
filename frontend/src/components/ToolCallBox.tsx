import { memo, useState, useCallback } from 'react';
import { Loader2, CheckCircle, AlertCircle, Square, ChevronDown, ChevronUp, Bot } from 'lucide-react';
import type { ToolCall } from '../types';
import { cn } from '../utils/cn';

interface ToolCallBoxProps {
  toolCall: ToolCall;
  defaultExpanded?: boolean;
}

const STATUS_CONFIG = {
  running: { icon: Loader2, iconClass: 'text-blue-600 animate-spin', badgeClass: 'bg-blue-100 text-blue-700', label: '运行中' },
  completed: { icon: CheckCircle, iconClass: 'text-emerald-600', badgeClass: 'bg-emerald-100 text-emerald-700', label: '已完成' },
  error: { icon: AlertCircle, iconClass: 'text-red-600', badgeClass: 'bg-red-100 text-red-700', label: '失败' },
  interrupted: { icon: Square, iconClass: 'text-amber-600', badgeClass: 'bg-amber-100 text-amber-700', label: '中断' },
};

function ToolCallBox({ toolCall, defaultExpanded = false }: ToolCallBoxProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [expandedParams, setExpandedParams] = useState<Set<string>>(new Set());

  const toggleParam = useCallback((key: string) => {
    setExpandedParams((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const status = STATUS_CONFIG[toolCall.status];
  const StatusIcon = status.icon;

  return (
    <div className="bg-slate-50 border border-slate-200 rounded-lg overflow-hidden text-[13px]">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2.5 hover:bg-slate-100 transition-colors text-left"
      >
        <StatusIcon size={14} className={cn('shrink-0', status.iconClass)} />
        {toolCall.isSubAgent && <Bot size={14} className="shrink-0 text-violet-500" />}
        <span className="font-mono font-medium text-slate-700 truncate">{toolCall.name}</span>
        {toolCall.isSubAgent && (
          <span className="shrink-0 text-[10px] px-1 py-0.5 rounded bg-violet-100 text-violet-700 font-medium">
            子代理
          </span>
        )}
        <span className={cn('ml-auto text-xs px-1.5 py-0.5 rounded shrink-0', status.badgeClass)}>
          {status.label}
        </span>
        {expanded ? <ChevronUp size={14} className="shrink-0 text-slate-400" /> : <ChevronDown size={14} className="shrink-0 text-slate-400" />}
      </button>

      {/* Body */}
      {expanded && (
        <div className="px-3 pb-3 space-y-2 border-t border-slate-100">
          {/* Parameters */}
          <div className="pt-2">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">参数</div>
            <div className="space-y-1">
              {Object.entries(toolCall.args).map(([key, value]) => {
                const isExpanded = expandedParams.has(key);
                const valueStr = JSON.stringify(value);
                const isLong = valueStr.length > 80;
                return (
                  <div key={key} className="bg-white border border-slate-100 rounded">
                    <button
                      onClick={() => isLong && toggleParam(key)}
                      className={cn('w-full flex items-start gap-2 px-2 py-1.5 text-left', isLong && 'hover:bg-slate-50')}
                    >
                      <span className="font-mono text-xs text-slate-600 shrink-0">{key}</span>
                      <span className="font-mono text-xs text-slate-500 break-all">
                        {isExpanded || !isLong ? valueStr : valueStr.slice(0, 80) + '...'}
                      </span>
                      {isLong && (
                        <span className="ml-auto shrink-0 text-slate-400">
                          {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                        </span>
                      )}
                    </button>
                  </div>
                );
              })}
              {Object.keys(toolCall.args).length === 0 && (
                <div className="text-xs text-slate-400 italic">无参数</div>
              )}
            </div>
          </div>

          {/* Result */}
          {toolCall.result !== undefined && (
            <div className="pt-1">
              <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">结果</div>
              <pre className="bg-white border border-slate-100 rounded px-2 py-1.5 text-xs font-mono text-slate-600 overflow-x-auto max-h-60">
                {toolCall.result}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default memo(ToolCallBox);
