import { memo, useState, useCallback } from 'react';
import { AlertTriangle, CheckCircle, XCircle, Edit3, ChevronDown, ChevronUp, Loader2 } from 'lucide-react';
import type { ToolCall, ToolApprovalAction } from '../types';
import { cn } from '../utils/cn';

interface ToolApprovalInterruptProps {
  toolCall: ToolCall;
  onAction: (action: ToolApprovalAction) => void;
  loading?: boolean;
}

function ToolApprovalInterrupt({ toolCall, onAction, loading = false }: ToolApprovalInterruptProps) {
  const [mode, setMode] = useState<'view' | 'edit' | 'reject'>('view');
  const [editedArgs, setEditedArgs] = useState<Record<string, unknown>>(toolCall.args);
  const [reason, setReason] = useState('');
  const [expandedParams, setExpandedParams] = useState<Set<string>>(new Set());

  const toggleParam = useCallback((key: string) => {
    setExpandedParams((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const handleApprove = useCallback(() => {
    onAction({ type: 'approve', toolCallId: toolCall.id });
  }, [onAction, toolCall.id]);

  const handleReject = useCallback(() => {
    if (mode === 'reject') {
      onAction({ type: 'reject', toolCallId: toolCall.id, reason: reason.trim() || undefined });
      setMode('view');
      setReason('');
    } else {
      setMode('reject');
    }
  }, [mode, onAction, toolCall.id, reason]);

  const handleEdit = useCallback(() => {
    if (mode === 'edit') {
      onAction({ type: 'edit', toolCallId: toolCall.id, editedArgs });
      setMode('view');
    } else {
      setEditedArgs(toolCall.args);
      setMode('edit');
    }
  }, [mode, onAction, toolCall.id, editedArgs, toolCall.args]);

  const handleCancel = useCallback(() => {
    setMode('view');
    setReason('');
    setEditedArgs(toolCall.args);
  }, [toolCall.args]);

  const isEditing = mode === 'edit';
  const isRejecting = mode === 'reject';

  return (
    <div className="bg-amber-50/60 border border-amber-200 rounded-lg overflow-hidden text-[13px]">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2.5 bg-amber-50">
        <AlertTriangle size={14} className="shrink-0 text-amber-600" />
        <span className="text-xs font-semibold text-amber-700 uppercase tracking-wider">需要审批</span>
        <span className="ml-auto text-xs text-amber-600">请点击下方按钮操作</span>
      </div>

      {/* Tool name */}
      <div className="px-3 py-2 border-t border-amber-100">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">工具</span>
          <span className="font-mono font-medium text-slate-700">{toolCall.name}</span>
        </div>
      </div>

      {/* Parameters */}
      <div className="px-3 pb-2 border-t border-amber-100">
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5 pt-2">参数</div>
        <div className="space-y-1">
          {Object.entries(isEditing ? editedArgs : toolCall.args).map(([key, value]) => {
            const isExpanded = expandedParams.has(key);
            const valueStr = typeof value === 'string' ? value : JSON.stringify(value);
            const isLong = valueStr.length > 60;

            return (
              <div key={key} className="bg-white border border-amber-100/80 rounded">
                <div className="flex items-start gap-2 px-2 py-1.5">
                  <span className="font-mono text-xs text-slate-600 shrink-0 pt-0.5">{key}</span>
                  {isEditing ? (
                    <input
                      type="text"
                      value={valueStr}
                      onChange={(e) => {
                        const raw = e.target.value;
                        let parsed: unknown = raw;
                        try { parsed = JSON.parse(raw); } catch { /* keep string */ }
                        setEditedArgs((prev) => ({ ...prev, [key]: parsed }));
                      }}
                      className="flex-1 min-w-0 text-xs font-mono text-slate-700 bg-slate-50 border border-slate-200 rounded px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-blue-500"
                    />
                  ) : (
                    <button
                      onClick={() => isLong && toggleParam(key)}
                      className={cn('flex-1 text-left', isLong && 'cursor-pointer')}
                    >
                      <span className="font-mono text-xs text-slate-500 break-all">
                        {isExpanded || !isLong ? valueStr : valueStr.slice(0, 60) + '...'}
                      </span>
                      {isLong && (
                        <span className="ml-1 text-slate-400 inline-block align-middle">
                          {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                        </span>
                      )}
                    </button>
                  )}
                </div>
              </div>
            );
          })}
          {Object.keys(isEditing ? editedArgs : toolCall.args).length === 0 && (
            <div className="text-xs text-slate-400 italic">无参数</div>
          )}
        </div>
      </div>

      {/* Rejection reason input */}
      {isRejecting && (
        <div className="px-3 py-2 border-t border-amber-100">
          <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">拒绝原因（可选）</label>
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="请填写拒绝原因..."
            rows={2}
            className="w-full text-xs text-slate-700 bg-white border border-slate-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500 resize-none"
          />
        </div>
      )}

      {/* Action buttons */}
      <div className="px-3 py-2.5 border-t border-amber-100 flex items-center gap-2">
        {isEditing || isRejecting ? (
          <>
            <button
              onClick={isEditing ? handleEdit : handleReject}
              disabled={loading}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading && <Loader2 size={12} className="animate-spin" />}
              <span>{isEditing ? '保存修改' : '确认拒绝'}</span>
            </button>
            <button
              onClick={handleCancel}
              disabled={loading}
              className="px-3 py-1.5 rounded-md text-xs font-medium bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              取消
            </button>
          </>
        ) : (
          <>
            <button
              onClick={handleApprove}
              disabled={loading}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading && <Loader2 size={12} className="animate-spin" />}
              <CheckCircle size={12} />
              <span>批准</span>
            </button>
            <button
              onClick={handleReject}
              disabled={loading}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <XCircle size={12} />
              <span>拒绝</span>
            </button>
            <button
              onClick={handleEdit}
              disabled={loading}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Edit3 size={12} />
              <span>编辑</span>
            </button>
          </>
        )}
      </div>
    </div>
  );
}

export default memo(ToolApprovalInterrupt);
