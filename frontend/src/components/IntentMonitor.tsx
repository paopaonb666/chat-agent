import { useEffect, useState } from 'react';
import type { IntentStep as IntentStepType, StepStatus } from '../types';

interface IntentMonitorProps {
  steps: IntentStepType[];
}

function StepIcon({ status }: { status: StepStatus }) {
  switch (status) {
    case 'running':
      return <div className="w-4 h-4 rounded-full border-2 border-blue-500 border-t-transparent animate-spin shrink-0" />;
    case 'completed':
      return <div className="w-4 h-4 rounded-full bg-green-500 flex items-center justify-center text-white text-[10px] font-bold shrink-0">✓</div>;
    case 'error':
      return <div className="w-4 h-4 rounded-full bg-red-500 flex items-center justify-center text-white text-[10px] font-bold shrink-0">✗</div>;
    default:
      return <div className="w-4 h-4 rounded-full border-2 border-slate-300 shrink-0" />;
  }
}

export default function IntentMonitor({ steps }: IntentMonitorProps) {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    if (steps.length > 0) {
      setVisible(true);
      if (steps.every(s => s.status === 'completed' || s.status === 'error')) {
        const timer = setTimeout(() => setVisible(false), 3000);
        return () => clearTimeout(timer);
      }
    }
  }, [steps]);

  if (steps.length === 0 || !visible) return null;

  const allDone = steps.every(s => s.status === 'completed');
  const hasError = steps.some(s => s.status === 'error');

  const statusText = hasError
    ? '意图分析异常'
    : allDone
      ? '意图分析完成'
      : '意图分析中...';

  const statusColor = hasError
    ? 'text-red-600'
    : allDone
      ? 'text-green-600'
      : 'text-blue-600';

  return (
    <div className="max-w-3xl mx-auto px-4 pt-3">
      <div className="flex items-center gap-3 px-3 py-2 bg-slate-50 rounded-lg border border-slate-200">
        <span className={`text-xs font-medium whitespace-nowrap ${statusColor}`}>
          {statusText}
        </span>
        <div className="h-4 w-px bg-slate-200" />
        <div className="flex items-center gap-3 overflow-x-auto">
          {steps.map((step) => (
            <div key={step.step} className="flex items-center gap-1.5 shrink-0">
              <StepIcon status={step.status} />
              <span className={`text-xs ${
                step.status === 'completed' ? 'text-green-700'
                : step.status === 'running' ? 'text-blue-700'
                : step.status === 'error' ? 'text-red-700'
                : 'text-slate-400'
              }`}>
                {step.label}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
