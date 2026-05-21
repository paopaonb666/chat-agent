import { useState } from 'react';
import { ChevronDown, ChevronUp, Sparkles } from 'lucide-react';

interface ReasoningPanelProps {
  content: string;
}

export default function ReasoningPanel({ content }: ReasoningPanelProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="mb-3 rounded-lg border border-gray-200 bg-gray-50 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-2.5 text-sm text-gray-600 hover:bg-gray-100 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Sparkles size={14} className="text-amber-500" />
          <span className="font-medium">已思考</span>
        </div>
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>
      {expanded && (
        <div className="px-4 pb-3 text-sm text-gray-500 whitespace-pre-wrap leading-relaxed">
          {content}
        </div>
      )}
    </div>
  );
}
