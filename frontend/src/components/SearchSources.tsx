import { Globe } from 'lucide-react';
import type { Source } from '../types';

interface SearchSourcesProps {
  sources: Source[];
}

export default function SearchSources({ sources }: SearchSourcesProps) {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-3 space-y-2 border-t border-slate-100 pt-2">
      <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-1">
        <Globe size={12} />
        <span>搜索结果 ({sources.length})</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {sources.map((s) => (
          <a
            key={s.position}
            href={s.url}
            target="_blank"
            rel="noopener noreferrer"
            className="group flex-1 min-w-[200px] max-w-full p-2.5 rounded-lg border border-slate-200 bg-slate-50/50 hover:bg-blue-50 hover:border-blue-200 transition-colors no-underline"
          >
            <div className="text-sm font-medium text-slate-800 group-hover:text-blue-700 line-clamp-1">
              {s.title}
            </div>
            <div className="text-xs text-slate-400 mt-0.5 truncate">{s.url}</div>
            <div className="text-xs text-slate-500 mt-1 line-clamp-2">{s.snippet}</div>
          </a>
        ))}
      </div>
    </div>
  );
}
