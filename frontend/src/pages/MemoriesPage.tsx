import { useState, useEffect, useCallback, useRef } from 'react';
import { Plus, Pencil, Trash2, Check, X, Search, ChevronLeft, ChevronRight } from 'lucide-react';
import type { UserMemoryItem } from '../types';

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem('chat_agent_token');
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  return headers;
}

export default function MemoriesPage() {
  const [memories, setMemories] = useState<UserMemoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [createContent, setCreateContent] = useState('');

  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const searchTimer = useRef<ReturnType<typeof setTimeout>>();

  const fetchMemories = useCallback(async (p: number) => {
    setLoading(true);
    setError('');
    try {
      const resp = await fetch(`/api/v1/memories?page=${p}&page_size=5`, { headers: authHeaders() });
      if (!resp.ok) throw new Error('获取失败');
      const data = await resp.json();
      setMemories(data.items);
      setPage(data.page);
      setTotalPages(data.total_pages);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchMemories(1); }, [fetchMemories]);

  // Debounced semantic search
  useEffect(() => {
    if (!searchQuery.trim()) {
      setIsSearching(false);
      fetchMemories(1);
      return;
    }
    setIsSearching(true);
    clearTimeout(searchTimer.current);
    searchTimer.current = setTimeout(async () => {
      setError('');
      try {
        const resp = await fetch(`/api/v1/memories/search?q=${encodeURIComponent(searchQuery)}&page=1&page_size=5`, { headers: authHeaders() });
        if (!resp.ok) throw new Error('搜索失败');
        const data = await resp.json();
        setMemories(data.items);
        setPage(data.page);
        setTotalPages(data.total_pages);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setIsSearching(false);
      }
    }, 300);
    return () => clearTimeout(searchTimer.current);
  }, [searchQuery, fetchMemories]);

  const handleCreate = async () => {
    if (!createContent.trim()) return;
    try {
      const resp = await fetch('/api/v1/memories', {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({ content: createContent.trim() }),
      });
      if (!resp.ok) throw new Error('创建失败');
      setCreateContent('');
      setShowCreate(false);
      await fetchMemories(1);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const handleUpdate = async (id: string) => {
    if (!editContent.trim()) return;
    try {
      const resp = await fetch(`/api/v1/memories/${id}`, {
        method: 'PUT',
        headers: authHeaders(),
        body: JSON.stringify({ content: editContent.trim() }),
      });
      if (!resp.ok) throw new Error('更新失败');
      setEditingId(null);
      await fetchMemories(page);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      const resp = await fetch(`/api/v1/memories/${id}`, { method: 'DELETE', headers: authHeaders() });
      if (!resp.ok) throw new Error('删除失败');
      // If it was the last item on the page, go back one page
      if (memories.length === 1 && page > 1) {
        await fetchMemories(page - 1);
      } else {
        await fetchMemories(page);
      }
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const goPage = (p: number) => {
    if (p >= 1 && p <= totalPages) {
      fetchMemories(searchQuery.trim() ? 1 : p);
    }
  };

  return (
    <div className="flex-1 flex flex-col bg-slate-50 h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-white shrink-0">
        <h1 className="text-lg font-semibold text-slate-800">记忆管理</h1>
      </div>

      {/* Search bar */}
      <div className="px-6 py-3 bg-white border-b border-slate-100 shrink-0">
        <div className="relative max-w-2xl mx-auto">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="语义搜索记忆..."
            className="w-full pl-10 pr-4 py-2 rounded-xl border border-slate-300 focus:border-blue-500 outline-none text-sm"
          />
          {isSearching && (
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-slate-400">搜索中...</span>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {error && <p className="text-red-500 text-sm mb-4">{error}</p>}

        {loading ? (
          <div className="text-center text-slate-400 py-12">加载中...</div>
        ) : memories.length === 0 && !showCreate ? (
          <div className="text-center text-slate-400 py-12">
            <p className="mb-4">{searchQuery ? '未找到匹配的记忆' : '暂无记忆'}</p>
            {!searchQuery && (
              <button
                onClick={() => setShowCreate(true)}
                className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-xl transition-colors"
              >
                <Plus size={18} /> 添加记忆
              </button>
            )}
          </div>
        ) : (
          <div className="space-y-3 max-w-2xl mx-auto">
            {showCreate && (
              <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
                <textarea
                  value={createContent}
                  onChange={(e) => setCreateContent(e.target.value)}
                  placeholder="输入你想让系统记住的信息..."
                  rows={3}
                  onKeyDown={(e) => { if (e.key === 'Enter' && e.ctrlKey) handleCreate(); }}
                  className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-blue-500 outline-none resize-none text-sm"
                />
                <div className="flex justify-end gap-2 mt-2">
                  <button
                    onClick={() => { setShowCreate(false); setCreateContent(''); }}
                    className="px-4 py-1.5 text-sm text-slate-600 hover:text-slate-800 transition-colors"
                  >
                    取消
                  </button>
                  <button
                    onClick={handleCreate}
                    className="flex items-center gap-1 px-4 py-1.5 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                  >
                    <Check size={14} /> 保存
                  </button>
                </div>
              </div>
            )}

            {memories.map((mem) => (
              <div key={mem.id} className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
                {editingId === mem.id ? (
                  <div>
                    <textarea
                      value={editContent}
                      onChange={(e) => setEditContent(e.target.value)}
                      rows={3}
                      onKeyDown={(e) => { if (e.key === 'Enter' && e.ctrlKey) handleUpdate(mem.id); }}
                      className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-blue-500 outline-none resize-none text-sm"
                    />
                    <div className="flex justify-end gap-2 mt-2">
                      <button onClick={() => setEditingId(null)} className="px-4 py-1.5 text-sm text-slate-600 hover:text-slate-800"><X size={14} /> 取消</button>
                      <button onClick={() => handleUpdate(mem.id)} className="flex items-center gap-1 px-4 py-1.5 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg"><Check size={14} /> 保存</button>
                    </div>
                  </div>
                ) : (
                  <div>
                    <p className="text-slate-800 whitespace-pre-wrap text-sm">{mem.content}</p>
                    <div className="flex items-center justify-between mt-3">
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <span className={`px-2 py-0.5 rounded-full ${mem.source === 'manual' ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-600'}`}>
                          {mem.source === 'manual' ? '手动添加' : '自动提取'}
                        </span>
                        {mem.distance !== undefined && (
                          <span className="bg-blue-50 text-blue-600 px-2 py-0.5 rounded-full">
                            相似度 {(mem.distance * 100).toFixed(0)}%
                          </span>
                        )}
                        <span>{new Date(mem.updated_at).toLocaleDateString('zh-CN')}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => { setEditingId(mem.id); setEditContent(mem.content); }}
                          className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors"
                          title="编辑"
                        >
                          <Pencil size={14} />
                        </button>
                        <button
                          onClick={() => handleDelete(mem.id)}
                          className="p-1.5 rounded-lg hover:bg-red-50 text-slate-400 hover:text-red-500 transition-colors"
                          title="删除"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-3 pt-2 pb-4">
                <button
                  onClick={() => goPage(page - 1)}
                  disabled={page <= 1}
                  className="p-1.5 rounded-lg hover:bg-slate-200 text-slate-500 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronLeft size={18} />
                </button>
                <span className="text-sm text-slate-500">
                  {page} / {totalPages}
                </span>
                <button
                  onClick={() => goPage(page + 1)}
                  disabled={page >= totalPages}
                  className="p-1.5 rounded-lg hover:bg-slate-200 text-slate-500 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronRight size={18} />
                </button>
              </div>
            )}

            {!showCreate && (
              <button
                onClick={() => setShowCreate(true)}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-xl border-2 border-dashed border-slate-300 text-slate-500 hover:border-blue-400 hover:text-blue-600 transition-colors"
              >
                <Plus size={18} /> 添加记忆
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
