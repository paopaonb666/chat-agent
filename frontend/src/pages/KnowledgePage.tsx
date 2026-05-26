import { useState, useEffect, useCallback, useRef } from 'react';
import { Search, Upload, Trash2, RefreshCw, ChevronDown, ChevronUp, ChevronLeft, ChevronRight, BookOpen, FileText, AlertCircle } from 'lucide-react';
import type { KnowledgeDocument, KnowledgeChunk } from '../types';
import { useAuth } from '../hooks/useAuth';

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem('chat_agent_token');
  const headers: Record<string, string> = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;
  return headers;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function statusBadge(status: KnowledgeDocument['status']) {
  const styles: Record<string, string> = {
    pending: 'bg-gray-100 text-gray-600',
    processing: 'bg-blue-100 text-blue-700',
    completed: 'bg-green-100 text-green-700',
    failed: 'bg-red-100 text-red-700',
  };
  const labels: Record<string, string> = {
    pending: '待处理',
    processing: '处理中',
    completed: '已完成',
    failed: '失败',
  };
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${styles[status] || styles.pending}`}>
      {labels[status] || status}
    </span>
  );
}

export default function KnowledgePage() {
  const [docs, setDocs] = useState<KnowledgeDocument[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<{ chunk_id: string; document_id: string; content: string; distance: number }[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const searchTimer = useRef<ReturnType<typeof setTimeout>>();

  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [chunks, setChunks] = useState<KnowledgeChunk[]>([]);
  const [chunksLoading, setChunksLoading] = useState(false);

  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [isDragOver, setIsDragOver] = useState(false);
  const dragCounterRef = useRef(0);

  const { isAdmin } = useAuth();

  const fetchDocs = useCallback(async (p: number) => {
    setLoading(true);
    setError('');
    try {
      const resp = await fetch(`/api/v1/knowledge/documents?page=${p}&page_size=10`, { headers: authHeaders() });
      if (!resp.ok) throw new Error('获取失败');
      const data = await resp.json();
      setDocs(data.items);
      setPage(data.page);
      setTotalPages(data.total_pages);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchDocs(1); }, [fetchDocs]);

  useEffect(() => {
    if (!searchQuery.trim()) {
      setIsSearching(false);
      setSearchResults([]);
      return;
    }
    setIsSearching(true);
    clearTimeout(searchTimer.current);
    searchTimer.current = setTimeout(async () => {
      setError('');
      try {
        const resp = await fetch('/api/v1/knowledge/search', {
          method: 'POST',
          headers: { ...authHeaders(), 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: searchQuery.trim(), top_k: 5 }),
        });
        if (!resp.ok) throw new Error('搜索失败');
        const data = await resp.json();
        setSearchResults(data);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setIsSearching(false);
      }
    }, 400);
    return () => clearTimeout(searchTimer.current);
  }, [searchQuery]);

  const uploadFiles = async (files: File[]) => {
    if (files.length === 0) return;
    setUploading(true);
    setError('');
    try {
      for (const file of files) {
        const form = new FormData();
        form.append('file', file);
        form.append('visibility', 'public');
        const resp = await fetch('/api/v1/knowledge/documents/upload', {
          method: 'POST',
          headers: authHeaders(),
          body: form,
        });
        if (!resp.ok) {
          const errData = await resp.json().catch(() => ({}));
          throw new Error(errData.detail || `上传失败: ${file.name}`);
        }
      }
      await fetchDocs(1);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    await uploadFiles(Array.from(e.target.files || []));
  };

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isAdmin) return;
    if (e.dataTransfer.types.includes('Files')) {
      dragCounterRef.current += 1;
      setIsDragOver(true);
    }
  }, [isAdmin]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isAdmin) return;
    dragCounterRef.current -= 1;
    if (dragCounterRef.current <= 0) {
      dragCounterRef.current = 0;
      setIsDragOver(false);
    }
  }, [isAdmin]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isAdmin) return;
    dragCounterRef.current = 0;
    setIsDragOver(false);
    uploadFiles(Array.from(e.dataTransfer.files));
  }, [isAdmin]);

  const handleDelete = async (id: string) => {
    if (!confirm('确定要删除这个文档吗？')) return;
    setError('');
    try {
      const resp = await fetch(`/api/v1/knowledge/documents/${id}`, {
        method: 'DELETE',
        headers: authHeaders(),
      });
      if (!resp.ok) throw new Error('删除失败');
      await fetchDocs(page);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const handleReprocess = async (id: string) => {
    setError('');
    try {
      const resp = await fetch(`/api/v1/knowledge/documents/${id}/reprocess`, {
        method: 'POST',
        headers: authHeaders(),
      });
      if (!resp.ok) throw new Error('重新处理失败');
      await fetchDocs(page);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const toggleExpand = async (id: string) => {
    if (expandedId === id) {
      setExpandedId(null);
      return;
    }
    setExpandedId(id);
    setChunksLoading(true);
    try {
      const resp = await fetch(`/api/v1/knowledge/documents/${id}/chunks`, { headers: authHeaders() });
      if (!resp.ok) throw new Error('获取切片失败');
      const data = await resp.json();
      setChunks(data);
    } catch (e) {
      setError((e as Error).message);
      setChunks([]);
    } finally {
      setChunksLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col bg-slate-50 h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-white shrink-0">
        <h1 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
          <BookOpen size={20} className="text-amber-600" />
          知识库
        </h1>
        {isAdmin && (
          <>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleUpload}
              className="hidden"
              accept=".txt,.pdf,.docx,.xlsx,.pptx,.md,.html,.csv"
              multiple
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              className="flex items-center gap-2 bg-amber-600 hover:bg-amber-700 disabled:opacity-50 text-white px-4 py-2 rounded-xl transition-colors text-sm"
            >
              <Upload size={16} />
              {uploading ? '上传中...' : '上传文档'}
            </button>
          </>
        )}
      </div>

      {/* Search bar */}
      <div className="px-6 py-3 bg-white border-b border-slate-100 shrink-0">
        <div className="relative max-w-2xl mx-auto">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="搜索知识库..."
            className="w-full pl-10 pr-4 py-2 rounded-xl border border-slate-300 focus:border-amber-500 outline-none text-sm"
          />
          {isSearching && (
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-slate-400">搜索中...</span>
          )}
        </div>
      </div>

      {/* Content */}
      <div
        className={`flex-1 overflow-y-auto px-6 py-4 relative transition-colors ${isDragOver ? 'bg-amber-50/80 border-2 border-dashed border-amber-400' : ''}`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        {isDragOver && (
          <div className="absolute inset-0 z-10 flex items-center justify-center pointer-events-none">
            <div className="bg-white/90 rounded-2xl px-8 py-6 shadow-lg border border-amber-200 text-center">
              <Upload size={40} className="mx-auto text-amber-500 mb-3" />
              <p className="text-amber-700 font-medium text-lg">释放以上传文档</p>
              <p className="text-amber-500 text-sm mt-1">支持 PDF、DOCX、TXT、XLSX、PPTX、MD、HTML、CSV</p>
            </div>
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 text-red-600 text-sm mb-4 bg-red-50 px-4 py-2 rounded-lg">
            <AlertCircle size={16} />
            {error}
          </div>
        )}

        {searchQuery.trim() ? (
          /* Search results */
          searchResults.length === 0 && !isSearching ? (
            <div className="text-center text-slate-400 py-12">未找到匹配结果</div>
          ) : (
            <div className="space-y-3 max-w-3xl mx-auto">
              {searchResults.map((r) => (
                <div key={r.chunk_id} className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
                  <p className="text-slate-800 text-sm whitespace-pre-wrap">{r.content}</p>
                  <div className="flex items-center justify-between mt-2">
                    <span className="text-xs text-slate-400">文档 ID: {r.document_id}</span>
                    <span className="text-xs text-slate-400">相似度: {(r.distance * 100).toFixed(1)}%</span>
                  </div>
                </div>
              ))}
            </div>
          )
        ) : loading ? (
          <div className="text-center text-slate-400 py-12">加载中...</div>
        ) : docs.length === 0 ? (
          <div className="text-center text-slate-400 py-12">
            <p>暂无文档</p>
            {isAdmin && <p className="text-sm mt-2">将文件拖放到此处，或点击右上角上传</p>}
          </div>
        ) : (
          <div className="max-w-4xl mx-auto space-y-3">
            {docs.map((doc) => (
              <div key={doc.id} className="bg-white rounded-xl border border-slate-200 shadow-sm">
                <div
                  className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-slate-50 transition-colors"
                  onClick={() => toggleExpand(doc.id)}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <FileText size={18} className="text-slate-400 shrink-0" />
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-slate-800 truncate">{doc.filename}</p>
                      <p className="text-xs text-slate-400">{formatSize(doc.file_size)} · {new Date(doc.created_at).toLocaleString('zh-CN')}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 shrink-0">
                    {statusBadge(doc.status)}
                    {doc.chunk_count > 0 && (
                      <span className="text-xs text-slate-500">{doc.chunk_count} 切片</span>
                    )}
                    {isAdmin && (
                      <div className="flex items-center gap-1">
                        <button
                          onClick={(e) => { e.stopPropagation(); handleReprocess(doc.id); }}
                          className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-amber-600 transition-colors"
                          title="重新处理"
                        >
                          <RefreshCw size={14} />
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); handleDelete(doc.id); }}
                          className="p-1.5 rounded-lg hover:bg-red-50 text-slate-400 hover:text-red-500 transition-colors"
                          title="删除"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    )}
                    {expandedId === doc.id ? <ChevronUp size={16} className="text-slate-400" /> : <ChevronDown size={16} className="text-slate-400" />}
                  </div>
                </div>

                {expandedId === doc.id && (
                  <div className="border-t border-slate-100 px-4 py-3 bg-slate-50">
                    {chunksLoading ? (
                      <div className="text-sm text-slate-400">加载切片...</div>
                    ) : chunks.length === 0 ? (
                      <div className="text-sm text-slate-400">暂无切片</div>
                    ) : (
                      <div className="space-y-2 max-h-96 overflow-y-auto">
                        {chunks.map((chunk) => (
                          <div key={chunk.id} className="bg-white rounded-lg border border-slate-200 p-3">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-xs font-medium text-slate-500">#{chunk.chunk_index + 1}</span>
                              {chunk.title_path && (
                                <span className="text-xs text-slate-400 truncate max-w-[300px]">{chunk.title_path}</span>
                              )}
                            </div>
                            <p className="text-sm text-slate-700 whitespace-pre-wrap">{chunk.content}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-3 pt-2 pb-4">
                <button
                  onClick={() => { if (page > 1) { setPage(page - 1); fetchDocs(page - 1); } }}
                  disabled={page <= 1}
                  className="p-1.5 rounded-lg hover:bg-slate-200 text-slate-500 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronLeft size={18} />
                </button>
                <span className="text-sm text-slate-500">{page} / {totalPages}</span>
                <button
                  onClick={() => { if (page < totalPages) { setPage(page + 1); fetchDocs(page + 1); } }}
                  disabled={page >= totalPages}
                  className="p-1.5 rounded-lg hover:bg-slate-200 text-slate-500 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronRight size={18} />
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
