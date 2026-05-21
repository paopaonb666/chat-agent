import { useState, useEffect, useCallback } from 'react';
import {
  Users, MessageSquare, Database, FileText, Brain, Activity,
  Plus, Pencil, Trash2, X, Check, Eye,
} from 'lucide-react';
import type { AdminStats, AdminUser, AdminConversation, AdminConversationDetail } from '../types';
import {
  fetchAdminStats, fetchAdminUsers, createAdminUser, updateAdminUser, deleteAdminUser,
  fetchAdminConversations, fetchAdminConversationDetail, deleteAdminConversation,
} from '../services/api';

type Tab = 'stats' | 'users' | 'conversations';

export default function AdminPage() {
  const [tab, setTab] = useState<Tab>('stats');
  const [error, setError] = useState('');
  return (
    <div className="flex-1 flex flex-col bg-slate-50 h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-white shrink-0">
        <h1 className="text-lg font-semibold text-slate-800">后台管理</h1>
        <nav className="flex gap-1 bg-slate-100 rounded-lg p-1">
          {([
            ['stats', '统计'],
            ['users', '用户管理'],
            ['conversations', '对话管理'],
          ] as [Tab, string][]).map(([key, label]) => (
            <button
              key={key}
              onClick={() => { setTab(key); setError(''); }}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
                tab === key ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              {label}
            </button>
          ))}
        </nav>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {error && <p className="text-red-500 text-sm mb-4">{error}</p>}
        {tab === 'stats' && <StatsDashboard />}
        {tab === 'users' && <UserManager setError={setError} />}
        {tab === 'conversations' && <ConversationManager setError={setError} />}
      </div>
    </div>
  );
}

// ── Stats Dashboard ──────────────────────────────────────────────────────

function StatsDashboard() {
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAdminStats().then(setStats).catch(() => {}).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-center text-slate-400 py-12">加载中...</div>;
  if (!stats) return <div className="text-center text-red-500 py-12">加载失败</div>;

  const cards: [string, number, React.ReactNode][] = [
    ['用户总数', stats.total_users, <Users size={20} />],
    ['对话总数', stats.total_conversations, <MessageSquare size={20} />],
    ['消息总数', stats.total_messages, <Database size={20} />],
    ['文件总数', stats.total_files, <FileText size={20} />],
    ['记忆总数', stats.total_memories, <Brain size={20} />],
    ['7日活跃用户', stats.active_users_7d, <Activity size={20} />],
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 max-w-4xl mx-auto">
      {cards.map(([label, value, icon]) => (
        <div key={label} className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-slate-500">{label}</span>
            <span className="text-slate-400">{icon}</span>
          </div>
          <p className="text-2xl font-semibold text-slate-800">{value}</p>
        </div>
      ))}
    </div>
  );
}

// ── User Manager ─────────────────────────────────────────────────────────

function UserManager({ setError }: { setError: (e: string) => void }) {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);

  const [editingId, setEditingId] = useState<number | null>(null);
  const [editUsername, setEditUsername] = useState('');
  const [editPassword, setEditPassword] = useState('');
  const [editRole, setEditRole] = useState('user');

  const [showCreate, setShowCreate] = useState(false);
  const [createUsername, setCreateUsername] = useState('');
  const [createPassword, setCreatePassword] = useState('');
  const [createRole, setCreateRole] = useState('user');

  const [deleteConfirm, setDeleteConfirm] = useState<number | null>(null);

  const loadUsers = useCallback(async () => {
    try {
      setUsers(await fetchAdminUsers());
    } catch {
      setError('加载用户失败');
    } finally {
      setLoading(false);
    }
  }, [setError]);

  useEffect(() => { loadUsers(); }, [loadUsers]);

  const handleCreate = async () => {
    if (!createUsername.trim() || !createPassword.trim()) return;
    try {
      await createAdminUser(createUsername.trim(), createPassword, createRole);
      setCreateUsername(''); setCreatePassword(''); setCreateRole('user'); setShowCreate(false);
      await loadUsers();
    } catch (e) { setError((e as Error).message); }
  };

  const handleUpdate = async (id: number) => {
    try {
      const data: Record<string, string> = {};
      if (editUsername.trim()) data.username = editUsername.trim();
      if (editPassword.trim()) data.password = editPassword;
      data.role = editRole;
      await updateAdminUser(id, data);
      setEditingId(null);
      await loadUsers();
    } catch (e) { setError((e as Error).message); }
  };

  const handleDelete = async (id: number) => {
    try {
      await deleteAdminUser(id);
      setDeleteConfirm(null);
      await loadUsers();
    } catch (e) { setError((e as Error).message); }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-3">
      {!showCreate && (
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-1.5 text-sm bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
        >
          <Plus size={16} /> 新建用户
        </button>
      )}

      {showCreate && (
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
            <input value={createUsername} onChange={(e) => setCreateUsername(e.target.value)} placeholder="用户名" className="px-3 py-2 rounded-lg border border-slate-300 outline-none text-sm focus:border-blue-500" />
            <input type="password" value={createPassword} onChange={(e) => setCreatePassword(e.target.value)} placeholder="密码" className="px-3 py-2 rounded-lg border border-slate-300 outline-none text-sm focus:border-blue-500" />
            <select value={createRole} onChange={(e) => setCreateRole(e.target.value)} className="px-3 py-2 rounded-lg border border-slate-300 outline-none text-sm focus:border-blue-500">
              <option value="user">user</option>
              <option value="admin">admin</option>
            </select>
            <div className="flex gap-2">
              <button onClick={handleCreate} className="flex items-center gap-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm"><Check size={14} /> 创建</button>
              <button onClick={() => { setShowCreate(false); setCreateUsername(''); setCreatePassword(''); }} className="px-4 py-2 text-slate-600 hover:text-slate-800 text-sm">取消</button>
            </div>
          </div>
        </div>
      )}

      {loading ? (
        <div className="text-center text-slate-400 py-12">加载中...</div>
      ) : users.length === 0 ? (
        <div className="text-center text-slate-400 py-12">暂无用户</div>
      ) : (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 border-b border-slate-200">
              <tr>
                <th className="text-left px-4 py-3 text-slate-600 font-medium">ID</th>
                <th className="text-left px-4 py-3 text-slate-600 font-medium">用户名</th>
                <th className="text-left px-4 py-3 text-slate-600 font-medium">角色</th>
                <th className="text-left px-4 py-3 text-slate-600 font-medium">创建时间</th>
                <th className="text-right px-4 py-3 text-slate-600 font-medium">操作</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {users.map((u) => (
                <tr key={u.id} className="hover:bg-slate-50">
                  {editingId === u.id ? (
                    <>
                      <td className="px-4 py-3 text-slate-400">{u.id}</td>
                      <td className="px-4 py-3"><input value={editUsername} onChange={(e) => setEditUsername(e.target.value)} className="w-full px-2 py-1 rounded border border-slate-300 outline-none text-sm focus:border-blue-500" /></td>
                      <td className="px-4 py-3">
                        <select value={editRole} onChange={(e) => setEditRole(e.target.value)} className="px-2 py-1 rounded border border-slate-300 outline-none text-sm">
                          <option value="user">user</option>
                          <option value="admin">admin</option>
                        </select>
                      </td>
                      <td className="px-4 py-3 text-slate-400">{new Date(u.created_at).toLocaleDateString('zh-CN')}</td>
                      <td className="px-4 py-3 text-right">
                        <button onClick={() => handleUpdate(u.id)} className="p-1.5 text-green-600 hover:bg-green-50 rounded" title="保存"><Check size={14} /></button>
                        <button onClick={() => setEditingId(null)} className="p-1.5 text-slate-400 hover:bg-slate-100 rounded ml-1" title="取消"><X size={14} /></button>
                      </td>
                    </>
                  ) : (
                    <>
                      <td className="px-4 py-3 text-slate-400">{u.id}</td>
                      <td className="px-4 py-3 font-medium text-slate-700">{u.username}</td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${u.role === 'admin' ? 'bg-purple-100 text-purple-700' : 'bg-slate-100 text-slate-600'}`}>
                          {u.role}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-slate-400">{new Date(u.created_at).toLocaleDateString('zh-CN')}</td>
                      <td className="px-4 py-3 text-right">
                        <button
                          onClick={() => { setEditingId(u.id); setEditUsername(u.username); setEditPassword(''); setEditRole(u.role); }}
                          className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded transition-colors"
                          title="编辑"
                        ><Pencil size={14} /></button>
                        {deleteConfirm === u.id ? (
                          <>
                            <button onClick={() => handleDelete(u.id)} className="p-1.5 text-red-500 hover:bg-red-50 rounded ml-1" title="确认删除"><Check size={14} /></button>
                            <button onClick={() => setDeleteConfirm(null)} className="p-1.5 text-slate-400 hover:bg-slate-100 rounded ml-1" title="取消"><X size={14} /></button>
                          </>
                        ) : (
                          <button onClick={() => setDeleteConfirm(u.id)} className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded ml-1 transition-colors" title="删除"><Trash2 size={14} /></button>
                        )}
                      </td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ── Conversation Manager ─────────────────────────────────────────────────

function ConversationManager({ setError }: { setError: (e: string) => void }) {
  const [convs, setConvs] = useState<AdminConversation[]>([]);
  const [loading, setLoading] = useState(true);
  const [detailId, setDetailId] = useState<string | null>(null);
  const [detail, setDetail] = useState<AdminConversationDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  const loadConvs = useCallback(async () => {
    try {
      setConvs(await fetchAdminConversations());
    } catch {
      setError('加载对话失败');
    } finally {
      setLoading(false);
    }
  }, [setError]);

  useEffect(() => { loadConvs(); }, [loadConvs]);

  const openDetail = async (convId: string) => {
    setDetailId(convId);
    setDetailLoading(true);
    try {
      setDetail(await fetchAdminConversationDetail(convId));
    } catch {
      setError('加载对话详情失败');
    } finally {
      setDetailLoading(false);
    }
  };

  const handleDelete = async (convId: string) => {
    try {
      await deleteAdminConversation(convId);
      setDeleteConfirm(null);
      if (detailId === convId) { setDetailId(null); setDetail(null); }
      await loadConvs();
    } catch (e) { setError((e as Error).message); }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-3">
      {loading ? (
        <div className="text-center text-slate-400 py-12">加载中...</div>
      ) : convs.length === 0 ? (
        <div className="text-center text-slate-400 py-12">暂无对话</div>
      ) : (
        <>
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 border-b border-slate-200">
                <tr>
                  <th className="text-left px-4 py-3 text-slate-600 font-medium">标题</th>
                  <th className="text-left px-4 py-3 text-slate-600 font-medium">用户</th>
                  <th className="text-left px-4 py-3 text-slate-600 font-medium">模型</th>
                  <th className="text-center px-4 py-3 text-slate-600 font-medium">消息数</th>
                  <th className="text-right px-4 py-3 text-slate-600 font-medium">操作</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {convs.map((c) => (
                  <tr key={c.id} className="hover:bg-slate-50">
                    <td className="px-4 py-3 font-medium text-slate-700 max-w-[200px] truncate">{c.title}</td>
                    <td className="px-4 py-3 text-slate-500">{c.username || '匿名'}</td>
                    <td className="px-4 py-3 text-slate-500">{c.model}</td>
                    <td className="px-4 py-3 text-center text-slate-500">{c.message_count}</td>
                    <td className="px-4 py-3 text-right">
                      <button onClick={() => openDetail(c.id)} className="p-1.5 text-slate-400 hover:text-blue-500 hover:bg-blue-50 rounded transition-colors" title="查看"><Eye size={14} /></button>
                      {deleteConfirm === c.id ? (
                        <>
                          <button onClick={() => handleDelete(c.id)} className="p-1.5 text-red-500 hover:bg-red-50 rounded ml-1" title="确认删除"><Check size={14} /></button>
                          <button onClick={() => setDeleteConfirm(null)} className="p-1.5 text-slate-400 hover:bg-slate-100 rounded ml-1" title="取消"><X size={14} /></button>
                        </>
                      ) : (
                        <button onClick={() => setDeleteConfirm(c.id)} className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded ml-1 transition-colors" title="删除"><Trash2 size={14} /></button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Detail panel */}
          {detailId && (
            <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-medium text-slate-800">对话消息</h3>
                <button onClick={() => { setDetailId(null); setDetail(null); }} className="p-1 text-slate-400 hover:text-slate-600"><X size={16} /></button>
              </div>
              {detailLoading ? (
                <div className="text-center text-slate-400 py-8">加载中...</div>
              ) : detail ? (
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {detail.messages.map((m, i) => (
                    <div key={i} className={`p-3 rounded-lg ${m.role === 'user' ? 'bg-blue-50 ml-8' : 'bg-slate-50 mr-8'}`}>
                      <div className="flex items-center justify-between mb-1">
                        <span className={`text-xs font-medium ${m.role === 'user' ? 'text-blue-600' : 'text-green-600'}`}>
                          {m.role === 'user' ? '用户' : '助手'}
                        </span>
                        <span className="text-xs text-slate-400">{new Date(m.created_at).toLocaleString('zh-CN')}</span>
                      </div>
                      <p className="text-sm text-slate-700 whitespace-pre-wrap">{m.content}</p>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          )}
        </>
      )}
    </div>
  );
}
