import { useState, useRef, useEffect } from 'react';
import { Plus, MessageSquare, Pencil, Brain, Shield, LogOut, Search, Users } from 'lucide-react';
import type { Conversation } from '../types';

interface SidebarProps {
  conversations: Conversation[];
  currentConvId: string | null;
  currentPage: 'chat' | 'memories' | 'admin';
  isAdmin: boolean;
  username: string;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onRename: (id: string, title: string) => void;
  onNavigate: (page: 'chat' | 'memories' | 'admin') => void;
  onSwitchUser: (username: 'admin' | 'local') => void;
  onLogout: () => void;
}

export default function Sidebar({
  conversations, currentConvId, currentPage, isAdmin, username,
  onSelect, onCreate, onRename, onNavigate, onSwitchUser, onLogout,
}: SidebarProps) {
  const [editingId, setEditingId] = useState<string | null>(null);

  return (
    <aside className="w-64 bg-gray-50 text-gray-700 flex flex-col h-full shrink-0 border-r border-gray-200">
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-lg font-bold text-gray-800">Chat Agent</h1>
          <button className="p-1.5 rounded-lg hover:bg-gray-200 text-gray-500 transition-colors">
            <Search size={18} />
          </button>
        </div>
        <button
          onClick={() => { onCreate(); if (currentPage !== 'chat') onNavigate('chat'); }}
          className="w-full flex items-center justify-center gap-2 bg-white hover:bg-gray-100 text-gray-700 py-3 px-4 rounded-xl border border-gray-200 transition-colors duration-200"
        >
          <Plus size={18} />
          <span>开启新对话</span>
        </button>
      </div>

      <div className="px-3 pb-2">
        <button
          onClick={() => onNavigate(currentPage === 'memories' ? 'chat' : 'memories')}
          className={`w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl border transition-colors duration-200 ${
            currentPage === 'memories'
              ? 'bg-blue-600 hover:bg-blue-700 text-white border-blue-500'
              : 'bg-white hover:bg-gray-100 text-gray-600 border-gray-200'
          }`}
        >
          <Brain size={18} />
          <span>记忆管理</span>
        </button>
        {isAdmin && (
          <button
            onClick={() => onNavigate('admin')}
            className={`w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl border transition-colors duration-200 mt-2 ${
              currentPage === 'admin'
                ? 'bg-purple-600 hover:bg-purple-700 text-white border-purple-500'
                : 'bg-white hover:bg-gray-100 text-gray-600 border-gray-200'
            }`}
          >
            <Shield size={18} />
            <span>后台管理</span>
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-3 pb-4">
        {conversations.length === 0 && (
          <div className="mt-8 text-center text-gray-400 text-sm">
            暂无历史对话
          </div>
        )}
        <ul className="space-y-1">
          {conversations.map((conv) => (
            <li key={conv.id} className="group">
              {editingId === conv.id ? (
                <RenameInput
                  initial={conv.title}
                  onSave={(title) => {
                    onRename(conv.id, title);
                    setEditingId(null);
                  }}
                  onCancel={() => setEditingId(null)}
                />
              ) : (
                <div
                  onClick={() => { onSelect(conv.id); if (currentPage !== 'chat') onNavigate('chat'); }}
                  onDoubleClick={() => setEditingId(conv.id)}
                  className={`flex items-center gap-3 cursor-pointer px-3 py-2.5 rounded-lg text-sm transition-colors duration-200 ${
                    conv.id === currentConvId
                      ? 'bg-gray-200 text-gray-900'
                      : 'hover:bg-gray-100 text-gray-500 hover:text-gray-900'
                  }`}
                >
                  <MessageSquare size={16} className="shrink-0 opacity-60" />
                  <span className="truncate flex-1">{conv.title}</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setEditingId(conv.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-gray-200 text-gray-400 hover:text-gray-700"
                    title="重命名"
                  >
                    <Pencil size={12} />
                  </button>
                </div>
              )}
            </li>
          ))}
        </ul>
      </div>

      <div className="p-4 border-t border-gray-200 space-y-2">
        <div className="flex items-center gap-2 text-sm">
          <div className="w-7 h-7 rounded-full bg-gray-200 flex items-center justify-center text-xs font-medium text-gray-700">
            {username.charAt(0).toUpperCase()}
          </div>
          <span className="text-gray-600 truncate">{username}</span>
          {isAdmin && <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-100 text-purple-700 font-medium">ADMIN</span>}
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => onSwitchUser('admin')}
            className={`flex-1 flex items-center justify-center gap-1 py-2 text-xs rounded-lg border transition-colors ${
              username === 'admin'
                ? 'bg-purple-100 text-purple-700 font-medium border-purple-300 shadow-sm'
                : 'bg-white text-gray-500 border-gray-200 hover:bg-gray-100 hover:text-gray-700'
            }`}
          >
            <Users size={12} />
            admin
          </button>
          <button
            onClick={() => onSwitchUser('local')}
            className={`flex-1 flex items-center justify-center gap-1 py-2 text-xs rounded-lg border transition-colors ${
              username === 'local'
                ? 'bg-blue-100 text-blue-700 font-medium border-blue-300 shadow-sm'
                : 'bg-white text-gray-500 border-gray-200 hover:bg-gray-100 hover:text-gray-700'
            }`}
          >
            <Users size={12} />
            local
          </button>
        </div>
        <button
          onClick={onLogout}
          className="w-full flex items-center justify-center gap-2 py-2 text-xs text-gray-400 hover:text-red-500 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <LogOut size={14} />
          <span>退出登录</span>
        </button>
      </div>
    </aside>
  );
}

function RenameInput({ initial, onSave, onCancel }: { initial: string; onSave: (v: string) => void; onCancel: () => void }) {
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
    inputRef.current?.select();
  }, []);

  return (
    <input
      ref={inputRef}
      defaultValue={initial}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          const val = e.currentTarget.value.trim();
          if (val) onSave(val);
          else onCancel();
        }
        if (e.key === 'Escape') onCancel();
      }}
      onBlur={(e) => {
        const val = e.target.value.trim();
        if (val) onSave(val);
        else onCancel();
      }}
      className="w-full px-3 py-2.5 rounded-lg text-sm bg-white text-gray-800 border border-blue-500 outline-none"
    />
  );
}
