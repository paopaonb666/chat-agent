import type { Conversation, UploadedFile, AdminStats, AdminUser, AdminConversation, AdminConversationDetail } from '../types';

const BASE = '/api/v1';

function getToken(): string | null {
  return localStorage.getItem('chat_agent_token');
}

async function authFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const token = getToken();
  const headers: Record<string, string> = { ...((options.headers as Record<string, string>) || {}) };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  if (!headers['Content-Type'] && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }
  const resp = await fetch(url, { ...options, headers });
  if (resp.status === 401) {
    localStorage.removeItem('chat_agent_token');
  }
  return resp;
}

export async function fetchConversations(): Promise<Conversation[]> {
  const resp = await authFetch(`${BASE}/conversations`);
  if (!resp.ok) throw new Error('Failed to fetch conversations');
  return resp.json();
}

export async function createConversation(model: string): Promise<Conversation> {
  const resp = await authFetch(`${BASE}/conversations`, {
    method: 'POST',
    body: JSON.stringify({ title: '新对话', model }),
  });
  if (!resp.ok) throw new Error('Failed to create conversation');
  return resp.json();
}

export async function renameConversation(convId: string, title: string): Promise<void> {
  const resp = await authFetch(`${BASE}/conversations/${convId}`, {
    method: 'PATCH',
    body: JSON.stringify({ title }),
  });
  if (!resp.ok) throw new Error('Failed to rename conversation');
}

export async function uploadFile(convId: string, file: File): Promise<UploadedFile> {
  const form = new FormData();
  form.append('conversation_id', convId);
  form.append('file', file);
  const token = getToken();
  const headers: Record<string, string> = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const resp = await fetch(`${BASE}/files/upload`, { method: 'POST', body: form, headers });
  if (!resp.ok) throw new Error('Failed to upload file');
  return resp.json();
}

// ── Admin APIs ───────────────────────────────────────────────────────────

export async function fetchAdminStats(): Promise<AdminStats> {
  const resp = await authFetch(`${BASE}/admin/stats`);
  if (!resp.ok) throw new Error('Failed to fetch stats');
  return resp.json();
}

export async function fetchAdminUsers(): Promise<AdminUser[]> {
  const resp = await authFetch(`${BASE}/admin/users`);
  if (!resp.ok) throw new Error('Failed to fetch users');
  return resp.json();
}

export async function createAdminUser(username: string, password: string, role: string): Promise<AdminUser> {
  const resp = await authFetch(`${BASE}/admin/users`, {
    method: 'POST',
    body: JSON.stringify({ username, password, role }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: 'Failed' }));
    throw new Error(err.detail || 'Failed to create user');
  }
  return resp.json();
}

export async function updateAdminUser(id: number, data: { username?: string; password?: string; role?: string }): Promise<AdminUser> {
  const resp = await authFetch(`${BASE}/admin/users/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: 'Failed' }));
    throw new Error(err.detail || 'Failed to update user');
  }
  return resp.json();
}

export async function deleteAdminUser(id: number): Promise<void> {
  const resp = await authFetch(`${BASE}/admin/users/${id}`, { method: 'DELETE' });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: 'Failed' }));
    throw new Error(err.detail || 'Failed to delete user');
  }
}

export async function fetchAdminConversations(): Promise<AdminConversation[]> {
  const resp = await authFetch(`${BASE}/admin/conversations`);
  if (!resp.ok) throw new Error('Failed to fetch conversations');
  return resp.json();
}

export async function fetchAdminConversationDetail(convId: string): Promise<AdminConversationDetail> {
  const resp = await authFetch(`${BASE}/admin/conversations/${convId}`);
  if (!resp.ok) throw new Error('Failed to fetch conversation');
  return resp.json();
}

export async function deleteAdminConversation(convId: string): Promise<void> {
  const resp = await authFetch(`${BASE}/admin/conversations/${convId}`, { method: 'DELETE' });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: 'Failed' }));
    throw new Error(err.detail || 'Failed to delete conversation');
  }
}
