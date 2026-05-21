import { useState, useCallback, useEffect } from 'react';
import type { AuthUser } from '../types';

const TOKEN_KEY = 'chat_agent_token';

export interface AuthState {
  token: string | null;
  user: AuthUser | null;
  loading: boolean;
}

export function useAuth() {
  const [state, setState] = useState<AuthState>({ token: null, user: null, loading: true });

  const fetchMe = useCallback(async (token: string) => {
    const resp = await fetch('/api/v1/auth/me', {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!resp.ok) throw new Error('Token invalid');
    const user: AuthUser = await resp.json();
    setState({ token, user, loading: false });
  }, []);

  const doLogin = useCallback(async (username: string, password: string) => {
    const form = new URLSearchParams();
    form.append('username', username);
    form.append('password', password);
    const resp = await fetch('/api/v1/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: form.toString(),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Login failed' }));
      throw new Error(err.detail || 'Login failed');
    }
    const data = await resp.json();
    localStorage.setItem(TOKEN_KEY, data.access_token);
    window.dispatchEvent(new CustomEvent('auth:login', { detail: { token: data.access_token } }));
    setState({ token: data.access_token, user: null, loading: true });
    await fetchMe(data.access_token);
  }, [fetchMe]);

  // Init: try saved token first, fallback to admin login
  useEffect(() => {
    const saved = localStorage.getItem(TOKEN_KEY);
    if (saved) {
      fetchMe(saved).catch(() => {
        localStorage.removeItem(TOKEN_KEY);
        doLogin('admin', 'admin123').catch(() => {
          setState({ token: null, user: null, loading: false });
        });
      });
    } else {
      doLogin('admin', 'admin123').catch(() => {
        setState({ token: null, user: null, loading: false });
      });
    }
  }, [fetchMe, doLogin]);

  const switchUser = useCallback(async (username: 'admin' | 'local') => {
    const password = username === 'admin' ? 'admin123' : 'local123';
    await doLogin(username, password);
  }, [doLogin]);

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    setState({ token: null, user: null, loading: false });
  }, []);

  return {
    token: state.token,
    user: state.user,
    loading: state.loading,
    isAuthenticated: !!state.user,
    isAdmin: state.user?.role === 'admin',
    switchUser,
    logout,
  };
}
