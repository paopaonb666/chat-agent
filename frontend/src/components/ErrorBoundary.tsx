import { Component, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <div className="h-full flex items-center justify-center">
            <div className="text-center text-slate-500">
              <p className="text-lg font-medium mb-2">页面出错了</p>
              <p className="text-sm text-slate-400 mb-4">
                {this.state.error?.message || '未知错误'}
              </p>
              <button
                onClick={() => this.setState({ hasError: false, error: null })}
                className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700"
              >
                重试
              </button>
            </div>
          </div>
        )
      );
    }
    return this.props.children;
  }
}
