import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { User, Bot } from 'lucide-react';
import type { Message } from '../types';
import SearchSources from './SearchSources';

interface MessageItemProps {
  message: Message;
}

export default function MessageItem({ message }: MessageItemProps) {
  const isUser = message.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} py-3`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center mr-3 shrink-0 mt-1">
          <Bot size={18} className="text-emerald-600" />
        </div>
      )}

      <div
        className={`max-w-[80%] px-4 py-2.5 rounded-2xl text-[15px] leading-relaxed ${
          isUser
            ? 'bg-blue-600 text-white rounded-br-md'
            : 'bg-white text-slate-800 shadow-sm border border-slate-100 rounded-bl-md'
        }`}
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            code({ node, inline, className, children, ...props }: any) {
              if (inline) {
                return (
                  <code className="bg-slate-100 text-slate-700 px-1.5 py-0.5 rounded text-sm font-mono" {...props}>
                    {children}
                  </code>
                );
              }
              return (
                <pre className="bg-slate-900 text-slate-100 p-4 rounded-lg overflow-x-auto my-2 text-sm">
                  <code className="font-mono" {...props}>{children}</code>
                </pre>
              );
            },
            p({ children }) {
              return <p className="mb-1.5 last:mb-0">{children}</p>;
            },
          }}
        >
          {message.content || '​'}
        </ReactMarkdown>
        {!isUser && <SearchSources sources={message.sources || []} />}
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center ml-3 shrink-0 mt-1">
          <User size={18} className="text-blue-600" />
        </div>
      )}
    </div>
  );
}
