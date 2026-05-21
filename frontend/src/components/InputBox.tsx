import { useRef, useState } from 'react';
import { Send, Square, Paperclip, X, Globe } from 'lucide-react';

interface InputBoxProps {
  onSend: (message: string, files?: FileList) => void;
  onStop: () => void;
  isStreaming: boolean;
  enableWebSearch: boolean;
  onWebSearchToggle: (enabled: boolean) => void;
}

export default function InputBox({
  onSend,
  onStop,
  isStreaming,
  enableWebSearch,
  onWebSearchToggle,
}: InputBoxProps) {
  const [text, setText] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if ((!text.trim() && files.length === 0) || isStreaming) return;
    const dt = new DataTransfer();
    files.forEach((f) => dt.items.add(f));
    onSend(text.trim(), dt.files);
    setText('');
    setFiles([]);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles((prev) => [...prev, ...Array.from(e.target.files!)]);
    }
    e.target.value = '';
  };

  const removeFile = (idx: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  return (
    <div className="bg-white border-t border-gray-200 px-4 py-4 shrink-0">
      <div className="max-w-3xl mx-auto">
        {files.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-2">
            {files.map((f, idx) => (
              <span
                key={idx}
                className="inline-flex items-center gap-1 text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded-md"
              >
                {f.name}
                <button type="button" onClick={() => removeFile(idx)} className="hover:text-red-500">
                  <X size={12} />
                </button>
              </span>
            ))}
          </div>
        )}
        <div className="flex items-center gap-2 mb-2">
          <button
            type="button"
            onClick={() => onWebSearchToggle(!enableWebSearch)}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border transition-colors ${
              enableWebSearch
                ? 'bg-blue-50 border-blue-300 text-blue-700'
                : 'bg-gray-100 border-gray-200 text-gray-500 hover:bg-gray-200'
            }`}
          >
            <Globe size={12} />
            联网搜索
          </button>
        </div>
        <form onSubmit={handleSubmit} className="relative flex items-end gap-2">
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="shrink-0 w-10 h-10 flex items-center justify-center text-gray-400 hover:text-gray-600 rounded-full transition-colors"
            title="上传文件"
          >
            <Paperclip size={20} />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={handleFileChange}
          />
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入消息..."
            rows={1}
            className="flex-1 resize-none max-h-32 bg-gray-100 rounded-2xl px-4 py-3 pr-12 text-[15px] text-gray-800 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:bg-white transition-all duration-200"
            style={{ minHeight: '48px' }}
          />
          {isStreaming ? (
            <button
              type="button"
              onClick={onStop}
              className="shrink-0 w-10 h-10 flex items-center justify-center bg-gray-200 hover:bg-gray-300 text-gray-600 rounded-full transition-colors duration-200"
              title="停止生成"
            >
              <Square size={16} fill="currentColor" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!text.trim() && files.length === 0}
              className="shrink-0 w-10 h-10 flex items-center justify-center bg-blue-600 hover:bg-blue-700 disabled:bg-gray-200 disabled:text-gray-400 text-white rounded-full transition-colors duration-200"
              title="发送"
            >
              <Send size={18} />
            </button>
          )}
        </form>
        <p className="text-center text-xs text-gray-400 mt-2">
          AI 生成内容仅供参考
        </p>
      </div>
    </div>
  );
}
