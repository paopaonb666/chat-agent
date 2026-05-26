import { useState } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import ErrorBoundary from './components/ErrorBoundary';
import AdminPage from './pages/AdminPage';
import MemoriesPage from './pages/MemoriesPage';
import KnowledgePage from './pages/KnowledgePage';
import { useChatStream } from './hooks/useChatStream';
import { useAuth } from './hooks/useAuth';

type Page = 'chat' | 'memories' | 'admin' | 'knowledge';

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>('chat');
  const { user, isAdmin, loading, switchUser, logout } = useAuth();

  const {
    conversations,
    currentConvId,
    currentMessages,
    currentModel,
    isStreaming,
    pendingMemory,
    memoryStore,
    enableWebSearch,
    toolSteps,
    createConversation,
    selectConversation,
    setCurrentModel,
    setEnableWebSearch,
    sendMessage,
    stopGeneration,
    renameConversation,
    storeMemory,
    dismissMemory,
  } = useChatStream();

  const currentTitle = conversations.find((c) => c.id === currentConvId)?.title || '对话';

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-slate-100">
        <p className="text-slate-400">加载中...</p>
      </div>
    );
  }

  return (
    <div className="h-screen flex bg-white">
      <Sidebar
        conversations={conversations}
        currentConvId={currentConvId}
        currentPage={currentPage}
        isAdmin={isAdmin}
        username={user?.username || ''}
        onSelect={selectConversation}
        onCreate={createConversation}
        onRename={renameConversation}
        onNavigate={setCurrentPage}
        onSwitchUser={switchUser}
        onLogout={logout}
      />
      <ErrorBoundary>
        {currentPage === 'admin' ? (
          <AdminPage />
        ) : currentPage === 'memories' ? (
          <MemoriesPage />
        ) : currentPage === 'knowledge' ? (
          <KnowledgePage />
        ) : (
          <ChatArea
            messages={currentMessages}
            isStreaming={isStreaming}
            currentModel={currentModel}
            currentTitle={currentTitle}
            pendingMemory={pendingMemory}
            memoryStore={memoryStore}
            enableWebSearch={enableWebSearch}
            toolSteps={toolSteps}
            onModelChange={setCurrentModel}
            onWebSearchToggle={setEnableWebSearch}
            onSend={sendMessage}
            onStop={stopGeneration}
            onStoreMemory={storeMemory}
            onDismissMemory={dismissMemory}
          />
        )}
      </ErrorBoundary>
    </div>
  );
}
