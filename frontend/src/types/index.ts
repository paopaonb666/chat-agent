export interface Source {
  title: string;
  url: string;
  snippet: string;
  position: number;
}

export type ToolCallStatus = 'running' | 'completed' | 'error' | 'interrupted';

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  result?: string;
  status: ToolCallStatus;
  requiresApproval?: boolean;
  rejectionReason?: string;
  isSubAgent?: boolean;
}

export interface ToolApprovalAction {
  type: 'approve' | 'reject' | 'edit';
  toolCallId: string;
  editedArgs?: Record<string, unknown>;
  reason?: string;
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  reasoning?: string;
  toolCalls?: ToolCall[];
}

export interface Conversation {
  id: string;
  title: string;
  model: string;
  messages: Message[];
}

export interface UploadedFile {
  id: number;
  filename: string;
  mime_type: string;
  extracted_text: string;
}

export interface MemoryPrompt {
  memory_content: string;
  conversation_id: string;
}

export type StepStatus = 'pending' | 'running' | 'completed' | 'error';

export interface MemoryStep {
  id: string;
  label: string;
  status: StepStatus;
}

export interface MemoryStoreState {
  steps: MemoryStep[];
  message: string;
  done: boolean;
}

export interface IntentStep {
  step: string;
  label: string;
  status: StepStatus;
}

export interface UserMemoryItem {
  id: string;
  content: string;
  source: 'auto_extracted' | 'manual';
  created_at: string;
  updated_at: string;
  distance?: number;
}

export interface PaginatedMemories {
  items: UserMemoryItem[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface ToolCallEvent {
  type: 'tool_call';
  name: string;
  status: 'running' | 'completed' | 'error';
  query?: string;
}

export interface ToolStep {
  name: string;
  label: string;
  status: 'running' | 'completed' | 'error';
  detail: string;
}

// Auth types
export interface AuthUser {
  id: number;
  username: string;
  role: 'user' | 'admin';
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
}

// Admin types
export interface AdminStats {
  total_users: number;
  total_conversations: number;
  total_messages: number;
  total_files: number;
  total_memories: number;
  active_users_7d: number;
}

export interface AdminUser {
  id: number;
  username: string;
  role: string;
  created_at: string;
}

export interface AdminConversation {
  id: string;
  title: string;
  model: string;
  username: string | null;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface AdminConversationDetail extends AdminConversation {
  messages: { role: string; content: string; created_at: string }[];
}
