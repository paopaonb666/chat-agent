export interface Source {
  title: string;
  url: string;
  snippet: string;
  position: number;
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
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
