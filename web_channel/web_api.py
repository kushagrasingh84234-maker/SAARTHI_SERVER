import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import { config } from '../config'; // must expose config.apiUrl

/* ───────────────────────── Types ───────────────────────── */

export interface Source {
  title: string;
  url: string;
  domain: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: number;
  sources?: Source[];
  streaming?: boolean;
  error?: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

interface HistoryRow {
  id: number | string;
  role: string;
  content: string;
  created_at: string | number;
}

interface HistoryResponse {
  messages?: HistoryRow[];
}

interface ChatState {
  conversations: Conversation[];
  activeId: string | null;
}

type Action =
  | { type: 'SET_CONVERSATIONS'; payload: Conversation[] }
  | { type: 'SELECT_CHAT'; payload: string }
  | { type: 'NEW_CHAT'; conversationId?: string }
  | { type: 'ADD_MESSAGES'; conversationId: string; messages: Message[] }
  | { type: 'APPEND_TOKEN'; conversationId: string; messageId: string; text: string }
  | { type: 'SET_SOURCES'; conversationId: string; messageId: string; sources: Source[] }
  | { type: 'FINISH'; conversationId: string; messageId: string }
  | { type: 'FAIL'; conversationId: string; messageId: string; error: string };

/* ───────────────────────── Helpers ───────────────────────── */

const makeId = (): string =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

const mapHistoryRole = (role: string): 'user' | 'assistant' | null => {
  if (role === 'user') return 'user';
  if (role === 'model' || role === 'assistant') return 'assistant';
  return null; // ignore anything else (e.g. system)
};

const mapHistoryToMessages = (rows: HistoryRow[]): Message[] => {
  const result: Message[] = [];
  for (const row of rows) {
    const role = mapHistoryRole(row.role);
    if (!role) continue;
    result.push({
      id: String(row.id),
      role,
      content: row.content,
      createdAt: new Date(row.created_at).getTime(),
    });
  }
  return result;
};

const emptyConversation = (id: string): Conversation => {
  const now = Date.now();
  return { id, title: 'New chat', messages: [], createdAt: now, updatedAt: now };
};

const updateMessage = (
  conversations: Conversation[],
  conversationId: string,
  messageId: string,
  patch: (m: Message) => Message
): Conversation[] =>
  conversations.map((c) =>
    c.id !== conversationId
      ? c
      : {
          ...c,
          updatedAt: Date.now(),
          messages: c.messages.map((m) => (m.id === messageId ? patch(m) : m)),
        }
  );

/* ───────────────────────── Reducer ───────────────────────── */

const initialState: ChatState = { conversations: [], activeId: null };

function reducer(state: ChatState, action: Action): ChatState {
  switch (action.type) {
    case 'SET_CONVERSATIONS': {
      const stillExists = action.payload.some((c) => c.id === state.activeId);
      return {
        conversations: action.payload,
        activeId: stillExists ? state.activeId : null,
      };
    }

    case 'SELECT_CHAT':
      return { ...state, activeId: action.payload };

    case 'NEW_CHAT': {
      const id = action.conversationId ?? makeId();
      const exists = state.conversations.some((c) => c.id === id);
      return {
        conversations: exists
          ? state.conversations
          : [emptyConversation(id), ...state.conversations],
        activeId: id,
      };
    }

    case 'ADD_MESSAGES':
      return {
        ...state,
        conversations: state.conversations.map((c) => {
          if (c.id !== action.conversationId) return c;
          const firstUser = [...c.messages, ...action.messages].find(
            (m) => m.role === 'user'
          );
          return {
            ...c,
            title: firstUser ? firstUser.content.slice(0, 40) : c.title,
            updatedAt: Date.now(),
            messages: [...c.messages, ...action.messages],
          };
        }),
      };

    case 'APPEND_TOKEN':
      return {
        ...state,
        conversations: updateMessage(
          state.conversations,
          action.conversationId,
          action.messageId,
          (m) => ({ ...m, content: m.content + action.text })
        ),
      };

    case 'SET_SOURCES':
      return {
        ...state,
        conversations: updateMessage(
          state.conversations,
          action.conversationId,
          action.messageId,
          (m) => ({ ...m, sources: action.sources })
        ),
      };

    case 'FINISH':
      return {
        ...state,
        conversations: updateMessage(
          state.conversations,
          action.conversationId,
          action.messageId,
          (m) => ({ ...m, streaming: false })
        ),
      };

    case 'FAIL':
      // keep any partial answer that already arrived, and attach the error
      return {
        ...state,
        conversations: updateMessage(
          state.conversations,
          action.conversationId,
          action.messageId,
          (m) => ({ ...m, streaming: false, error: action.error })
        ),
      };

    default:
      return state;
  }
}

/* ───────────────────────── SSE parser ───────────────────────── */

interface SSEEvent {
  event: string;
  data: any;
}

async function* readSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<SSEEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n/g, '\n');

      let idx: number;
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);

        let event = 'message';
        const dataLines: string[] = [];
        for (const line of raw.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
        }
        if (dataLines.length === 0) continue;

        try {
          yield { event, data: JSON.parse(dataLines.join('\n')) };
        } catch {
          // ignore malformed event
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/* ───────────────────────── Hook ───────────────────────── */

export function useChat(authUserId: string | null | undefined) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState('');
  const [searchQueries, setSearchQueries] = useState<string[]>([]);
  const streamAbortRef = useRef<AbortController | null>(null);

  /* ── Initialization: load history from the cloud (no local storage) ── */
  useEffect(() => {
    // stop any running answer when the user changes or logs out
    streamAbortRef.current?.abort();

    // Logged out: clear everything
    if (!authUserId) {
      dispatch({ type: 'SET_CONVERSATIONS', payload: [] });
      return;
    }

    const controller = new AbortController();

    const loadHistory = async () => {
      try {
        const res = await fetch(
          `${config.apiUrl}/api/history/${encodeURIComponent(authUserId)}`,
          { signal: controller.signal }
        );
        if (!res.ok) {
          throw new Error(`History request failed with status ${res.status}`);
        }

        const data: HistoryResponse = await res.json();
        const rows = Array.isArray(data.messages) ? data.messages : [];
        const messages = mapHistoryToMessages(rows);

        if (messages.length > 0) {
          const firstUser = messages.find((m) => m.role === 'user');
          const conversation: Conversation = {
            id: authUserId,
            title: firstUser ? firstUser.content.slice(0, 40) : 'New chat',
            messages,
            createdAt: messages[0].createdAt,
            updatedAt: messages[messages.length - 1].createdAt,
          };
          dispatch({ type: 'SET_CONVERSATIONS', payload: [conversation] });
          dispatch({ type: 'SELECT_CHAT', payload: authUserId });
        } else {
          // New user: empty history
          dispatch({ type: 'SET_CONVERSATIONS', payload: [] });
          dispatch({ type: 'NEW_CHAT', conversationId: authUserId });
        }
      } catch (err) {
        if (controller.signal.aborted) return;
        console.error('Could not load chat history:', err);
        // Fall back to a clean chat so the UI stays usable
        dispatch({ type: 'SET_CONVERSATIONS', payload: [] });
        dispatch({ type: 'NEW_CHAT', conversationId: authUserId });
      }
    };

    loadHistory();
    return () => controller.abort();
  }, [authUserId]);

  /* ── Send a message and stream the answer ── */
  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || !authUserId || isStreaming) return;

      const conversationId = state.activeId ?? authUserId;
      if (!state.conversations.some((c) => c.id === conversationId)) {
        dispatch({ type: 'NEW_CHAT', conversationId });
      }

      const now = Date.now();
      const userMsg: Message = {
        id: makeId(),
        role: 'user',
        content: trimmed,
        createdAt: now,
      };
      const assistantMsg: Message = {
        id: makeId(),
        role: 'assistant',
        content: '',
        createdAt: now + 1,
        streaming: true,
      };
      dispatch({
        type: 'ADD_MESSAGES',
        conversationId,
        messages: [userMsg, assistantMsg],
      });

      const controller = new AbortController();
      streamAbortRef.current = controller;
      setIsStreaming(true);
      setStatus('');
      setSearchQueries([]);

      const fail = (error: string) =>
        dispatch({ type: 'FAIL', conversationId, messageId: assistantMsg.id, error });

      try {
        const res = await fetch(`${config.apiUrl}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: trimmed,
            channel: 'web',
            session_id: authUserId,
            conversation_id: conversationId,
          }),
          signal: controller.signal,
        });

        if (!res.ok || !res.body) {
          let msg = `Request failed (${res.status})`;
          try {
            const j = await res.json();
            if (j && typeof j.error === 'string') msg = j.error;
          } catch {
            /* body was not JSON */
          }
          fail(msg);
          return;
        }

        let finished = false;
        for await (const evt of readSSE(res.body)) {
          switch (evt.event) {
            case 'status':
              setStatus(String(evt.data?.text ?? ''));
              break;
            case 'search':
              if (evt.data?.query) {
                setSearchQueries((q) => [...q, String(evt.data.query)]);
              }
              break;
            case 'sources':
              dispatch({
                type: 'SET_SOURCES',
                conversationId,
                messageId: assistantMsg.id,
                sources: Array.isArray(evt.data?.items) ? evt.data.items : [],
              });
              break;
            case 'token':
              dispatch({
                type: 'APPEND_TOKEN',
                conversationId,
                messageId: assistantMsg.id,
                text: String(evt.data?.text ?? ''),
              });
              break;
            case 'done':
              finished = true;
              dispatch({ type: 'FINISH', conversationId, messageId: assistantMsg.id });
              break;
            case 'error':
              finished = true;
              fail(String(evt.data?.text ?? 'Something went wrong. Please try again.'));
              break;
          }
        }

        // stream closed without done/error
        if (!finished) {
          dispatch({ type: 'FINISH', conversationId, messageId: assistantMsg.id });
        }
      } catch (err) {
        if (controller.signal.aborted) {
          // user pressed Stop: keep the partial answer
          dispatch({ type: 'FINISH', conversationId, messageId: assistantMsg.id });
        } else {
          console.error('Chat stream failed:', err);
          fail('Could not reach the server. Please check your connection.');
        }
      } finally {
        setIsStreaming(false);
        setStatus('');
        setSearchQueries([]);
        if (streamAbortRef.current === controller) streamAbortRef.current = null;
      }
    },
    [authUserId, isStreaming, state.activeId, state.conversations]
  );

  const stopStreaming = useCallback(() => {
    streamAbortRef.current?.abort();
  }, []);

  const newChat = useCallback(() => {
    if (!authUserId) return;
    dispatch({ type: 'NEW_CHAT', conversationId: authUserId });
  }, [authUserId]);

  const selectChat = useCallback((id: string) => {
    dispatch({ type: 'SELECT_CHAT', payload: id });
  }, []);

  const activeConversation =
    state.conversations.find((c) => c.id === state.activeId) ?? null;

  return {
    conversations: state.conversations,
    activeConversation,
    messages: activeConversation?.messages ?? [],
    status,
    searchQueries,
    isStreaming,
    sendMessage,
    stopStreaming,
    newChat,
    selectChat,
  };
}
